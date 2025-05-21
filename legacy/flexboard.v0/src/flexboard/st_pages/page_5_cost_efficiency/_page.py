import json
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st
from core.data_manager import DataManager
from st_pages.page_4_comparison.plots_cost import create_system_cost_sensitivity_plot
from st_pages.page_5_cost_efficiency.plots import (
    create_fixed_result_cost_plot,
    create_training_cost_comparison_plot,
)
from streamlit import delta_generator

COLUMNS_TO_SHOW = [
    "benchmark_name",
    "benchmark_version_alias",
    "framework",
    "system_name",
    "accelerator_name",
    "number_of_nodes",
    "system.accelerators_per_node",
    "model_name",
    "result",
    "result_unit",
]


def get_valid_options(
    df: pl.DataFrame, selected_gpu: str = None, selected_nodes: int = None
) -> dict:
    """Get valid configuration options based on current selection"""
    filtered_df = df

    gpus = sorted(filtered_df["accelerator_name"].drop_nulls().unique().to_list())

    if selected_gpu:
        filtered_df = filtered_df.filter(pl.col("accelerator_name") == selected_gpu)
        nodes = sorted(filtered_df["number_of_nodes"].drop_nulls().unique().to_list())

        if selected_nodes is not None:
            filtered_df = filtered_df.filter(pl.col("number_of_nodes") == selected_nodes)
            accs = sorted(
                filtered_df["system.accelerators_per_node"]
                .drop_nulls()
                .unique()
                .to_list()
            )
        else:
            accs = []
    else:
        nodes = []
        accs = []

    return {
        "gpus": gpus,
        "nodes": nodes,
        "accs": accs,
    }


def select_system_hardware(
    container: delta_generator.DeltaGenerator,
    label: str,
    df: pl.DataFrame,
    key_prefix: str,
    disabled: bool = False,
    filter_opts: dict = None,
) -> pl.DataFrame:
    """Select system hardware configuration"""
    container.subheader(f"{label} System", anchor=False, divider="gray")

    available_gpus = (
        filter_opts["gpus"]
        if filter_opts
        else sorted(df["accelerator_name"].drop_nulls().unique().to_list())
    )
    gpu = container.selectbox(
        "Accelerator",
        options=[gpu for gpu in available_gpus if gpu != ""],
        key=f"{key_prefix}.gpu",
        disabled=disabled,
        on_change=lambda: (
            setattr(st.session_state, "versus.comp.gpu", None)
            if label == "Current"
            else None
        ),
        index=0 if label == "Current" else 1
    )

    if disabled or not gpu:
        return pl.DataFrame()

    filtered_df = df.filter(pl.col("accelerator_name") == gpu)

    slider_cols = container.columns(2)

    available_nodes = (
        filter_opts["nodes"]
        if filter_opts
        else sorted(filtered_df["number_of_nodes"].drop_nulls().unique().to_list())
    )
    nodes = slider_cols[0].select_slider(
        "Number of Nodes",
        options=prepare_slider_options(available_nodes, integer=True),
        key=f"{key_prefix}.nodes",
        disabled=disabled,
    )

    filtered_df = filtered_df.filter(pl.col("number_of_nodes") == nodes)

    available_accs = (
        filter_opts["accs"]
        if filter_opts
        else sorted(
            filtered_df["system.accelerators_per_node"].drop_nulls().unique().to_list()
        )
    )
    accs_per_node = slider_cols[1].select_slider(
        "Engines per Node",
        options=prepare_slider_options(available_accs, integer=True),
        key=f"{key_prefix}.accs_per_node",
        disabled=disabled,
    )

    filtered_df = filtered_df.filter(
        pl.col("system.accelerators_per_node") == accs_per_node
    )

    if filtered_df.is_empty():
        container.warning(
            f"No {label.lower()} systems found with selected configuration."
        )

    return filtered_df


def select_final_system(
    container: delta_generator.DeltaGenerator,
    label: str,
    df: pl.DataFrame,
    key_prefix: str,
) -> dict:
    """Show available systems and let user select one"""
    if df.is_empty():
        return {}

    if len(df) == 1:
        system_dict = {
            col: (df[0, col].item() if isinstance(df[0, col], pl.Series) else df[0, col])
            for col in sorted(df.columns)
        }
        container.info(
            f"Only one {label.lower()} system found with above configuration: `{system_dict['system_name']}`"
        )
        container.caption("MLPerf submission details:")
        container.json(system_dict, expanded=False)
        return system_dict

    results = df["result"]
    stats = {
        "min": results.min(),
        "max": results.max(),
        "avg": results.mean(),
        "med": results.median(),
        "count": len(results),
    }

    fig = go.Figure(data=[go.Histogram(x=results, nbinsx=20)])
    fig.update_layout(
        title=f"Result Distribution - All {label} Systems",
        xaxis_title=df["result_unit"].unique()[0],
        yaxis_title="Count",
        showlegend=False,
        height=300,
    )

    container.warning(
        f"Multiple (n={len(results)}) {label.lower()} systems found with above configuration. "
        "User can select a specific system for further analysis."
    )
    container.caption(
        f"Showing aggregated stats for all {len(results)} {label.lower()} systems:"
    )
    col1, col2 = container.columns([1, 2])
    col1.dataframe(stats, use_container_width=True)
    col2.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}.histogram")

    with container.expander("View All Matches", expanded=False):
        st.dataframe(df.select(COLUMNS_TO_SHOW), use_container_width=True)

    container.divider()

    selection_method = container.radio(
        "Selection Method",
        options=["By System", "By Result"],
        key=f"{key_prefix}.selection_method",
        horizontal=True,
    )

    system_dict = {}

    if selection_method == "By System":
        available_systems = sorted(df["system_name"].unique().to_list())
        selected_system = container.selectbox(
            f"Select {label} System",
            options=available_systems,
            key=f"{key_prefix}.system_final",
        )

        if not selected_system:
            return {}

        system_data = df.filter(pl.col("system_name") == selected_system)

    else:
        available_results = sorted(df["result"].unique().to_list())
        target_result = container.select_slider(
            f"Select Target {df['result_unit'].unique()[0]}",
            options=prepare_slider_options(available_results),
            key=f"{key_prefix}.result_final",
            format_func=lambda x: f"{x:.2f}",
        )

        system_data = df.filter(pl.col("result") == target_result)

        container.write(
            f"Selected system with result: `{system_data['system_name'][0]}` "
            f"({system_data['result'][0]:.4f})"
        )

    if not system_data.is_empty():
        system_dict = {
            col: (
                system_data[0, col].item()
                if isinstance(system_data[0, col], pl.Series)
                else system_data[0, col]
            )
            for col in sorted(system_data.columns)
        }

        container.caption("MLPerf submission details:")
        container.json(system_dict, expanded=False)

    return system_dict


def prepare_slider_options(options: list, integer: bool = False) -> list:
    """Prepare options for select_slider by ensuring at least 2 identical values"""
    if integer:
        options = [int(opt) if opt is not None else opt for opt in options]
    if len(options) == 0:
        return [0, 0]
    elif len(options) == 1:
        return [options[0], options[0]]
    return options


def show_model_metrics(
    container: delta_generator.DeltaGenerator, df: pl.DataFrame, label: str
) -> None:
    """Display available models and their metrics in a container"""
    container.caption(f"Available **{label.lower()}**1 models and their metrics:")
    for model in sorted(df["model_name"].drop_nulls().unique()):
        model_metrics = df.filter(pl.col("model_name") == model)["result_unit"].unique()
        container.markdown(
            f"- **{model}**  \n" f"  └─ Metrics: {', '.join(sorted(model_metrics))}"
        )


def get_default_prices() -> dict[str, float]:
    """Load default accelerator prices from JSON file"""
    with open(
        Path(__file__).parent.parent / "page_4_comparison/accelerator_prices.json"
    ) as f:
        return json.load(f)


st.title("Cost Efficiency Analysis", anchor=False)
st.caption(
    "Use this page to compare the performance and cost efficiency of two systems. "
    "The current system represent the one you want to evaluate, while the reference system is the one you want to compare against."
)

data_manager: DataManager = st.session_state["data_manager"]
df = data_manager.active_df

if df.is_empty():
    st.error("No data available for analysis.")
    st.stop()

left_col, right_col = st.columns(2)

curr_df = select_system_hardware(
    left_col.container(border=True), "Current", df, "versus.curr"
)

if curr_df.is_empty():
    st.stop()

st.subheader("Benchmark Configuration", anchor=False, divider="gray")
model_col, metric_col = st.columns(2)

available_models = sorted(curr_df["model_name"].drop_nulls().unique().to_list())
model = model_col.selectbox(
    "Model",
    options=available_models,
    key="versus.model",
    index=0,
)

curr_df = curr_df.filter(pl.col("model_name") == model)

available_metrics = sorted(curr_df["result_unit"].drop_nulls().unique().to_list())
metric = metric_col.selectbox(
    "Target Metric",
    options=available_metrics,
    key="versus.metric",
    index=0,
)

curr_df = curr_df.filter(pl.col("result_unit") == metric)

if all([model, metric]):
    filtered_df = df.filter(
        (pl.col("model_name") == model) & (pl.col("result_unit") == metric)
    )

    ref_df = select_system_hardware(
        right_col.container(border=True),
        "Reference",
        filtered_df,
        "versus.ref",
        disabled=False,
    )
else:
    ref_df = pl.DataFrame()
    right_col.warning("Please complete benchmark configuration first.")

if ref_df.is_empty():
    st.stop()

st.subheader("Available Systems", anchor=False, divider="gray")
left_col, right_col = st.columns(2)
curr_system = select_final_system(
    left_col.container(border=True), "Current", curr_df, "versus.curr"
)
ref_system = select_final_system(
    right_col.container(border=True), "Reference", ref_df, "versus.ref"
)

if not curr_system or not ref_system:
    st.stop()

st.subheader("System Pricing", anchor=False, divider="gray")
price_cols = st.columns(2)
left_price_col, right_price_col = price_cols[0].container(border=True), price_cols[
    1
].container(border=True)

default_prices = get_default_prices()

curr_acc_name = curr_system["accelerator_name"]
curr_acc_count = curr_system["accelerator_count"]
curr_acc_price = left_price_col.number_input(
    f"`{curr_acc_name}` GPU Price per Hour (USD)",
    min_value=0.01,
    value=st.session_state.get(
        f"versus.curr.price.{curr_acc_name}", default_prices.get(curr_acc_name)
    ),
    step=0.1,
    format="%.2f",
    key=f"versus.curr.price.{curr_acc_name}",
)
ref_acc_name = ref_system["accelerator_name"]
ref_acc_price = right_price_col.number_input(
    f"`{ref_acc_name}` GPU Price per Hour (USD)",
    min_value=0.01,
    value=st.session_state.get(
        f"versus.ref.price.{ref_acc_name}", default_prices.get(ref_acc_name)
    ),
    step=0.1,
    format="%.2f",
    key=f"versus.ref.price.{ref_acc_name}",
)

if not all([curr_acc_price, ref_acc_price]):
    st.warning("Please provide prices for both accelerators.")
    st.stop()

curr_system["system_hourly_price"] = curr_acc_price * curr_acc_count
left_price_col.caption(
    f"Total system cost: ${curr_system['system_hourly_price']:.2f}/hour ({curr_acc_count} GPUs)"
)
ref_acc_count = ref_system["accelerator_count"]

ref_system["system_hourly_price"] = ref_acc_price * ref_acc_count
right_price_col.caption(
    f"Total system cost: ${ref_system['system_hourly_price']:.2f}/hour ({ref_acc_count} GPUs)"
)

if curr_system["result_unit"] == "TTA (min)":
    st.subheader("Fixed Result Cost Analysis", anchor=False, divider="gray")
    fixed_result_fig = create_fixed_result_cost_plot(ref_system, curr_system)
    st.plotly_chart(fixed_result_fig, use_container_width=True)
    st.info(
        "Reading the Fixed Result Plot:\n"
        "- Each line shows how total training cost varies with hourly price\n"
        "- Stars mark actual system configurations\n"
        "- Systems with faster training times have flatter curves"
    )

    st.subheader("Training Cost Comparison", anchor=False, divider="gray")
    st.write("This plot uses default prices for accelerators.")
    all_systems_df = (
        filtered_df.filter(
            (pl.col("model_name") == model) & (pl.col("result_unit") == metric)
        )
        .with_columns(
            system_hourly_price=(
                pl.col("accelerator_count")
                * pl.col("accelerator_name").map_elements(
                    lambda x: default_prices.get(x), return_dtype=pl.Float64
                )
            )
        )
        .drop_nulls(subset=["system_hourly_price"])
        .unique(subset="system_name")
    )

    cost_comparison_fig = create_training_cost_comparison_plot(
        all_systems_df,
        reference_name=ref_system["system_name"],
        current_name=curr_system["system_name"],
    )
    st.plotly_chart(cost_comparison_fig, use_container_width=True)
    st.expander("View Data", expanded=False).dataframe(
        all_systems_df.select(COLUMNS_TO_SHOW)
    )

st.subheader("Cost Sensitivity Analysis", anchor=False, divider="gray")
fig = create_system_cost_sensitivity_plot(
    ref_system=ref_system,
    curr_system=curr_system,
    x_column="system_hourly_price",
    y_column="result",
    x_title="System Price per Hour (USD)",
    y_title=curr_system["result_unit"],
    color_title=f"Cost ({curr_system['cost_unit']})",
)

st.plotly_chart(fig, use_container_width=True, key="versus.cost_sensitivity")
st.info(
    "Reading the Comparison Plot:\n"
    "- **Green regions**: Configurations where the cost is lower than current\n"
    "- **Red regions**: Configurations where the cost is higher than current\n"
    "- **Black line**: Points where costs are equal\n"
    "- Use this plot to understand what price or performance improvements are needed for cost parity"
)
