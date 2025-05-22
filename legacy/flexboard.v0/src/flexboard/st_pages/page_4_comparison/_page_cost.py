import polars as pl
import streamlit as st
from core.data_manager import DataManager
from st_pages.page_4_comparison.plots_cost import (
    create_cost_breakdown_plots,
    create_cost_vs_performance_plots,
    create_system_cost_sensitivity_plot,
)
from st_pages.page_4_comparison.selection import get_selection
from st_pages.page_4_comparison.utils import create_color_mapping

st.title("Cost Analysis", anchor=False)
st.write("Compare and analyze cost metrics across different accelerators.")

data_manager: DataManager = st.session_state["data_manager"]
df = data_manager.active_df

if df.is_empty():
    st.error("No data available for analysis.")
    st.stop()

acc_color_mapping = create_color_mapping(
    sorted(df["accelerator_name"].drop_nulls().unique().to_list())
)

selection = get_selection(df, mode="cost")

data_manager.update_prices(selection.accelerator_prices, selection.system_prices)

df = data_manager.active_df


if not all([selection.accelerators, selection.model, selection.systems]):
    st.warning("Please select at least one accelerator with its systems.")
    st.stop()

filtered_df = df.filter(
    (pl.col("model_name") == selection.model)
    & (pl.col("submitter").is_in(selection.submitters))
    & (pl.col("accelerator_name").is_in(selection.accelerators))
    & (pl.col("system_name").is_in(selection.systems))
).with_columns(
    color=pl.col("accelerator_name").map_elements(
        lambda x: acc_color_mapping.get(x), return_dtype=pl.Utf8
    )
)

if filtered_df.is_empty():
    st.warning("No data available for the selected combination.")
    st.stop()

plot_df = (
    filtered_df.select(
        "benchmark_version",
        "benchmark_name",
        "scenario",
        "model_name",
        "submitter",
        "accelerator_name",
        "accelerator_count",
        "accelerator_hourly_price",
        "system_name",
        "result",
        "result_unit",
        "result_norm",
        "result_norm_unit",
        "cost",
        "cost_unit",
        "system_hourly_price",
        "color",
    )
    .unique()
    .sort("accelerator_name", "system_name", "benchmark_version", "cost_unit", "cost")
)

st.subheader("Data", anchor=False, divider="gray")
with st.expander(label="Click to view data", expanded=False, icon="🧮"):
    data_manager.render_dataframe(plot_df)

plot_df = plot_df.to_pandas()

st_tabs = st.tabs(["General comparison", "Compare 2 systems"])

with st_tabs[0]:
    st.subheader("Cost Breakdown", anchor=False, divider="gray")
    fig = create_cost_breakdown_plots(plot_df, selection.model, acc_color_mapping)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Compare system costs across different scenarios and metrics")

    st.subheader("Cost vs Performance", anchor=False, divider="gray")
    figs = create_cost_vs_performance_plots(plot_df, selection.model, acc_color_mapping)
    for i, fig in enumerate(figs):
        st.plotly_chart(fig, use_container_width=True)
        if i < len(figs) - 1:
            st.divider()
    st.info(
        "Compare cost efficiency and performance trade-offs across scenarios and metrics"
    )

with st_tabs[1]:
    st.subheader("System Cost Sensitivity Analysis", anchor=False, divider="gray")

    show_norm = st.toggle(
        "Use Normalized Metrics",
        help="Toggle between base and normalized metrics",
        key="comparison.cost.sensitivity.use_norm",
    )

    result_col = "result_norm" if show_norm else "result"
    unit_col = "result_norm_unit" if show_norm else "result_unit"

    available_benchmarks = sorted(plot_df["benchmark_name"].unique())
    cont1 = st.container(border=True)
    cont2 = st.container(border=True)

    selected_benchmark = cont1.selectbox(
        "Select Benchmark",
        options=available_benchmarks,
        key="comparison.cost.sensitivity.benchmark",
        index=0,
    )

    if selected_benchmark:
        benchmark_df = plot_df[plot_df["benchmark_name"] == selected_benchmark]
        available_scenarios = sorted(benchmark_df["scenario"].unique())
        selected_scenario = cont1.selectbox(
            "Select Scenario",
            options=available_scenarios,
            key="comparison.cost.sensitivity.scenario",
            index=0,
        )

        if selected_scenario or available_scenarios == [None]:
            scenario_df = (
                benchmark_df[benchmark_df["scenario"] == selected_scenario]
                if selected_scenario
                else benchmark_df
            )
            available_metrics = sorted(scenario_df[unit_col].unique())
            selected_metric = cont1.selectbox(
                "Select Metric",
                options=available_metrics,
                key="comparison.cost.sensitivity.metric",
                index=0,
            )

            if selected_metric:
                valid_systems = scenario_df[
                    scenario_df[unit_col] == selected_metric
                ].dropna(subset=["system_hourly_price"])

                if len(valid_systems["system_name"].unique()) < 2:
                    st.error("Need at least 2 different systems to compare.")
                    st.stop()

                available_systems = sorted(valid_systems["system_name"].unique())

                cont2.write("**System selection**")
                col1, col2 = cont2.columns(2)
                curr_system_name = col1.selectbox(
                    "Current System",
                    options=available_systems,
                    help="Select the primary system to compare against the reference",
                    key="comparison.cost.sensitivity.curr_system",
                    index=(
                        available_systems.index(
                            st.session_state.get(
                                "comparison.cost.sensitivity.curr_system"
                            )
                        )
                        if st.session_state.get(
                            "comparison.cost.sensitivity.curr_system"
                        )
                        in available_systems
                        else None
                    ),
                )

                ref_system_name = col2.selectbox(
                    "Reference System",
                    options=available_systems,
                    help="Select the reference system for comparison",
                    key="comparison.cost.sensitivity.ref_system",
                    index=(
                        available_systems.index(
                            st.session_state.get(
                                "comparison.cost.sensitivity.ref_system"
                            )
                        )
                        if st.session_state.get("comparison.cost.sensitivity.ref_system")
                        in available_systems
                        else None
                    ),
                )

                if curr_system_name and ref_system_name:
                    if curr_system_name == ref_system_name:
                        st.warning("Please select two different systems.")
                        st.stop()

                    comp_candidates = valid_systems[
                        valid_systems["system_name"] == curr_system_name
                    ]
                    ref_candidates = valid_systems[
                        valid_systems["system_name"] == ref_system_name
                    ]

                    if len(comp_candidates) > 1:
                        col1.warning(
                            f"Multiple matches found for comparison system '{curr_system_name}'. Using first match."
                        )
                        col1.caption("Available matches for comparison system:")
                        col1.dataframe(comp_candidates, use_container_width=True)

                    if len(ref_candidates) > 1:
                        col2.warning(
                            f"Multiple matches found for reference system '{ref_system_name}'. Using first match."
                        )
                        col2.caption("Available matches for reference system:")
                        col2.dataframe(ref_candidates, use_container_width=True)

                    curr_system = comp_candidates.iloc[0].to_dict()
                    ref_system = ref_candidates.iloc[0].to_dict()

                    col1, col2 = cont2.columns(2)
                    col1.write(f"Selected System: `{curr_system_name}`")
                    col1.json(curr_system)
                    col2.write(f"Selected System: `{ref_system_name}`")
                    col2.json(ref_system)

                    matching_systems = valid_systems[
                        ~valid_systems["system_name"].isin(
                            [curr_system_name, ref_system_name]
                        )
                    ]

                    additional_systems = []
                    if not matching_systems.empty:
                        selected_additional = cont2.multiselect(
                            "Additional Systems (optional)",
                            options=sorted(matching_systems["system_name"].unique()),
                            help="Select additional systems to include in the comparison",
                            key="comparison.cost.sensitivity.additional_systems",
                        )
                        additional_systems = [
                            matching_systems[matching_systems["system_name"] == name]
                            .iloc[0]
                            .to_dict()
                            for name in selected_additional
                        ]

                    fig = create_system_cost_sensitivity_plot(
                        ref_system=ref_system,
                        curr_system=curr_system,
                        x_column="system_hourly_price",
                        y_column=result_col,
                        x_title="System Price per Hour (USD)",
                        y_title=ref_system[unit_col],
                        color_title=f"Cost ({ref_system['cost_unit']})",
                        additional_systems=additional_systems,
                    )

                    fig.update_layout(
                        title=dict(
                            text=(
                                f"System Comparison: {curr_system['system_name']} vs. {ref_system['system_name']}<br>"
                                f"Cost Metric: {ref_system['cost_unit']}"
                            )
                        )
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.info(
                        "Reading the Comparison Plot:\n"
                        "- **Green regions**: Configurations where the cost is lower than reference\n"
                        "- **Red regions**: Configurations where the cost is higher than reference\n"
                        "- **Black line**: Points where costs are equal\n"
                        "- Use this plot to understand what price or performance improvements are needed for cost parity"
                    )
