import polars as pl
import streamlit as st
from core.data_manager import DataManager
from st_pages.page_4_comparison.plots_performance import metric_comparison_bar_plot
from st_pages.page_4_comparison.selection import get_selection
from st_pages.page_4_comparison.utils import create_color_mapping

st.title("Performance Analysis", anchor=False)
st.write("Compare and analyze performance metrics across different accelerators.")

data_manager: DataManager = st.session_state["data_manager"]
df = data_manager.active_df

if df.is_empty():
    st.error("No data available for analysis.")
    st.stop()

acc_color_mapping = create_color_mapping(
    sorted(df["accelerator_name"].drop_nulls().unique().to_list())
)

selection = get_selection(df, mode="performance")

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
        "system_name",
        "result",
        "result_unit",
        "result_norm",
        "result_norm_unit",
        "color",
    )
    .unique()
    .sort(
        "accelerator_name", "system_name", "benchmark_version", "result_unit", "result"
    )
)

st.subheader("Data", anchor=False, divider="gray")
with st.expander(label="Click to view data", expanded=False, icon="🧮"):
    data_manager.render_dataframe(plot_df)

plot_df = plot_df.to_pandas()

st.subheader("Performance Plots", anchor=False, divider="gray")
show_norm = st.radio(
    "Metric Type",
    ["Base Metrics", "Normalized Metrics"],
    horizontal=True,
)

st.subheader("Bar Plots", anchor=False, divider="gray")
figs = metric_comparison_bar_plot(plot_df, selection.model, show_norm)
for i, fig in enumerate(figs):
    st.plotly_chart(fig, use_container_width=True)
    if i < len(figs) - 1:
        st.divider()
st.info("Compare base or normalized metrics across systems")
