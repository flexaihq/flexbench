import streamlit as st

from flexboard.plots import metric_comparison_bar_plot
from flexboard.processor import DataProcessor

st.title("Performance Analysis", anchor=False)

processor: DataProcessor = st.session_state["processor"]
df = processor.df

# Sort models by number of rows (runs)
model_counts = {m: int(df.filter(df["model.name"] == m).height) for m in df["model.name"].unique()}
models = sorted(model_counts.keys(), key=lambda m: model_counts[m], reverse=True)

submitters = sorted(df["submission.organization"].unique())
accelerators = sorted(df["system.accelerator.name"].unique())

# def model_format_func(m: str) -> str:
#     return f"{m} ({model_counts[m]})"

selected_model = st.selectbox(
    "Select Model",
    models,
    index=0 if models else None,
    format_func=lambda m: f"{m} ({model_counts[m]} runs)",
)
selected_submitters = st.multiselect("Select Submitters", submitters, default=submitters)
selected_accelerators = st.multiselect("Select Accelerators", accelerators, default=accelerators)

if not selected_model:
    st.warning("Please select a model.")
    st.stop()

filtered_df = df.filter(
    (df["model.name"] == selected_model)
    & (df["submission.organization"].is_in(selected_submitters))
    & (df["system.accelerator.name"].is_in(selected_accelerators))
)

if filtered_df.height == 0:
    st.warning("No data available for the selected combination.")
    st.stop()

st.subheader("Performance Bar Plots", anchor=False, divider="gray")
figs = metric_comparison_bar_plot(filtered_df, selected_model)
for i, fig in enumerate(figs):
    st.plotly_chart(fig, use_container_width=True)
    if i < len(figs) - 1:
        st.divider()
