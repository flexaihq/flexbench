import streamlit as st

from flexboard.plots import (
    cost_breakdown_bar_plots,
    cost_vs_performance_scatter_plot,
    system_cost_sensitivity_plot,
)
from flexboard.processor import DataProcessor

processor: DataProcessor = st.session_state["processor"]
df = processor.df

gpu_names = sorted(df["system.accelerator.name"].drop_nulls().unique())

st.title("Cost Efficiency Analysis", anchor=False)

model_counts_df = df["model.name"].value_counts(sort=True)
models = [(row["model.name"], row["count"]) for row in model_counts_df.to_dicts()]
selected_model = st.selectbox(
    "Select Model",
    models,
    key="cost.model",
    format_func=lambda m: f"{m[0]} ({m[1]} runs)",
)
if not selected_model:
    st.warning("Please select a model.")
    st.stop()

df = df.filter(df["model.name"] == selected_model[0])

st.subheader("Cost Breakdown by System (Cost per 1M Tokens)", anchor=False, divider="gray")
for fig in cost_breakdown_bar_plots(df, selected_model[0]):
    st.plotly_chart(fig, use_container_width=True)
fig = cost_vs_performance_scatter_plot(df, selected_model[0])
st.plotly_chart(fig, use_container_width=True)

st.subheader(
    "Select GPUs and Accelerator Counts for Cost Sensitivity", anchor=False, divider="gray"
)
col1, col2 = st.columns(2)

col1.markdown("#### System A (Reference)")
gpu_a = col1.selectbox(
    "GPU A",
    gpu_names,
    key="cost.gpu_a",
    index=0,
    format_func=lambda gpu: f"{gpu} ({processor.price_mapping.get(gpu, 1.0)} USD/hr)",
)
col2.markdown("#### System B (Comparison)")
gpu_b = col2.selectbox(
    "GPU B",
    gpu_names,
    key="cost.gpu_b",
    index=1,
    format_func=lambda gpu: f"{gpu} ({processor.price_mapping.get(gpu, 1.0)} USD/hr)",
)

acc_counts_a = sorted(
    df.filter(df["system.accelerator.name"] == gpu_a)["system.total_accelerators"].unique()
)
acc_counts_b = sorted(
    df.filter(df["system.accelerator.name"] == gpu_b)["system.total_accelerators"].unique()
)

acc_count_a = col1.selectbox("Number of Accelerators (A)", acc_counts_a, key="cost.acc_count_a")
acc_count_b = col2.selectbox("Number of Accelerators (B)", acc_counts_b, key="cost.acc_count_b")

sys_a_df = df.filter(
    (df["system.accelerator.name"] == gpu_a) & (df["system.total_accelerators"] == acc_count_a)
)
sys_b_df = df.filter(
    (df["system.accelerator.name"] == gpu_b) & (df["system.total_accelerators"] == acc_count_b)
)

if sys_a_df.height == 0 or sys_b_df.height == 0:
    st.warning("No systems found for selected GPU and accelerator count.")
    st.stop()

idx_a = sys_a_df["result.tokens_per_second"].arg_sort(descending=True)[0]
idx_b = sys_b_df["result.tokens_per_second"].arg_sort(descending=True)[0]

system_a = {col: sys_a_df[idx_a, col] for col in sys_a_df.columns}
system_b = {col: sys_b_df[idx_b, col] for col in sys_b_df.columns}

tokens_a = system_a.get("result.tokens_per_second")
price_a = system_a.get("system.price_per_hour")
tokens_b = system_b.get("result.tokens_per_second")
price_b = system_b.get("system.price_per_hour")

tokens_str_a = f"{tokens_a:.2f}" if tokens_a is not None else "N/A"
price_str_a = f"${price_a:.2f}/hr" if price_a is not None else "N/A"
tokens_str_b = f"{tokens_b:.2f}" if tokens_b is not None else "N/A"
price_str_b = f"${price_b:.2f}/hr" if price_b is not None else "N/A"

col1_metrics = col1.columns(3)
col1_metrics[0].metric("Tokens/s", tokens_str_a)
col1_metrics[1].metric("Price/hr", price_str_a)
col1_metrics[2].metric(
    "Cost/M Tokens",
    f"${system_a.get('result.cost_per_million_tokens', float('nan')):.2f}"
    if system_a.get("result.cost_per_million_tokens") is not None
    else "N/A",
)
col1.json(system_a, expanded=False)

col2_metrics = col2.columns(3)
col2_metrics[0].metric("Tokens/s", tokens_str_b)
col2_metrics[1].metric("Price/hr", price_str_b)
col2_metrics[2].metric(
    "Cost/M Tokens",
    f"${system_b.get('result.cost_per_million_tokens', float('nan')):.2f}"
    if system_b.get("result.cost_per_million_tokens") is not None
    else "N/A",
)
col2.json(system_b, expanded=False)

st.info(
    "For each GPU and accelerator count, the best system (highest tokens/s) is selected for comparison."
)

st.subheader("Cost Sensitivity Plot", anchor=False, divider="gray")
fig = system_cost_sensitivity_plot(
    reference_system=system_a,
    comparison_system=system_b,
    x_column="system.price_per_hour",
    y_column="result.tokens_per_second",
    x_title="System Price per Hour (USD)",
    y_title="Tokens/s",
    color_title="Cost (USD)",
)
st.plotly_chart(fig, use_container_width=True)
