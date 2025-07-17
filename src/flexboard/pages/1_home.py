import streamlit as st

from flexboard.processor import DataProcessor

st.title("FlexBoard", anchor=False)
st.write(
    "FlexBoard helps you analyze MLPerf inference results and compare them to FlexBench runs. "
    "Focus on both inference speed (tokens/s) and accuracy (rouge, etc.) to ensure models are fast and correct."
)

processor: DataProcessor = st.session_state["processor"]
df = processor.df

st.header("Key Metrics", anchor=False, divider="gray")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Systems", df["system.name"].n_unique())
col2.metric("Submitters", df["submission.organization"].n_unique())
col3.metric("Accelerators", df["system.accelerator.name"].n_unique())
col4.metric("Results", len(df))
col5.metric("Total Accelerators", int(df["system.total_accelerators"].sum()))

st.header("Data Overview", anchor=False, divider="gray")
st.dataframe(
    df.select(
        [
            "benchmark.name",
            "benchmark.version",
            "model.name",
            "system.name",
            "system.accelerator.name",
            "system.total_accelerators",
            "result.tokens_per_second",
            "metrics.accuracy",
        ]
    ),
    use_container_width=True,
)
