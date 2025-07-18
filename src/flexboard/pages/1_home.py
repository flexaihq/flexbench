import streamlit as st

from flexboard.processor import DataProcessor

st.title("FlexBoard :chart_with_upwards_trend:", anchor=False)
st.markdown(
    """
    <div style='font-size:1.2em; margin-bottom:1em;'>
        FlexBoard helps you analyze <b>MLPerf inference results</b> and compare them to <b>FlexBench</b> runs.<br>
        Focus on both <span style='color:#636EFA'><b>inference speed</b> (tokens/s)</span> and <span style='color:#EF553B'><b>accuracy</b> (rouge, etc.)</span> to ensure models are fast and correct.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style='background:linear-gradient(90deg,#636EFA10,#EF553B10);padding:0.7em 1em;border-radius:8px;margin-bottom:1em;'>
        <b>Creators:</b> <a href='https://www.linkedin.com/in/daltunay' target='_blank'>Daniel Altunay</a> &amp; <a href='https://cKnowledge.org/gfursin' target='_blank'>Grigori Fursin</a> (FCS Labs)
    </div>
    """,
    unsafe_allow_html=True,
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
st.caption(
    "These metrics are based on https://huggingface.co/datasets/ctuning/OpenMLPerf dataset."
    " Future version will also include FlexBench runs for more comprehensive analysis."
)

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

st.markdown(
    """
    <div style='margin-top:2em;font-size:0.95em;color:#888;'>
        <b>Tip:</b> Use the sidebar to adjust accelerator prices and explore cost efficiency.<br>
        For more info, see <a href='https://github.com/flexaihq/flexbench' target='_blank'>FlexBench on GitHub</a>.
    </div>
    """,
    unsafe_allow_html=True,
)
