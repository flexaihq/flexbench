import polars as pl
import streamlit as st
from core.data_manager import DataManager

st.title("FlexBoard: Universal MLPerf dashboard", anchor=False)
st.write(
    "FlexBoard provides a unified platform for analyzing MLPerf inference and training benchmarks. "
    "Compare performance across different AI systems, analyze cost-efficiency metrics, and make "
    "data-driven decisions by leveraging comprehensive MLPerf benchmark results."
)

data_manager: DataManager = st.session_state["data_manager"]

st.header("🎯 Key Features", anchor=False, divider="gray")
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
        ### MLPerf Data Integration
        - Consolidated view of MLPerf inference and training results
        - Performance analysis across diverse AI tasks and models
        - Standardized metrics for systematic comparisons
        """
    )
with col2:
    st.markdown(
        """
        ### Performance Analysis
        - Compare system efficiency across MLPerf benchmarks
        - Analyze cost-performance trade-offs
        - Track performance trends across MLPerf submissions
        """
    )

st.header("📊 Summary Metrics", divider="gray", anchor=False)

benchmark_name = st.session_state.get("data.selected_benchmark_name")
if benchmark_name:
    st.subheader(benchmark_name, anchor=False)

df = data_manager.active_df

if not df.is_empty():
    stats_cols = st.columns(4)
    try:
        stats_cols[0].metric("Systems", df["system.system_name"].n_unique())
        stats_cols[1].metric("Submitters", df["system.submitter"].n_unique())
        stats_cols[2].metric("Accelerators", df["accelerator_name"].n_unique())
        stats_cols[3].metric("Total Results", len(df))
    except pl.exceptions.ColumnNotFoundError as e:
        st.warning(f"Missing column: {e}")

st.header("🔍 Data Overview", divider="gray", anchor=False)
data_manager.render_dataframe(df)
