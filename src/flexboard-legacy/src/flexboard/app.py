import argparse
import os

import streamlit as st
from core.data_manager import DataManager
from utils.styling import streamlit_styling

parser = argparse.ArgumentParser(description="FlexBoard dashboard configuration")
parser.add_argument("--disable-pages", nargs="+", help="List of pages to disable")
args = parser.parse_args()
disable_pages = args.disable_pages or []

st.set_page_config(
    page_title="FlexBoard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

streamlit_styling()

all_pages = [
    {
        "page": "st_pages/page_1_home/_page.py",
        "title": "Home",
        "icon": "🏠",
        "url_path": "home",
        "default": True,
    },
    # {
    #     "page": "st_pages/page_2_features_and_filters/_page.py",
    #     "title": "Add features & filters",
    #     "url_path": "customize-data",
    #     "icon": "⚙️",
    # },
    {
        "page": "st_pages/page_3_dashboard/_page.py",
        "title": "Universal dashboard",
        "url_path": "dashboard",
        "icon": "📊",
    },
    # {
    #     "page": "st_pages/page_4_comparison/_page_performance.py",
    #     "title": "Comparison - Performance",
    #     "url_path": "comparison-performance",
    #     "icon": "🚀",
    # },
    # {
    #     "page": "st_pages/page_4_comparison/_page_cost.py",
    #     "title": "Comparison - Cost",
    #     "url_path": "comparison-cost",
    #     "icon": "💰",
    # },
    {
        "page": "st_pages/page_5_cost_efficiency/_page.py",
        "title": "Cost Efficiency Analysis",
        "url_path": "cost-efficiency",
        "icon": "⚖️",
    },
]

pg = st.navigation(
    [st.Page(**page) for page in all_pages if page["url_path"] not in disable_pages]
)

with st.sidebar:
    data_manager: DataManager = st.session_state.setdefault(
        "data_manager", DataManager()
    )

    sources_container = st.container(border=True)
    sources_container.subheader("Data Sources", divider="gray")

    refresh_button = sources_container.button(
        "Refresh Data",
        help="Clear and reload data from all sources",
        on_click=lambda: (
            data_manager.fetch_data_from_source.clear(),
            data_manager.remove_source("all"),
            data_manager.update_source("all"),
        ),
        use_container_width=True,
    )

    use_cmx = sources_container.toggle(
        "Use CMX",
        value=st.session_state.get("data.use_cmx", os.getenv("CMX_FLEXBOARD_USE_CMX", True)),
        key="data.use_cmx",
        help="Enable CMX as data source",
        on_change=lambda: data_manager.update_source("cmx"),
    )
    if st.session_state.get("data.use_cmx") and "cmx" not in data_manager.source_data:
        # for 1st page load (cmx is enabled but source is updated on_change only)
        data_manager.add_source("cmx")

    use_database = sources_container.toggle(
        "Use Database (WIP)",
        value=st.session_state.get("data.use_database", False),
        key="data.use_database",
        help="Enable PostgreSQL database as data source",
        on_change=lambda: data_manager.update_source("database"),
        disabled=True,
    )

    use_local = sources_container.toggle(
        "Use Local Data",
        value=st.session_state.get("data.use_local", False),
        key="data.use_local",
        help="Enable local JSON file(s) as data source",
        on_change=lambda: (
            data_manager.fetch_data_from_source.clear(),  # TODO: not clear all cache
            data_manager.update_source("local")
        ),
    )

    if use_local:
        files = sources_container.file_uploader(
            "Upload JSON File",
            type="json",
            help="Upload one or several JSON files",
            accept_multiple_files=True,
            key="data.use_local.files",
            on_change=lambda: (
                data_manager.fetch_data_from_source.clear(),  # TODO: not clear all cache
                data_manager.update_source("local")
            ),
        )
        if not files:
            sources_container.error("Please upload a JSON file")

    if not data_manager.source_data:
        sources_container.error("Please enable at least one data source")

    data_manager.process_data()

    if "prices.accelerator" in st.session_state and "prices.system" in st.session_state:
        data_manager.update_prices(
            st.session_state["prices.accelerator"], st.session_state["prices.system"]
        )

    filters_container = st.container(border=True)
    filters_container.subheader("Benchmark Selection", divider="gray")

    filters_container.info("Use filters that will be applied to all pages.")

    benchmark_name = filters_container.selectbox(
        "Benchmark",
        options=(
            sorted(
                data_manager.base_df["benchmark_name"].drop_nulls().unique().to_list()
            )
            if "benchmark_name" in data_manager.base_df.columns
            else []
        ),
        key="data.selected_benchmark_name",
        help="Choose which MLPerf benchmark to analyze.",
        index=1,  # mlperf-training
        on_change=lambda: data_manager.process_data(),
    )

    scenario = filters_container.selectbox(
        "Scenario",
        options=(
            sorted(data_manager.base_df["scenario"].drop_nulls().unique().to_list())
            if "scenario" in data_manager.base_df.columns
            else []
        ),
        key="data.selected_scenario",
        help="Choose which scenario to analyze",
        index=None,
        disabled=not benchmark_name,
        on_change=lambda: data_manager.process_data(),
    )

    version = filters_container.selectbox(
        "Version",
        options=(
            sorted(
                data_manager.base_df["benchmark_version"].drop_nulls().unique().to_list()
            )
            if "benchmark_version" in data_manager.base_df.columns
            else []
        ),
        key="data.selected_version",
        help="Choose which version to analyze",
        index=None,
        disabled=not benchmark_name,
        on_change=lambda: data_manager.process_data(),
    )

    metric = filters_container.selectbox(
        "Metric",
        options=(
            # TODO: rework this (not use base_df?)
            sorted(data_manager.base_df["units"].drop_nulls().unique().to_list() + ["TTA (min)"])
            if "units" in data_manager.base_df.columns
            else []
        ),
        key="data.selected_metric",
        help="Choose which metric to analyze",
        index=None,
        disabled=not benchmark_name,
        on_change=lambda: data_manager.process_data(),
    )

    data_manager.process_data()

    with st.expander("DEBUG"):
        st.caption("Current session state")
        st.json(dict(sorted(st.session_state.items(), key=lambda x: x[0])))

pg.run()
