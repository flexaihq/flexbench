import json

import streamlit as st

from flexboard.processor import DataProcessor

st.set_page_config(
    page_title="FlexBoard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "processor" not in st.session_state:
    st.session_state.processor = DataProcessor()
    st.session_state.processor.run()
processor = st.session_state.processor

# Accelerator color mapping (shared for all pages)
df = processor.df
gpu_names = sorted(df["system.accelerator.name"].drop_nulls().unique())
color_palette = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]
acc_color_mapping = {gpu: color_palette[i % len(color_palette)] for i, gpu in enumerate(gpu_names)}
st.session_state.acc_color_mapping = acc_color_mapping

with st.sidebar:
    with st.container(border=True):
        st.header("Accelerator Prices", anchor=False)
        st.caption(
            "Set the hourly price (USD/hr) for each accelerator type. "
            "This is used to calculate the cost per million tokens processed."
        )

        def reset_prices() -> None:
            for key in processor.price_mapping.keys():
                st.session_state.pop(f"price_{key}", None)

        st.button(
            "Reset to default",
            use_container_width=True,
            on_click=reset_prices,
            help="Default values can be found in [`accelerator_prices.json`]"
            "(https://github.com/flexaihq/flexbench/tree/main/src/flexboard/accelerator_prices.json)",
        )

        tabs = st.tabs(["Widgets", "JSON"])
        with tabs[0]:
            items = list(processor.price_mapping.items())
            for i in range(0, len(items), 3):
                cols = st.columns(3, vertical_alignment="bottom")
                for j, col in enumerate(cols):
                    if i + j < len(items):
                        key, value = items[i + j]
                        col.number_input(
                            key,
                            value=st.session_state.get(f"price_{key}", value),
                            step=0.01,
                            min_value=0.01,
                            key=f"price_{key}",
                            format="%.2f",
                        )
        with tabs[1]:
            default_json = json.dumps(
                {
                    key: round(st.session_state.get(f"price_{key}", 0.0), 2)
                    for key in processor.price_mapping.keys()
                },
                indent=2,
            )

            def sync_prices_tabs() -> None:
                try:
                    prices = json.loads(st.session_state["json_config"])
                    if not isinstance(prices, dict):
                        raise ValueError("JSON must be a dictionary")
                    for key, value in prices.items():
                        if f"price_{key}" in st.session_state:
                            st.session_state[f"price_{key}"] = float(value)
                except Exception as e:
                    tabs[1].error(f"Invalid JSON: {e}")

            json_config = st.text_area(
                "JSON Configuration",
                value=default_json,
                height=len(items) * 20,
                key="json_config",
                on_change=sync_prices_tabs,
            )
            st.download_button(
                "Download JSON",
                data=json_config,
                file_name="accelerator_prices.json",
                mime="application/json",
                use_container_width=True,
            )

pg = st.navigation(
    pages=[
        st.Page(
            "pages/1_home.py",
            title="Home",
            icon="🏠",
            default=True,
        ),
        st.Page(
            "pages/2_performance.py",
            title="Performance",
            # chart icon
            icon="📊",
        ),
        st.Page(
            "pages/3_cost.py",
            title="Cost",
            icon="💰",
        ),
    ],
    position="top",
)

pg.run()
