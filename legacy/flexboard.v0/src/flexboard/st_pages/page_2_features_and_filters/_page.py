import streamlit as st
from core.data_manager import DataManager
from st_pages.page_2_features_and_filters.features import FeaturesUI
from st_pages.page_2_features_and_filters.filters import FiltersUI

st.title("Customize Data", anchor=False)
st.write("Create features and apply filters to the data.")

data_manager: DataManager = st.session_state["data_manager"]

st.header("Preprocessing", anchor=False, divider="gray")
st.markdown(
    """
Rules:
- unify all frequency columns to Hertz unit (ex: `"1980MHz"` to `1,980,000,000`)
- unify all memory columns to bytes unit
- convert to correct data types (ex: `"1"` to `1`)
- merge same values with different cases (ex: `"HBM3E"` and `"HBM3e"`)
"""
)

features_ui = FeaturesUI(data_manager=data_manager)
features_ui.render()

filters_ui = FiltersUI(data_manager=data_manager)
filters_ui.render()

df = data_manager.active_df

if not df.is_empty():
    data_manager.render_dataframe(df)
