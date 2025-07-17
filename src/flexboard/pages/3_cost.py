import streamlit as st

from flexboard.processor import DataProcessor

st.title("Cost Analysis", anchor=False)

processor: DataProcessor = st.session_state["processor"]
