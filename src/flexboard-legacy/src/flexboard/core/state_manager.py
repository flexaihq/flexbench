import fnmatch
import json
import typing as t
from dataclasses import asdict, is_dataclass

import streamlit as st
from core.models import CategoricalFilter, Column, Feature, NumericFilter


class StateManager:
    """Manages application state serialization and persistence"""

    def __init__(self):
        self.session_key = "state"
        if "_benchmark_states" not in st.session_state:
            st.session_state._benchmark_states = {}

    def render(self) -> None:
        """Render state management UI"""
        with st.container(border=True):
            st.header("State Management", divider="gray", anchor=False)
            st.caption("Download and upload the current state of the app.")

            download_tab, upload_tab = st.tabs(["Download State", "Upload State"])

            with download_tab:
                state_json = json.dumps(
                    self.download_state(
                        exclude=["*.*.delete", "*.*.toggle", "*.*.reset", "_*"]
                    ),
                    indent=2,
                )
                st.download_button(
                    "Download Current State",
                    data=state_json,
                    file_name="flexboard_state.json",
                    mime="application/json",
                    use_container_width=True,
                )

            with upload_tab:
                st.file_uploader(
                    "Upload State",
                    type="json",
                    help="Upload a previously saved state file",
                    label_visibility="collapsed",
                    key=f"_upload_{st.session_state.setdefault('_uploader_key', 0)}",
                    on_change=lambda: setattr(
                        # needed to avoid infinite usage of uploaded file
                        st.session_state,
                        "_uploader_key",
                        st.session_state["_uploader_key"] + 1,
                    ),
                )
                if (
                    uploaded_file := st.session_state.get(
                        f"_upload_{st.session_state['_uploader_key'] - 1}"
                    )
                ) is not None:
                    state_dict = json.load(uploaded_file)
                    self.upload_state(state_dict)
                    st.success("State loaded successfully!")

    def download_state(self, exclude: list[str] | None = None) -> dict[str, t.Any]:
        """Export current session state as a downloadable dict"""
        encoded = {}
        exclude = exclude or []

        for key, value in st.session_state.items():
            if any(fnmatch.fnmatch(key, pattern) for pattern in exclude):
                continue

            if isinstance(value, list):
                encoded[key] = [
                    asdict(item) if is_dataclass(item) else item for item in value
                ]
            elif isinstance(value, dict) and key == "columns":
                encoded[key] = {k: asdict(v) for k, v in value.items()}
            elif is_dataclass(value):
                encoded[key] = asdict(value)
            else:
                encoded[key] = value

        return encoded

    def upload_state(self, state_dict: dict[str, t.Any]) -> None:
        """Update session state from uploaded dict"""
        decoded = {}

        for key, value in state_dict.items():
            if not isinstance(value, (dict, list)):
                decoded[key] = value
                continue

            if key == "features":
                decoded[key] = [Feature(**item) for item in value]
            elif key == "filters":
                decoded[key] = [
                    (
                        NumericFilter(**item)
                        if item["type"] == "Numerical"
                        else CategoricalFilter(**item)
                    )
                    for item in value
                ]
            elif key == "columns":
                decoded[key] = {name: Column(**data) for name, data in value.items()}
            else:
                decoded[key] = value

        for key in list(st.session_state.keys()):
            del st.session_state[key]

        for key, value in decoded.items():
            st.session_state[key] = value
