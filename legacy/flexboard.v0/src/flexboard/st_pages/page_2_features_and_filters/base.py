import typing as t
from abc import ABC, abstractmethod

import streamlit as st
from core.models import CategoricalFilter, Feature, NumericFilter


class BaseUI(ABC):
    SESSION_KEY: str
    HEADER: str
    CONFIGS: dict[str, dict[str, t.Any]]

    def __init__(self) -> None:
        st.session_state.setdefault(self.SESSION_KEY, [])

    def render(self) -> None:
        st.header(self.HEADER, divider="gray", anchor=False)
        self._render_item_controls()
        with st.container(border=True):
            self._render_form()

    def _render_item_controls(self) -> None:
        """Render toggle and delete controls for each item"""
        with st.container(border=True):
            # Get items from benchmark state instead of session state
            data_manager = self.data_manager  # Assuming this is set in child classes
            items = getattr(data_manager.current_state, self.SESSION_KEY)

            for idx, item in enumerate(items):
                st_cols = st.columns([1, 5, 2, 1, 1], vertical_alignment="center")
                with st_cols[0]:
                    st.metric("_", f"#{idx + 1}", label_visibility="collapsed")
                with st_cols[1]:
                    st.text(self.get_item_label(item))
                with st_cols[2]:
                    st.popover("Details").write(item)
                with st_cols[3]:
                    toggle_key = f"{self.SESSION_KEY}.{idx}.toggle"
                    new_state = st.toggle(
                        ("Enabled" if st.session_state.get(toggle_key) else "Disabled"),
                        value=st.session_state.get(toggle_key, True),
                        key=toggle_key,
                    )
                    if new_state != item.enabled:
                        item.enabled = new_state
                        data_manager._active_df = None  # Force recompute
                        st.rerun()
                with st_cols[4]:
                    if st.button(
                        "",
                        key=f"{self.SESSION_KEY}.{idx}.delete",
                        type="secondary",
                        icon="❌",
                        use_container_width=False,
                    ):
                        items.pop(idx)
                        data_manager.invalidate_cache()
                        st.rerun()

    def _has_duplicate(
        self, new_item: CategoricalFilter | NumericFilter | Feature
    ) -> bool:
        """Check if an item already exists in benchmark state"""
        data_manager = self.data_manager
        existing_items = getattr(data_manager.current_state, self.SESSION_KEY)
        return any(item == new_item for item in existing_items)

    @abstractmethod
    def _render_form(self) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_item_label(item: t.Any) -> str:
        raise NotImplementedError
