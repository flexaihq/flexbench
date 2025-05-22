import streamlit as st
from core.data_manager import DataManager
from core.models import Feature
from st_pages.page_2_features_and_filters.base import BaseUI


class FeaturesUI(BaseUI):
    def __init__(self, data_manager: DataManager):
        super().__init__()
        self.data_manager = data_manager

    SESSION_KEY = "features"
    HEADER = "Features"
    CONFIGS = {
        "Numerical": {
            "methods": ["add", "sub", "mul", "truediv"],
            "column_type": "Numerical",
        },
        "Categorical": {"methods": ["concat"], "column_type": "Categorical"},
    }

    @staticmethod
    def get_item_label(feature: Feature) -> str:
        return feature.name

    def _render_form(self) -> None:
        st_cols = st.columns(
            [1 / 10, 3 / 10, 2 / 10, 4 / 10], vertical_alignment="center"
        )

        feature_type = st_cols[0].radio(
            "Feature type",
            self.CONFIGS,
            index=(
                list(self.CONFIGS.keys()).index(
                    st.session_state.get(f"{self.SESSION_KEY}.type")
                )
                if st.session_state.get(f"{self.SESSION_KEY}.type")
                else 0
            ),
            key=f"{self.SESSION_KEY}.type",
        )
        config = self.CONFIGS[feature_type]

        available_cols = self.data_manager.get_columns(type=config["column_type"])
        col_a = st_cols[1].selectbox(
            "Column A",
            available_cols,
            index=(
                available_cols.index(st.session_state.get(f"{self.SESSION_KEY}.col_a"))
                if st.session_state.get(f"{self.SESSION_KEY}.col_a") in available_cols
                else None
            ),
            key=f"{self.SESSION_KEY}.col_a",
        )
        col_b = st_cols[1].selectbox(
            "Column B",
            available_cols,
            index=(
                available_cols.index(st.session_state.get(f"{self.SESSION_KEY}.col_b"))
                if st.session_state.get(f"{self.SESSION_KEY}.col_b")
                else None
            ),
            key=f"{self.SESSION_KEY}.col_b",
        )

        operator = st_cols[2].selectbox(
            "Method",
            config["methods"],
            index=(
                config["methods"].index(
                    st.session_state.get(f"{self.SESSION_KEY}.operator")
                )
                if st.session_state.get(f"{self.SESSION_KEY}.operator")
                in config["methods"]
                else None
            ),
            key=f"{self.SESSION_KEY}.operator",
        )

        suggested_name = (
            f"{operator or '<operator>'}({col_a or '<col_a>'},{col_b or '<col_b>'})"
        )
        name = st_cols[3].text_input("Name", value=suggested_name)

        temp_feature = Feature(
            type=feature_type,
            col_a=col_a,
            col_b=col_b,
            operator=operator,
            name=name,
        )

        button_disabled = not all([col_a, col_b, operator, name])
        if not button_disabled and self._has_duplicate(temp_feature):
            st.warning("A feature with these parameters already exists!")
            button_disabled = True

        if st.button(
            f"Add {feature_type} Feature",
            use_container_width=True,
            disabled=button_disabled,
            type="primary",
        ):
            # Store in benchmark-specific state
            self.data_manager.current_state.features.append(temp_feature)
            # self.data_manager.invalidate_cache()  # Add this line
            for key in [
                f"{self.SESSION_KEY}.type",
                f"{self.SESSION_KEY}.col_a",
                f"{self.SESSION_KEY}.col_b",
                f"{self.SESSION_KEY}.operator",
            ]:
                st.session_state.pop(key, None)
            st.rerun()
