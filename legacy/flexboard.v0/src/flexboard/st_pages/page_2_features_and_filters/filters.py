import streamlit as st
from core.data_manager import DataManager
from core.models import CategoricalFilter, NumericFilter
from st_pages.page_2_features_and_filters.base import BaseUI


class FiltersUI(BaseUI):
    def __init__(self, data_manager: DataManager):
        super().__init__()
        self.data_manager = data_manager

    SESSION_KEY = "filters"
    HEADER = "Filters"
    CONFIGS = {
        "Numerical": {
            "filterclass": NumericFilter,
            "column_type": "Numerical",
        },
        "Categorical": {
            "filterclass": CategoricalFilter,
            "column_type": "Categorical",
        },
    }

    @staticmethod
    def get_item_label(filter: CategoricalFilter | NumericFilter) -> str:
        match filter.type:
            case "Numerical":
                return f"{filter.column} in {filter.range}"
            case "Categorical":
                return f"{filter.column} in {filter.values}"
            case _:
                raise ValueError(f"Unknown filter type: {filter.type}")

    def _render_form(self) -> None:
        st_cols = st.columns([1 / 10, 3 / 10, 6 / 10], vertical_alignment="center")

        filtertype = st_cols[0].radio("Filter type", self.CONFIGS)
        config = self.CONFIGS[filtertype]

        available_cols = self.data_manager.get_columns(type=config["column_type"])

        existing_filters = st.session_state.get(self.SESSION_KEY, [])
        available_cols = [
            col
            for col in available_cols
            if not any(f.column == col for f in existing_filters)
        ]

        column = st_cols[1].selectbox(
            "Column",
            available_cols,
            index=(
                available_cols.index(st.session_state.get(f"{self.SESSION_KEY}.col"))
                if st.session_state.get(f"{self.SESSION_KEY}.col") in available_cols
                else None
            ),
            key=f"{self.SESSION_KEY}.col",
        )

        filtervalue = None
        if isinstance(column, str):
            if config["column_type"] == "Numerical":
                min_val, max_val = self.data_manager.get_column_range(column)
                if min_val != max_val:
                    min_val = min_val[0] if isinstance(min_val, list) else min_val
                    max_val = max_val[0] if isinstance(max_val, list) else max_val
                    filtervalue = st_cols[2].slider(
                        "Range",
                        min_val,
                        max_val,
                        (min_val, max_val),
                        key=f"{self.SESSION_KEY}.range",
                    )
                else:
                    st_cols[2].warning(
                        f"Column '{column}' has only one unique value ({min_val})."
                    )
            elif config["column_type"] == "Categorical":
                values = self.data_manager.get_column_values(column)
                if not values:
                    st_cols[2].warning(f"Column '{column}' is always null.")
                elif len(values) == 1:
                    st_cols[2].warning(
                        f"Column '{column}' has only one unique value ({values[0]})."
                    )
                else:
                    filtervalue = st_cols[2].multiselect(
                        "Values",
                        values,
                        default=values,
                        key=f"{self.SESSION_KEY}.values",
                    )
        else:
            st_cols[2].info("Please select a column.")

        if st.button(
            f"Add {filtertype} Filter",
            use_container_width=True,
            disabled=not all([column, filtervalue]),
            type="primary",
        ):
            filterkey = "range" if config["column_type"] == "Numerical" else "values"
            filterargs = {
                "type": filtertype,
                "column": column,
                filterkey: filtervalue,
                "enabled": True,
            }
            filter: CategoricalFilter | NumericFilter = config["filterclass"](
                **filterargs
            )
            # Store in benchmark-specific state
            self.data_manager.current_state.filters.append(filter)
            # self.data_manager.invalidate_cache()  # Add this line
            for key in [
                f"{self.SESSION_KEY}.col",
                f"{self.SESSION_KEY}.range",
                f"{self.SESSION_KEY}.values",
            ]:
                st.session_state.pop(key, None)
            st.rerun()
