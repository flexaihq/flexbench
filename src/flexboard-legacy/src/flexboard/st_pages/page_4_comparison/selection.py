import json
import typing as t
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import streamlit as st


@dataclass
class Selection:
    """Container for selection state"""

    model: str
    submitters: list[str]
    accelerators: list[str]
    systems: list[str]
    accelerator_prices: dict[str, float] = None
    system_prices: dict[str, float] = None

    @property
    def has_prices(self) -> bool:
        return self.accelerator_prices is not None


class Selector:
    """Handles selection of models, submitters, accelerators and systems"""

    def __init__(self, df: pl.DataFrame, mode: t.Literal["performance", "cost"]):
        self.df = df
        self.mode = mode
        self._load_default_prices()
        self.acc_prices = {}

    def _load_default_prices(self) -> None:
        """Load default accelerator prices if in cost mode"""
        if self.mode == "cost":
            with open(Path(__file__).parent / "accelerator_prices.json") as f:
                self.default_prices = json.load(f)
        else:
            self.default_prices = {}

    def select_model(self) -> str | None:
        """Let user select a model"""
        all_models = sorted(self.df["model_name"].drop_nulls().unique().to_list())

        return st.selectbox(
            "Model",
            options=all_models,
            key=f"comparison.{self.mode}.model",
            label_visibility="collapsed",
            index=all_models.index(
                next(filter(lambda x: "llama" in x, all_models), all_models[0])
            ),
        )

    def select_submitters(self, model: str) -> list[str]:
        """Let user select submitters for given model"""
        available = sorted(
            self.df.filter(pl.col("model_name") == model)["submitter"].unique().to_list()
        )

        return st.multiselect(
            "Submitter(s)",
            options=available,
            key=f"comparison.{self.mode}.submitters",
            default=available,
            label_visibility="collapsed",
        )

    def _get_filtered_data(_self, model: str, submitters: list[str]) -> tuple[list[str], dict[str, list[str]], int]:
        """Get filtered accelerators and their systems (cached)"""
        filtered_df = _self.df.filter(
            (pl.col("model_name") == model) & pl.col("submitter").is_in(submitters)
        )

        all_accelerators = sorted(
            filtered_df["accelerator_name"].drop_nulls().unique().to_list()
        )
        acc_to_systems = {}
        total_systems = 0
        for acc in all_accelerators:
            systems = (
                filtered_df.filter(pl.col("accelerator_name") == acc)["system_name"]
                .unique()
                .to_list()
            )
            if systems:
                acc_to_systems[acc] = sorted(systems)
                total_systems += len(systems)
        
        return all_accelerators, acc_to_systems, total_systems

    def select_accelerators(
        self, model: str, submitters: list[str]
    ) -> tuple[list[str], dict]:
        """Let user select accelerators and their prices if in cost mode"""
        all_accelerators, acc_to_systems, total_systems = self._get_filtered_data(model, submitters)

        st.caption(
            f"Found {len(all_accelerators)} accelerators with {total_systems} systems total"
        )
        
        self.acc_prices = {}
        selected_accelerators = []
        selected_systems = []

        for idx, acc in enumerate(all_accelerators, 1):
            cols = (
                st.columns([0.25, 1.25, 0.75, 3], gap="large")
                if self.mode == "cost"
                else st.columns([0.25, 2, 3], gap="large")
            )
            
            cols[0].write(f"{idx}.")
            
            is_selected = cols[1].toggle(
                acc,
                key=f"comparison.{self.mode}.toggle.{acc}",
                value=True,
            )

            if self.mode == "cost":
                price = cols[2].number_input(
                    f"Price for {acc}",
                    value=st.session_state.get(
                        f"price.{acc}", self.default_prices.get(acc, 0.01)
                    ),
                    min_value=0.01,
                    step=0.1,
                    format="%.2f",
                    key=f"comparison.{self.mode}.price.{acc}",
                    label_visibility="collapsed",
                    disabled=not is_selected,
                )
                if is_selected:
                    self.acc_prices[acc] = price

            selected = cols[-1].multiselect(
                "Systems",
                options=acc_to_systems.get(acc, []),
                default=acc_to_systems.get(acc, []),
                key=f"comparison.{self.mode}.systems.{acc}",
                label_visibility="collapsed",
                disabled=not is_selected,
            )
            
            if is_selected:
                selected_accelerators.append(acc)
                selected_systems.extend(selected)

        return selected_accelerators, self.acc_prices if self.mode == "cost" else {}

    def select_systems(
        self, model: str, submitters: list[str], accelerators: list[str]
    ) -> tuple[list[str], dict]:
        """Let user select custom system prices if in cost mode"""
        if not self.mode == "cost":
            return [], {}

        filtered_df = self.df.filter(
            (pl.col("model_name") == model)
            & pl.col("submitter").is_in(submitters)
            & pl.col("accelerator_name").is_in(accelerators)
        )

        all_systems = sorted(filtered_df["system_name"].unique().to_list())
        system_prices = {}

        st.divider()
        st.subheader("Custom System Prices", anchor=False)
        with st.expander("Click to expand", expanded=False):
            st.info(
                "Custom system prices will override the calculated prices for selected systems."
            )
            custom_systems = st.multiselect(
                "Select systems for custom pricing",
                options=all_systems,
                key=f"comparison.{self.mode}.custom_systems",
                help="Select systems to override their calculated prices",
            )

            if custom_systems:
                for system in custom_systems:
                    system_data = filtered_df.filter(pl.col("system_name") == system)
                    acc_name = system_data.select("accelerator_name").item()
                    acc_count = system_data.select("accelerator_count").item()

                    base_price = acc_count * self.acc_prices.get(acc_name, 0.0)

                    system_prices[system] = st.number_input(
                        f"Price for {system}",
                        min_value=0.01,
                        value=base_price,
                        step=0.1,
                        format="%.2f",
                        key=f"comparison.{self.mode}.system_price.{system}",
                    )

        return all_systems, system_prices

    def get_selection(self) -> Selection:
        """Get complete selection from user"""
        with st.container(border=True):
            cols = st.columns([2, 3], gap="large")

            with cols[0]:
                st.subheader("Model", anchor=False, divider="gray")
                model = self.select_model()

            with cols[1]:
                st.subheader("Submitters", anchor=False, divider="gray")
                submitters = self.select_submitters(model) if model else []

            st.subheader("Accelerators & Systems", anchor=False, divider="gray")
            accelerators, acc_prices = (
                self.select_accelerators(model, submitters)
                if model and submitters
                else ([], {})
            )

            systems, sys_prices = (
                self.select_systems(model, submitters, accelerators)
                if model and submitters and accelerators and self.mode == "cost"
                else ([], {})
            )

        return Selection(
            model=model,
            submitters=submitters,
            accelerators=accelerators,
            systems=systems,
            accelerator_prices=acc_prices if self.mode == "cost" else None,
            system_prices=sys_prices if self.mode == "cost" else None,
        )


def get_selection(df: pl.DataFrame, mode: t.Literal["performance", "cost"]) -> Selection:
    """Helper function to get selection"""
    selector = Selector(df, mode)
    return selector.get_selection()
