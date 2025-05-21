"""Configuration recommendation module for MLPerf benchmarks."""

import logging

import pandas as pd
from utils import get_feature_type

logger = logging.getLogger(__name__)


class ConfigurationFinder:
    """Finds optimal hardware configurations based on user requirements."""

    def __init__(self, dataset: pd.DataFrame):
        """Initialize with benchmark dataset."""
        self.df = dataset
        self.perf_metric = "metrics.result_per_accelerator"
        self.cost_metric = "cost_per_million_tokens"
        self.total_perf_metric = "metrics.result"

    def is_within_tolerance(
        self, value1: float, value2: float, tolerance: float = 0.1
    ) -> bool:
        """Check if two values are within a specified percentage tolerance."""
        if value1 is None or value2 is None:
            return False

        try:
            if value1 == 0 or value2 == 0:
                return value1 == value2
            percentage_diff = abs(value1 - value2) / max(abs(value1), abs(value2))
            return percentage_diff <= tolerance
        except:
            return False

    def find_configurations(
        self, constraints: dict, tolerance: float = 0.1
    ) -> pd.DataFrame:
        """Find configurations matching the given constraints."""
        if self.df.empty:
            return pd.DataFrame()

        filtered_df = self.df.copy()

        for feature, value in constraints.items():
            if feature not in filtered_df.columns or value is None or value == "Any":
                continue

            if get_feature_type(feature) == "continuous":
                try:
                    target_value = float(value)
                    lower_bound = target_value * (1 - tolerance)
                    upper_bound = target_value * (1 + tolerance)
                    filtered_df = filtered_df[
                        (filtered_df[feature] >= lower_bound)
                        & (filtered_df[feature] <= upper_bound)
                    ]
                except:
                    filtered_df = filtered_df[filtered_df[feature] == value]
            else:
                filtered_df = filtered_df[filtered_df[feature] == value]

        if "min_accelerators" in constraints and constraints["min_accelerators"]:
            min_acc = constraints["min_accelerators"]
            filtered_df = filtered_df[
                filtered_df["system.accelerator.total_count"] >= min_acc
            ]

        if "max_accelerators" in constraints and constraints["max_accelerators"]:
            max_acc = constraints["max_accelerators"]
            filtered_df = filtered_df[
                filtered_df["system.accelerator.total_count"] <= max_acc
            ]

        return filtered_df

    def rank_configurations(
        self,
        df: pd.DataFrame,
        metric: str = "metrics.result_per_accelerator",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """Rank configurations by the specified metric."""
        if df.empty or metric not in df.columns:
            return df
        return df.sort_values(by=metric, ascending=ascending)

    def recommend(self, constraints: dict, top_n: int = 10) -> pd.DataFrame:
        """Find and rank configurations based on constraints."""
        filtered_configs = self.find_configurations(constraints)
        ranked_configs = self.rank_configurations(
            filtered_configs, metric=self.perf_metric, ascending=False
        )

        if len(ranked_configs) > top_n:
            return ranked_configs.head(top_n)
        return ranked_configs
