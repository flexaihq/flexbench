"""
Cost calculation module for MLPerf configurations.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_HOURLY_COST = 1.0

DEFAULT_DEVICE_COSTS = {
    "NVIDIA H100": 3.00,
    "NVIDIA H200": 4.00,
    "NVIDIA GH200": 5.00,
    "NVIDIA B200/GB200": 7.00,
    "AMD MI300X": 3.50,
    "AMD MI325X": 4.50,
    "NVIDIA RTX 4090": 1.20,
    "NVIDIA L40S": 1.80,
    "NVIDIA Jetson AGX": 0.30,
}

device_costs = {}


def normalize_gpu_name(name: str) -> str:
    """Normalize GPU names by identifying common patterns for the same device families."""
    if not name:
        return name

    name_upper = name.upper()

    gpu_families = {
        "H100": "NVIDIA H100",
        "H200": "NVIDIA H200",
        "GH200": "NVIDIA GH200",
        "GRACE HOPPER": "NVIDIA GH200",
        "B200": "NVIDIA B200/GB200",
        "GB200": "NVIDIA B200/GB200",
        "MI300X": "AMD MI300X",
        "MI325X": "AMD MI325X",
        "RTX 4090": "NVIDIA RTX 4090",
        "L40S": "NVIDIA L40S",
    }

    if "JETSON" in name_upper and ("ORIN" in name_upper or "THOR" in name_upper):
        return "NVIDIA Jetson AGX"

    for keyword, normalized_name in gpu_families.items():
        if keyword in name_upper:
            return normalized_name

    return name


def initialize_device_costs(df: pd.DataFrame) -> None:
    """Initialize device costs from dataset with default values."""
    global device_costs

    accelerators = set()

    if df is not None and not df.empty and "system.accelerator.name" in df.columns:
        for acc in df["system.accelerator.name"].dropna().unique():
            normalized_name = normalize_gpu_name(acc)
            accelerators.add(normalized_name)

    device_costs = {}
    for device in accelerators:
        if device in DEFAULT_DEVICE_COSTS:
            device_costs[device] = DEFAULT_DEVICE_COSTS[device]
        else:
            device_costs[device] = DEFAULT_HOURLY_COST

    logger.info(f"Initialized costs for {len(device_costs)} unique device families")


def get_device_costs() -> dict[str, float]:
    """Return a copy of the current device costs."""
    return device_costs.copy()


def update_device_costs(new_costs: dict[str, float]) -> None:
    """Update device costs with new values."""
    global device_costs
    device_costs.update(new_costs)
    logger.info(f"Updated costs for {len(new_costs)} devices")


def calculate_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Add cost metrics to the DataFrame."""
    if df is None or df.empty:
        return df

    result_df = df.copy()

    result_df["hourly_cost"] = None
    result_df["cost_per_million_tokens"] = None

    for idx, row in result_df.iterrows():
        hourly_cost = estimate_hourly_cost(row)
        result_df.at[idx, "hourly_cost"] = hourly_cost

        if hourly_cost and "metrics.result" in row and row["metrics.result"]:
            tokens_per_hour = row["metrics.result"] * 3600
            if tokens_per_hour > 0:
                cost_per_million = (hourly_cost / tokens_per_hour) * 1000000
                result_df.at[idx, "cost_per_million_tokens"] = cost_per_million

    return result_df


def estimate_hourly_cost(row: pd.Series) -> float:
    """Estimate hourly cost for a single configuration."""
    try:
        acc_name = row.get("system.accelerator.name")
        acc_vendor = row.get("system.accelerator.vendor")
        acc_count = row.get("system.accelerator.total_count")

        if not acc_count:
            return None

        base_cost = DEFAULT_HOURLY_COST

        if acc_name:
            normalized_name = normalize_gpu_name(acc_name)
            if normalized_name in device_costs:
                base_cost = device_costs[normalized_name]
            elif acc_vendor and acc_vendor in device_costs:
                base_cost = device_costs[acc_vendor]

        return base_cost * acc_count

    except Exception as e:
        logger.warning(f"Error calculating cost: {e}")
        return None
