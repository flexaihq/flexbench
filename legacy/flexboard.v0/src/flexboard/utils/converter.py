import re
import sys

import polars as pl

UNITS = {
    "Hz": {
        "G": 1e9,
        "M": 1e6,
        "": 1,
    },
    "B": {
        "T": 1024**4,
        "G": 1024**3,
        "M": 1024**2,
        "K": 1024,
        "": 1,
    },
}


def analyze_series(series: pl.Series, threshold: float) -> str | None:
    """Determine predominant unit type in series."""
    str_series = series.cast(str)

    # Count occurrences of each unit type
    unit_counts = {"Hz": 0, "B": 0}

    for value in str_series:
        # if value is empty or has less than half digits, skip
        n_digits = sum(c.isdigit() for c in value)
        if (
            not value
            or n_digits
            < len(value.replace("Hz", "").replace("B", "").replace(".", "")) / 2
        ):
            continue
        value = value.strip().upper()
        if "HZ" in value:
            unit_counts["Hz"] += 1
        # Consider lone G/T/M/K as bytes if not followed by Hz
        elif (
            "B" in value
            or any(
                (
                    "G" in value,
                    "T" in value,
                    "M" in value,
                    "K" in value,
                )
            )
            and "HZ" not in value
        ):
            unit_counts["B"] += 1

    # Check if any unit type meets threshold
    series_len = len(str_series)
    for unit, count in unit_counts.items():
        if count / (series_len + sys.float_info.epsilon) >= threshold:
            return unit
    return None


def detect_and_convert(
    value: str, target_unit: str | None = None
) -> tuple[float | None, str | None]:
    """Detect unit type and convert value to base unit."""
    if not value or not isinstance(value, str):
        return None, None

    value = value.strip().upper()

    # If lone G/T/M/K and target is bytes, append B
    if target_unit == "B" and any(value.endswith(x) for x in ["G", "T", "M", "K"]):
        value = value + "B"

    # If we have a target unit, try that first
    if target_unit:
        multipliers = UNITS[target_unit]
        prefix_pattern = "|".join(multipliers.keys())
        pattern = rf"^(\d+(?:\.\d+)?)\s*({prefix_pattern})?\s*{target_unit}"
        match = re.match(pattern, value, re.IGNORECASE)
        if match:
            number, prefix = match.groups()
            prefix = prefix.upper() if prefix else ""
            scale = multipliers.get(prefix, 1)
            return float(number) * scale, target_unit

    # If no target unit or no match, try all units
    for base_unit, multipliers in UNITS.items():
        prefix_pattern = "|".join(multipliers.keys())
        pattern = rf"^(\d+(?:\.\d+)?)\s*({prefix_pattern})?\s*{base_unit}"
        match = re.match(pattern, value, re.IGNORECASE)
        if match:
            number, prefix = match.groups()
            prefix = prefix.upper() if prefix else ""
            scale = multipliers.get(prefix, 1)
            return float(number) * scale, base_unit

    return None, None


def convert_series(series: pl.Series, threshold: float = 0.5) -> pl.Series:
    """Convert a series to appropriate numerical type."""
    if series.is_empty() or series.dtype.is_numeric():
        return series

    # Clean and standardize
    str_series = series.cast(str).map_elements(
        lambda x: (
            None
            if (not x) or (x.strip() == "") or (x.strip().upper() == "N/A")
            else x.strip()
        ),
        return_dtype=pl.Utf8,
    )

    # First determine predominant unit type
    target_unit = analyze_series(str_series.drop_nulls(), threshold)
    if not target_unit:
        return str_series

    # Convert values using target unit
    values_with_units = [
        detect_and_convert(x, target_unit) if x else (None, None) for x in str_series
    ]
    values, units = zip(*values_with_units)

    return pl.Series(series.name, values).cast(pl.Float64, strict=False)
