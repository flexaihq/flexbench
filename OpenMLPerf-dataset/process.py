"""
Data processing module for MLPerf benchmark data.
"""

import glob
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime

import polars as pl
from datasets import Dataset

logger = logging.getLogger(__name__)

FEATURES = {
    "Performance": {
        "metrics.result": "continuous",
        "metrics.result_per_accelerator": "continuous",
        "metrics.accuracy": "continuous",
    },
    "Model": {
        "model.name": "categorical",
        "model.mlperf_name": "categorical",
        "model.architecture": "categorical",
        "model.number_of_parameters": "continuous",
        "model.weight_data_types": "categorical",
    },
    "Accelerator": {
        "system.accelerator.vendor": "categorical",
        "system.accelerator.name": "categorical",
        "system.accelerator.count_per_node": "continuous",
        "system.accelerator.total_count": "continuous",
        "system.accelerator.memory_capacity": "continuous",
        "system.accelerator.memory_config": "text",
        "system.interconnect.accelerator": "categorical",
    },
    "CPU": {
        "system.cpu.vendor": "categorical",
        "system.cpu.model": "categorical",
        "system.cpu.core_count": "continuous",
        "system.cpu.count_per_node": "continuous",
        "system.cpu.frequency": "continuous",
        "system.cpu.caches": "text",
        "system.cpu.vcpu_count": "continuous",
    },
    "System": {
        "system.name": "text",
        "system.type": "categorical",
        "system.cooling": "categorical",
        "system.number_of_nodes": "continuous",
        "system.memory.capacity": "continuous",
        "system.memory.configuration": "text",
        "system.interconnect.accelerator_host": "categorical",
    },
    "Software": {
        "software.framework": "categorical",
        "software.version": "categorical",
        "software.operating_system": "categorical",
    },
    "Submission": {
        "submission.organization": "categorical",
        "submission.division": "categorical",
        "submission.scenario": "categorical",
        "submission.availability": "boolean",
        "submission.debug_uid": "text",
    },
}

MISSING_VALUES = defaultdict(set)


def get_feature_type(feature_name: str) -> str:
    """Get the type of a feature from the FEATURES dictionary."""
    for group in FEATURES.values():
        if feature_name in group:
            return group[feature_name]
    return "categorical"


def find_result_files(base_path: str = "semi-raw-mlperf-data") -> list[str]:
    """Find all cmx-result-summary.json files."""
    return glob.glob(os.path.join(base_path, "**/cmx-result-summary.json"), recursive=True)


def load_raw_data(base_path: str = "semi-raw-mlperf-data") -> pl.DataFrame:
    """Load and merge data from MLPerf result files."""
    result_files = find_result_files(base_path)
    logger.info(f"Found {len(result_files)} result files")
    all_records = []

    for file_path in result_files:
        with open(file_path, "r") as f:
            all_records.extend(json.loads(f.read()))

    df = pl.DataFrame(all_records, infer_schema_length=None)
    logger.info(f"Loaded {len(df)} raw benchmark records")

    rename_map = {
        "Accuracy": "metrics.accuracy",
        "Availability": "submission.availability",
        "Organization": "submission.organization",
        "Division": "submission.division",
        "Scenario": "submission.scenario",
        "Result": "metrics.result",
        "Units": "metrics.units",
        "MlperfModel": "model.mlperf_name",
        "Model": "model.name",
        "weight_data_types": "model.weight_data_types",
        "framework": "software.framework",
        "operating_system": "software.operating_system",
        "SystemName": "system.name",
        "system.system_name": "system.name",
        "SystemType": "system.type",
        "system.system_type": "system.type",
        "accelerator_model_name": "system.accelerator.name",
        "system.accelerator_model_name": "system.accelerator.name",
        "number_of_nodes": "system.number_of_nodes",
        "accelerators_per_node": "system.accelerator.count_per_node",
        "system.accelerators_per_node": "system.accelerator.count_per_node",
        "host_processor_core_count": "system.cpu.core_count",
        "system.host_processor_core_count": "system.cpu.core_count",
        "host_processor_model_name": "system.cpu.model",
        "system.host_processor_model_name": "system.cpu.model",
        "host_processors_per_node": "system.cpu.count_per_node",
        "system.host_processors_per_node": "system.cpu.count_per_node",
        "cooling": "system.cooling",
        "system.cooling": "system.cooling",
        "system.accelerator_host_interconnect": "system.interconnect.accelerator_host",
        "system.accelerator_interconnect": "system.interconnect.accelerator",
        "system.accelerator_memory_capacity": "system.accelerator.memory_capacity",
        "system.accelerator_memory_configuration": "system.accelerator.memory_config",
        "system.host_memory_capacity": "system.memory.capacity",
        "system.host_memory_configuration": "system.memory.configuration",
        "system.host_processor_frequency": "system.cpu.frequency",
        "system.host_processor_caches": "system.cpu.caches",
        "system.host_processor_vcpu_count": "system.cpu.vcpu_count",
        "benchmark_name": "benchmark.name",
        "benchmark_version": "benchmark.version",
        "datetime_last_commit": "datetime",
        "debug_uid": "submission.debug_uid",
    }

    for old_name, new_name in rename_map.items():
        if old_name in df.columns:
            if new_name in df.columns:
                df = df.drop(new_name)
            df = df.rename({old_name: new_name})

    columns_to_select = list(set(rename_map.values()))
    return df.select([col for col in columns_to_select if col in df.columns])


def is_within_tolerance(value1: float, value2: float, tolerance: float = 0.1) -> bool:
    """Check if two values are within a specified tolerance."""
    if value1 is None or value2 is None:
        return value1 == value2

    if value1 == 0 or value2 == 0:
        return value1 == value2

    percentage_diff = abs(value1 - value2) / max(abs(value1), abs(value2))
    return percentage_diff <= tolerance


def find_similar_configurations(
    df: pl.DataFrame, query_config: dict, continuous_tolerance: float = 0.1
) -> pl.DataFrame:
    """Find configurations similar to the query_config."""
    mask = pl.lit(True)

    for feature, value in query_config.items():
        if value is None:
            continue

        if get_feature_type(feature) == "continuous":
            lower_bound = value * (1 - continuous_tolerance)
            upper_bound = value * (1 + continuous_tolerance)
            feature_mask = (pl.col(feature) >= lower_bound) & (pl.col(feature) <= upper_bound)
        else:
            feature_mask = pl.col(feature) == value

        mask = mask & feature_mask

    return df.filter(mask)


def convert_datetime_to_iso(value: str) -> str | None:
    """Convert datetime string to ISO 8601 format."""
    if not value or value in ["", "N/A", "null"]:
        MISSING_VALUES["datetime_values"].add(str(value))
        return None

    try:
        # Handle format like "2025/04/03_22:56:53"
        if "/" in value and "_" in value:
            # Replace / with - and _ with T for ISO format
            iso_value = value.replace("/", "-").replace("_", "T")
            # Validate by parsing
            datetime.fromisoformat(iso_value)
            return iso_value

        # Try to parse other common formats and convert to ISO
        # Add more format patterns as needed
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
            try:
                dt = datetime.strptime(value, fmt)
                return dt.isoformat()
            except ValueError:
                continue

        # If no format matches, log as missing value
        MISSING_VALUES["datetime_values"].add(str(value))
        return None

    except Exception:
        MISSING_VALUES["datetime_values"].add(str(value))
        return None


def convert_memory_to_gb(value: str) -> float | None:
    """Convert memory string to GB."""
    if value is None:
        return None

    if "+" in value:
        left, right = value.split("+", 1)
        return (convert_memory_to_gb(left) or 0.0) + (convert_memory_to_gb(right) or 0.0) or None

    value = value.replace(" ", "").upper()
    numeric = ""
    for char in value:
        if char.isdigit() or char == ".":
            numeric += char
        else:
            break

    if not numeric:
        return None

    number = float(numeric)
    if "TB" in value or "TIB" in value:
        return number * 1024
    elif "GB" in value or "GIB" in value:
        return number
    else:
        return None


def convert_frequency_to_ghz(value: str) -> float | None:
    """Convert frequency string to GHz."""
    if not value or value == "undefined":
        MISSING_VALUES["frequency_values"].add(str(value))
        return None

    value = value.strip().upper()
    if value.isdigit():
        return float(value) / 1000

    matches = re.findall(r"([\d.]+)\s*(?:GHZ|MHZ)?", value, re.IGNORECASE)
    if not matches:
        MISSING_VALUES["frequency_values"].add(str(value))
        return None

    frequencies = [float(match) for match in matches]
    max_freq = max(frequencies)
    if "MHZ" in value.upper():
        max_freq /= 1000

    return max_freq


def extract_accelerator_vendor(name: str) -> str | None:
    """Extract vendor from accelerator name."""
    if name is None:
        MISSING_VALUES["accelerator_names"].add(None)
        return None

    name_upper = name.upper()
    known_vendors = {
        "NVIDIA": ["NVIDIA", "TESLA", "A100", "H100", "T4"],
        "AMD": ["AMD"],
        "Intel": ["INTEL", "HABANA", "GAUDI"],
        "Google": ["TPU", "TRILLIUM", "LPU", "VPU"],
        "Qualcomm": ["QUALCOMM", "ADRENO", "HEXAGON", "CLOUD AI 100", "SNAPDRAGON"],
        "UntetherAI": ["UNTETHERAIR", "SPEEDAI"],
        "Huawei": ["DAVINCI"],
        "Moffett": ["MOFFETT"],
    }

    for vendor, keywords in known_vendors.items():
        if any(keyword in name_upper for keyword in keywords):
            return vendor

    MISSING_VALUES["accelerator_vendors"].add(name)
    return None


def extract_cpu_vendor(name: str) -> str | None:
    """Extract vendor from CPU model name."""
    if name is None:
        MISSING_VALUES["cpu_names"].add(None)
        return None

    name_upper = name.upper()
    known_vendors = {
        "AMD": ["AMD", "EPYC"],
        "Intel": ["INTEL", "XEON"],
        "NVIDIA": ["NVIDIA", "GRACE"],
        "ARM": ["ARM", "CORTEX", "NEOVERSE", "ARMV8"],
        "AWS": ["AWS", "GRAVITON"],
        "Apple": ["APPLE", "M1", "M2", "M3"],
        "Qualcomm": ["QUALCOMM", "SNAPDRAGON"],
    }

    for vendor, keywords in known_vendors.items():
        if any(keyword in name_upper for keyword in keywords):
            return vendor

    MISSING_VALUES["cpu_vendors"].add(name)
    return None


def extract_framework_info(framework_str: str) -> list[tuple[str, str]]:
    """Extract framework name-version pairs."""
    if not framework_str:
        return []

    results = []
    for item in framework_str.split(","):
        item = item.strip()
        name_match = re.search(r"(\w+)\s+", item)
        version_match = re.search(r"\s+([\d\.]+)", item)

        if name_match and version_match:
            name = name_match.group(1).lower()
            version = version_match.group(1)
            results.append((name, version.strip(".")))

    return results


def clean_string_value(value: str) -> str | None:
    """Clean empty and N/A string values."""
    if value.upper() in ["", "N/A", "DUMMY"]:
        return None
    return value


def normalize_interconnect_type(value: str) -> str | None:
    """Normalize interconnect type strings."""
    if value is None or value.upper() in ["TBD", "TODO", "TODD"]:
        MISSING_VALUES["interconnect_types"].add(str(value))
        return None

    value_upper = value.upper()
    if "NVLINK" in value_upper:
        if any(gen in value_upper for gen in ["5TH", "5TH-GEN"]):
            return "NVLink-5"
        elif any(gen in value_upper for gen in ["4TH", "4TH-GEN"]):
            return "NVLink-4"
        else:
            return "NVLink"

    if "PCIE" in value_upper:
        if "GEN5" in value_upper.replace(" ", ""):
            return "PCIe-5"
        else:
            return "PCIe"

    if "INFINIBAND" in value_upper:
        return "InfiniBand"
    if "XGMI" in value_upper:
        return "XGMI"

    return value


def clean_string_values(df: pl.DataFrame, string_columns: list[str] | None = None) -> pl.DataFrame:
    """Clean string values in specified columns."""
    if string_columns is None:
        string_columns = [col for col in df.columns if df[col].dtype == pl.String]
    return df.with_columns(
        [
            pl.col(col).map_elements(clean_string_value, return_dtype=str).alias(col)
            for col in string_columns
        ]
    )


def filter_submissions(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only valid token/s submissions."""
    return df.filter(
        (pl.col("metrics.units") == "Tokens/s")
        & (pl.col("metrics.result").is_not_null())
        & (pl.col("metrics.result") != 0)
        & (pl.col("metrics.result").is_finite())
        & (pl.col("system.accelerator.total_count") > 0)
    )


def normalize_memory_values(df: pl.DataFrame) -> pl.DataFrame:
    """Convert memory values to GB."""
    return df.with_columns(
        [
            pl.col("system.accelerator.memory_capacity")
            .map_elements(convert_memory_to_gb, return_dtype=float)
            .alias("system.accelerator.memory_capacity"),
            pl.col("system.memory.capacity")
            .map_elements(convert_memory_to_gb, return_dtype=float)
            .alias("system.memory.capacity"),
        ]
    )


def normalize_datetime_values(df: pl.DataFrame) -> pl.DataFrame:
    """Convert datetime values to ISO 8601 format."""
    if "datetime" in df.columns:
        return df.with_columns(
            pl.col("datetime")
            .map_elements(convert_datetime_to_iso, return_dtype=str)
            .alias("datetime")
        )
    return df


def add_vendor_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add vendor columns based on model names."""
    return df.with_columns(
        [
            pl.col("system.accelerator.name")
            .map_elements(extract_accelerator_vendor, return_dtype=str)
            .alias("system.accelerator.vendor"),
            pl.col("system.cpu.model")
            .map_elements(extract_cpu_vendor, return_dtype=str)
            .alias("system.cpu.vendor"),
        ]
    )


def normalize_interconnect_values(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize interconnect values."""
    return df.with_columns(
        [
            pl.col("system.interconnect.accelerator")
            .map_elements(normalize_interconnect_type, return_dtype=str)
            .alias("system.interconnect.accelerator"),
            pl.col("system.interconnect.accelerator_host")
            .map_elements(normalize_interconnect_type, return_dtype=str)
            .alias("system.interconnect.accelerator_host"),
        ]
    )


def extract_framework_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Extract framework versions into separate columns."""
    df_with_id = df.with_columns(pl.Series(name="row_id", values=range(len(df))))
    framework_info = []

    for idx, framework_str in enumerate(df["software.framework"]):
        if framework_str is not None:
            for name, version in extract_framework_info(framework_str):
                framework_info.append({"row_id": idx, "name": name, "version": version})

    if not framework_info:
        return df

    df_frameworks = pl.DataFrame(framework_info)
    df_pivoted = df_frameworks.pivot(
        values="version",
        index="row_id",
        on="name",
        aggregate_function="first",
    )

    rename_dict = {
        col: f"software.framework.{col}" for col in df_pivoted.columns if col != "row_id"
    }
    df_pivoted = df_pivoted.rename(rename_dict)

    return df_with_id.join(df_pivoted, on="row_id", how="left").drop("row_id")


def cast_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Cast columns to proper types."""
    column_types = {
        "system.cpu.core_count": pl.Int64,
        "system.accelerator.count_per_node": pl.Int64,
        "system.cpu.count_per_node": pl.Int64,
        "system.number_of_nodes": pl.Int64,
    }

    df = df.with_columns(
        pl.col("system.cpu.frequency")
        .map_elements(convert_frequency_to_ghz, return_dtype=float)
        .alias("system.cpu.frequency")
    )

    return df.cast(column_types)


def add_model_parameters(df: pl.DataFrame) -> pl.DataFrame:
    """Add number of parameters column based on model name."""
    model_parameters = {
        "llama2-70b": 70,
        "llama-2-70b": 70,
        "llama3_1-405b": 405,
        "llama3_1-70b": 70,
        "gptj": 6,
        "mixtral-8x7b": 47,
        "DeepSeek-R1-Distill-Llama-8B": 8,
        "Llama-3.3-70B": 70,
        "deepseek-v3": 671,
    }

    def extract_parameters(model_name: str) -> float | None:
        if not model_name:
            return None
        for base_name, params in model_parameters.items():
            if model_name.lower().startswith(base_name.lower()):
                return float(params)
        return None

    return df.with_columns(
        pl.col("model.name")
        .map_elements(extract_parameters, return_dtype=float)
        .alias("model.number_of_parameters")
    )


def add_model_architecture(df: pl.DataFrame) -> pl.DataFrame:
    """Add model architecture classification."""
    model_architectures = {
        "llama": "LLM",
        "gpt": "LLM",
        "mixtral": "LLM",
        "deepseek": "LLM",
        "falcon": "LLM",
        "mistral": "LLM",
    }

    def classify_architecture(model_name: str) -> str | None:
        if not model_name:
            return None
        model_name_lower = model_name.lower()
        for pattern, arch in model_architectures.items():
            if pattern in model_name_lower:
                return arch
        return "Other"

    return df.with_columns(
        pl.col("model.name")
        .map_elements(classify_architecture, return_dtype=str)
        .alias("model.architecture")
    )


def add_total_accelerator_count(df: pl.DataFrame) -> pl.DataFrame:
    """Compute total number of accelerators."""
    return df.with_columns(
        (pl.col("system.number_of_nodes") * pl.col("system.accelerator.count_per_node")).alias(
            "system.accelerator.total_count"
        )
    )


def add_normalized_performance(df: pl.DataFrame) -> pl.DataFrame:
    """Add performance per accelerator metric."""
    return df.with_columns(
        (pl.col("metrics.result") / pl.col("system.accelerator.total_count")).alias(
            "metrics.result_per_accelerator"
        )
    )


def sort_columns_alphabetically(df: pl.DataFrame) -> pl.DataFrame:
    """Sort columns alphabetically."""
    return df.select(sorted(df.columns))


def log_missing_values() -> None:
    """Log all collected missing values once."""
    for category, values in MISSING_VALUES.items():
        if values:
            logger.warning(
                f"Could not determine {len(values)} unique {category}: {sorted(str(v) for v in values)}"
            )


def upload_to_huggingface_hub(
    df: pl.DataFrame, dataset_name: str = "OpenMLPerf", private: bool = True
) -> None:
    """Upload the processed dataset to HuggingFace Hub."""
    logger.info(f"Preparing dataset '{dataset_name}' for upload to HuggingFace Hub")
    data_dict = {col: df[col].to_list() for col in df.columns}
    dataset = Dataset.from_dict(data_dict)

    try:
        dataset.push_to_hub(dataset_name, private=private)
        logger.info(f"Successfully uploaded dataset to HuggingFace Hub as '{dataset_name}'")
    except Exception as e:
        logger.error(f"Failed to upload dataset to HuggingFace Hub: {e}")


def process_data(base_path: str = "semi-raw-mlperf-data") -> pl.DataFrame:
    """Main data processing pipeline."""
    logger.info("Starting data processing pipeline")

    MISSING_VALUES.clear()

    df = (
        load_raw_data(base_path)
        .pipe(clean_string_values)
        .pipe(normalize_memory_values)
        .pipe(normalize_datetime_values)
        .pipe(cast_columns)
        .pipe(add_vendor_columns)
        .pipe(normalize_interconnect_values)
        .pipe(extract_framework_columns)
        .pipe(add_model_parameters)
        .pipe(add_model_architecture)
        .pipe(add_total_accelerator_count)
        .pipe(add_normalized_performance)
        .pipe(sort_columns_alphabetically)
        .pipe(filter_submissions)
    )

    log_missing_values()

    logger.info(f"Processed {len(df)} records")
    return df


def export_data(df: pl.DataFrame) -> None:
    """Export processed data to JSON file."""
    with open("data.json", "w") as f:
        json.dump(df.to_dicts(), f, indent=2)
        logger.info("Exported data to data.json")
    df.write_parquet("data.parquet")
    logger.info("Exported data to data.parquet")


def main(
    base_path: str = "semi-raw-mlperf-data",
    upload_to_hub: bool = False,
    dataset_name: str = "OpenMLPerf",
    push_to_hub: bool = True,
    private: bool = True,
):
    """Run the complete data processing pipeline."""
    logging.basicConfig(level=logging.INFO)
    df = process_data(base_path)
    export_data(df)

    if upload_to_hub:
        upload_to_huggingface_hub(df, dataset_name, private)

    return df


if __name__ == "__main__":
    main(upload_to_hub=False, private=True)
