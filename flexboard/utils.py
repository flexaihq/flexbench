import json
import logging

import polars as pl

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
    },
}


def get_features_by_type(feature_type: str) -> list[str]:
    """Get all features of a specific type."""
    result = []
    for group in FEATURES.values():
        for feature, typ in group.items():
            if typ == feature_type:
                result.append(feature)
    return result


FEATURE_TYPES = {
    "continuous": get_features_by_type("continuous"),
    "categorical": get_features_by_type("categorical"),
    "boolean": get_features_by_type("boolean"),
    "text": get_features_by_type("text"),
}

UI_FEATURE_GROUPS = {
    group: list(features.keys()) for group, features in FEATURES.items()
}


def get_feature_type(feature_name: str) -> str:
    """Get the type of a feature from the FEATURES dictionary."""
    for group in FEATURES.values():
        if feature_name in group:
            return group[feature_name]
    return "categorical"


def load_data(file_path: str = "../OpenMLPerf-dataset/data.json") -> pl.DataFrame:
    """Load processed benchmark data."""
    logger.info(f"Loading processed data from {file_path}")

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        for item in data:
            for key, value in item.items():
                if isinstance(value, str):
                    if value.isdigit():
                        item[key] = int(value)
                    elif value.replace(".", "", 1).isdigit():
                        item[key] = float(value)

        df = pl.DataFrame(data, infer_schema_length=None)
        logger.info(f"Loaded {len(df)} benchmark results")
        return df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pl.DataFrame()
