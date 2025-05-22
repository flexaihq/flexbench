"""MLPerf Hardware Configuration Finder application."""

import logging
import os

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from cost_calculator import (
    calculate_costs,
    get_device_costs,
    initialize_device_costs,
    update_device_costs,
)
from plotly.subplots import make_subplots
from predictor import PerformancePredictor
from recommender import ConfigurationFinder

from utils import get_feature_type, load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading benchmark data...")
df = load_data()
pd_df = df.to_pandas() if not df.is_empty() else pd.DataFrame()
logger.info(f"Loaded {len(pd_df)} benchmark records total")

initialize_device_costs(pd_df)

predictor = PerformancePredictor(pd_df) if not pd_df.empty else None
config_finder = ConfigurationFinder(pd_df) if not pd_df.empty else None


def extract_metadata(df: pl.DataFrame) -> dict:
    """Extract metadata for UI filters from dataset."""
    metadata = {}
    if df.is_empty():
        return metadata

    metadata["architectures"] = sorted(
        df.filter(pl.col("model.architecture").is_not_null())
        .get_column("model.architecture")
        .unique()
        .to_list()
    )

    model_sizes = sorted(
        df.filter(pl.col("model.number_of_parameters").is_not_null())
        .get_column("model.number_of_parameters")
        .unique()
        .to_list()
    )
    if model_sizes:
        metadata["model_sizes"] = model_sizes
        metadata["model_size_min"] = min(model_sizes)
        metadata["model_size_max"] = max(model_sizes)
        metadata["model_size_values"] = sorted(model_sizes)

    metadata["weight_data_types"] = sorted(
        df.filter(pl.col("model.weight_data_types").is_not_null())
        .get_column("model.weight_data_types")
        .unique()
        .to_list()
    )

    metadata["accelerator_vendors"] = sorted(
        df.filter(pl.col("system.accelerator.vendor").is_not_null())
        .get_column("system.accelerator.vendor")
        .unique()
        .to_list()
    )

    metadata["cpu_vendors"] = sorted(
        df.filter(pl.col("system.cpu.vendor").is_not_null())
        .get_column("system.cpu.vendor")
        .unique()
        .to_list()
    )

    metadata["accelerator_models"] = sorted(
        df.filter(pl.col("system.accelerator.name").is_not_null())
        .get_column("system.accelerator.name")
        .unique()
        .to_list()
    )

    metadata["cpu_models"] = sorted(
        df.filter(pl.col("system.cpu.model").is_not_null())
        .get_column("system.cpu.model")
        .unique()
        .to_list()
    )

    memory_values = df.filter(
        pl.col("system.accelerator.memory_capacity").is_not_null()
    )
    metadata["gpu_memory_min"] = max(
        1,
        round(
            float(memory_values.get_column("system.accelerator.memory_capacity").min())
        ),
    )
    metadata["gpu_memory_max"] = min(
        1024,
        round(
            float(memory_values.get_column("system.accelerator.memory_capacity").max())
        ),
    )

    memory_values = df.filter(pl.col("system.memory.capacity").is_not_null())
    metadata["cpu_memory_min"] = max(
        1, round(float(memory_values.get_column("system.memory.capacity").min()))
    )
    metadata["cpu_memory_max"] = min(
        16384, round(float(memory_values.get_column("system.memory.capacity").max()))
    )

    metadata["interconnect_types"] = sorted(
        df.filter(pl.col("system.interconnect.accelerator").is_not_null())
        .get_column("system.interconnect.accelerator")
        .unique()
        .to_list()
    )

    acc_counts = sorted(
        df.filter(pl.col("system.accelerator.total_count").is_not_null())
        .get_column("system.accelerator.total_count")
        .unique()
        .cast(pl.Int64)
        .to_list()
    )
    metadata["accelerator_counts"] = acc_counts
    metadata["min_accelerators"] = min(acc_counts)
    metadata["max_accelerators"] = max(acc_counts)

    metadata["node_counts"] = sorted(
        df.filter(pl.col("system.number_of_nodes").is_not_null())
        .get_column("system.number_of_nodes")
        .unique()
        .cast(pl.Int64)
        .to_list()
    )

    frameworks = []
    for col in df.columns:
        if col.startswith("software.framework.") and col != "software.framework":
            framework_name = col.replace("software.framework.", "")
            frameworks.append(framework_name)
            versions = (
                df.filter(pl.col(col).is_not_null()).get_column(col).unique().to_list()
            )
            if versions:
                metadata[f"{framework_name}_versions"] = sorted(versions)

    metadata["frameworks"] = sorted(frameworks)

    metadata["operating_systems"] = sorted(
        df.filter(pl.col("software.operating_system").is_not_null())
        .get_column("software.operating_system")
        .unique()
        .to_list()
    )

    result_per_acc = df.filter(pl.col("metrics.result_per_accelerator").is_not_null())
    metadata["result_per_accelerator_ranges"] = {
        "min": float(result_per_acc.get_column("metrics.result_per_accelerator").min()),
        "max": float(result_per_acc.get_column("metrics.result_per_accelerator").max()),
        "median": float(
            result_per_acc.get_column("metrics.result_per_accelerator").median()
        ),
    }

    return metadata


metadata = extract_metadata(df)


def apply_continuous_feature_tolerance(
    df: pd.DataFrame, feature: str, value: float, tolerance: float = 0.1
) -> pd.DataFrame:
    """Apply tolerance for continuous feature searches."""
    lower_bound = value * (1 - tolerance)
    upper_bound = value * (1 + tolerance)
    return df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]


def find_best_configs(
    workload_specs: dict,
    constraints: dict,
    include_predictions: bool = True,
    optimization_metric: str = "performance",
) -> pd.DataFrame:
    """Find best hardware configurations for workload."""
    if pd_df.empty:
        return pd.DataFrame()

    filtered_df = pd_df.copy()

    if workload_specs.get("model_size") is not None:
        filtered_df = apply_continuous_feature_tolerance(
            filtered_df,
            "model.number_of_parameters",
            float(workload_specs["model_size"]),
        )

    if (
        workload_specs.get("weight_data_type")
        and workload_specs["weight_data_type"] != "Any"
    ):
        filtered_df = filtered_df[
            filtered_df["model.weight_data_types"] == workload_specs["weight_data_type"]
        ]

    if workload_specs.get("architecture") and workload_specs["architecture"] != "Any":
        filtered_df = filtered_df[
            filtered_df["model.architecture"] == workload_specs["architecture"]
        ]

    clean_constraints = {k: v for k, v in constraints.items() if v and v != "Any"}

    for feature, value in clean_constraints.items():
        if feature in filtered_df.columns:
            if get_feature_type(feature) == "continuous":
                filtered_df = apply_continuous_feature_tolerance(
                    filtered_df, feature, float(value)
                )
            else:
                filtered_df = filtered_df[filtered_df[feature] == value]

    if constraints.get("min_gpu_memory") is not None:
        filtered_df = filtered_df[
            filtered_df["system.accelerator.memory_capacity"]
            >= constraints["min_gpu_memory"]
        ]

    if constraints.get("max_gpu_memory") is not None:
        filtered_df = filtered_df[
            filtered_df["system.accelerator.memory_capacity"]
            <= constraints["max_gpu_memory"]
        ]

    if constraints.get("min_cpu_memory") is not None:
        filtered_df = filtered_df[
            filtered_df["system.memory.capacity"] >= constraints["min_cpu_memory"]
        ]

    if constraints.get("max_cpu_memory") is not None:
        filtered_df = filtered_df[
            filtered_df["system.memory.capacity"] <= constraints["max_cpu_memory"]
        ]

    if constraints.get("min_accelerators") is not None:
        filtered_df = filtered_df[
            filtered_df["system.accelerator.total_count"]
            >= constraints["min_accelerators"]
        ]

    if constraints.get("max_accelerators") is not None:
        filtered_df = filtered_df[
            filtered_df["system.accelerator.total_count"]
            <= constraints["max_accelerators"]
        ]

    if (
        include_predictions
        and predictor
        and workload_specs.get("model_size")
        and workload_specs.get("architecture")
    ):
        predicted_df = predictor.generate_predictions(
            architecture=workload_specs["architecture"],
            parameters=float(workload_specs["model_size"]),
            constraints=clean_constraints,
            num_configs=20,
        )

        if not predicted_df.empty:
            predicted_df = calculate_costs(predicted_df)

            if not filtered_df.empty:
                filtered_df = calculate_costs(filtered_df)
                filtered_df["predicted"] = False
                combined_df = pd.concat([filtered_df, predicted_df], ignore_index=True)
            else:
                combined_df = predicted_df

            sort_col = (
                "cost_per_million_tokens"
                if optimization_metric == "cost"
                else "metrics.result_per_accelerator"
            )
            asc = optimization_metric == "cost"
            return combined_df.sort_values(by=sort_col, ascending=asc)

    if not filtered_df.empty:
        filtered_df = calculate_costs(filtered_df)
        filtered_df["predicted"] = False

        sort_col = (
            "cost_per_million_tokens"
            if optimization_metric == "cost"
            else "metrics.result_per_accelerator"
        )
        asc = optimization_metric == "cost"
        return filtered_df.sort_values(by=sort_col, ascending=asc)

    return pd.DataFrame()


def format_recommendations(configs_df: pd.DataFrame) -> pd.DataFrame:
    """Format recommendations for display."""
    if configs_df.empty:
        return pd.DataFrame(
            columns=[
                "System",
                "Accelerator",
                "Count",
                "Nodes",
                "GPU Memory (GB)",
                "Model",
                "Architecture",
                "Parameters (B)",
                "Weight Data Type",
                "Total Performance (Tokens/s)",
                "Per-GPU Performance (Tokens/s)",
                "Hourly Cost ($)",
                "Cost/Million Tokens",
                "Predicted",
            ]
        )

    display_columns = {
        "system.name": "System",
        "system.accelerator.name": "Accelerator",
        "system.accelerator.total_count": "Count",
        "system.number_of_nodes": "Nodes",
        "system.accelerator.memory_capacity": "GPU Memory (GB)",
        "model.name": "Model",
        "model.architecture": "Architecture",
        "model.number_of_parameters": "Parameters (B)",
        "model.weight_data_types": "Weight Data Type",
        "metrics.result": "Total Performance (Tokens/s)",
        "metrics.result_per_accelerator": "Per-GPU Performance (Tokens/s)",
        "hourly_cost": "Hourly Cost ($)",
        "cost_per_million_tokens": "Cost/Million Tokens",
        "predicted": "Predicted",
    }

    result_df = pd.DataFrame()
    for col_name, display_name in display_columns.items():
        if col_name in configs_df.columns:
            result_df[display_name] = configs_df[col_name]
        else:
            result_df[display_name] = "N/A" if col_name != "predicted" else "No"

    numeric_columns = [
        "Count",
        "Nodes",
        "GPU Memory (GB)",
        "Parameters (B)",
        "Total Performance (Tokens/s)",
        "Per-GPU Performance (Tokens/s)",
        "Hourly Cost ($)",
        "Cost/Million Tokens",
    ]

    for col in numeric_columns:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors="coerce")

    result_df["Total Performance (Tokens/s)"] = result_df[
        "Total Performance (Tokens/s)"
    ].round(4)
    result_df["Per-GPU Performance (Tokens/s)"] = result_df[
        "Per-GPU Performance (Tokens/s)"
    ].round(4)
    result_df["GPU Memory (GB)"] = result_df["GPU Memory (GB)"].round(2)
    result_df["Cost/Million Tokens"] = result_df["Cost/Million Tokens"].round(4)
    result_df["Hourly Cost ($)"] = result_df["Hourly Cost ($)"].round(4)

    if "Parameters (B)" in result_df.columns:
        result_df["Parameters (B)"] = result_df["Parameters (B)"].round(2)

    if "Predicted" in result_df.columns:
        result_df["Predicted"] = result_df["Predicted"].map(
            lambda x: "Yes" if x is True else "No"
        )

    result_df = result_df.drop_duplicates()

    return result_df


def get_top_config_details(configs_df: pd.DataFrame) -> pd.DataFrame:
    """Extract details for the top recommendation."""
    if configs_df.empty:
        return pd.DataFrame(columns=["Feature", "Value"])

    top_config = configs_df.iloc[0]
    is_predicted = "predicted" in top_config and top_config["predicted"]

    details = {
        "Feature": [
            "System",
            "Accelerator",
            "Accelerator Count",
            "Accelerator Vendor",
            "Memory Capacity",
            "CPU",
            "CPU Vendor",
            "Nodes",
            "Devices per Node",
            "Interconnect",
            "Total Performance (Tokens/s)",
            "Per-Accelerator Performance (Tokens/s)",
            "Hourly Cost (estimated)",
            "Cost per Million Tokens",
            "Prediction Status",
        ],
        "Value": [
            top_config.get("system.name", "N/A"),
            top_config.get("system.accelerator.name", "N/A"),
            top_config.get("system.accelerator.total_count", "N/A"),
            top_config.get("system.accelerator.vendor", "N/A"),
            (
                f"{float(top_config.get('system.accelerator.memory_capacity', 0)):.1f}GB"
                if top_config.get("system.accelerator.memory_capacity") is not None
                else "N/A"
            ),
            top_config.get("system.cpu.model", "N/A"),
            top_config.get("system.cpu.vendor", "N/A"),
            top_config.get("system.number_of_nodes", "N/A"),
            top_config.get("system.accelerator.count_per_node", "N/A"),
            top_config.get("system.interconnect.accelerator", "N/A"),
            (
                f"{float(top_config.get('metrics.result', 0)):.4f}"
                if top_config.get("metrics.result") is not None
                else "N/A"
            ),
            (
                f"{float(top_config.get('metrics.result_per_accelerator', 0)):.4f}"
                if top_config.get("metrics.result_per_accelerator") is not None
                else "N/A"
            ),
            (
                f"${float(top_config.get('hourly_cost', 0)):.4f}"
                if top_config.get("hourly_cost") is not None
                else "N/A"
            ),
            (
                f"${float(top_config.get('cost_per_million_tokens', 0)):.4f}"
                if top_config.get("cost_per_million_tokens") is not None
                else "N/A"
            ),
            "Predicted" if is_predicted else "Actual data",
        ],
    }

    return pd.DataFrame(details)


def create_top_configs_plot(
    configs_df: pd.DataFrame, optimization_metric: str = "performance", top_n: int = 10
) -> go.Figure:
    """Create a bar plot of top configurations based on the optimization metric."""
    if configs_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No configurations found",
            xaxis_title="Value",
            yaxis_title="Rank",
            template="plotly_white",
            height=600,
        )
        return fig

    if optimization_metric == "cost":
        sort_col = "cost_per_million_tokens"
        display_col = "Cost/Million Tokens ($)"
        configs_df = configs_df.sort_values(by=sort_col, ascending=True)
    else:
        sort_col = "metrics.result_per_accelerator"
        display_col = "Performance (Tokens/s per device)"
        configs_df = configs_df.sort_values(by=sort_col, ascending=False)

    top_configs = configs_df.head(top_n)

    ranks = [f"#{i + 1}" for i in range(len(top_configs))]

    if optimization_metric == "cost":
        x_values = top_configs["cost_per_million_tokens"]
        color = "crimson"
    else:
        x_values = top_configs["metrics.result_per_accelerator"]
        color = "royalblue"

    hover_text = []
    for _, row in top_configs.iterrows():
        system = row.get("system.name", "Unknown")
        acc_name = row.get("system.accelerator.name", "Unknown")
        acc_count = row.get("system.accelerator.total_count", "?")
        total_perf = row.get("metrics.result", 0)
        per_acc_perf = row.get("metrics.result_per_accelerator", 0)
        cost = row.get("hourly_cost", 0)
        cost_per_million = row.get("cost_per_million_tokens", 0) or 0
        predicted = "Yes" if row.get("predicted", False) else "No"

        info = f"System: {system}<br>"
        info += f"Config: {acc_count}× {acc_name}<br>"
        info += f"Tokens/s (total): {total_perf:.4f}<br>"
        info += f"Tokens/s (per device): {per_acc_perf:.4f}<br>"
        info += f"Hourly cost: ${cost:.4f}<br>"
        info += f"Cost per million tokens: ${cost_per_million:.4f}<br>"
        info += f"Predicted: {predicted}"
        hover_text.append(info)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=ranks,
            x=x_values,
            text=x_values.apply(lambda x: f"{x:.4f}"),
            textposition="auto",
            marker=dict(color=color),
            hovertext=hover_text,
            hoverinfo="text",
            orientation="h",
        )
    )

    title = f"Top {len(ranks)} Configurations by {'Cost' if optimization_metric == 'cost' else 'Performance'}"
    fig.update_layout(
        title=title,
        xaxis_title=display_col,
        yaxis_title="Rank",
        template="plotly_white",
        height=max(400, min(20 * len(ranks), 800)),
        margin=dict(l=50),
    )

    return fig


def recommend_hardware(
    model_size: float,
    weight_data_type: str,
    architecture: str,
    accelerator_vendor: str,
    accelerator_model: str,
    min_gpu_memory: float | None,
    max_gpu_memory: float | None,
    interconnect: str,
    min_accelerators: int | None,
    max_accelerators: int | None,
    cpu_vendor: str,
    cpu_model: str,
    nodes: str,
    min_cpu_memory: float | None,
    max_cpu_memory: float | None,
    os: str,
    include_predictions: bool = True,
    optimization_metric: str = "performance",
    top_n_configs: int = 10,
    **framework_versions,
) -> tuple[pd.DataFrame, pd.DataFrame, str, go.Figure]:
    """Find hardware configurations matching requirements."""
    workload_specs = {
        "model_size": model_size,
        "weight_data_type": weight_data_type,
        "architecture": architecture,
    }

    constraints = {
        "system.accelerator.vendor": accelerator_vendor,
        "system.accelerator.name": accelerator_model,
        "system.interconnect.accelerator": interconnect,
        "system.cpu.vendor": cpu_vendor,
        "system.cpu.model": cpu_model,
        "system.number_of_nodes": nodes if nodes != "Any" else None,
        "software.operating_system": os,
        "min_gpu_memory": min_gpu_memory,
        "max_gpu_memory": max_gpu_memory,
        "min_cpu_memory": min_cpu_memory,
        "max_cpu_memory": max_cpu_memory,
        "min_accelerators": min_accelerators,
        "max_accelerators": max_accelerators,
    }

    for fw_name, version in framework_versions.items():
        if version != "Any":
            constraints[f"software.framework.{fw_name}"] = version

    best_configs = find_best_configs(
        workload_specs, constraints, include_predictions, optimization_metric
    )
    recommendations_df = format_recommendations(best_configs)
    details_df = get_top_config_details(best_configs)

    top_configs_chart = create_top_configs_plot(
        best_configs, optimization_metric, top_n_configs
    )

    if best_configs.empty:
        summary = "No matching configurations found. Try relaxing some constraints or changing the model parameters."
    else:
        actual_count = (
            sum(~best_configs["predicted"])
            if "predicted" in best_configs.columns
            else len(best_configs)
        )
        predicted_count = (
            sum(best_configs["predicted"]) if "predicted" in best_configs.columns else 0
        )

        top_config = best_configs.iloc[0]
        is_predicted = "predicted" in top_config and top_config["predicted"]

        if optimization_metric == "cost":
            metric_value = f"${float(top_config.get('cost_per_million_tokens', 0)):.4f} per million tokens"
            metric_name = "cost"
        else:
            metric_value = f"{float(top_config.get('metrics.result_per_accelerator', 0)):.4f} tokens/s per device"
            metric_name = "performance"

        acc = top_config.get("system.accelerator.name", "Unknown")
        count = top_config.get("system.accelerator.total_count", "Unknown")

        summary = f"Found {actual_count} actual and {predicted_count} predicted configurations. "
        summary += f"\nTop recommendation optimized for {metric_name}: {count}× {acc} with {metric_value}"
        if is_predicted:
            summary += " (Predicted)"

    return recommendations_df, details_df, summary, top_configs_chart


def create_model_performance_plot(
    predictor: PerformancePredictor,
) -> tuple[go.Figure, dict, pd.DataFrame]:
    """Create performance visualization for the ML model using Plotly."""
    logger.info("Starting to create model performance plot")

    empty_metrics = {"rmse": 0, "mae": 0, "r2": 0, "mape": 0}
    empty_df = pd.DataFrame(columns=["Feature", "Importance"])

    empty_fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Predicted vs Actual Performance",
            "Residual Plot (% Error)",
            "Distribution of Prediction Errors",
            "Top 10 Feature Importance",
        ),
    )
    empty_fig.update_layout(
        height=800,
        width=1200,
        showlegend=False,
        title_text="No Model Evaluation Data Available",
        annotations=[
            dict(
                text="Train the model with test data to see evaluation metrics",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
            )
        ],
    )

    if predictor is None:
        logger.warning("No predictor available for performance plot")
        return empty_fig, empty_metrics, empty_df

    if (
        not hasattr(predictor, "evaluation_data")
        or predictor.evaluation_data is None
        or predictor.evaluation_data.empty
    ):
        logger.warning("Evaluation data not found, attempting to re-train model")
        try:
            predictor._train_model()
        except Exception as e:
            logger.error(f"Error re-training model: {e}")

    eval_data = predictor.get_evaluation_data()
    metrics = predictor.get_evaluation_metrics()
    feature_importance = predictor.get_feature_importance()

    logger.info(f"Retrieved evaluation data: {type(eval_data)}")
    if eval_data is not None:
        logger.info(
            f"Evaluation data shape: {eval_data.shape if not eval_data.empty else 'empty'}"
        )

    if eval_data is None or eval_data.empty:
        logger.warning("Evaluation data is not available")
        return (
            empty_fig,
            empty_metrics,
            feature_importance if feature_importance is not None else empty_df,
        )

    logger.info(f"First few rows of evaluation data: {eval_data.head(3).to_dict()}")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Predicted vs Actual Performance",
            "Residual Plot (% Error)",
            "Distribution of Prediction Errors",
            "Top 10 Feature Importance",
        ),
    )

    hover_text = [
        f"Accelerator: {acc}<br>"
        f"Vendor: {vendor}<br>"
        f"Count: {count}<br>"
        f"Actual: {actual:.4f}<br>"
        f"Predicted: {pred:.4f}<br>"
        f"Error: {error:.2f} ({err_pct:.2f}%)"
        for acc, vendor, count, actual, pred, error, err_pct in zip(
            eval_data["system.accelerator.name"],
            eval_data["system.accelerator.vendor"],
            eval_data["system.accelerator.total_count"],
            eval_data["actual"],
            eval_data["predicted"],
            eval_data["error"],
            eval_data["error_percent"],
        )
    ]

    fig.add_trace(
        go.Scatter(
            x=eval_data["actual"],
            y=eval_data["predicted"],
            mode="markers",
            marker=dict(
                opacity=0.6,
                color=eval_data["error_percent"],
                colorscale="RdBu_r",
                colorbar=dict(title="Error %"),
                cmin=-30,
                cmax=30,
            ),
            text=hover_text,
            hoverinfo="text",
            name="Predictions",
        ),
        row=1,
        col=1,
    )

    max_val = max(eval_data["actual"].max(), eval_data["predicted"].max())
    min_val = min(eval_data["actual"].min(), eval_data["predicted"].min())

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Perfect Prediction",
            hoverinfo="none",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=eval_data["predicted"],
            y=eval_data["error_percent"],
            mode="markers",
            marker=dict(
                opacity=0.6,
                color=eval_data["error_percent"],
                colorscale="RdBu_r",
                colorbar=dict(title="Error %"),
                showscale=False,
                cmin=-30,
                cmax=30,
            ),
            text=hover_text,
            hoverinfo="text",
            name="Errors",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Histogram(
            x=eval_data["error_percent"],
            nbinsx=20,
            marker=dict(color="blue", opacity=0.7, line=dict(color="black", width=1)),
            name="Error Distribution",
        ),
        row=2,
        col=1,
    )

    fig.add_vline(x=0, line_dash="dash", line_color="red", row=2, col=1)

    top_features = feature_importance.head(10).sort_values("Importance")

    fig.add_trace(
        go.Bar(
            y=top_features["Feature"],
            x=top_features["Importance"],
            orientation="h",
            marker=dict(color="blue"),
            name="Feature Importance",
        ),
        row=2,
        col=2,
    )

    fig.update_xaxes(title_text="Actual Performance (tokens/s)", row=1, col=1)
    fig.update_yaxes(title_text="Predicted Performance (tokens/s)", row=1, col=1)

    fig.update_xaxes(title_text="Predicted Value", row=1, col=2)
    fig.update_yaxes(title_text="Error (%)", row=1, col=2)

    fig.update_xaxes(title_text="Prediction Error (%)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)

    fig.update_xaxes(title_text="Importance", row=2, col=2)

    fig.update_layout(
        height=800,
        width=1200,
        autosize=True,
        showlegend=False,
        title_text="Model Performance Analysis",
    )

    logger.info("Successfully created model performance plot")
    return fig, metrics, feature_importance.head(10)


with gr.Blocks(title="MLPerf Configuration Finder") as interface:
    gr.Markdown(
        """
    # 🔍 MLPerf Configuration Finder (ongoing preliminary work)
    
    Find the optimal configurations for your AI workloads by specifying your model and constraints.
    Results are ranked by performance and include both real benchmark data and AI-generated predictions.
    
    *All configurations include a ±10% tolerance for continuous features like model size, memory capacity, etc.*
    """
    )

    with gr.Row():
        status_msg = gr.Markdown(
            "*Ready to search. Enter your criteria and click 'Search Configurations'.*"
        )

    with gr.Tabs():
        with gr.TabItem("Workload Specifications"):
            with gr.Accordion("Model Specifications", open=True):
                with gr.Row():
                    architecture = gr.Dropdown(
                        choices=["Any"] + metadata.get("architectures", []),
                        label="Architecture",
                        value="LLM",
                        info="Model architecture type",
                    )
                    weight_data_type = gr.Dropdown(
                        choices=["Any"] + metadata.get("weight_data_types", []),
                        label="Weight Data Type",
                        value="Any",
                        info="Precision format for model weights",
                    )

                model_size = gr.Slider(
                    minimum=metadata.get("model_size_min"),
                    maximum=metadata.get("model_size_max"),
                    value=70,
                    step=1,
                    label="Model Size (billions of parameters)",
                    info="Number of parameters in billions",
                )

            with gr.Accordion("Accelerator (GPU/TPU) Specifications", open=False):
                with gr.Row():
                    accelerator_vendor = gr.Dropdown(
                        choices=["Any"] + metadata.get("accelerator_vendors", []),
                        label="Vendor",
                        value="Any",
                        info="Hardware manufacturer",
                    )
                    accelerator_model = gr.Dropdown(
                        choices=["Any"] + metadata.get("accelerator_models", []),
                        label="Model",
                        value="Any",
                        info="Specific accelerator model",
                    )

                with gr.Row():
                    min_gpu_memory = gr.Slider(
                        minimum=metadata.get("gpu_memory_min"),
                        maximum=metadata.get("gpu_memory_max"),
                        value=metadata.get("gpu_memory_min"),
                        step=1,
                        label="Min GPU Memory (GB)",
                        info="Minimum GPU memory capacity needed",
                    )
                    max_gpu_memory = gr.Slider(
                        minimum=metadata.get("gpu_memory_min"),
                        maximum=metadata.get("gpu_memory_max"),
                        value=metadata.get("gpu_memory_max"),
                        step=1,
                        label="Max GPU Memory (GB)",
                        info="Maximum GPU memory capacity to consider",
                    )

                with gr.Row():
                    interconnect = gr.Dropdown(
                        choices=["Any"] + metadata.get("interconnect_types", []),
                        label="Interconnect",
                        value="Any",
                        info="GPU-to-GPU connection type",
                    )

                with gr.Row():
                    min_accelerators = gr.Slider(
                        minimum=metadata.get("min_accelerators"),
                        maximum=metadata.get("max_accelerators"),
                        value=metadata.get("min_accelerators"),
                        step=1,
                        label="Minimum Accelerators",
                        info="Minimum number of accelerators needed",
                    )
                    max_accelerators = gr.Slider(
                        minimum=metadata.get("min_accelerators"),
                        maximum=metadata.get("max_accelerators"),
                        value=metadata.get("max_accelerators"),
                        step=1,
                        label="Maximum Accelerators",
                        info="Maximum number of accelerators to consider",
                    )

            with gr.Accordion("CPU & System Specifications", open=False):
                with gr.Row():
                    cpu_vendor = gr.Dropdown(
                        choices=["Any"] + metadata.get("cpu_vendors", []),
                        label="CPU Vendor",
                        value="Any",
                        info="CPU manufacturer",
                    )
                    cpu_model = gr.Dropdown(
                        choices=["Any"] + metadata.get("cpu_models", []),
                        label="CPU Model",
                        value="Any",
                        info="Specific CPU model",
                    )

                nodes = gr.Dropdown(
                    choices=["Any"] + [str(n) for n in metadata.get("node_counts", [])],
                    label="Number of Nodes",
                    value="Any",
                    info="Number of physical servers in the system",
                )

                with gr.Row():
                    min_cpu_memory = gr.Slider(
                        minimum=metadata.get("cpu_memory_min"),
                        maximum=metadata.get("cpu_memory_max"),
                        value=metadata.get("cpu_memory_min"),
                        step=1,
                        label="Min System Memory (GB)",
                        info="Minimum system RAM needed",
                    )
                    max_cpu_memory = gr.Slider(
                        minimum=metadata.get("cpu_memory_min"),
                        maximum=metadata.get("cpu_memory_max"),
                        value=metadata.get("cpu_memory_max"),
                        step=1,
                        label="Max System Memory (GB)",
                        info="Maximum system RAM to consider",
                    )

            with gr.Accordion("Software Environment", open=False):
                os = gr.Dropdown(
                    choices=["Any"] + metadata.get("operating_systems", []),
                    label="Operating System",
                    value="Any",
                    info="Host operating system",
                )

                frameworks = [
                    fw
                    for fw in metadata.get("frameworks", [])
                    if f"{fw}_versions" in metadata
                ]
                n_frameworks = len(frameworks)
                column_size = (n_frameworks + 1) // 2

                framework_dropdowns = []
                with gr.Row():
                    for i in range(0, 2):
                        with gr.Column():
                            start_idx = i * column_size
                            end_idx = min((i + 1) * column_size, n_frameworks)

                            if start_idx < n_frameworks:
                                column_frameworks = frameworks[start_idx:end_idx]
                                for fw in column_frameworks:
                                    version_key = f"{fw}_versions"
                                    dropdown = gr.Dropdown(
                                        choices=["Any"] + metadata.get(version_key),
                                        label=fw,
                                        value="Any",
                                        info=f"Select {fw} framework version",
                                    )
                                    framework_dropdowns.append((fw, dropdown))

        with gr.TabItem("Device Cost Settings 💰"):
            gr.Markdown(
                """
            ## Configure Device Hourly Costs

            Customize the hourly cost (in USD) for each accelerator type. These values will be used to
            calculate the cost metrics for hardware configurations.

            Default values may not reflect actual current market prices. Please adjust them according to your needs.
            """
            )

            with gr.Column():
                with gr.Row():
                    save_costs_button = gr.Button(
                        "💾 Save Cost Settings", variant="primary"
                    )
                    reset_costs_button = gr.Button("↻ Reset to Defaults")

                current_costs = get_device_costs()
                cost_data = pd.DataFrame(
                    {
                        "Device": list(current_costs.keys()),
                        "Hourly Cost ($)": list(current_costs.values()),
                    }
                ).sort_values("Device")

                device_costs_df = gr.DataFrame(
                    value=cost_data,
                    datatype=["str", "number"],
                    col_count=(2, "fixed"),
                    interactive=True,
                    wrap=True,
                    show_copy_button=True,
                    show_search="filter",
                )

                costs_status = gr.Markdown("*Device costs ready for customization*")

                def update_costs_callback(df):
                    """Update device costs with values from dataframe."""
                    if isinstance(df, list):
                        new_costs = {
                            row[0]: float(row[1]) for row in df if len(row) >= 2
                        }
                    else:
                        new_costs = {
                            df.loc[i, "Device"]: float(df.loc[i, "Hourly Cost ($)"])
                            for i in range(len(df))
                        }

                    update_device_costs(new_costs)
                    return "*Device costs successfully updated!*"

                def reset_costs_callback():
                    """Reset all costs to defaults."""
                    initialize_device_costs(pd_df)
                    current_costs = get_device_costs()
                    cost_data = pd.DataFrame(
                        {
                            "Device": list(current_costs.keys()),
                            "Hourly Cost ($)": list(current_costs.values()),
                        }
                    ).sort_values("Device")
                    return cost_data, "*Device costs reset to defaults*"

                save_costs_button.click(
                    fn=update_costs_callback,
                    inputs=device_costs_df,
                    outputs=costs_status,
                )

                reset_costs_button.click(
                    fn=reset_costs_callback,
                    inputs=[],
                    outputs=[device_costs_df, costs_status],
                )

    with gr.Row():
        with gr.Accordion("Options", open=True):
            with gr.Row():
                include_predictions = gr.Checkbox(
                    label="Include AI-generated predictions",
                    value=True,
                    info="When enabled, AI will predict performance for configurations not in the benchmark database",
                )
                optimization_metric = gr.Radio(
                    choices=["performance", "cost"],
                    label="Optimization Target",
                    value="performance",
                    info="Choose whether to optimize for highest performance or lowest cost per token",
                )

    with gr.Row():
        search_button = gr.Button(
            "🔍 Search Configurations", variant="primary", scale=3
        )

    with gr.Group():
        summary = gr.Markdown(
            "Enter your requirements and click 'Search Configurations' to find suitable hardware.",
            label="Summary",
        )

    with gr.Tabs():
        with gr.TabItem("Top Configuration Details 🏆"):
            details = gr.DataFrame(
                headers=["Feature", "Value"],
                datatype=["str", "str"],
                label="Configuration Details",
            )

        with gr.TabItem("All Matching Configurations 📊"):
            recommendations = gr.DataFrame(
                headers=[
                    "System",
                    "Accelerator",
                    "Count",
                    "Nodes",
                    "GPU Memory (GB)",
                    "Model",
                    "Architecture",
                    "Parameters (B)",
                    "Weight Data Type",
                    "Total Performance (Tokens/s)",
                    "Per-GPU Performance (Tokens/s)",
                    "Hourly Cost ($)",
                    "Cost/Million Tokens",
                    "Predicted",
                ],
                datatype=[
                    "str",
                    "str",
                    "number",
                    "number",
                    "number",
                    "str",
                    "str",
                    "number",
                    "str",
                    "number",
                    "number",
                    "number",
                    "number",
                    "str",
                ],
                label="Hardware Configurations",
            )

        with gr.TabItem("ML Model Performance 📈"):
            gr.Markdown(
                """
            ## Model Performance Analysis
            This tab shows how well our machine learning model can predict performance for unseen hardware configurations.
            The evaluation is based on a test set that was not used to train the model.
            
            **Hover over data points in the plots to see detailed information about each prediction.**
            """
            )

            model_metrics = gr.Dataframe(
                headers=["Metric", "Value"],
                value=[
                    ["Root Mean Squared Error (RMSE)", 0],
                    ["Mean Absolute Error (MAE)", 0],
                    ["R² Score", 0],
                    ["Mean Absolute Percentage Error (MAPE)", 0],
                ],
                label="Model Performance Metrics",
            )

            feature_importance_df = gr.Dataframe(
                headers=["Feature", "Importance"], label="Feature Importance"
            )

            performance_plot = gr.Plot(
                label="Performance Visualization", elem_id="performance_plot"
            )

    with gr.Row():
        gr.Markdown("## Top Configurations Comparison")

    with gr.Row():
        top_n_configs = gr.Slider(
            minimum=1,
            maximum=100,
            value=10,
            step=1,
            label="Number of configurations to show",
            info="Adjust to see more or fewer configurations in the chart",
        )

    with gr.Row():
        top_configs_chart = gr.Plot(label="")

    current_configs_state = gr.State(pd.DataFrame())

    all_inputs = [
        model_size,
        weight_data_type,
        architecture,
        accelerator_vendor,
        accelerator_model,
        min_gpu_memory,
        max_gpu_memory,
        interconnect,
        min_accelerators,
        max_accelerators,
        cpu_vendor,
        cpu_model,
        nodes,
        min_cpu_memory,
        max_cpu_memory,
        os,
        include_predictions,
        optimization_metric,
        top_n_configs,
    ]

    framework_input_components = [dropdown for _, dropdown in framework_dropdowns]

    def process_framework_inputs(*args):
        base_args = args[: -len(framework_dropdowns)]
        framework_args = args[-len(framework_dropdowns) :]

        framework_versions = {}
        for (framework_name, _), version in zip(framework_dropdowns, framework_args):
            if version != "Any":
                framework_versions[framework_name] = version

        opt_metric = base_args[16]

        results = recommend_hardware(*base_args, **framework_versions)
        recommendations_df, details_df, summary, top_chart = results

        best_configs = find_best_configs(
            {
                "model_size": base_args[0],
                "weight_data_type": base_args[1],
                "architecture": base_args[2],
            },
            constraints=get_constraints_from_args(*base_args),
            include_predictions=base_args[15],
            optimization_metric=opt_metric,
        )

        return (
            recommendations_df,
            details_df,
            summary,
            top_chart,
            best_configs,
        )

    def get_constraints_from_args(*args):
        """Helper function to convert args to constraints dict."""
        return {
            "system.accelerator.vendor": args[3],
            "system.accelerator.name": args[4],
            "system.interconnect.accelerator": args[7],
            "system.cpu.vendor": args[10],
            "system.cpu.model": args[11],
            "system.number_of_nodes": args[12] if args[12] != "Any" else None,
            "software.operating_system": args[15],
            "min_gpu_memory": args[5],
            "max_gpu_memory": args[6],
            "min_cpu_memory": args[13],
            "max_cpu_memory": args[14],
            "min_accelerators": args[8],
            "max_accelerators": args[9],
        }

    def update_chart(n: int, configs_df: pd.DataFrame, metric: str) -> go.Figure:
        """Update the configurations chart based on the slider value."""
        return create_top_configs_plot(configs_df, metric, n)

    search_button.click(
        fn=process_framework_inputs,
        inputs=all_inputs + framework_input_components,
        outputs=[
            recommendations,
            details,
            summary,
            top_configs_chart,
            current_configs_state,
        ],
        show_progress="full",
    )

    top_n_configs.change(
        fn=update_chart,
        inputs=[top_n_configs, current_configs_state, optimization_metric],
        outputs=top_configs_chart,
    )

    def initial_load():
        logger.info("Starting initial load of app")
        default_values = []
        for input_component in all_inputs:
            default_values.append(input_component.value)

        for _, dropdown in framework_dropdowns:
            default_values.append(dropdown.value)

        (
            recommendations_df,
            details_df,
            summary_text,
            top_chart,
            best_configs,
        ) = process_framework_inputs(*default_values)

        if not recommendations_df.empty:
            top_n_configs.maximum = min(100, len(recommendations_df))

        if predictor:
            logger.info("Predictor available, generating performance visualization")
            try:
                plot_fig, metrics, feature_importance = create_model_performance_plot(
                    predictor
                )

                metrics_df = pd.DataFrame(
                    {
                        "Metric": [
                            "Root Mean Squared Error (RMSE)",
                            "Mean Absolute Error (MAE)",
                            "R² Score",
                            "Mean Absolute Percentage Error (MAPE)",
                        ],
                        "Value": [
                            f"{metrics.get('rmse', 0):.4f}",
                            f"{metrics.get('mae', 0):.4f}",
                            f"{metrics.get('r2', 0):.4f}",
                            f"{metrics.get('mape', 0):.2f}%",
                        ],
                    }
                )
                logger.info(f"Created metrics_df with values: {metrics_df.to_dict()}")
            except Exception as e:
                logger.error(f"Error creating performance plot: {e}", exc_info=True)
                plot_fig = go.Figure()
                metrics_df = pd.DataFrame(
                    {
                        "Metric": [
                            "Root Mean Squared Error (RMSE)",
                            "Mean Absolute Error (MAE)",
                            "R² Score",
                            "Mean Absolute Percentage Error (MAPE)",
                        ],
                        "Value": ["N/A", "N/A", "N/A", "N/A"],
                    }
                )
                feature_importance = pd.DataFrame(columns=["Feature", "Importance"])
        else:
            logger.warning("No predictor available for initial load")
            plot_fig = go.Figure()
            plot_fig.update_layout(
                title="No model available",
                annotations=[
                    dict(
                        text="No prediction model available",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                    )
                ],
            )

            metrics_df = pd.DataFrame(
                {
                    "Metric": [
                        "Root Mean Squared Error (RMSE)",
                        "Mean Absolute Error (MAE)",
                        "R² Score",
                        "Mean Absolute Percentage Error (MAPE)",
                    ],
                    "Value": ["N/A", "N/A", "N/A", "N/A"],
                }
            )
            feature_importance = pd.DataFrame(columns=["Feature", "Importance"])

        logger.info("Completed initial load")
        return (
            recommendations_df,
            details_df,
            summary_text,
            plot_fig,
            metrics_df,
            feature_importance,
            top_chart,
            best_configs,
        )

    interface.load(
        fn=initial_load,
        outputs=[
            recommendations,
            details,
            summary,
            performance_plot,
            model_metrics,
            feature_importance_df,
            top_configs_chart,
            current_configs_state,
        ],
        api_name=False,
    )

    gr.Markdown("---")
    gr.HTML("""
        <div style="text-align: center;">
            Authors: <a href="https://www.linkedin.com/in/daltunay">Daniel Altunay</a> and 
            <a href="https://cKnowledge.org/gfursin">Grigori Fursin</a> (FCS Labs)
        </div>
    """)

if __name__ == "__main__":
    interface.launch(share=False)
