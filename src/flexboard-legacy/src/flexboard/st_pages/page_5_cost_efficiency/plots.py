import numpy as np
import plotly.graph_objects as go
import polars as pl


def create_fixed_result_cost_plot(
    ref_system: dict,
    curr_system: dict,
    x_range: tuple[float, float] = None,
    range_factor: float = 1.25,
) -> go.Figure:
    """Create a comparison plot showing cost curves with fixed performance results."""
    if not x_range:
        max_price = max(
            ref_system["system_hourly_price"], curr_system["system_hourly_price"]
        )
        x_range = (0, max_price * range_factor)

    x = np.linspace(x_range[0], x_range[1], 100)

    ref_gpu_cost = x / ref_system["accelerator_count"]
    curr_gpu_cost = x / curr_system["accelerator_count"]

    ref_y = x * ref_system["result"] / 60
    curr_y = x * curr_system["result"] / 60
    max_y = max(ref_y.max(), curr_y.max())

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=ref_y,
            name=f'Reference: {ref_system["system_name"]}',
            line=dict(color="rgba(31, 119, 180, 0.8)", width=2),
            customdata=np.column_stack((
                ref_gpu_cost,
                [ref_system["system_name"]] * len(ref_gpu_cost)
            )),
            hovertemplate=(
                "<b>Reference System: %{customdata[1]}</b><br>"
                f"Accelerator Name: {ref_system['accelerator_name']}<br>"
                f"Accelerator Count: {ref_system['accelerator_count']}<br>"
                "Accelerator Cost: $%{customdata[0]:.2f}/hour<br>"
                "System Cost: $%{x:.2f}/hour<br>"
                "Training Cost: $%{y:.2f}<br>"
                f"Time to Accuracy (TTA): {ref_system['result']:.2f} min<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=curr_y,
            name=f'Current: {curr_system["system_name"]}',
            line=dict(color="rgba(255, 127, 14, 0.8)", width=2),
            customdata=np.column_stack((
                curr_gpu_cost,
                [curr_system["system_name"]] * len(curr_gpu_cost)
            )),
            hovertemplate=(
                "<b>Current System: %{customdata[1]}</b><br>"
                f"Accelerator Name: {curr_system['accelerator_name']}<br>"
                f"Accelerator Count: {curr_system['accelerator_count']}<br>"
                "Accelerator Cost: $%{customdata[0]:.2f}/hour<br>"
                "System Cost: $%{x:.2f}/hour<br>"
                "Training Cost: $%{y:.2f}<br>"
                f"Time to Accuracy (TTA): {curr_system['result']:.2f} min<br>"
                "<extra></extra>"
            ),
        )
    )

    for system, color in [
        (ref_system, "rgb(31, 119, 180)"),
        (curr_system, "rgb(255, 127, 14)"),
    ]:
        gpu_cost = system["system_hourly_price"] / system["accelerator_count"]
        fig.add_trace(
            go.Scatter(
                x=[system["system_hourly_price"]],
                y=[system["system_hourly_price"] * system["result"] / 60],
                mode="markers",
                name=f'Actual {system["system_name"]}',
                marker=dict(color=color, size=12, symbol="star"),
                customdata=[[gpu_cost]],
                hovertemplate=(
                    f"<b>{system['system_name']}</b><br>"
                    f"Accelerator Name: {system['accelerator_name']}<br>"
                    f"Accelerator Count: {system['accelerator_count']}<br>"
                    "Accelerator Cost: $%{customdata[0]:.2f}/hour<br>"
                    f"System Cost: ${system['system_hourly_price']:.2f}/hour<br>"
                    f"Training Cost: ${system['system_hourly_price'] * system['result'] / 60:.2f}<br>"
                    f"Time to Accuracy (TTA): {system['result']:.2f} min<br>"
                    "<extra></extra>"
                ),
            )
        )

        for val, axis in [
            (system["system_hourly_price"], "x"),
            (system["system_hourly_price"] * system["result"] / 60, "y"),
        ]:
            fig.add_shape(
                type="line",
                x0=x_range[0] if axis == "y" else val,
                y0=0 if axis == "x" else val,
                x1=x_range[1] if axis == "y" else val,
                y1=max_y if axis == "x" else val,
                line=dict(color=color, dash="dot", width=1),
            )

    fig.update_layout(
        title="Fixed Result Cost Analysis",
        xaxis_title="System Cost per Hour (USD)",
        yaxis_title="Training Cost (USD)",
        height=600,
        showlegend=True,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )

    return fig


def create_training_cost_comparison_plot(
    df: pl.DataFrame,
    reference_name: str | None = None,
    current_name: str | None = None,
) -> go.Figure:
    """Create a bar plot comparing training costs across all systems."""
    df = df.with_columns(training_cost=df["system_hourly_price"] * df["result"] / 60).sort("training_cost", descending=False)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["system_name"],
            y=df["training_cost"],
            marker_color=[
                "rgba(31, 119, 180, 0.8)" if name == reference_name
                else "rgba(255, 127, 14, 0.8)" if name == current_name
                else "rgba(128, 128, 128, 0.6)"
                for name in df["system_name"]
            ],
            customdata=df.select([
                "system_name",
                "accelerator_name",
                "accelerator_count",
                "system_hourly_price",
                "result",
            ]),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Accelerator Name: %{customdata[1]}<br>"
                "Accelerator Count: %{customdata[2]}<br>"
                "System Cost: $%{customdata[3]:.2f}/hour<br>"
                "Time to Accuracy (TTA): %{customdata[4]:.2f} min<br>"
                "Training Cost: $%{y:.2f}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Training Cost Comparison - All Systems",
        xaxis_title="Systems",
        yaxis_title="Total Training Cost (USD)",
        height=600,  # Increased from 400 to 600
        showlegend=False,
        xaxis_tickangle=45,
    )

    return fig
