import numpy as np
import pandas as pd
from plotly import graph_objects as go
from streamlit import cache_data

HOVER_TEMPLATE = "<br>".join(
    [
        "<b>%{customdata[0]}</b>",
        "Type: %{customdata[1]}",
        "<br>",
        "<b>Metadata:</b>",
        "└─ Submitter: %{customdata[2]}",
        "└─ Model: %{customdata[3]}",
        "└─ MLPerf %{customdata[4]}",
        "<br>",
        "<b>System Configuration:</b>",
        "└─ Accelerator: %{customdata[5]}",
        "└─ Count: %{customdata[6]} units",
        "└─ Price per accelerator: %{customdata[7]:.2f} USD/h",
        "└─ Total system price: %{customdata[8]:.2f} USD/h",
        "<br>",
        "<b>Performance (%{customdata[15]}):</b>",
        "└─ Value: %{customdata[9]:.2f}",
        "└─ vs Reference: %{customdata[10]:.2f} (%{customdata[11]:.1f}%)",
        "<br>",
        "<b>Cost (%{customdata[16]}):</b>",
        "└─ Value: %{customdata[12]:.2f}",
        "└─ vs Reference: %{customdata[13]:.2f} (%{customdata[14]:.1f}%)",
        "<extra></extra>",
    ]
)


def create_system_cost_sensitivity_plot(
    ref_system: dict,
    curr_system: dict,
    x_column: str,
    y_column: str,
    x_title: str,
    y_title: str,
    color_title: str,
    additional_systems: list[dict] = None,
    range_factor: float = 1.25,
) -> go.Figure:
    """Create a 2D comparison plot between multiple systems with flexible axes and target metric."""
    all_systems = [ref_system, curr_system] + (additional_systems or [])

    max_x = max(s[x_column] for s in all_systems) * range_factor
    max_y = max(s[y_column] for s in all_systems) * range_factor
    x_range = np.linspace(0.0, max_x, 100)
    y_range = np.linspace(0.0, max_y, 100)

    X, Y = np.meshgrid(x_range, y_range)
    total_delta = ((X * Y) - (ref_system[x_column] * ref_system[y_column])) / 60

    max_abs_diff = max(abs(np.min(total_delta)), abs(np.max(total_delta)))

    fig = go.Figure()

    contour_customdata = np.zeros((X.shape[0], X.shape[1], 17), dtype=object)

    contour_customdata[:, :, 0] = curr_system["system_name"]
    contour_customdata[:, :, 1] = "Simulated configuration"
    contour_customdata[:, :, 2] = curr_system["submitter"]
    contour_customdata[:, :, 3] = curr_system["model_name"]
    contour_customdata[:, :, 4] = curr_system["benchmark_version_alias"]  # TODO: fix 2 columns (_alias)
    contour_customdata[:, :, 5] = curr_system["accelerator_name"]
    contour_customdata[:, :, 6] = curr_system["accelerator_count"]
    contour_customdata[:, :, 7] = X / curr_system["accelerator_count"]
    contour_customdata[:, :, 8] = X
    contour_customdata[:, :, 9] = Y
    contour_customdata[:, :, 10] = Y - ref_system[y_column]
    contour_customdata[:, :, 11] = (
        (Y - ref_system[y_column]) / ref_system[y_column] * 100
    )
    contour_customdata[:, :, 12] = X * Y / 60
    contour_customdata[:, :, 13] = total_delta
    contour_customdata[:, :, 14] = total_delta / ref_system[x_column] * 100
    contour_customdata[:, :, 15] = curr_system["result_unit"]
    contour_customdata[:, :, 16] = curr_system["cost_unit"]

    fig.add_trace(
        go.Contour(
            x=x_range,
            y=y_range,
            z=total_delta,
            customdata=contour_customdata,
            hovertemplate=HOVER_TEMPLATE,
            contours=dict(
                showlabels=True,
                labelfont=dict(size=12, color="black"),
            ),
            colorscale=[
                [0, "rgb(0,109,44)"],
                [0.4, "rgb(116,196,118)"],
                [0.5, "rgb(255,255,255)"],
                [0.6, "rgb(251,106,74)"],
                [1, "rgb(165,0,38)"],
            ],
            contours_coloring="heatmap",
            zmin=-max_abs_diff,
            zmax=max_abs_diff,
            zauto=False,
            colorbar=dict(
                title=dict(
                    text=f"{color_title} delta vs. Reference",
                    font=dict(size=12),
                    side="right",
                ),
                tickmode="array",
                tickvals=[-max_abs_diff, 0, max_abs_diff],
                ticktext=[
                    f"-{max_abs_diff:.2f}<br>(lower)",
                    "0\n(equal)",
                    f"+{max_abs_diff:.2f} and more<br>(higher)",
                ],
                ypad=150,
                thickness=35,
            ),
        )
    )

    for system in all_systems:
        is_reference = system == ref_system
        is_current = system == curr_system

        system_name = system["system_name"]
        system_type = (
            "Reference" if is_reference else "Current" if is_current else "Additional"
        ) + " configuration"
        submitter = system["submitter"]
        model = system["model_name"]
        mlperf_version = system["benchmark_version"]

        acc_name = system["accelerator_name"]
        acc_count = system["accelerator_count"]
        system_price = system[x_column]
        acc_price = system_price / acc_count

        perf_value = system[y_column]
        perf_delta = perf_value - ref_system[y_column]
        perf_delta_pct = (perf_delta / ref_system[y_column]) * 100

        cost_value = (perf_value * system_price) / 60
        ref_cost = (ref_system[y_column] * ref_system[x_column]) / 60
        cost_delta = cost_value - ref_cost
        cost_delta_pct = (cost_delta / ref_cost) * 100

        perf_unit = system["result_unit"]
        cost_unit = system["cost_unit"]

        point_customdata = [
            system_name,
            system_type,
            submitter,
            model,
            mlperf_version,
            acc_name,
            acc_count,
            acc_price,
            system_price,
            perf_value,
            perf_delta,
            perf_delta_pct,
            cost_value,
            cost_delta,
            cost_delta_pct,
            perf_unit,
            cost_unit,
        ]

        marker_symbol = "star" if is_reference else "diamond" if is_current else "circle"

        fig.add_trace(
            go.Scatter(
                x=[system_price],
                y=[perf_value],
                mode="markers+text",
                name=f"{system_name} ({system_type})",
                marker=dict(
                    symbol=marker_symbol,
                    size=20,
                    color="white",
                    line=dict(color="black", width=2),
                ),
                customdata=[point_customdata],
                hovertemplate=HOVER_TEMPLATE,
                showlegend=True,
            )
        )

    _add_system_guide_lines(
        fig=fig, system=ref_system, x_range=(0, max_x), y_range=(0, max_y)
    )
    _add_system_guide_lines(
        fig=fig, system=curr_system, x_range=(0, max_x), y_range=(0, max_y)
    )

    fig.update_layout(
        title=f"System Comparison: {curr_system['system_name']} vs. {ref_system['system_name']}",
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="right",
            x=1
        ),
    )

    return fig


def _add_system_guide_lines(
    fig: go.Figure,
    system: dict,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> None:
    """Add horizontal and vertical guide lines for a system point."""
    for val, axis in [
        (system["system_hourly_price"], "x"),
        (system["result"], "y"),
    ]:
        fig.add_shape(
            type="line",
            x0=x_range[0] if axis == "y" else val,
            y0=y_range[0] if axis == "x" else val,
            x1=x_range[1] if axis == "y" else val,
            y1=y_range[1] if axis == "x" else val,
            line=dict(color="black", dash="dot", width=1),
        )


@cache_data
def create_cost_breakdown_plots(
    plot_df: pd.DataFrame, selected_model: str, acc_color_mapping: dict
) -> go.Figure:
    """Create cost breakdown plot showing system costs."""
    plot_df = plot_df.drop_duplicates(subset=["system_name", "system_hourly_price"])
    plot_df = plot_df.sort_values("system_hourly_price")

    fig = go.Figure()

    acc_groups = plot_df.groupby("accelerator_name")

    for acc_name, acc_df in acc_groups:
        fig.add_trace(
            go.Bar(
                x=acc_df["system_name"],
                y=acc_df["system_hourly_price"],
                name=acc_name,
                marker_color=acc_color_mapping[acc_name],
                customdata=acc_df[
                    [
                        "system_name",
                        "submitter",
                        "accelerator_name",
                        "accelerator_count",
                        "accelerator_hourly_price",
                        "system_hourly_price",
                    ]
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "<br>"
                    "<b>System Information:</b><br>"
                    "└─ Submitter: %{customdata[1]}<br>"
                    "<br>"
                    "<b>Hardware Configuration:</b><br>"
                    "└─ Accelerator: %{customdata[2]}<br>"
                    "└─ Count: %{customdata[3]} units<br>"
                    "└─ Price per accelerator: $%{customdata[4]:.2f}/h<br>"
                    "<br>"
                    "<b>Total System Cost:</b><br>"
                    "└─ $%{customdata[5]:.2f}/hour<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=600,
        title_text=f"System Costs per Hour - {selected_model}",
        showlegend=True,
        legend_title_text="Accelerators",
        xaxis_title="System",
        yaxis_title="Cost per Hour (USD)",
    )

    fig.update_xaxes(tickangle=45)

    return fig


@cache_data
def create_cost_vs_performance_plots(
    plot_df: pd.DataFrame, selected_model: str, acc_color_mapping: dict
) -> list[go.Figure]:
    """Create cost vs performance scatter plots grouped by scenario and metric."""
    metric_groups = plot_df.fillna("N/A").groupby(["scenario", "result_unit"])
    figures = []

    for (scenario, metric_unit), group_df in metric_groups:
        fig = go.Figure()

        for acc_name in group_df["accelerator_name"].unique():
            acc_df = group_df[group_df["accelerator_name"] == acc_name]

            fig.add_trace(
                go.Scatter(
                    x=acc_df["system_hourly_price"],
                    y=acc_df["result"],
                    mode="markers+text",
                    name=acc_name,
                    text=acc_df["system_name"],
                    textposition="top center",
                    marker=dict(
                        color=acc_color_mapping[acc_name],
                        size=12,
                        symbol="diamond",
                    ),
                    customdata=acc_df[
                        [
                            "system_name",
                            "submitter",
                            "accelerator_name",
                            "accelerator_count",
                            "system_hourly_price",
                            "result",
                            "result_unit",
                        ]
                    ],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "<br>"
                        "<b>System Information:</b><br>"
                        "└─ Submitter: %{customdata[1]}<br>"
                        "└─ Accelerator: %{customdata[2]}<br>"
                        "└─ Count: %{customdata[3]} units<br>"
                        "<br>"
                        "<b>Metrics:</b><br>"
                        "└─ Cost: $%{customdata[4]:.2f}/hour<br>"
                        "└─ Performance: %{customdata[5]:.2f} %{customdata[6]}<br>"
                        "<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            height=600,
            title_text=f"Cost vs Performance Trade-off - {selected_model}<br>Scenario: {scenario} ({metric_unit})",
            showlegend=True,
            legend_title_text="Accelerators",
            xaxis_title="System Cost per Hour (USD)",
            yaxis_title=f"Performance ({metric_unit})",
        )

        figures.append(fig)

    return figures
