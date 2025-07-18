import numpy as np
import plotly.graph_objects as go
import polars as pl

COLOR_PALETTE = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


def get_acc_color_mapping(df: pl.DataFrame) -> dict[str, str]:
    gpu_names = sorted(df["system.accelerator.name"].drop_nulls().unique())
    return {gpu: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, gpu in enumerate(gpu_names)}


def metric_comparison_bar_plot(
    df: pl.DataFrame,
    selected_model: str,
) -> dict[str, list[go.Figure]]:
    plot_df = df.filter(pl.col("model.name") == selected_model)
    acc_color_mapping = get_acc_color_mapping(plot_df)
    figures_tokens: list[go.Figure] = []
    figures_tokens_per_acc: list[go.Figure] = []
    scenarios = sorted(plot_df["submission.scenario"].unique())
    for scenario in scenarios:
        scenario_df = plot_df.filter(pl.col("submission.scenario") == scenario)
        systems = scenario_df["system.name"].unique()
        best_rows = []
        for sys in systems:
            sys_df = scenario_df.filter(pl.col("system.name") == sys)
            if sys_df.height == 0:
                continue
            idx = sys_df["result.tokens_per_second"].arg_sort(descending=True)[0]
            best_rows.append({col: sys_df[idx, col] for col in sys_df.columns})
        if not best_rows:
            continue
        # Tokens/s
        x_vals = [row["system.name"] for row in best_rows]
        y_vals = [row["result.tokens_per_second"] for row in best_rows]
        acc_vals = [row["system.accelerator.name"] for row in best_rows]
        customdata_vals = list(
            zip(
                x_vals,
                acc_vals,
                [row["system.total_accelerators"] for row in best_rows],
                y_vals,
                ["Tokens/s"] * len(best_rows),
                [row["submission.scenario"] for row in best_rows],
                [row.get("system.price_per_hour") for row in best_rows],
                [row.get("result.cost_per_million_tokens") for row in best_rows],
            )
        )
        fig = go.Figure()
        for acc in sorted(set(acc_vals)):
            indices = [i for i, a in enumerate(acc_vals) if a == acc]
            fig.add_trace(
                go.Bar(
                    x=[x_vals[i] for i in indices],
                    y=[y_vals[i] for i in indices],
                    name=acc,
                    marker=dict(color=acc_color_mapping.get(acc)),  # ty: ignore[no-matching-overload]
                    customdata=[customdata_vals[i] for i in indices],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "<br>"
                        "<b>Accelerator:</b> %{customdata[1]}<br>"
                        "<b>Accelerator Count:</b> %{customdata[2]}<br>"
                        "<b>Scenario:</b> %{customdata[5]}<br>"
                        "<b>Tokens/s:</b> %{customdata[3]:.2f}<br>"
                        "<b>System Price per Hour:</b> $%{customdata[6]:.2f}<br>"
                        "<b>Cost per Million Tokens:</b> $%{customdata[7]:.2f}<br>"
                        "<extra></extra>"
                    ),
                    showlegend=True,
                ),
            )
        fig.update_layout(
            height=600,
            title_text=f"Tokens/s for {selected_model} - Scenario: {scenario}",
            showlegend=True,
            yaxis_title="Tokens/s",
            xaxis_title="System Name",
        )
        fig.update_xaxes(tickangle=45)
        figures_tokens.append(fig)
        # Tokens/s/acc
        y_vals_acc = [
            row["result.tokens_per_second"] / row["system.total_accelerators"]
            if row["system.total_accelerators"]
            else 0.0
            for row in best_rows
        ]
        customdata_vals_acc = list(
            zip(
                x_vals,
                acc_vals,
                [row["system.total_accelerators"] for row in best_rows],
                y_vals_acc,
                ["Tokens/s/acc"] * len(best_rows),
                [row["submission.scenario"] for row in best_rows],
                [row.get("system.price_per_hour") for row in best_rows],
                [row.get("result.cost_per_million_tokens") for row in best_rows],
            )
        )
        fig_acc = go.Figure()
        for acc in sorted(set(acc_vals)):
            indices = [i for i, a in enumerate(acc_vals) if a == acc]
            fig_acc.add_trace(
                go.Bar(
                    x=[x_vals[i] for i in indices],
                    y=[y_vals_acc[i] for i in indices],
                    name=acc,
                    marker=dict(color=acc_color_mapping.get(acc)),  # ty: ignore[no-matching-overload]
                    customdata=[customdata_vals_acc[i] for i in indices],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "<br>"
                        "<b>Accelerator:</b> %{customdata[1]}<br>"
                        "<b>Accelerator Count:</b> %{customdata[2]}<br>"
                        "<b>Scenario:</b> %{customdata[5]}<br>"
                        "<b>Tokens/s/acc:</b> %{customdata[3]:.2f}<br>"
                        "<b>System Price per Hour:</b> $%{customdata[6]:.2f}<br>"
                        "<b>Cost per Million Tokens:</b> $%{customdata[7]:.2f}<br>"
                        "<extra></extra>"
                    ),
                    showlegend=True,
                ),
            )
        fig_acc.update_layout(
            height=600,
            title_text=f"Tokens/s/acc for {selected_model} - Scenario: {scenario}",
            showlegend=True,
            yaxis_title="Tokens/s/acc",
            xaxis_title="System Name",
        )
        fig_acc.update_xaxes(tickangle=45)
        figures_tokens_per_acc.append(fig_acc)
    return {"tokens": figures_tokens, "tokens_per_acc": figures_tokens_per_acc}


def cost_breakdown_bar_plots(
    df: pl.DataFrame,
    selected_model: str,
) -> list[go.Figure]:
    plot_df = df.filter(pl.col("model.name") == selected_model)
    acc_color_mapping = get_acc_color_mapping(plot_df)
    systems = plot_df["system.name"].unique()
    best_rows = []
    for sys in systems:
        sys_df = plot_df.filter(pl.col("system.name") == sys)
        if sys_df.height == 0:
            continue
        idx = sys_df["result.tokens_per_second"].arg_sort(descending=True)[0]
        best_rows.append({col: sys_df[idx, col] for col in sys_df.columns})
    if not best_rows:
        return []
    x_vals = [row["system.name"] for row in best_rows]
    y_vals = [row["result.cost_per_million_tokens"] for row in best_rows]
    acc_vals = [row["system.accelerator.name"] for row in best_rows]
    acc_count_vals = [row["system.total_accelerators"] for row in best_rows]
    price_per_hour = [row["system.price_per_hour"] for row in best_rows]
    tokens_per_sec = [row["result.tokens_per_second"] for row in best_rows]
    fig = go.Figure()
    for acc in sorted(set(acc_vals)):
        indices = [i for i, a in enumerate(acc_vals) if a == acc]
        fig.add_trace(
            go.Bar(
                x=[x_vals[i] for i in indices],
                y=[y_vals[i] for i in indices],
                name=acc,
                marker=dict(color=acc_color_mapping.get(acc)),  # ty: ignore[no-matching-overload]
                customdata=[
                    (
                        x_vals[i],
                        acc_vals[i],
                        acc_count_vals[i],
                        price_per_hour[i],
                        tokens_per_sec[i],
                        y_vals[i],
                    )
                    for i in indices
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "<b>Accelerator:</b> %{customdata[1]}<br>"
                    "<b>Accelerator Count:</b> %{customdata[2]}<br>"
                    "<b>System Price per Hour:</b> $%{customdata[3]:.2f}<br>"
                    "<b>Tokens/s:</b> %{customdata[4]:.2f}<br>"
                    "<b>Cost per Million Tokens:</b> $%{customdata[5]:.2f}<br>"
                    "<extra></extra>"
                ),
                showlegend=True,
            ),
        )
    fig.update_layout(
        height=500,
        title_text=f"System Cost per 1M Tokens - {selected_model}",
        showlegend=True,
        xaxis_title="System Name",
        yaxis_title="Cost per 1M Tokens (USD)",
    )
    fig.update_xaxes(tickangle=45)
    return [fig]


def cost_vs_performance_scatter_plot(
    df: pl.DataFrame,
    selected_model: str,
) -> go.Figure:
    plot_df = df.filter(pl.col("model.name") == selected_model)
    acc_color_mapping = get_acc_color_mapping(plot_df)
    systems = plot_df["system.name"].unique()
    best_rows = []
    for sys in systems:
        sys_df = plot_df.filter(pl.col("system.name") == sys)
        if sys_df.height == 0:
            continue
        idx = sys_df["result.tokens_per_second"].arg_sort(descending=True)[0]
        best_rows.append({col: sys_df[idx, col] for col in sys_df.columns})
    if not best_rows:
        return go.Figure()
    x_vals = [row["system.price_per_hour"] for row in best_rows]
    y_vals = [row["result.tokens_per_second"] for row in best_rows]
    system_names = [row["system.name"] for row in best_rows]
    acc_vals = [row["system.accelerator.name"] for row in best_rows]
    acc_count_vals = [row["system.total_accelerators"] for row in best_rows]
    cost_per_million = [row["result.cost_per_million_tokens"] for row in best_rows]
    colors = [acc_color_mapping.get(acc, None) for acc in acc_vals]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers+text",
            name="Systems",
            text=system_names,
            textposition="top center",
            marker=dict(size=14, symbol="diamond", color=colors),  # ty: ignore[no-matching-overload]
            customdata=list(
                zip(
                    system_names,
                    acc_vals,
                    acc_count_vals,
                    x_vals,
                    y_vals,
                    cost_per_million,
                )
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "<b>Accelerator:</b> %{customdata[1]}<br>"
                "<b>Accelerator Count:</b> %{customdata[2]}<br>"
                "<b>System Price per Hour:</b> $%{customdata[3]:.2f}<br>"
                "<b>Tokens/s:</b> %{customdata[4]:.2f}<br>"
                "<b>Cost per Million Tokens:</b> $%{customdata[5]:.2f}<br>"
                "<extra></extra>"
            ),
        ),
    )
    fig.update_layout(
        height=600,
        title_text=f"Cost vs Performance - {selected_model}",
        showlegend=False,
        xaxis_title="System Price per Hour (USD)",
        yaxis_title="Tokens/s",
    )
    fig.update_xaxes(tickangle=45)
    return fig


def system_cost_sensitivity_plot(
    reference_system: dict,
    comparison_system: dict,
    x_column: str,
    y_column: str,
    x_title: str,
    y_title: str,
    color_title: str,
    range_factor: float = 1.25,
) -> go.Figure:
    """Create a 2D cost sensitivity plot between two systems with rich hover info."""
    all_systems = [reference_system, comparison_system]
    max_x = max(s[x_column] for s in all_systems) * range_factor
    max_y = max(s[y_column] for s in all_systems) * range_factor
    x_range = np.linspace(0.0, max_x, 100)
    y_range = np.linspace(0.0, max_y, 100)
    X, Y = np.meshgrid(x_range, y_range)
    total_delta = ((X * Y) - (reference_system[x_column] * reference_system[y_column])) / 60
    max_abs_diff = max(abs(np.min(total_delta)), abs(np.max(total_delta)))

    # Build customdata for contour hover
    contour_customdata = np.empty(X.shape + (17,), dtype=object)
    # Use comparison_system for hover info
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            contour_customdata[i, j, 0] = comparison_system.get("system.name", "Unknown")
            contour_customdata[i, j, 1] = "Comparison (B)"
            contour_customdata[i, j, 2] = comparison_system.get("submission.organization", "?")
            contour_customdata[i, j, 3] = comparison_system.get("model.name", "?")
            contour_customdata[i, j, 4] = comparison_system.get("benchmark.version", "?")
            contour_customdata[i, j, 5] = comparison_system.get("system.accelerator.name", "?")
            contour_customdata[i, j, 6] = comparison_system.get("system.total_accelerators", "?")
            contour_customdata[i, j, 7] = X[i, j] / (
                comparison_system.get("system.total_accelerators", 1) or 1
            )
            contour_customdata[i, j, 8] = X[i, j]
            contour_customdata[i, j, 9] = Y[i, j]
            contour_customdata[i, j, 10] = Y[i, j] - reference_system[y_column]
            contour_customdata[i, j, 11] = (
                ((Y[i, j] - reference_system[y_column]) / reference_system[y_column] * 100)
                if reference_system[y_column]
                else 0.0
            )
            contour_customdata[i, j, 12] = X[i, j] * Y[i, j] / 60
            contour_customdata[i, j, 13] = total_delta[i, j]
            contour_customdata[i, j, 14] = (
                (total_delta[i, j] / reference_system[x_column] * 100)
                if reference_system[x_column]
                else 0.0
            )
            contour_customdata[i, j, 15] = "Tokens/s"
            contour_customdata[i, j, 16] = "USD"
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
            "└─ vs Reference (A): %{customdata[10]:.2f} (%{customdata[11]:.1f}%)",
            "<br>",
            "<b>Cost (%{customdata[16]}):</b>",
            "└─ Value: %{customdata[12]:.2f}",
            "└─ vs Reference (A): %{customdata[13]:.2f} (%{customdata[14]:.1f}%)",
            "<extra></extra>",
        ]
    )
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=x_range,
            y=y_range,
            z=total_delta,
            customdata=contour_customdata,
            hovertemplate=HOVER_TEMPLATE,
            contours=dict(showlabels=True, labelfont=dict(size=12, color="black")),  # ty: ignore[no-matching-overload]
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
            colorbar=dict(  # ty: ignore[no-matching-overload]
                title=dict(  # ty: ignore[no-matching-overload]
                    text=f"{color_title} delta vs. Reference (A)",
                    font=dict(size=12),  # ty: ignore[no-matching-overload]
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
    # Add system markers with rich hover info
    for system in all_systems:
        is_reference = system == reference_system
        system_name = system.get("system.name", "Unknown")
        system_type = "Reference (A)" if is_reference else "Comparison (B)"
        submitter = system.get("submission.organization", "?")
        model = system.get("model.name", "?")
        mlperf_version = system.get("benchmark.version", "?")
        acc_name = system.get("system.accelerator.name", "?")
        acc_count = system.get("system.total_accelerators", "?")
        system_price = system[x_column]
        acc_price = system_price / (acc_count or 1)
        perf_value = system[y_column]
        perf_delta = perf_value - reference_system[y_column]
        perf_delta_pct = (
            (perf_delta / reference_system[y_column] * 100) if reference_system[y_column] else 0.0
        )
        cost_value = (perf_value * system_price) / 60
        ref_cost = (reference_system[y_column] * reference_system[x_column]) / 60
        cost_delta = cost_value - ref_cost
        cost_delta_pct = (cost_delta / ref_cost * 100) if ref_cost else 0.0
        perf_unit = "Tokens/s"
        cost_unit = "USD"
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
        marker_symbol = "star" if is_reference else "diamond"
        fig.add_trace(
            go.Scatter(
                x=[system_price],
                y=[perf_value],
                mode="markers+text",
                name=f"{system_name} ({system_type})",
                marker=dict(  # ty: ignore[no-matching-overload]
                    symbol=marker_symbol,
                    size=20,
                    color="white",
                    line=dict(color="black", width=2),  # ty: ignore[no-matching-overload]
                ),
                customdata=[point_customdata],
                hovertemplate=HOVER_TEMPLATE,
                showlegend=True,
            )
        )
    fig.update_layout(
        title="System Comparison: B vs. Reference (A)",
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),  # ty: ignore[no-matching-overload]
    )
    return fig
