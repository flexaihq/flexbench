import plotly.graph_objects as go
import polars as pl


def metric_comparison_bar_plot(
    df: pl.DataFrame,
    selected_model: str,
) -> list[go.Figure]:
    plot_df = df.filter(pl.col("model.name") == selected_model)
    metric_unit = "Tokens/s"
    figures: list[go.Figure] = []
    scenarios = sorted(plot_df["submission.scenario"].unique())
    for scenario in scenarios:
        scenario_df = plot_df.filter(pl.col("submission.scenario") == scenario)
        acc_groups = scenario_df.group_by("system.accelerator.name")
        fig = go.Figure()
        for acc_name, acc_df in acc_groups:
            acc_name_str = acc_name[0] if isinstance(acc_name, tuple) else acc_name
            # Sort by performance descending
            sorted_idx = acc_df["result.tokens_per_second"].arg_sort(descending=True)
            x_vals = [acc_df["system.name"].to_list()[i] for i in sorted_idx]
            y_vals = [acc_df["result.tokens_per_second"].to_list()[i] for i in sorted_idx]
            customdata_vals = list(
                zip(
                    [acc_df["system.name"].to_list()[i] for i in sorted_idx],
                    [acc_df["system.accelerator.name"].to_list()[i] for i in sorted_idx],
                    [acc_df["system.accelerator.total_count"].to_list()[i] for i in sorted_idx],
                    [acc_df["result.tokens_per_second"].to_list()[i] for i in sorted_idx],
                    [metric_unit] * len(sorted_idx),
                    [acc_df["submission.scenario"].to_list()[i] for i in sorted_idx],
                )
            )
            fig.add_trace(
                go.Bar(
                    x=x_vals,
                    y=y_vals,
                    name=acc_name_str,
                    marker=dict(color=None),  # ty: ignore[no-matching-overload]
                    customdata=customdata_vals,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "<br>"
                        "<b>Model Information:</b><br>"
                        f"└─ Model: {selected_model}<br>"
                        "└─ Accelerator: %{customdata[1]}<br>"
                        "└─ Number of accelerators: %{customdata[2]}<br>"
                        "└─ Scenario: %{customdata[5]}<br>"
                        "<br>"
                        "<b>Performance Metric:</b><br>"
                        "└─ %{customdata[4]}: %{customdata[3]:.2f}<br>"
                        "<extra></extra>"
                    ),
                    showlegend=True,
                ),
            )
        fig.update_layout(
            height=600,
            title_text=f"Tokens/s for {selected_model} - Scenario: {scenario}",
            showlegend=True,
            legend_title_text="Accelerators",
            yaxis_title=metric_unit,
            xaxis_title="System Name",
        )
        fig.update_xaxes(tickangle=45)
        figures.append(fig)
    return figures
