import pandas as pd
from plotly import graph_objects as go
from plotly import subplots
from streamlit import cache_data


@cache_data
def metric_comparison_bar_plot(
    plot_df: pd.DataFrame,
    selected_model: str,
    show_norm: str = "Base Metrics",
) -> list[go.Figure]:
    metric_groups = plot_df.groupby("result_unit")
    figures = []

    for metric_unit, group_df in metric_groups:
        metric_type = "result_norm" if show_norm == "Normalized Metrics" else "result"
        unit_col = f"{metric_type}_unit"
        current_unit = (
            group_df[unit_col].iloc[0]
            if show_norm == "Normalized Metrics"
            else metric_unit
        )

        # Group by scenario first
        scenarios = sorted(group_df["scenario"].unique())
        n_scenarios = len(scenarios)

        fig = subplots.make_subplots(
            rows=1,
            cols=n_scenarios,
            subplot_titles=[f"Scenario: {s}" for s in scenarios],
            shared_yaxes=True,
        )

        # For each scenario
        for col, scenario in enumerate(scenarios, 1):
            scenario_df = (
                group_df[group_df["scenario"] == scenario]
                if scenario is not None
                else group_df[group_df["scenario"].isnull()]
            )

            # Then group by accelerator within each scenario
            acc_groups = scenario_df.groupby("accelerator_name")
            
            # TODO: for inference, also group by processor info

            for acc_name, acc_df in acc_groups:
                fig.add_trace(
                    go.Bar(
                        x=acc_df["system_name"],
                        y=acc_df[metric_type],
                        name=acc_name,
                        marker=dict(
                            color=acc_df["color"].iloc[0],
                        ),
                        customdata=acc_df[
                            [
                                "system_name",
                                "accelerator_name",
                                "accelerator_count",
                                metric_type,
                                unit_col,
                                "scenario",
                            ]
                        ],
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
                        showlegend=col == 1,  # Show legend only for first column
                    ),
                    row=1,
                    col=col,
                )

        fig.update_layout(
            height=600,
            title_text=f"{metric_unit} for {selected_model}",
            showlegend=True,
            legend_title_text="Accelerators",
            yaxis_title=current_unit,
        )

        # Update all x-axes
        for i in range(n_scenarios):
            fig.update_xaxes(tickangle=45, row=1, col=i + 1)

        figures.append(fig)

    return figures


@cache_data
def create_parallel_coordinates_plot(
    plot_df: pd.DataFrame,
    selected_model: str,
) -> go.Figure:
    """Create parallel coordinates plot for multi-dimensional analysis."""
    dimensions = [
        dict(
            range=[0, plot_df["result"].max()],
            label=plot_df["result_unit"].iloc[0],
            values=plot_df["result"],
        ),
        dict(
            range=[0, plot_df["result_norm"].max()],
            label=plot_df["result_norm_unit"].iloc[0],
            values=plot_df["result_norm"],
        ),
        dict(
            range=[0, plot_df["accelerator_count"].max()],
            label="Accelerator Count",
            values=plot_df["accelerator_count"],
        ),
    ]

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=plot_df["color"],
                showscale=False,
            ),
            dimensions=dimensions,
            customdata=plot_df[["system_name", "submitter"]],
        )
    )

    fig.update_layout(
        title=f"Multi-dimensional Performance Analysis for {selected_model}",
        height=600,
    )
    return fig
