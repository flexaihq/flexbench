import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from streamlit import cache_data


def calculate_plot_height(base_height: int, n_items: int, min_height: int = 400) -> int:
    return max(min_height, base_height + 50 * n_items)


@cache_data
def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: list[str],
    triangle: bool = False,
    method: str = "pearson",
    hide_all_nulls: bool = False,
) -> go.Figure | None:
    if not columns:
        return
    corr = df[columns].corr(method=method)
    if hide_all_nulls:
        corr = corr.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if triangle:
        mask = np.triu(np.ones_like(corr, dtype=bool))
        corr = corr.mask(mask)

    fig = px.imshow(
        corr,
        x=corr.columns,
        y=corr.columns,
        text_auto=True,
        color_continuous_scale="RdBu",
        range_color=[-1, 1],
        title=f"Correlation matrix ({method.title()})",
        labels=dict(color="Correlation"),
    )
    fig.update_layout(height=calculate_plot_height(400, len(columns)))
    return fig


@cache_data
def plot_scatter(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: str | None,
    marker_column: str | None,
) -> go.Figure | None:
    if not x_column or not y_column:
        return
    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        color=color_column,
        symbol=marker_column,
        title=f"Scatter plot: {x_column} vs {y_column}",
    )
    fig.update_layout(
        margin={"t": 50, "b": 0, "r": 0, "l": 0, "pad": 0},
        height=600,
        legend=dict(x=1.15),
    )
    return fig


@cache_data
def plot_bar(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_columns: list[str] | None = None,
    facet_row: str | None = None,
    facet_col: str | None = None,
    barmode: str = "group",
    agg_func: str | None = None,
) -> go.Figure | None:
    if not x_column or not y_column:
        return

    if agg_func is not None:
        group_keys = [x_column] + (color_columns if color_columns else [])
        df = df.groupby(group_keys)[y_column].agg(agg_func).reset_index()

    color = None
    if color_columns:
        color = df[color_columns].astype(str).agg("-".join, axis=1)

    fig = px.bar(
        df,
        x=x_column,
        y=y_column,
        color=color,
        facet_row=facet_row,
        facet_col=facet_col,
        barmode=barmode,
        title=f"Bar plot: {x_column} vs {y_column}",
    )
    fig.update_layout(
        margin={"t": 50, "b": 0, "r": 0, "l": 0, "pad": 0},
        height=600,
        legend=dict(x=1.15),
    )
    return fig


@cache_data
def plot_box(
    df: pd.DataFrame, x_column: str, y_column: str, color_column: str | None
) -> go.Figure | None:
    if not x_column or not y_column:
        return
    fig = px.box(
        df,
        x=x_column,
        y=y_column,
        color=color_column,
        title=f"Box plot: {x_column} vs {y_column}",
        points="outliers",
    )
    fig.update_layout(
        margin={"t": 50, "b": 0, "r": 0, "l": 0, "pad": 0},
        height=calculate_plot_height(400, df[x_column].nunique()),
        legend=dict(x=1.15),
    )
    return fig


@cache_data
def plot_histogram(
    df: pd.DataFrame,
    x_column: str,
    color_column: str | None,
    nbins: int = 30,
    barmode: str = "group",
) -> go.Figure | None:
    if not x_column:
        return
    fig = px.histogram(
        df,
        x=x_column,
        color=color_column,
        title=f"Histogram: {x_column}",
        nbins=nbins,
        marginal="box",
    )
    fig.update_layout(
        margin={"t": 50, "b": 0, "r": 0, "l": 0, "pad": 0},
        height=600,
        legend=dict(x=1.15),
        barmode=barmode,
    )
    if color_column:
        fig.update_traces(opacity=0.75)
    return fig
