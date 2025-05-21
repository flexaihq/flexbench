import streamlit as st
from core.data_manager import DataManager
from st_pages.page_3_dashboard.plots import (plot_bar, plot_box,
                                             plot_correlation_matrix,
                                             plot_histogram, plot_scatter)

st.title("Dashboard", anchor=False)
st.write(
    "Visualize and analyze the data using interactive plots."
    " Select the plot type and configure the parameters to generate the plot."
)


data_manager: DataManager = st.session_state["data_manager"]

df = data_manager.active_df

with st.expander(label="Click to expand", expanded=False, icon="🧮"):
    data_manager.render_dataframe(df, title="Processed Data")


st.header("Plots", divider="gray", anchor=False)

numerical_cols = set(data_manager.get_columns(type="Numerical"))
categorical_cols = set(data_manager.get_columns(type="Categorical"))


def render_correlation_plot(data_manager: DataManager) -> None:
    st.subheader("Correlation Matrix", anchor=False)

    reset_col, config_col = st.columns([1, 10], vertical_alignment="center")
    reset_col.button(
        "Reset",
        key="dashboard.control.correlation.reset",
        use_container_width=True,
        on_click=lambda: st.session_state.pop("dashboard.correlation.columns", None),
    )

    with config_col.container(border=True):
        cols = st.columns([4, 1, 1], vertical_alignment="center")
        numerical_cols = data_manager.get_columns(type="Numerical")

        columns = cols[0].multiselect(
            "Columns to plot",
            options=numerical_cols,
            default=(
                st.session_state.get("dashboard.correlation.columns")
                if st.session_state.get("dashboard.correlation.columns")
                and all(
                    col in numerical_cols
                    for col in st.session_state.get("dashboard.correlation.columns")
                )
                else numerical_cols
            ),
            key="dashboard.correlation.columns",
            help="Numerical columns only",
        )

        method = cols[1].selectbox(
            "Method",
            ["pearson", "spearman"],
            key="dashboard.correlation.method",
            index=0,
            help="Correlation method to use",
        )

        triangle = cols[2].checkbox(
            "Triangle",
            value=st.session_state.get("dashboard.correlation.triangle", False),
            key="dashboard.correlation.triangle",
            help="Show only lower triangle",
        )

        hide_all_nulls = cols[2].checkbox(
            "Hide all nulls",
            value=st.session_state.get("dashboard.correlation.hide_all_nulls", True),
            key="dashboard.correlation.hide_all_nulls",
            help="Hide rows and columns with all null values",
        )

    if df is None or df.is_empty():
        st.error("No data to plot. Try removing some filters.", icon="⚠️")
        return

    fig = plot_correlation_matrix(
        df=df.to_pandas(),
        columns=columns,
        triangle=triangle,
        method=method,
        hide_all_nulls=hide_all_nulls,
    )
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing arguments for correlation matrix.")


def render_scatter_plot(data_manager: DataManager) -> None:
    st.subheader("Scatter Plot", anchor=False)

    reset_col, config_col = st.columns([1, 10], vertical_alignment="center")
    reset_col.button(
        "Reset",
        key="dashboard.control.scatter.reset",
        use_container_width=True,
        on_click=lambda: (
            st.session_state.pop("dashboard.scatter.x", None),
            st.session_state.pop("dashboard.scatter.y", None),
            st.session_state.pop("dashboard.scatter.color", None),
            st.session_state.pop("dashboard.scatter.marker", None),
        ),
    )

    with config_col.container(border=True):
        cols = st.columns(3, vertical_alignment="center")
        numerical_cols = data_manager.get_columns(type="Numerical")
        categorical_cols = data_manager.get_columns(type="Categorical")

        numerical_cols = [col for col in numerical_cols if col in df.columns]
        categorical_cols = [col for col in categorical_cols if col in df.columns]

        for key in ["scatter.x", "scatter.y"]:
            if (
                f"dashboard.{key}" in st.session_state
                and st.session_state[f"dashboard.{key}"] not in numerical_cols
            ):
                st.session_state.pop(f"dashboard.{key}", None)

        x_col = cols[0].selectbox(
            "X column",
            options=numerical_cols,
            key="dashboard.scatter.x",
            index=(
                numerical_cols.index(st.session_state.get("dashboard.scatter.x"))
                if st.session_state.get("dashboard.scatter.x") in numerical_cols
                else None
            ),
            help="Numerical columns only",
        )

        y_col = cols[0].selectbox(
            "Y column",
            options=numerical_cols,
            index=(
                numerical_cols.index(st.session_state.get("dashboard.scatter.y"))
                if st.session_state.get("dashboard.scatter.y") in numerical_cols
                else None
            ),
            key="dashboard.scatter.y",
            help="Numerical columns only",
        )

        color_options = categorical_cols + numerical_cols
        color_col = cols[1].selectbox(
            "Color column",
            options=color_options,
            key="dashboard.scatter.color",
            index=(
                color_options.index(st.session_state.get("dashboard.scatter.color"))
                if st.session_state.get("dashboard.scatter.color") in color_options
                else None
            ),
            help="Categorical or numerical columns",
        )

        marker_col = cols[2].selectbox(
            "Marker column",
            options=categorical_cols,
            key="dashboard.scatter.marker",
            index=(
                categorical_cols.index(st.session_state.get("dashboard.scatter.marker"))
                if st.session_state.get("dashboard.scatter.marker") in categorical_cols
                else None
            ),
            help="Categorical columns only",
        )

    if df is None or df.is_empty():
        st.error("No data to plot. Try removing some filters.", icon="⚠️")
        return

    fig = plot_scatter(
        df=df.to_pandas(),
        x_column=x_col,
        y_column=y_col,
        color_column=color_col,
        marker_column=marker_col,
    )
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing arguments for scatter plot.")


def render_bar_plot(data_manager: DataManager) -> None:
    st.subheader("Bar Plot", anchor=False)

    reset_col, config_col = st.columns([1, 10], vertical_alignment="center")
    reset_col.button(
        "Reset",
        key="dashboard.control.bar.reset",
        use_container_width=True,
        on_click=lambda: (
            st.session_state.pop("dashboard.bar.x", None),
            st.session_state.pop("dashboard.bar.y", None),
            st.session_state.pop("dashboard.bar.color", None),
            st.session_state.pop("dashboard.bar.facet_row", None),
            st.session_state.pop("dashboard.bar.facet_col", None),
            st.session_state.pop("dashboard.bar.mode", None),
        ),
    )

    with config_col.container(border=True):
        cols = st.columns(3, vertical_alignment="center")
        numerical_cols = data_manager.get_columns(type="Numerical")
        categorical_cols = data_manager.get_columns(type="Categorical")
        all_cols = categorical_cols + numerical_cols

        numerical_cols = [col for col in numerical_cols if col in df.columns]
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        all_cols = categorical_cols + numerical_cols

        for key in ["bar.x", "bar.y"]:
            if st.session_state.get(f"dashboard.{key}") not in all_cols:
                st.session_state.pop(f"dashboard.{key}", None)

        x_col = cols[0].selectbox(
            "X column",
            options=all_cols,
            key="dashboard.bar.x",
            index=(
                all_cols.index(st.session_state.get("dashboard.bar.x"))
                if st.session_state.get("dashboard.bar.x") in all_cols
                else None
            ),
        )

        y_col = cols[0].selectbox(
            "Y column",
            options=numerical_cols,
            key="dashboard.bar.y",
            index=None,
        )

        color_cols = cols[1].multiselect(
            "Color column(s)",
            options=all_cols,
            default=(
                st.session_state.get("dashboard.bar.color")
                if st.session_state.get("dashboard.bar.color")
                and all(
                    col in all_cols
                    for col in st.session_state.get("dashboard.bar.color")
                )
                else []
            ),
            key="dashboard.bar.color",
        )

        agg_func = cols[1].selectbox(
            "Aggregation function",
            options=["mean", "median", "min", "max"],
            key="dashboard.bar.agg_func",
            index=(
                ["mean", "median", "min", "max"].index(
                    st.session_state.get("dashboard.bar.agg_func")
                )
                if st.session_state.get("dashboard.bar.agg_func") in all_cols
                else None
            ),
        )

        barmode = cols[1].radio(
            "Bar mode",
            ["group", "stack"],
            key="dashboard.bar.mode",
            index=0,
            help="Used when one or multiple color columns are selected",
            horizontal=True,
        )

        facet_row = cols[2].selectbox(
            "Facet row",
            options=categorical_cols,
            key="dashboard.bar.facet_row",
            index=(
                categorical_cols.index(st.session_state.get("dashboard.bar.facet_row"))
                if st.session_state.get("dashboard.bar.facet_row") in categorical_cols
                else None
            ),
        )

        facet_col = cols[2].selectbox(
            "Facet column",
            options=categorical_cols,
            key="dashboard.bar.facet_col",
            index=(
                categorical_cols.index(st.session_state.get("dashboard.bar.facet_col"))
                if st.session_state.get("dashboard.bar.facet_col") in categorical_cols
                else None
            ),
        )

    if df is None or df.is_empty():
        st.error("No data to plot. Try removing some filters.", icon="⚠️")
        return

    fig = plot_bar(
        df=df.to_pandas(),
        x_column=x_col,
        y_column=y_col,
        color_columns=color_cols,
        facet_row=facet_row,
        facet_col=facet_col,
        barmode=barmode,
        agg_func=agg_func,
    )
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing arguments for bar plot.")


def render_box_plot(data_manager: DataManager) -> None:
    st.subheader("Box Plot", anchor=False)

    reset_col, config_col = st.columns([1, 10], vertical_alignment="center")
    reset_col.button(
        "Reset",
        key="dashboard.control.box.reset",
        use_container_width=True,
        on_click=lambda: (
            st.session_state.pop("dashboard.box.x", None),
            st.session_state.pop("dashboard.box.y", None),
            st.session_state.pop("dashboard.box.color", None),
        ),
    )

    with config_col.container(border=True):
        cols = st.columns(3, vertical_alignment="center")
        numerical_cols = data_manager.get_columns(type="Numerical")
        categorical_cols = data_manager.get_columns(type="Categorical")

        numerical_cols = [col for col in numerical_cols if col in df.columns]
        categorical_cols = [col for col in categorical_cols if col in df.columns]

        for key in ["box.x", "box.y"]:
            if (
                f"dashboard.{key}" in st.session_state
                and st.session_state[f"dashboard.{key}"] not in numerical_cols
            ):
                st.session_state.pop(f"dashboard.{key}", None)

        x_col = cols[0].selectbox(
            "X column",
            options=categorical_cols,
            key="dashboard.box.x",
            index=(
                categorical_cols.index(st.session_state.get("dashboard.box.x"))
                if st.session_state.get("dashboard.box.x") in categorical_cols
                else None
            ),
            help="Categorical columns only",
        )

        y_col = cols[0].selectbox(
            "Y column",
            options=numerical_cols,
            key="dashboard.box.y",
            index=(
                numerical_cols.index(st.session_state.get("dashboard.box.y"))
                if st.session_state.get("dashboard.box.y") in numerical_cols
                else None
            ),
            help="Numerical columns only",
        )

        color_col = cols[1].selectbox(
            "Color column",
            options=categorical_cols,
            key="dashboard.box.color",
            index=(
                categorical_cols.index(st.session_state.get("dashboard.box.color"))
                if st.session_state.get("dashboard.box.color") in categorical_cols
                else None
            ),
            help="Categorical columns only",
        )

    if df is None or df.is_empty():
        st.error("No data to plot. Try removing some filters.", icon="⚠️")
        return

    fig = plot_box(
        df=df.to_pandas(),
        x_column=x_col,
        y_column=y_col,
        color_column=color_col,
    )
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing arguments for box plot.")


def render_histogram_plot(data_manager: DataManager) -> None:
    st.subheader("Histogram Plot", anchor=False)

    reset_col, config_col = st.columns([1, 10], vertical_alignment="center")
    reset_col.button(
        "Reset",
        key="dashboard.control.histogram.reset",
        use_container_width=True,
        on_click=lambda: (
            st.session_state.pop("dashboard.histogram.x", None),
            st.session_state.pop("dashboard.histogram.color", None),
            st.session_state.pop("dashboard.histogram.nbins", None),
            st.session_state.pop("dashboard.histogram.mode", None),
        ),
    )

    with config_col.container(border=True):
        cols = st.columns(3, vertical_alignment="center")
        numerical_cols = data_manager.get_columns(type="Numerical")
        categorical_cols = data_manager.get_columns(type="Categorical")

        numerical_cols = [col for col in numerical_cols if col in df.columns]
        categorical_cols = [col for col in categorical_cols if col in df.columns]

        for key in ["histogram.x"]:
            if (
                f"dashboard.{key}" in st.session_state
                and st.session_state[f"dashboard.{key}"] not in numerical_cols
            ):
                st.session_state.pop(f"dashboard.{key}", None)
            key = ("dashboard.histogram.nbins",)

        x_col = cols[0].selectbox(
            "X column",
            options=numerical_cols,
            key="dashboard.histogram.x",
            index=(
                numerical_cols.index(st.session_state.get("dashboard.histogram.x"))
                if st.session_state.get("dashboard.histogram.x") in numerical_cols
                else None
            ),
            help="Numerical columns only",
        )

        color_col = cols[1].selectbox(
            "Color column",
            options=categorical_cols,
            key="dashboard.histogram.color",
            index=None,
            help="Categorical columns only",
        )

        barmode = cols[1].radio(
            "Bar mode",
            ["group", "stack"],
            key="dashboard.histogram.mode",
            index=["group", "stack"].index(
                st.session_state.get("dashboard.histogram.mode", "group")
            ),
            horizontal=True,
            help="Used when color column is selected",
        )

        nbins = cols[2].slider(
            "Number of bins",
            min_value=5,
            max_value=100,
            value=st.session_state.get("dashboard.histogram.nbins", 30),
            key="dashboard.histogram.nbins",
        )

    if df is None or df.is_empty():
        st.error("No data to plot. Try removing some filters.", icon="⚠️")
        return

    fig = plot_histogram(
        df=df.to_pandas(),
        x_column=x_col,
        color_column=color_col,
        nbins=nbins,
        barmode=barmode,
    )
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing arguments for histogram plot.")


PLOT_TYPES = ["Correlation", "Scatter", "Bar", "Box", "Histogram"]
for i, tab in enumerate(st.tabs(PLOT_TYPES)):
    plot_method_str = PLOT_TYPES[i].lower()
    with tab:
        match plot_method_str:
            case "correlation":
                render_correlation_plot(data_manager)
            case "scatter":
                render_scatter_plot(data_manager)
            case "bar":
                render_bar_plot(data_manager)
            case "box":
                render_box_plot(data_manager)
            case "histogram":
                render_histogram_plot(data_manager)
            case _:
                st.error(f"Unknown plot method: {plot_method_str}")
