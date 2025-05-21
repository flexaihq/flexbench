import json
import os
import typing as t
from pathlib import Path

import cmind
import polars as pl
import requests
import streamlit as st

SCHEMA_PATH = (
    Path(__file__).parent.parent / "st_pages/page_4_comparison/schema_mapping.json"
)
PRICES_PATH = (
    Path(__file__).parent.parent / "st_pages/page_4_comparison/accelerator_prices.json"
)


class DataManager:
    def __init__(self):
        self.source_data: dict[str, pl.DataFrame] = {}
        self.active_df: pl.DataFrame = pl.DataFrame()
        with open(SCHEMA_PATH) as f:
            self.schema_mapping = json.load(f)
        self.prices = {
            "accelerator": self._load_default_prices(),
            "system": {},
        }

    def _load_default_prices(self) -> dict[str, float]:
        """Load default accelerator prices from JSON file"""
        with open(PRICES_PATH) as f:
            prices = json.load(f)
        return prices

    def update_prices(
        self, acc_prices: dict[str, float], sys_prices: dict[str, float]
    ) -> None:
        """Update price configurations"""
        self.prices["accelerator"].update(acc_prices)
        self.prices["system"].update(sys_prices)
        self.process_data()

    def add_source(self, source: str) -> None:
        """Add a source and fetch its data"""
        if self.source_data.get(source, pl.DataFrame()).is_empty():
            json_data = self.fetch_data_from_source(source)
            df = pl.DataFrame(json_data, infer_schema_length=None)
            self.source_data[source] = df

    def remove_source(self, source: str) -> None:
        """Remove a source and its data"""
        if source == "all":
            for source in ["database", "local", "cmx"]:
                self.remove_source(source)
        self.source_data.pop(source, None)

    def update_source(self, source: str) -> None:
        """Update source and fetch its data"""
        if source == "all":
            for source in ["database", "local", "cmx"]:
                self.update_source(source)

        use_source = st.session_state.get(f"data.use_{source}")
        if use_source:
            try:
                self.add_source(source)
            except Exception as e:
                st.session_state[f"data.use_{source}"] = False
                st.toast(f"Error fetching data from {source}: {e}", icon="⚠️")
        else:
            self.remove_source(source)

    @staticmethod
    @st.cache_data
    def fetch_data_from_source(source: str) -> list[dict[str, t.Any]]:
        """Fetch data from specified source and return as DataFrame"""
        match source:

            case "database":
                response = requests.get("http://localhost:8000/pull")
                response.raise_for_status()
                json_data = response.json()

            case "local":
                local_files = st.session_state.get("data.use_local.files", [])
                json_data = [item for file in local_files for item in json.load(file)]

            case "cmx":
                # Load data from CMX
                experiment_name = os.environ.get("CMX_FLEXBOARD_EXPERIMENT_NAME", "")
                experiment_tags = os.environ.get("CMX_FLEXBOARD_EXPERIMENT_TAGS", "")

                ii = {"action": "get", "automation": "flex.experiment"}

                if experiment_name != "":
                    ii["artifact"] = experiment_name
                if experiment_tags != "":
                    ii["tags"] = experiment_tags

                cmind.repos = None
                cmind.index = None

                r = cmind.x(ii)

                if r["return"] > 0:
                    raise ValueError(r["error"])

                json_data = r["summary"]

                # TBD: handle CMX errors here

        return json_data

    def combine_active_sources(self) -> pl.DataFrame:
        """Combine all active data sources into a single DataFrame"""
        dfs = list(self.source_data.values())
        if not dfs:
            return pl.DataFrame()
        return pl.concat(dfs, how="diagonal_relaxed")

    @property
    def base_df(self) -> pl.DataFrame:
        """Get the base DataFrame without any processing"""
        return (
            self.combine_active_sources()
            .pipe(self._clean_types)
            .pipe(self._map_schema)  # TODO: metric creation happens after schema mapping
            .pipe(self._handle_case_duplicates)
        )

    def process_data(self) -> None:
        """Process data in a single pipeline"""
        self.active_df = self.base_df.pipe(self._calculate_metrics).pipe(
            self._filter_by_selections
        )

    def _clean_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """Clean data types and remove empty columns"""
        if df.is_empty():
            return df

        df = df.with_columns(
            [
                pl.when(pl.col(col) == "N/A")
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
                for col in df.columns
                if not df[col].dtype.is_numeric()
            ]
        )

        for col in df.columns:
            if df[col].drop_nulls().is_empty():
                df.drop_in_place(col)
                continue

            if df[col].dtype.is_numeric():
                continue

            try:
                df = df.with_columns(
                    pl.col(col).cast(pl.Float64, strict=True).alias(col)
                )
            except pl.exceptions.InvalidOperationError:
                pass

        return df

    def _filter_by_selections(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filter DataFrame based on session state selections"""
        if df.is_empty():
            return df

        filters = []
        if benchmark := st.session_state.get("data.selected_benchmark_name"):
            filters.append(pl.col("benchmark_name") == benchmark)
        if scenario := st.session_state.get("data.selected_scenario"):
            filters.append(pl.col("scenario") == scenario)
        if version := st.session_state.get("data.selected_version"):
            filters.append(pl.col("benchmark_version") == version)
        if metric := st.session_state.get("data.selected_metric"):
            filters.append(pl.col("units") == metric)  # TODO: fix

        return df.filter(pl.all_horizontal(filters)) if filters else df

    def _map_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize column names based on schema mapping"""
        if df.is_empty() or "benchmark_name" not in df:
            return df

        for benchmark_name in df["benchmark_name"].drop_nulls().unique().to_list():
            mapping = self.schema_mapping[benchmark_name]

            for ref_name, initial_name in mapping.items():
                if initial_name is None or initial_name not in df.columns:
                    continue

                df = df.with_columns(
                    pl.when(pl.col("benchmark_name") == benchmark_name)
                    .then(pl.col(initial_name))
                    .otherwise(pl.col(ref_name) if ref_name in df.columns else None)
                    .alias(ref_name)
                )

        df = df.with_columns(
            accelerator_count=pl.coalesce(
                pl.col("accelerator_count"),
                pl.col("system.number_of_nodes")
                * pl.col("system.accelerators_per_node"),
            )
        )

        df = df.with_columns(
            system_name=pl.concat_str(
                pl.col("submitter"),
                pl.col("accelerator_name"),
                pl.col("system_name").str.replace(".json", ""),
                separator=" | ",
            )
        )

        return df

    def _handle_case_duplicates(self, df: pl.DataFrame) -> pl.DataFrame:
        """Handle case-insensitive column name duplicates using coalesce."""
        column_groups = {}
        for col in df.columns:
            column_groups.setdefault(col.lower(), []).append(col)

        for lowercase_name, columns in column_groups.items():
            if len(columns) > 1:
                df = df.with_columns(
                    pl.coalesce([pl.col(col) for col in columns]).alias(
                        f"{lowercase_name}_temp"
                    )
                )
                df = df.drop(columns).rename({f"{lowercase_name}_temp": lowercase_name})
            elif columns[0] != lowercase_name:
                df = df.rename({columns[0]: lowercase_name})

        df = df.rename({col: col.lower() for col in df.columns})

        return df

    def _calculate_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df

        # TBD GF: should depend on the input source (universal, MLPerf, etc)
        if "benchmark_name" in df:
            if any(self.prices.values()):
                df = df.with_columns(
                    [
                        pl.col("accelerator_name")
                        .map_elements(
                            lambda x: self.prices["accelerator"].get(x),
                            return_dtype=pl.Float64,
                        )
                        .alias("accelerator_hourly_price"),
                        pl.col("system_name")
                        .map_elements(
                            lambda x: self.prices["system"].get(x),
                            return_dtype=pl.Float64,
                        )
                        .alias("system_hourly_price_override"),
                        (
                            pl.col("accelerator_name").map_elements(
                                lambda x: self.prices["accelerator"].get(x),
                                return_dtype=pl.Float64,
                            )
                            * pl.col("accelerator_count")
                        ).alias("system_hourly_price_base"),
                    ]
                )

                df = df.with_columns(
                    pl.coalesce(
                        "system_hourly_price_override", "system_hourly_price_base"
                    ).alias("system_hourly_price")
                )

            metric_expr = self._metric_expressions

            if metric_expr:
                df = df.with_columns(metric_expr)

        return df

    @property
    def _metric_expressions(self) -> list[pl.Expr] | None:
        """Get metric calculation expressions based on benchmark type"""
        return [
            pl.when(pl.col("benchmark_name") == "mlperf-training")
            .then(pl.lit("TTA (min)"))
            .when(pl.col("benchmark_name") == "mlperf-inference")
            .then(pl.col("result_unit"))
            .alias("result_unit"),
            pl.when(pl.col("benchmark_name") == "mlperf-training")
            .then(pl.col("result") * pl.col("accelerator_count"))
            .when(pl.col("benchmark_name") == "mlperf-inference")
            .then(
                pl.when(
                    pl.col("result_unit").is_in(["Samples/s", "Queries/s", "Tokens/s"])
                )
                .then(pl.col("result") / pl.col("accelerator_count"))
                .when(
                    pl.col("result_unit").is_in(["Latency (ms)", "Watts", "millijoules"])
                )
                .then(pl.col("result") * pl.col("accelerator_count"))
            )
            .alias("result_norm"),
            pl.when(pl.col("benchmark_name") == "mlperf-training")
            .then(pl.lit("TTA (min)*acc"))
            .when(pl.col("benchmark_name") == "mlperf-inference")
            .then(
                pl.when(pl.col("result_unit") == "Samples/s")
                .then(pl.lit("Samples/s/acc"))
                .when(pl.col("result_unit") == "Queries/s")
                .then(pl.lit("Queries/s/acc"))
                .when(pl.col("result_unit") == "Tokens/s")
                .then(pl.lit("Tokens/s/acc"))
                .when(pl.col("result_unit") == "Latency (ms)")
                .then(pl.lit("Latency (ms)*acc"))
                .when(pl.col("result_unit") == "Watts")
                .then(pl.lit("Watts/acc"))
                .when(pl.col("result_unit") == "millijoules")
                .then(pl.lit("millijoules/acc"))
            )
            .alias("result_norm_unit"),
            pl.when(pl.col("benchmark_name") == "mlperf-training")
            .then(pl.col("result") / 60 * pl.col("system_hourly_price"))
            .when(pl.col("benchmark_name") == "mlperf-inference")
            .then(
                pl.when(pl.col("result_unit").is_in(["Samples/s", "Queries/s"]))
                .then(
                    (pl.col("system_hourly_price") / pl.col("result"))
                    * 1_000_000
                    / 3_600
                )
                .when(pl.col("result_unit") == "Latency (ms)")
                .then(pl.col("result") * pl.col("system_hourly_price") / 3_600_000)
                .when(pl.col("result_unit").is_in(["Watts", "millijoules"]))
                .then(pl.col("result"))
            )
            .alias("cost"),
            pl.when(pl.col("benchmark_name") == "mlperf-training")
            .then(pl.lit("USD/training"))
            .when(pl.col("benchmark_name") == "mlperf-inference")
            .then(
                pl.when(pl.col("result_unit") == "Samples/s")
                .then(pl.lit("USD/million samples"))
                .when(pl.col("result_unit") == "Queries/s")
                .then(pl.lit("USD/million queries"))
                .when(pl.col("result_unit") == "Latency (ms)")
                .then(pl.lit("USD*ms/query"))
                .when(pl.col("result_unit") == "Watts")
                .then(pl.lit("Watts/acc"))
                .when(pl.col("result_unit") == "millijoules")
                .then(pl.lit("millijoules/acc"))
            )
            .alias("cost_unit"),
        ]

    def get_columns(
        self, type: t.Literal["Numerical", "Categorical"] | None = None
    ) -> list[str]:
        """Get columns of specified type"""
        if type == "Numerical":
            return [
                col
                for col in self.active_df.columns
                if self.active_df[col].dtype.is_numeric()
            ]
        return [
            col
            for col in self.active_df.columns
            if not self.active_df[col].dtype.is_numeric()
        ]

    @staticmethod
    def render_dataframe(df: pl.DataFrame, title: str | None = None) -> None:
        """Render a DataFrame with statistics"""
        if df.is_empty():
            st.warning("No data to display.", icon="⚠️")
            return

        if title:
            st.caption(title)

        df = df[[s.name for s in df if not (s.null_count() == df.height)]].clone()
        df = df.select(sorted(df.columns))

        selector_cols = st.columns([5, 1, 3], vertical_alignment="center", gap="medium")
        nrows = selector_cols[0].slider(
            "Number of rows to display",
            min_value=1 if len(df) > 1 else 0,
            max_value=len(df) if len(df) > 1 else 1,
            value=st.session_state.get("datamanager.nrows", 100),
            key="datamanager.nrows",
            disabled=len(df) == 1,
        )
        nrows = min(nrows, len(df))
        selector_cols[1].button(
            "Display all",
            key="datamanager.display_all",
            on_click=lambda: setattr(st.session_state, "datamanager.nrows", len(df)),
            use_container_width=True,
        )
        selector_cols[2].caption(f"Showing {nrows} rows out of {len(df)} total")
        st.dataframe(df.head(nrows), use_container_width=True)

        st_cols = st.columns([2, 1])
        stats_cols = st_cols[0].columns(2, gap="large")

        num_cols = [col for col in df.columns if df[col].dtype.is_numeric()]
        num_stats = df.select(num_cols).describe() if num_cols else pl.DataFrame()
        stats_cols[0].popover(
            "Numerical columns",
            use_container_width=True,
        ).dataframe(num_stats, use_container_width=True)

        cat_cols = [col for col in df.columns if not df[col].dtype.is_numeric()]
        cat_stats = df.select(cat_cols).describe() if cat_cols else pl.DataFrame()
        stats_cols[1].popover(
            "Categorical columns",
            use_container_width=True,
        ).dataframe(cat_stats, use_container_width=True)

        download_col = st_cols[1].columns(2)[1]
        download_col.download_button(
            "Download as JSON",
            data=df.write_json(),
            file_name="session_state.json",
            mime="application/json",
            icon="📥",
            use_container_width=True,
            type="primary",
        )
