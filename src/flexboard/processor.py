import json
import time

import polars as pl
import structlog
from datasets import load_dataset

logger = structlog.get_logger()


class DataProcessor:
    def __init__(
        self,
        dataset_name: str = "ctuning/OpenMLPerf",
        split: str = "train",
        price_file: str = "accelerator_prices.json",
    ):
        logger.info(
            "Initializing DataProcessor",
            dataset_name=dataset_name,
            split=split,
            price_file=price_file,
        )
        self.dataset_name = dataset_name
        self.split = split
        self.price_file = price_file
        self.original_df = self.df = self._load_dataset()
        self.price_mapping = self._load_price_mapping()

    def _load_dataset(self) -> pl.DataFrame:
        # TODO: load flexbench dataset too
        logger.info("Loading dataset", dataset_name=self.dataset_name, split=self.split)
        mlperf_dataset = load_dataset(self.dataset_name, split=self.split)
        df = mlperf_dataset.to_polars()
        logger.info("Loaded dataset to polars DataFrame", shape=df.shape)
        return df

    def _load_price_mapping(self) -> dict[str, float]:
        logger.info("Loading price mapping from file", price_file=self.price_file)
        with open(self.price_file, "r") as f:
            price_dict = json.load(f)
        return price_dict

    def _add_total_accelerators(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            (
                pl.col("system.number_of_nodes").fill_null(1)
                * pl.col("system.accelerator.count_per_node").fill_null(1)
            ).alias("system.total_accelerators")
        )
        logger.info("Added total accelerators column")
        return df

    def _add_price_per_hour(self, df: pl.DataFrame) -> pl.DataFrame:
        price_df = pl.DataFrame(
            {
                "system.accelerator.name": list(self.price_mapping.keys()),
                "accelerator.price_per_hour": list(self.price_mapping.values()),
            }
        )
        df = df.join(price_df, on="system.accelerator.name", how="left")
        df = df.with_columns(
            (pl.col("system.total_accelerators") * pl.col("accelerator.price_per_hour")).alias(
                "system.price_per_hour"
            )
        )
        logger.info("Added price per hour column based on price mapping")
        return df

    def _focus_tokens_and_cost(self, df: pl.DataFrame) -> pl.DataFrame:
        df = (
            df.filter(pl.col("metrics.units") == "Tokens/s")
            .with_columns(pl.col("metrics.result").alias("result.tokens_per_second"))
            .with_columns(
                (
                    pl.col("system.price_per_hour") / pl.col("result.tokens_per_second") * 1_000_000
                ).alias("result.cost_per_million_tokens")
            )
        )
        logger.info("Filtered for tokens per second and calculated cost per million tokens")
        return df

    def _extract_accuracy_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            pl.col("metrics.accuracy")
            .str.split("  ")
            .map_elements(
                lambda x: {
                    k.strip().lower(): float(v.strip())
                    for k, v in (item.split(": ") for item in x if ": " in item)
                    if k and v
                },
                return_dtype=pl.Object,
            )
            .alias("metrics.accuracy.structured")
        )
        all_accuracy_keys: set[str] = set()
        for accuracy in df["metrics.accuracy.structured"].to_list():
            if isinstance(accuracy, dict):
                all_accuracy_keys.update(accuracy.keys())
        logger.info("Extracted accuracy metrics", keys=sorted(all_accuracy_keys))
        return df

    def update_price(self, price_dict: dict[str, float]) -> None:
        old_price_mapping = self.price_mapping.copy()
        self.price_mapping.update(price_dict)
        changed = {
            k: (old_price_mapping.get(k), v)
            for k, v in price_dict.items()
            if old_price_mapping.get(k) != v
        }
        if not changed:
            logger.info("No price changes detected")
            return
        logger.info("Accelerator prices updated", changed=changed)
        self.run()

    def run(self) -> None:
        logger.info("Starting full processing pipeline")
        t0 = time.perf_counter()
        self.df = (
            self.original_df.pipe(lambda df: df)
            .pipe(self._add_total_accelerators)
            .pipe(self._add_price_per_hour)
            .pipe(self._focus_tokens_and_cost)
            .pipe(self._extract_accuracy_metrics)
        )
        logger.info(
            "Processing complete",
            shape=self.df.shape,
            duration=f"{time.perf_counter() - t0:.3f}s",
        )
        logger.info(
            "Columns after processing",
            columns={col: str(self.df[col].dtype) for col in self.df.columns},
        )


if __name__ == "__main__":
    processor = DataProcessor()
    processor.run()
