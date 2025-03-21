import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from flexbench.utils import get_logger

log = get_logger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and column mapping."""

    path: str
    input_column: str
    output_column: str | None = None  # Only required for accuracy
    system_prompt_column: str | None = None
    image_column: str | None = None
    split: str = "train"
    accuracy_mode: bool = False

    def __post_init__(self):
        if self.accuracy_mode and not self.output_column:
            raise ValueError("output_column is required when running in accuracy mode")


@dataclass
class ReferenceData:
    """Container for reference data used in accuracy evaluation."""

    references: list[str]
    inputs: list[str]
    system_prompts: list[str | None]


class MLPerfDataset(ABC):
    """Base class for MLPerf inference datasets."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
    ) -> None:
        self.config = dataset_config
        self.raw_samples = []
        self.samples = []

        if dataset_config.accuracy_mode and not dataset_config.output_column:
            raise ValueError("output_column must be specified for accuracy evaluation")

        if dataset_config.path.endswith((".pkl.gz", ".pkl")):
            self.load_from_pickle(dataset_config.path)
        else:
            self.load_from_huggingface(dataset_config.path, dataset_config.split)

        log.info(f"Loaded {len(self)} samples from {dataset_config.path}")

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)

    def get_sample(self, index: int) -> tp.Any:
        """Get a single sample by index."""
        return self.samples[index]

    def get_batch(self, indices: tp.Sequence[int]) -> list[tp.Any]:
        """Get multiple samples by indices."""
        return [self.samples[i] for i in indices]

    @abstractmethod
    def _format_sample(self, sample: dict) -> tp.Any:
        """Format a raw sample into the desired format."""
        pass

    def load_from_huggingface(self, dataset_path: str, split: str = "train") -> None:
        """Load and format data from HuggingFace dataset."""
        log.info(f"Loading dataset from HuggingFace: {dataset_path} ({split})")
        from datasets import load_dataset

        dataset = load_dataset(dataset_path, split=split)
        self.raw_samples = list(dataset)
        self.samples = [
            self._format_sample(sample)
            for sample in tqdm(dataset, desc="Formatting samples")
        ]
        log.info(f"Loaded {len(self)} samples")

    def load_from_pickle(self, filepath: str) -> None:
        """Load preprocessed dataset from pickle file."""
        if not Path(filepath).is_file():
            raise FileNotFoundError(f"Processed pickle file {filepath} not found.")

        log.info(f"Loading dataset from pickle file: {filepath}")
        import pandas as pd

        data = pd.read_pickle(filepath)
        self.raw_samples = data.to_dict("records")
        self.samples = [
            self._format_sample(row)
            for _, row in tqdm(data.iterrows(), desc="Formatting samples")
        ]
        log.info(f"Loaded {len(self)} samples")

    def get_references(self) -> ReferenceData:
        """Get raw reference data for accuracy evaluation."""
        if not self.config.accuracy_mode:
            return ReferenceData([], [], [])

        if not self.raw_samples or not self.config.output_column:
            log.warning("No samples or output column found for reference data")
            return ReferenceData([], [], [])

        return ReferenceData(
            references=[
                sample[self.config.output_column] for sample in self.raw_samples
            ],
            inputs=[sample[self.config.input_column] for sample in self.raw_samples],
            system_prompts=[
                (
                    sample.get(self.config.system_prompt_column)
                    if self.config.system_prompt_column
                    else None
                )
                for sample in self.raw_samples
            ],
        )

    def LoadSamplesToRam(self, sample_list: list) -> None:
        """MLPerf LoadGen callback - not used but required."""
        pass

    def UnloadSamplesFromRam(self, sample_list: list) -> None:
        """MLPerf LoadGen callback - not used but required."""
        pass
