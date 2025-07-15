import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from flexbench.config import DatasetConfig
from flexbench.utils import get_logger

log = get_logger(__name__)


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

        if dataset_config.path.endswith((".pkl.gz", ".pkl")):
            self._load_from_pickle(dataset_config.path)
        else:
            self._load_from_huggingface(dataset_config.path, dataset_config.split)

        self._format_samples()

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
    def _format_samples_batch(self, samples: list[dict]) -> list[tp.Any]:
        """Format multiple samples at once, with potential parallelization.

        All dataset implementations must provide a batch formatting method.
        This should handle all sample processing efficiently in batch.
        """
        pass

    def _format_samples(self) -> None:
        """Format all raw samples into processed samples."""
        if not self.raw_samples:
            log.warning("No raw samples to format")
            return

        log.info(f"Formatting {len(self.raw_samples)} samples")
        self.samples = self._format_samples_batch(self.raw_samples)

    def _load_from_huggingface(self, dataset_path: str, split: str = "train") -> None:
        """Load raw data from HuggingFace dataset."""
        log.info(f"Loading dataset from HuggingFace: {dataset_path} ({split})")
        from datasets import load_dataset

        dataset = load_dataset(dataset_path, split=split)
        self.raw_samples = list(dataset)
        log.info(f"Loaded {len(self.raw_samples)} raw samples")

    def _load_from_pickle(self, filepath: str) -> None:
        """Load raw data from pickle file."""
        if not Path(filepath).is_file():
            raise FileNotFoundError(f"Processed pickle file {filepath} not found.")

        log.info(f"Loading dataset from pickle file: {filepath}")
        import pandas as pd

        data = pd.read_pickle(filepath)
        self.raw_samples = data.to_dict("records")
        log.info(f"Loaded {len(self.raw_samples)} raw samples")

    def get_references(self) -> ReferenceData:
        """Get raw reference data for accuracy evaluation."""
        if (
            self.config.mode not in ("accuracy", "all")
            or not self.raw_samples
            or not self.config.output_column
        ):
            log.debug("Cannot generate references: accuracy mode disabled or missing data")
            return ReferenceData([], [], [])

        log.debug(f"Processing {len(self.raw_samples)} raw samples for references")
        try:
            references = [sample[self.config.output_column] for sample in self.raw_samples]
            inputs = [sample[self.config.input_column] for sample in self.raw_samples]
            system_prompts = [
                (
                    sample.get(self.config.system_prompt_column)
                    if self.config.system_prompt_column
                    else None
                )
                for sample in self.raw_samples
            ]
            log.debug(f"Created reference data with {len(references)} entries")
            return ReferenceData(references, inputs, system_prompts)
        except KeyError as e:
            log.error(
                f"Column access error: {e}. Available columns: {list(self.raw_samples[0].keys()) if self.raw_samples else 'no samples'}"
            )
            return ReferenceData([], [], [])

    def LoadSamplesToRam(self, sample_list: list) -> None:
        """MLPerf LoadGen callback - not used but required."""
        pass

    def UnloadSamplesFromRam(self, sample_list: list) -> None:
        """MLPerf LoadGen callback - not used but required."""
        pass
