import threading
import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests
from transformers import AutoTokenizer

from flexbench.dataset.base import DatasetConfig
from flexbench.dataset.text import TextDataset
from flexbench.dataset.vision import VisionDataset
from flexbench.utils import get_logger

log = get_logger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for MLPerf benchmark runs."""

    task: str
    model_path: str
    api_server: str
    dataset_config: DatasetConfig
    scenario: tp.Literal["Offline", "Server"]
    target_qps: float | None = None

    sweep_mode: bool = False
    num_sweep_points: int = 10
    tokenizer_path_override: str | None = None
    api_token: str | None = None
    batch_size: int | None = None
    max_generated_tokens: int | None = None
    max_input_tokens: int | None = None
    fixed_input_length: bool = False
    accuracy: bool = False
    total_sample_count: int | None = None
    model_name: str = "llama2-70b"
    config_path: str = "user.conf"
    enable_trace: bool = False
    log_output_to_stdout: bool = True
    output_dir: str | None = None

    def __post_init__(self):
        if not self.sweep_mode and self.target_qps is None:
            raise ValueError(
                "Either sweep_mode must be True or target_qps must be specified"
            )

        if self.sweep_mode and self.target_qps is not None:
            raise ValueError(
                f"Cannot specify both sweep_mode={self.sweep_mode} "
                f"and target_qps={self.target_qps}"
            )

        if self.scenario == "Server" and self.batch_size is not None:
            raise ValueError("Batch size is not applicable for Server scenario")

        if self.sweep_mode and self.accuracy:
            raise ValueError(
                "Sweep mode is not compatible with accuracy testing. "
                "Use --target-qps for accuracy mode."
            )


class BaseRunner(ABC):
    """Base class for benchmark runners."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

        self.results_dir = (
            Path(config.output_dir)
            if config.output_dir
            else Path("results") / datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def run(self) -> dict:
        """Run benchmark and return results."""
        pass


class BaseBackend(ABC):
    """Base class for benchmark backends."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

        if config.task == "text":
            self.dataset = TextDataset(
                dataset_config=config.dataset_config,
                model_path=config.model_path,
                max_generated_tokens=config.max_generated_tokens,
                max_input_tokens=config.max_input_tokens,
                fixed_input_length=config.fixed_input_length,
            )
        elif config.task == "vision":
            self.dataset = VisionDataset(
                dataset_config=config.dataset_config,
                model_path=config.model_path,
                max_generated_tokens=config.max_generated_tokens,
            )
        else:
            raise ValueError(f"Unsupported task type: {config.task}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path_override or config.model_path,
            use_fast=True,
            padding_side="right",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self._active = False
        self.total_sample_count = config.total_sample_count or len(self.dataset)
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    @abstractmethod
    def start(self):
        """Start the backend."""
        self._active = True

    @abstractmethod
    def stop(self):
        """Stop the backend."""
        self._active = False

    @abstractmethod
    def process_query(self, query: dict) -> dict:
        """Process a single query."""
        pass

    def _update_counter(self) -> None:
        """Update and log sample counter in a thread-safe way."""
        with self.sample_counter_lock:
            self.sample_counter += 1
            self.log_progress(self.sample_counter)

    def log_progress(self, count: int):
        """Log processing progress."""
        percent = count / self.total_sample_count * 100
        if count == 1 or count % max(1, self.total_sample_count // 10) == 0:
            log.info(f"Progress: {count}/{self.total_sample_count} ({percent:.1f}%)")

    def _make_api_request(
        self, inputs: str | dict | list, stream: bool = False
    ) -> dict | requests.Response:
        """Make API request to vLLM server with proper headers and error handling."""

        with requests.Session() as s:
            resp = s.post(
                url=f"{self.config.api_server}/v1/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": (
                        f"Bearer {self.config.api_token}"
                        if self.config.api_token
                        else None
                    ),
                },
                json={
                    "model": self.config.model_path,
                    "prompt": inputs,
                    "max_tokens": getattr(self, "max_tokens", 1024),
                    "temperature": 0,
                    "stream": stream,
                    "min_tokens": 1,
                },
                verify=False,
                stream=stream,
            )
            resp.raise_for_status()
            return resp if stream else resp.json()

    def _process_response(self, response: dict | str, streaming: bool = False) -> str:
        """Process API response and extract text content."""
        if isinstance(response, str):
            return response

        if streaming:
            return response["choices"][0]["text"]
        return "".join(choice["text"] for choice in response["choices"])
