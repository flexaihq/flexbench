import threading
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import requests
from transformers import AutoTokenizer

from flexbench.configs import BenchmarkConfig
from flexbench.dataset.factory import create_dataset
from flexbench.utils import get_logger

log = get_logger(__name__)


class BaseRunner(ABC):
    """Base class for benchmark runners."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results_dir = Path("results") / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def run(self) -> dict:
        """Run benchmark and return results."""
        pass


class BaseBackend(ABC):
    """Base class for benchmark backends."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.dataset = create_dataset(
            config.task,
            config.dataset_config,
            model_path=config.model_path,
            max_generated_tokens=config.max_generated_tokens,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path or config.model_path,
            use_fast=True,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Common tracking
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

    def _create_api_payload(
        self, inputs: str | dict | list, stream: bool = False
    ) -> tuple[str, dict]:
        """Create API payload for vLLM request."""
        endpoint = f"{self.config.api_server}/v1/completions"
        payload = {
            "model": self.config.model_path,
            "prompt": inputs,
            "max_tokens": getattr(self, "max_tokens", 1024),
            "temperature": 0,
            "stream": stream,
            "min_tokens": 1,
        }
        return endpoint, payload

    def _make_api_request(
        self, inputs: str | dict | list, stream: bool = False
    ) -> dict | requests.Response:
        """Make API request with proper headers and error handling."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": (
                f"Bearer {self.config.api_token}" if self.config.api_token else None
            ),
        }
        endpoint, payload = self._create_api_payload(inputs, stream)

        with requests.Session() as s:
            resp = s.post(
                endpoint, headers=headers, json=payload, verify=False, stream=stream
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
