import typing as tp
from dataclasses import dataclass

from flexbench.dataset.base import DatasetConfig


@dataclass
class BenchmarkConfig:
    """Configuration for MLPerf benchmark runs."""

    # Required settings
    task: str
    model_path: str
    api_server: str
    dataset_config: DatasetConfig
    scenario: tp.Literal["Offline", "Server"]
    target_qps: float

    # Optional settings
    tokenizer_path: str | None = None
    api_token: str | None = None
    batch_size: int | None = None
    max_generated_tokens: int | None = None
    accuracy: bool = False
    total_sample_count: int | None = None
    model_name: str = "llama2-70b"
    config_path: str = "user.conf"
    enable_trace: bool = False
    log_output_to_stdout: bool = True
