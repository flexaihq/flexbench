"""Configuration classes and builders for FlexBench."""

import typing as tp
from dataclasses import dataclass

from flexbench.utils import get_logger

log = get_logger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and column mapping."""

    path: str
    input_column: str
    output_column: str | None = None
    system_prompt_column: str | None = None
    split: str = "train"
    mode: str = "performance"

    def __post_init__(self):
        if self.mode in ("accuracy", "all") and not self.output_column:
            raise ValueError("output_column is required when running in accuracy mode")


@dataclass
class BenchmarkConfig:
    """Configuration for MLPerf benchmark runs."""

    # Required parameters
    model_path: str
    dataset_config: DatasetConfig
    scenario: tp.Literal["Offline", "Server", "SingleStream"]

    # Performance configuration
    target_qps: float | None = None
    sweep: bool = False
    num_sweep_points: int = 10
    batch_size: int | None = None
    max_generated_tokens: int | None = None
    max_input_tokens: int | None = None
    fixed_input_length: bool = False
    total_sample_count: int | None = None

    # Model and API configuration
    remote_model_path: str | None = None
    tokenizer_path_override: str | None = None
    api_server: str = "http://localhost:8000"
    api_token: str | None = None
    hf_token: str | None = None
    vllm_server_token: str | None = None
    backend: str = "loadgen"

    # Accuracy and output configuration
    mode: str = "performance"
    output_dir: str | None = None

    # MLPerf configuration
    model_name: str = "llama2-70b"
    config_path: str = "user.conf"
    enable_trace: bool = False
    log_output_to_stdout: bool = True

    def __post_init__(self):
        if self.scenario in ("Offline", "Server"):
            if not self.sweep and self.target_qps is None:
                raise ValueError(
                    "Either sweep must be True or target_qps must be specified for Offline/Server scenarios"
                )
            if self.sweep and self.target_qps is not None:
                # raise ValueError(
                #     f"Cannot specify both sweep={self.sweep} and target_qps={self.target_qps} for Offline/Server scenarios"
                # )
                log.warning(
                    f"Both sweep={self.sweep} and target_qps={self.target_qps} specified. Discarding target_qps for sweep mode."
                )
            if self.scenario == "Server" and self.batch_size is not None:
                raise ValueError("Batch size is not applicable for Server scenario")
        elif self.scenario == "SingleStream":
            if self.sweep or self.target_qps is not None:
                pass  # Just ignore these for SingleStream
            if self.mode == "accuracy":
                raise ValueError("Accuracy mode is not supported for SingleStream scenario.")

        if self.sweep and self.mode in ("accuracy", "all"):
            raise ValueError(
                "Sweep mode is not compatible with accuracy testing. Use --target-qps for accuracy mode."
            )
        if self.remote_model_path is None:
            self.remote_model_path = self.model_path


def create_dataset_config(args) -> DatasetConfig:
    """Create DatasetConfig from parsed arguments."""
    return DatasetConfig(
        path=args.dataset_path,
        input_column=args.dataset_input_column,
        output_column=getattr(args, "dataset_output_column", None),
        system_prompt_column=getattr(args, "dataset_system_prompt_column", None),
        split=getattr(args, "dataset_split", "train"),
        mode=getattr(args, "mode", "performance"),
    )


def create_benchmark_config(args, dataset_config: DatasetConfig | None = None) -> BenchmarkConfig:
    """Create BenchmarkConfig from parsed arguments."""

    if dataset_config is None:
        dataset_config = create_dataset_config(args)

    return BenchmarkConfig(
        # Required parameters
        model_path=args.model_path,
        dataset_config=dataset_config,
        scenario=args.scenario,
        # Performance configuration
        target_qps=getattr(args, "target_qps", None),
        sweep=getattr(args, "sweep", False),
        num_sweep_points=getattr(args, "num_sweep_points", 10),
        batch_size=getattr(args, "batch_size", None),
        max_generated_tokens=getattr(args, "max_generated_tokens", None),
        max_input_tokens=getattr(args, "max_input_tokens", None),
        fixed_input_length=getattr(args, "fixed_input_length", False),
        total_sample_count=getattr(args, "total_sample_count", None),
        # Model and API configuration
        remote_model_path=getattr(args, "remote_model_path", args.model_path),
        tokenizer_path_override=getattr(args, "tokenizer_path_override", None),
        api_server=getattr(args, "api_server", "http://localhost:8000"),
        api_token=getattr(args, "api_token", None),
        hf_token=getattr(args, "hf_token", None),
        vllm_server_token=getattr(args, "vllm_server_token", None),
        backend=getattr(args, "backend", "loadgen"),
        # Accuracy and output configuration
        mode=getattr(args, "mode", "performance"),
        output_dir=getattr(args, "output_dir", None),
        # MLPerf configuration (use defaults from dataclass)
        model_name=getattr(args, "model_name", "llama2-70b"),
        config_path=getattr(args, "config_path", "user.conf"),
        enable_trace=getattr(args, "enable_trace", False),
        log_output_to_stdout=getattr(args, "log_output_to_stdout", True),
    )
