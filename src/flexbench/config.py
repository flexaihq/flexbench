"""Configuration classes and builders for FlexBench (text-only)."""

import typing as tp
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and column mapping (text-only)."""

    path: str
    input_column: str
    output_column: str | None = None
    system_prompt_column: str | None = None
    split: str = "train"
    accuracy_mode: bool = False

    def __post_init__(self):
        if self.accuracy_mode and not self.output_column:
            raise ValueError("output_column is required when running in accuracy mode")


@dataclass
class BenchmarkConfig:
    """Configuration for MLPerf benchmark runs (text-only)."""

    model_path: str
    api_server: str
    dataset_config: DatasetConfig
    scenario: tp.Literal["Offline", "Server", "SingleStream"]
    target_qps: float | None = None

    sweep_mode: bool = False
    num_sweep_points: int = 10
    tokenizer_path_override: str | None = None
    remote_model_path: str | None = None
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
        if self.scenario in ("Offline", "Server"):
            if not self.sweep_mode and self.target_qps is None:
                raise ValueError(
                    "Either sweep_mode must be True or target_qps must be specified for Offline/Server scenarios"
                )
            if self.sweep_mode and self.target_qps is not None:
                raise ValueError(
                    f"Cannot specify both sweep_mode={self.sweep_mode} and target_qps={self.target_qps} for Offline/Server scenarios"
                )
            if self.scenario == "Server" and self.batch_size is not None:
                raise ValueError("Batch size is not applicable for Server scenario")
        elif self.scenario == "SingleStream":
            if self.sweep_mode or self.target_qps is not None:
                pass  # Just ignore these for SingleStream
            if self.accuracy:
                raise ValueError("Accuracy mode is not supported for SingleStream scenario.")

        if self.sweep_mode and self.accuracy:
            raise ValueError(
                "Sweep mode is not compatible with accuracy testing. Use --target-qps for accuracy mode."
            )
        if self.remote_model_path is None:
            self.remote_model_path = self.model_path


def create_dataset_config(args) -> DatasetConfig:
    """Create DatasetConfig from parsed arguments (text-only)."""
    return DatasetConfig(
        path=args.dataset_path,
        input_column=args.dataset_input_column,
        output_column=getattr(args, "dataset_output_column", None),
        system_prompt_column=getattr(args, "dataset_system_prompt_column", None),
        split=getattr(args, "dataset_split", "train"),
        accuracy_mode=getattr(args, "accuracy", False),
    )


def create_benchmark_config(args, dataset_config: DatasetConfig | None = None) -> BenchmarkConfig:
    """Create BenchmarkConfig from parsed arguments (text-only)."""

    if dataset_config is None:
        dataset_config = create_dataset_config(args)

    return BenchmarkConfig(
        model_path=args.model_path,
        remote_model_path=getattr(args, "remote_model_path", args.model_path),
        tokenizer_path_override=getattr(args, "tokenizer_path_override", None),
        api_server=getattr(args, "api_server", "http://localhost:8000"),
        api_token=getattr(args, "api_token", None),
        dataset_config=dataset_config,
        scenario=args.scenario,
        target_qps=getattr(args, "target_qps", None),
        sweep_mode=getattr(args, "sweep", False),
        num_sweep_points=getattr(args, "num_points", 10),
        batch_size=getattr(args, "batch_size", None),
        max_generated_tokens=getattr(args, "max_generated_tokens", None),
        max_input_tokens=getattr(args, "max_input_tokens", None),
        fixed_input_length=getattr(args, "fixed_input_length", False),
        accuracy=getattr(args, "accuracy", False),
        total_sample_count=getattr(args, "total_sample_count", None),
        output_dir=getattr(args, "output_dir", None),
    )
