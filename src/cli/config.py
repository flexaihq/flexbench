"""Configuration classes for FlexBench CLI."""

import typing as tp
from dataclasses import dataclass

from cli.utils import get_logger

log = get_logger(__name__)

IMAGE_DEFAULTS = {
    "cuda": "vllm/vllm-openai:latest",  # https://hub.docker.com/r/vllm/vllm-openai
    "cpu": "public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.9.1",  # https://gallery.ecr.aws/q9t5s3a7/vllm-cpu-release-repo
    "rocm": "rocm/vllm:latest",  # https://hub.docker.com/r/rocm/vllm
}


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and column mapping."""

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
    hf_token: str | None = None
    vllm_server_token: str | None = None
    backend: str = "loadgen"

    # Accuracy and output configuration
    accuracy: bool = False
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
                raise ValueError(
                    f"Cannot specify both sweep={self.sweep} and target_qps={self.target_qps} for Offline/Server scenarios"
                )
            if self.scenario == "Server" and self.batch_size is not None:
                raise ValueError("Batch size is not applicable for Server scenario")
        elif self.scenario == "SingleStream":
            if self.sweep or self.target_qps is not None:
                pass  # Just ignore these for SingleStream
            if self.accuracy:
                raise ValueError("Accuracy mode is not supported for SingleStream scenario.")

        if self.sweep and self.accuracy:
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
        accuracy_mode=getattr(args, "accuracy", False),
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
        hf_token=getattr(args, "hf_token", None),
        vllm_server_token=getattr(args, "vllm_server_token", None),
        backend=getattr(args, "backend", "loadgen"),
        # Accuracy and output configuration
        accuracy=getattr(args, "accuracy", False),
        output_dir=getattr(args, "output_dir", None),
        # MLPerf configuration (use defaults from dataclass)
        model_name=getattr(args, "model_name", "llama2-70b"),
        config_path=getattr(args, "config_path", "user.conf"),
        enable_trace=getattr(args, "enable_trace", False),
        log_output_to_stdout=getattr(args, "log_output_to_stdout", True),
    )


@dataclass
class DockerConfig:
    """Configuration for Docker containers."""

    # External vLLM server configuration
    vllm_server: str | None = None  # If specified, use existing vLLM server instead of creating one

    # Docker image configuration
    vllm_image: str | None = None  # Will be set based on device_type in __post_init__
    flexbench_image: str = "flexbench:latest"
    network_name: str = "flexbench-network"

    # Device and hardware configuration
    device_type: str = "auto"  # auto, cpu, cuda, rocm, arm
    gpu_devices: list[str] | None = None  # e.g., ["0", "1"] for specific GPUs
    tensor_parallel_size: int | None = None  # Number of GPUs for tensor parallelism

    # vLLM build configuration (only used for ARM)
    vllm_repo: str = "https://github.com/vllm-project/vllm.git"
    vllm_branch: str = "main"
    vllm_build_args: str | dict[str, str] | None = None  # Additional build arguments

    # vLLM server configuration
    vllm_port: int = 8000
    vllm_max_model_len: int = 2048
    vllm_disable_log_requests: bool = True
    vllm_gpu_memory_utilization: float = 0.9  # GPU memory utilization for vLLM

    # Volume mounts and directories
    model_cache_dir: str = "~/.cache/huggingface"  # HuggingFace cache directory
    results_dir: str | None = None  # Host directory for results

    # Container resource limits
    vllm_memory_limit: str | None = None  # e.g., "16g"
    flexbench_memory_limit: str | None = None

    @property
    def custom_vllm_image_name(self) -> str:
        """Return the custom vLLM image name for building from source."""
        return f"vllm-{self.device_type}:latest"

    def __post_init__(self):
        self._resolve_device_type()
        self._validate_gpu_config()
        self._parse_build_args()
        self._set_default_vllm_image()

    def _resolve_device_type(self):
        """Resolve device type if set to 'auto'."""
        if self.device_type == "auto":
            from cli.utils import detect_device_type

            self.device_type = detect_device_type()
            log.info(f"Auto-detected device type: {self.device_type}")

    def _validate_gpu_config(self):
        """Validate GPU configuration consistency."""
        pass

    def _parse_build_args(self):
        """Parse vLLM build arguments from string to dict if needed."""
        if isinstance(self.vllm_build_args, str):
            build_args = {}
            for arg in self.vllm_build_args.split():
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    build_args[key] = value
            self.vllm_build_args = build_args

    def _set_default_vllm_image(self):
        """Set default vLLM image based on device type if not specified."""
        if not self.vllm_image:
            if self.device_type == "arm":
                # ARM builds from source, use custom image name
                self.vllm_image = self.custom_vllm_image_name
            else:
                # Other devices use public images
                if self.device_type not in IMAGE_DEFAULTS:
                    raise ValueError(
                        f"Unsupported device type: {self.device_type}. Supported types: {list(IMAGE_DEFAULTS.keys()) + ['arm']}"
                    )
                self.vllm_image = IMAGE_DEFAULTS[self.device_type]


@dataclass
class FlexBenchDockerConfig:
    """Complete configuration for FlexBench CLI with Docker orchestration."""

    # Core configuration components
    benchmark_config: BenchmarkConfig
    docker_config: DockerConfig

    # CLI execution configuration
    cleanup: bool = True  # Clean up containers after run
    pull_images: bool = True  # Pull latest images before run
    build_flexbench: bool = True  # Build flexbench image if needed
    wait_timeout: int = 300  # Timeout for container startup (seconds)
