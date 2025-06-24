"""Docker configuration for FlexBench CLI."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class DatasetConfig:
    """Dataset configuration for benchmark runs."""
    path: str
    input_column: str
    output_column: str | None = None
    system_prompt_column: str | None = None
    image_column: str | None = None
    split: str = "train"
    accuracy_mode: bool = False


@dataclass 
class BenchmarkConfig:
    """Configuration for MLPerf benchmark runs."""
    task: str
    model_path: str
    api_server: str
    dataset_config: DatasetConfig
    scenario: Literal["Offline", "Server", "SingleStream"]
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


@dataclass
class DockerConfig:
    """Configuration for Docker containers."""

    # Docker-specific settings
    vllm_image: str = "vllm/vllm-openai:latest"
    flexbench_image: str = "flexbench:latest"
    network_name: str = "flexbench-network"

    # Device configuration
    device_type: str = "cpu"  # cpu, nvidia, rocm

    # GPU settings
    gpu_devices: list[str] | None = None  # e.g., ["0", "1"] for specific GPUs
    gpu_count: int | None = None  # Total GPU count for tensor parallelism

    # vLLM build settings (for non-nvidia devices)
    vllm_repo: str = "https://github.com/vllm-project/vllm.git"
    vllm_branch: str = "main"
    vllm_build_args: dict[str, str] | None = None  # Additional build arguments

    # vLLM container settings
    vllm_port: int = 8000
    vllm_max_model_len: int = 2048
    vllm_disable_log_requests: bool = True

    # Volume mounts
    model_cache_dir: str | None = None  # Host directory for model cache
    results_dir: str | None = None  # Host directory for results

    # Container resource limits
    vllm_memory_limit: str | None = None  # e.g., "16g"
    flexbench_memory_limit: str | None = None

    @property
    def needs_build_from_source(self) -> bool:
        """Check if we need to build vLLM from source."""
        return self.device_type != "nvidia"

    @property
    def vllm_dockerfile(self) -> str:
        """Get the appropriate Dockerfile for the device type."""
        dockerfile_map = {
            "nvidia": "Dockerfile",
            "cpu": "Dockerfile.cpu",
            "rocm": "Dockerfile.rocm",
            "arm": "Dockerfile.cpu",  # ARM uses CPU dockerfile as fallback
        }
        return dockerfile_map.get(self.device_type, "Dockerfile")

    @property
    def custom_vllm_image_name(self) -> str:
        """Get the custom image name when building from source."""
        return f"vllm-{self.device_type}:latest"

    def __post_init__(self):
        if self.gpu_devices and self.gpu_count:
            if len(self.gpu_devices) != self.gpu_count:
                raise ValueError(
                    f"gpu_devices length ({len(self.gpu_devices)}) must match gpu_count ({self.gpu_count})"
                )

        # Parse build args string into dict
        if isinstance(self.vllm_build_args, str):
            build_args = {}
            for arg in self.vllm_build_args.split():
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    build_args[key] = value
            self.vllm_build_args = build_args


@dataclass
class FlexBenchDockerConfig:
    """Complete configuration for FlexBench CLI with Docker orchestration."""

    # Core benchmark configuration
    benchmark_config: BenchmarkConfig

    # Docker-specific configuration
    docker_config: DockerConfig

    # CLI-specific settings
    cleanup: bool = True  # Clean up containers after run
    pull_images: bool = True  # Pull latest images before run
    build_flexbench: bool = True  # Build flexbench image if needed
    wait_timeout: int = 300  # Timeout for container startup (seconds)

    def __post_init__(self):
        # Override api_server to use Docker network
        self.benchmark_config.api_server = (
            f"http://vllm-server:{self.docker_config.vllm_port}"
        )


def create_dataset_config(args) -> DatasetConfig:
    """Create DatasetConfig from parsed arguments."""
    return DatasetConfig(
        path=args.dataset_path,
        input_column=args.dataset_input_column,
        output_column=getattr(args, 'dataset_output_column', None),
        system_prompt_column=getattr(args, 'dataset_system_prompt_column', None),
        image_column=getattr(args, 'dataset_image_column', None),
        split=getattr(args, 'dataset_split', 'train'),
        accuracy_mode=getattr(args, 'accuracy', False),
    )


def create_benchmark_config(args, dataset_config: DatasetConfig | None = None) -> BenchmarkConfig:
    """Create BenchmarkConfig from parsed arguments."""
    
    if dataset_config is None:
        dataset_config = create_dataset_config(args)
    
    remote_model_path = getattr(args, 'remote_model_path', None) or args.model_path

    return BenchmarkConfig(
        task=args.task,
        model_path=args.model_path,
        remote_model_path=remote_model_path,
        tokenizer_path_override=getattr(args, 'tokenizer_path_override', None),
        api_server=getattr(args, 'api_server', 'http://localhost:8000'),
        api_token=getattr(args, 'api_token', None),
        dataset_config=dataset_config,
        scenario=args.scenario,
        target_qps=getattr(args, 'target_qps', None),
        sweep_mode=getattr(args, 'sweep', False),
        num_sweep_points=getattr(args, 'num_points', 10),
        batch_size=getattr(args, 'batch_size', None),
        max_generated_tokens=getattr(args, 'max_generated_tokens', None),
        max_input_tokens=getattr(args, 'max_input_tokens', None),
        fixed_input_length=getattr(args, 'fixed_input_length', False),
        accuracy=getattr(args, 'accuracy', False),
        total_sample_count=getattr(args, 'total_sample_count', None),
        output_dir=args.output_dir,
    )


def create_docker_config_from_args(args) -> FlexBenchDockerConfig:
    """Create FlexBenchDockerConfig from parsed CLI arguments."""

    # Create BenchmarkConfig using CLI builder
    benchmark_config = create_benchmark_config(args)

    # Parse vLLM build args if provided
    vllm_build_args = getattr(args, "vllm_build_args", None)
    
    # Determine device type
    device_type = getattr(args, "device_type", "cpu")
    
    # Set vLLM image based on device type
    if device_type == "nvidia":
        # Use published image for NVIDIA
        vllm_image = getattr(args, "vllm_image", "vllm/vllm-openai:latest")
    else:
        # Use custom image name for CPU and ROCm (will be built from source)
        vllm_image = f"vllm-{device_type}:latest"

    # Create DockerConfig with CLI-specific options
    docker_config = DockerConfig(
        vllm_image=vllm_image,
        flexbench_image=getattr(args, "flexbench_image", DockerConfig.flexbench_image),
        device_type=device_type,
        gpu_devices=getattr(args, "gpu_devices", None),
        gpu_count=getattr(args, "gpu_count", None),
        vllm_repo=getattr(args, "vllm_repo", DockerConfig.vllm_repo),
        vllm_branch=getattr(args, "vllm_branch", DockerConfig.vllm_branch),
        vllm_build_args=vllm_build_args,
        vllm_port=getattr(args, "vllm_port", DockerConfig.vllm_port),
        vllm_max_model_len=getattr(
            args, "vllm_max_model_len", DockerConfig.vllm_max_model_len
        ),
        model_cache_dir=getattr(args, "model_cache_dir", None),
        results_dir=args.output_dir,
        vllm_memory_limit=getattr(args, "vllm_memory_limit", None),
        flexbench_memory_limit=getattr(args, "flexbench_memory_limit", None),
    )

    return FlexBenchDockerConfig(
        benchmark_config=benchmark_config,
        docker_config=docker_config,
        cleanup=not getattr(args, "no_cleanup", False),
        pull_images=not getattr(args, "no_pull", False),
        build_flexbench=not getattr(args, "no_build", False),
        wait_timeout=getattr(args, "wait_timeout", 300),
    )
