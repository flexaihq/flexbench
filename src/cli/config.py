"""CLI-specific configuration classes for FlexBench."""

from dataclasses import dataclass

from cli.utils import get_logger
from flexbench.config import BenchmarkConfig

log = get_logger(__name__)

IMAGE_DEFAULTS = {
    "cuda": "vllm/vllm-openai:latest",  # https://hub.docker.com/r/vllm/vllm-openai
    "cpu": "public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.9.1",  # https://gallery.ecr.aws/q9t5s3a7/vllm-cpu-release-repo
    "rocm": "rocm/vllm:latest",  # https://hub.docker.com/r/rocm/vllm
}


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

    # vLLM server configuration
    vllm_port: int = 8000  # Host port to forward vLLM server (internal port is always 8000)
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
        self._set_default_vllm_image()

    def _resolve_device_type(self):
        """Resolve device type if set to 'auto'."""
        if self.device_type == "auto":
            # Skip device detection if using an existing vLLM server
            if self.vllm_server:
                log.info("Using existing vLLM server - skipping device detection")
                self.device_type = "external"  # Use a placeholder device type
            else:
                from cli.utils import detect_device_type

                self.device_type = detect_device_type()
                log.info(f"Auto-detected device type: {self.device_type}")

    def _validate_gpu_config(self):
        """Validate GPU configuration consistency."""
        pass

    def _set_default_vllm_image(self):
        """Set default vLLM image based on device type if not specified."""
        if not self.vllm_image:
            # Skip image selection if using an external vLLM server
            if self.device_type == "external":
                self.vllm_image = None  # No image needed for external server
            elif self.device_type == "arm":
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
