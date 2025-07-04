"""Docker configuration for FlexBench CLI (text-only)."""

import sys
from dataclasses import dataclass
from pathlib import Path

from cli.utils import get_logger
from flexbench.config import BenchmarkConfig

log = get_logger(__name__)


@dataclass
class DockerConfig:
    """Configuration for Docker containers."""

    # External vLLM server configuration
    api_server: str | None = None  # If specified, use existing vLLM server instead of creating one

    # Docker-specific settings
    vllm_image: str | None = None  # Will be set based on device_type in __post_init__
    flexbench_image: str = "flexbench:latest"
    network_name: str = "flexbench-network"

    # Device configuration
    device_type: str = "auto"  # auto, cpu, cuda, rocm, arm

    # GPU settings
    gpu_devices: list[str] | None = None  # e.g., ["0", "1"] for specific GPUs
    tensor_parallel_size: int | None = None  # Number of GPUs for tensor parallelism

    # vLLM build settings (only used for ARM)
    vllm_repo: str = "https://github.com/vllm-project/vllm.git"
    vllm_branch: str = "main"
    vllm_build_args: dict[str, str] | None = None  # Additional build arguments

    # vLLM container settings
    vllm_port: int = 8000
    vllm_max_model_len: int = 2048
    vllm_disable_log_requests: bool = True

    # Volume mounts
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
                image_defaults = {
                    "cuda": "vllm/vllm-openai:latest",
                    "cpu": "public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.9.1",
                    "rocm": "rocm/vllm:latest",
                }
                if self.device_type not in image_defaults:
                    raise ValueError(
                        f"Unsupported device type: {self.device_type}. Supported types: {list(image_defaults.keys()) + ['arm']}"
                    )
                self.vllm_image = image_defaults[self.device_type]


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
