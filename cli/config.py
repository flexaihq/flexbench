"""Docker configuration for FlexBench CLI (text-only)."""

import typing as tp
from dataclasses import dataclass
import sys
from pathlib import Path
from cli.utils import get_logger

log = get_logger(__name__)

# Add src directory to path for flexbench imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import the main configs from flexbench
from flexbench.config import DatasetConfig, BenchmarkConfig, create_dataset_config, create_benchmark_config


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

    # vLLM build settings (only used for ARM)
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
        """Return True if vLLM must be built from source (only for ARM)."""
        return self.device_type == "arm"

    def __post_init__(self):
        if self.gpu_devices and self.gpu_count:
            if len(self.gpu_devices) != self.gpu_count:
                raise ValueError(
                    f"gpu_devices length ({len(self.gpu_devices)}) must match gpu_count ({self.gpu_count})"
                )

        # Parse build args string into dict if it's a string
        if isinstance(self.vllm_build_args, str):
            build_args = {}
            for arg in self.vllm_build_args.split():
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    build_args[key] = value
            self.vllm_build_args = build_args

        # Final fallback for vllm_image if None is passed
        if not self.vllm_image:
            if self.device_type == "arm":
                log.warning("No vLLM image specified, falling back to default for ARM: vllm-arm:latest")
                self.vllm_image = "vllm-arm:latest"
            elif self.device_type == "nvidia":
                log.warning("No vLLM image specified, falling back to default for NVIDIA: vllm/vllm-openai:latest")
                self.vllm_image = "vllm/vllm-openai:latest"
            elif self.device_type == "cpu":
                log.warning("No vLLM image specified, falling back to default for CPU: public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.9.1")
                self.vllm_image = "public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.9.1"
            elif self.device_type == "rocm":
                log.warning("No vLLM image specified, falling back to default for ROCm: rocm/vllm:latest")
                self.vllm_image = "rocm/vllm:latest"
            else:
                log.warning(f"No vLLM image specified, falling back to vllm-{self.device_type}:latest")
                self.vllm_image = f"vllm-{self.device_type}:latest"

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

