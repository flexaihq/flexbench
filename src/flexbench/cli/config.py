"""Docker configuration for FlexBench."""

from dataclasses import dataclass

from flexbench.config import create_benchmark_config
from flexbench.runners.base import BenchmarkConfig


@dataclass
class DockerConfig:
    """Configuration for Docker containers."""
    
    # Docker-specific settings
    vllm_image: str = "vllm/vllm-openai:latest"
    flexbench_image: str = "flexbench:latest"
    network_name: str = "flexbench-network"
    
    # GPU settings
    gpu_devices: list[str] | None = None  # e.g., ["0", "1"] for specific GPUs
    gpu_count: int | None = None  # Total GPU count for tensor parallelism
    
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
    
    def __post_init__(self):
        if self.gpu_devices and self.gpu_count:
            if len(self.gpu_devices) != self.gpu_count:
                raise ValueError(
                    f"gpu_devices length ({len(self.gpu_devices)}) must match gpu_count ({self.gpu_count})"
                )


@dataclass
class FlexBenchDockerConfig:
    """Complete configuration for FlexBench CLI with Docker orchestration."""
    
    # Core benchmark configuration (reuse existing structure)
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
        self.benchmark_config.api_server = f"http://vllm-server:{self.docker_config.vllm_port}"


def create_docker_config_from_args(args) -> FlexBenchDockerConfig:
    """Create FlexBenchDockerConfig from parsed CLI arguments."""
    
    # Create BenchmarkConfig using shared builder
    benchmark_config = create_benchmark_config(args)
    
    # Create DockerConfig with CLI-specific options
    docker_config = DockerConfig(
        vllm_image=getattr(args, 'vllm_image', DockerConfig.vllm_image),
        flexbench_image=getattr(args, 'flexbench_image', DockerConfig.flexbench_image),
        gpu_devices=getattr(args, 'gpu_devices', None),
        gpu_count=getattr(args, 'gpu_count', None),
        vllm_port=getattr(args, 'vllm_port', DockerConfig.vllm_port),
        vllm_max_model_len=getattr(args, 'vllm_max_model_len', DockerConfig.vllm_max_model_len),
        model_cache_dir=getattr(args, 'model_cache_dir', None),
        results_dir=args.output_dir,
        vllm_memory_limit=getattr(args, 'vllm_memory_limit', None),
        flexbench_memory_limit=getattr(args, 'flexbench_memory_limit', None),
    )
    
    return FlexBenchDockerConfig(
        benchmark_config=benchmark_config,
        docker_config=docker_config,
        cleanup=not getattr(args, 'no_cleanup', False),
        pull_images=not getattr(args, 'no_pull', False),
        build_flexbench=not getattr(args, 'no_build', False),
        wait_timeout=getattr(args, 'wait_timeout', 300),
    )
