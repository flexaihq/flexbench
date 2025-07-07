"""Typer CLI for FlexBench Docker orchestration."""

import asyncio
from enum import Enum

import typer
from typing_extensions import Annotated

from cli.utils import get_logger

log = get_logger(__name__)


class DeviceType(str, Enum):
    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"
    rocm = "rocm"
    arm = "arm"


# Create the Docker orchestration CLI app
app = typer.Typer(
    help="FlexBench - Docker orchestration for MLPerf-style text benchmarking",
    epilog="""
Examples:
  # Basic benchmark with auto device detection
  flexbench --model-path HuggingFaceTB/SmolLM2-135M-Instruct --dataset-path ctuning/MLPerf-OpenOrca --dataset-input-column question --scenario Server --target-qps 1
  
  # With specific GPU configuration
  flexbench --model-path meta-llama/Llama-2-7b-chat-hf --dataset-path ctuning/MLPerf-OpenOrca --dataset-input-column question --scenario Server --target-qps 10 --device-type cuda --gpu-devices "0,1" --tensor-parallel-size 2
    """,
    rich_markup_mode="markdown",
)


@app.command()
def run(
    # === REQUIRED PARAMETERS ===
    model_path: str = typer.Option(..., help="Model name on HuggingFace or local path"),
    dataset_path: str = typer.Option(..., help="Dataset path on HuggingFace or local pickle file"),
    dataset_input_column: str = typer.Option(..., help="Input text column name in dataset"),
    scenario: str = typer.Option(..., help="MLPerf scenario: Offline, Server, or SingleStream"),
    
    # === PERFORMANCE CONFIGURATION ===
    target_qps: float | None = typer.Option(None, help="Target queries per second (required unless using sweep mode)"),
    sweep: bool = typer.Option(
        False, help="Run sweep mode: find max QPS then sweep different values"
    ),
    num_sweep_points: int = typer.Option(10, help="Number of QPS points to test in sweep mode"),
    total_sample_count: int | None = typer.Option(None, help="Number of samples to process"),
    batch_size: int | None = typer.Option(None, help="Batch size for offline scenario"),
    max_generated_tokens: int = typer.Option(1024, help="Maximum tokens to generate"),
    max_input_tokens: int | None = typer.Option(
        None, help="Maximum input tokens (longer inputs truncated)"
    ),
    fixed_input_length: bool = typer.Option(False, help="Pad inputs to max-input-tokens"),
    
    # === MODEL AND API CONFIGURATION ===
    remote_model_path: str | None = typer.Option(None, help="Model name for remote endpoint"),
    tokenizer_path_override: str | None = typer.Option(None, help="Custom tokenizer path"),
    hf_token: Annotated[
        str | None, 
        typer.Option(help="HuggingFace authentication token", envvar="HF_TOKEN")
    ] = None,
    backend: str = typer.Option("loadgen", help="Benchmark backend: loadgen or vllm"),
    
    # === DATASET CONFIGURATION ===
    dataset_output_column: str | None = typer.Option(
        None, help="Reference text column (required for accuracy mode)"
    ),
    dataset_split: str = typer.Option("train", help="Dataset split to use"),
    dataset_system_prompt_column: str | None = typer.Option(None, help="System prompt column name"),
    
    # === ACCURACY AND OUTPUT CONFIGURATION ===
    accuracy: bool = typer.Option(False, help="Run accuracy evaluation"),
    output_dir: str | None = typer.Option(None, help="Directory to store results"),
    
    # === MLPERF CONFIGURATION ===
    model_name: str = typer.Option("llama2-70b", help="Model name for MLPerf configuration"),
    config_path: str = typer.Option("user.conf", help="MLPerf configuration file path"),
    enable_trace: bool = typer.Option(False, help="Enable MLPerf trace logging"),
    log_output_to_stdout: bool = typer.Option(True, help="Log MLPerf output to stdout"),
    
    # === DOCKER CONFIGURATION ===
    vllm_server: str | None = typer.Option(
        None,
        help="Existing vLLM server URL (e.g., 'http://localhost:8000'). If specified, FlexBench will use this server instead of creating its own.",
    ),
    vllm_server_token: str | None = typer.Option(
        None,
        help="Authentication token for existing vLLM server (if required).",
    ),
    vllm_image: str | None = typer.Option(
        None,
        help="Full vLLM Docker image name (e.g. 'vllm/vllm-openai:latest', 'public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.9.1', 'rocm/vllm:latest'). Overrides default for device type. Highly recommended for reproducibility.",
    ),
    flexbench_image: str = typer.Option("flexbench:latest", help="FlexBench Docker image"),
    network_name: str = typer.Option("flexbench-network", help="Docker network name"),
    device_type: DeviceType = typer.Option(
        DeviceType.auto, help="Hardware device type (auto-detects: cuda -> rocm -> arm -> cpu)"
    ),
    
    # === GPU CONFIGURATION ===
    gpu_devices: str | None = typer.Option(
        None, help="Comma-separated GPU device IDs (e.g., '0,1,2'). Auto-detects if not specified."
    ),
    tensor_parallel_size: int | None = typer.Option(
        None, help="Number of GPUs to use for tensor parallelism (e.g., 2, 4, 8)"
    ),
    
    # === VLLM SERVER CONFIGURATION ===
    vllm_port: int = typer.Option(8000, help="Port for vLLM server"),
    vllm_max_model_len: int = typer.Option(2048, help="Maximum model length"),
    vllm_disable_log_requests: bool = typer.Option(
        True, help="Disable vLLM request logging for better performance"
    ),
    vllm_gpu_memory_utilization: float = typer.Option(
        0.9, help="GPU memory utilization for vLLM (0.1-1.0)"
    ),
    
    # === VOLUME MOUNTS AND DIRECTORIES ===
    model_cache_dir: Annotated[
        str, typer.Option(help="Model cache directory (HuggingFace cache)", envvar="HF_HOME")
    ] = "~/.cache/huggingface",
    
    # === CONTAINER RESOURCE LIMITS ===
    vllm_memory_limit: str | None = typer.Option(None, help="vLLM memory limit (e.g., '8g')"),
    flexbench_memory_limit: str | None = typer.Option(
        None, help="FlexBench memory limit (e.g., '4g')"
    ),
    
    # === BUILD CONFIGURATION ===
    vllm_repo: str = typer.Option(
        "https://github.com/vllm-project/vllm.git", help="vLLM repository URL"
    ),
    vllm_branch: str = typer.Option("main", help="vLLM branch/tag to build"),
    vllm_build_args: str | None = typer.Option(None, help="Additional vLLM build arguments"),
    
    # === CLI EXECUTION CONFIGURATION ===
    cleanup: bool = typer.Option(True, "--cleanup/--no-cleanup", help="Clean up containers after run (default: True)"),
    pull_images: bool = typer.Option(True, "--pull/--no-pull", help="Pull latest Docker images before run (default: True)"),
    build_flexbench: bool = typer.Option(True, "--build/--no-build", help="Build FlexBench image if needed (default: True)"),
    wait_timeout: int = typer.Option(300, help="Container startup timeout (seconds)"),
    dry_run: bool = typer.Option(False, help="Show config without running"),
):
    """
    Run FlexBench benchmarking with Docker orchestration.

    This command orchestrates vLLM and FlexBench containers to run MLPerf-style
    benchmarks on language models with automatic hardware detection and optimization.
    """

    # Parse GPU devices - gpu_count is auto-calculated from gpu_devices
    gpu_device_list = None
    if gpu_devices:
        gpu_device_list = [device.strip() for device in gpu_devices.split(",")]

    # Import here to avoid circular imports
    from cli.config import (
        DockerConfig,
        FlexBenchDockerConfig,
        create_benchmark_config,
    )
    from cli.main import run_benchmark_async

    # Create minimal args object for benchmark config
    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    # Only pass parameters needed for benchmark config
    benchmark_args = Args(
        # Required parameters
        model_path=model_path,
        dataset_path=dataset_path,
        dataset_input_column=dataset_input_column,
        scenario=scenario,
        
        # Performance configuration
        target_qps=target_qps,
        sweep=sweep,
        num_sweep_points=num_sweep_points,
        total_sample_count=total_sample_count,
        batch_size=batch_size,
        max_generated_tokens=max_generated_tokens,
        max_input_tokens=max_input_tokens,
        fixed_input_length=fixed_input_length,
        
        # Model and API configuration
        remote_model_path=remote_model_path,
        tokenizer_path_override=tokenizer_path_override,
        hf_token=hf_token,
        backend=backend,
        vllm_server_token=vllm_server_token,
        
        # Dataset configuration
        dataset_output_column=dataset_output_column,
        dataset_split=dataset_split,
        dataset_system_prompt_column=dataset_system_prompt_column,
        
        # Accuracy and output configuration
        accuracy=accuracy,
        output_dir=output_dir,
        
        # MLPerf configuration
        model_name=model_name,
        config_path=config_path,
        enable_trace=enable_trace,
        log_output_to_stdout=log_output_to_stdout,
    )

    log.debug(f"Benchmark args: {vars(benchmark_args)}")

    # Create benchmark config
    benchmark_config = create_benchmark_config(benchmark_args)

    # Create docker config - pass only the necessary parameters, not duplicating
    docker_config = DockerConfig(
        # External vLLM server configuration
        vllm_server=vllm_server,
        
        # Docker image configuration
        vllm_image=vllm_image,
        flexbench_image=flexbench_image,
        network_name=network_name,
        
        # Device and hardware configuration
        device_type=device_type.value,
        gpu_devices=gpu_device_list,
        tensor_parallel_size=tensor_parallel_size,
        
        # vLLM build configuration
        vllm_repo=vllm_repo,
        vllm_branch=vllm_branch,
        vllm_build_args=vllm_build_args,
        
        # vLLM server configuration
        vllm_port=vllm_port,
        vllm_max_model_len=vllm_max_model_len,
        vllm_disable_log_requests=vllm_disable_log_requests,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        
        # Volume mounts and directories
        model_cache_dir=model_cache_dir,
        results_dir=output_dir,
        
        # Container resource limits
        vllm_memory_limit=vllm_memory_limit,
        flexbench_memory_limit=flexbench_memory_limit,
    )

    # Create complete config
    config = FlexBenchDockerConfig(
        benchmark_config=benchmark_config,
        docker_config=docker_config,
        cleanup=cleanup,
        pull_images=pull_images,
        build_flexbench=build_flexbench,
        wait_timeout=wait_timeout,
    )

    # Run the async benchmark function
    exit_code = asyncio.run(run_benchmark_async(config, dry_run))
    if exit_code != 0:
        raise typer.Exit(exit_code)


def main():
    """Main entry point for FlexBench Docker CLI."""
    app(prog_name="flexbench")


if __name__ == "__main__":
    main()
