"""Typer CLI for FlexBench Docker orchestration."""

import typer
import asyncio
from enum import Enum
from typing_extensions import Annotated
from cli.utils import get_logger

log = get_logger(__name__)

"""
🚀 Run FlexBench text benchmarking with Docker orchestration.

This command orchestrates vLLM and FlexBench containers to run MLPerf-style 
text generation benchmarks with automatic hardware detection and optimization.

**Examples:**

```bash
# Basic CPU benchmark
flexbench --model-path HuggingFaceTB/SmolLM2-135M-Instruct --dataset-path ctuning/MLPerf-OpenOrca --dataset-input-column question --scenario Server

# GPU benchmark with environment variables
export FLEXBENCH_DEVICE_TYPE=cuda
export FLEXBENCH_GPU_DEVICES="0,1"
flexbench --model-path meta-llama/Llama-2-7b-chat-hf --dataset-path ctuning/MLPerf-OpenOrca --dataset-input-column question --scenario Server --target-qps 10
```
"""

class DeviceType(str, Enum):
    cpu = "cpu"
    cuda = "cuda"
    rocm = "rocm"
    arm = "arm"


# Create the Docker orchestration CLI app
app = typer.Typer(
    help="FlexBench - Docker orchestration for MLPerf-style text benchmarking",
    epilog="""
🐳 **Docker orchestration:** This CLI automatically manages vLLM and FlexBench containers.
� **Text-only tasks:** Only text generation and completion tasks are supported.
    """,
    rich_markup_mode="markdown"
)


@app.command()
def run(
    # Core FlexBench arguments (text-only)
    model_path: str = typer.Option(..., help="Model name on HuggingFace or local path"),
    dataset_path: str = typer.Option(..., help="Dataset path on HuggingFace or local pickle file"),
    dataset_input_column: str = typer.Option(..., help="Input text column name in dataset"),
    scenario: str = typer.Option(..., help="MLPerf scenario: Offline, Server, or SingleStream"),
    
    # Core benchmark options
    remote_model_path: str | None = typer.Option(None, help="Model name for remote endpoint"),
    target_qps: float = typer.Option(..., help="Target queries per second (required)"),
    sweep: bool = typer.Option(False, help="Run sweep mode: find max QPS then sweep different values"),
    num_points: int = typer.Option(10, help="Number of QPS points to test in sweep mode"),
    backend: str = typer.Option("loadgen", help="Benchmark backend: loadgen or vllm"),
    
    # Dataset options (text-only)
    dataset_output_column: str | None = typer.Option(None, help="Reference text column (required for accuracy mode)"),
    accuracy: bool = typer.Option(False, help="Run accuracy evaluation"),
    dataset_split: str = typer.Option("train", help="Dataset split to use"),
    dataset_system_prompt_column: str | None = typer.Option(None, help="System prompt column name"),
    
    # Model and tokenizer options
    tokenizer_path_override: str | None = typer.Option(None, help="Custom tokenizer path"),
    api_token: str | None = typer.Option(None, help="API authentication token"),
    
    # Performance options
    total_sample_count: int | None = typer.Option(None, help="Number of samples to process"),
    batch_size: int | None = typer.Option(None, help="Batch size for offline scenario"),
    max_generated_tokens: int = typer.Option(1024, help="Maximum tokens to generate"),
    max_input_tokens: int | None = typer.Option(None, help="Maximum input tokens (longer inputs truncated)"),
    fixed_input_length: bool = typer.Option(False, help="Pad inputs to max-input-tokens"),
    output_dir: str | None = typer.Option(None, help="Directory to store results"),
    
    # Docker-specific configuration
    vllm_image: str = typer.Option(
        None,
        help="Full vLLM Docker image name (e.g. 'vllm/vllm-openai:latest', 'public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.9.1', 'rocm/vllm:latest'). Overrides default for device type. Highly recommended for reproducibility."
    ),
    flexbench_image: str = typer.Option("flexbench:latest", help="FlexBench Docker image"),
    device_type: DeviceType = typer.Option(DeviceType.cpu, help="Hardware device type"),
    
    # GPU configuration
    gpu_devices: str | None = typer.Option(None, help="Comma-separated GPU device IDs (e.g., '0,1,2')"),
    gpu_count: int | None = typer.Option(None, help="Number of GPUs to use (first N GPUs)"),
    
    # vLLM configuration
    vllm_port: int = typer.Option(8000, help="Port for vLLM server"),
    vllm_max_model_len: int = typer.Option(2048, help="Maximum model length"),
    model_cache_dir: Annotated[str | None, typer.Option(
        help="Model cache directory",
        envvar="HF_HOME"
    )] = None,
    vllm_memory_limit: str | None = typer.Option(None, help="vLLM memory limit (e.g., '8g')"),
    flexbench_memory_limit: str | None = typer.Option(None, help="FlexBench memory limit (e.g., '4g')"),
    
    # Build configuration
    vllm_repo: str = typer.Option("https://github.com/vllm-project/vllm.git", help="vLLM repository URL"),
    vllm_branch: str = typer.Option("main", help="vLLM branch/tag to build"),
    vllm_build_args: str | None = typer.Option(None, help="Additional vLLM build arguments"),
    
    # CLI configuration
    no_cleanup: bool = typer.Option(False, help="Don't clean up containers (useful for debugging)"),
    no_pull: bool = typer.Option(False, help="Don't pull latest Docker images"),
    no_build: bool = typer.Option(False, help="Don't build FlexBench image"),
    wait_timeout: int = typer.Option(300, help="Container startup timeout (seconds)"),
    compose_file: str | None = typer.Option(None, help="Custom docker-compose.yml path"),
    dry_run: bool = typer.Option(False, help="Show config without running"),
):
    """
    🚀 Run FlexBench benchmarking with Docker orchestration.
    
    This command orchestrates vLLM and FlexBench containers to run MLPerf-style 
    benchmarks on language models with automatic hardware detection and optimization.
    
    **Examples:**
    
    ```bash
    # Basic CPU benchmark
    flexbench --model-path HuggingFaceTB/SmolLM2-135M-Instruct --dataset-path ctuning/MLPerf-OpenOrca --dataset-input-column question --scenario Server
    
    # GPU benchmark with environment variables
    export FLEXBENCH_DEVICE_TYPE=cuda
    export FLEXBENCH_GPU_DEVICES="0,1"
    flexbench --model-path meta-llama/Llama-2-7b-chat-hf --dataset-path ctuning/MLPerf-OpenOrca --dataset-input-column question --scenario Server --target-qps 10
    ```
    """
    
    # Validate mutually exclusive GPU options
    if gpu_devices and gpu_count:
        typer.echo("❌ Error: Cannot specify both --gpu-devices and --gpu-count", err=True)
        raise typer.Exit(1)
    
    # Parse GPU devices
    gpu_device_list = None
    if gpu_devices:
        gpu_device_list = [device.strip() for device in gpu_devices.split(",")]
        gpu_count = len(gpu_device_list)
    
    # Import here to avoid circular imports
    from cli.config import (
        DatasetConfig, BenchmarkConfig, DockerConfig, FlexBenchDockerConfig,
        create_dataset_config, create_benchmark_config
    )
    from cli.main import run_benchmark_async
    
    # Create fake args object to use config builders
    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    # Only add vllm_image to Args if it is not None or empty
    args_kwargs = dict(
        model_path=model_path,
        dataset_path=dataset_path,
        dataset_input_column=dataset_input_column,
        scenario=scenario,
        remote_model_path=remote_model_path,
        target_qps=target_qps,
        sweep=sweep,
        num_points=num_points,
        backend=backend,
        dataset_output_column=dataset_output_column,
        accuracy=accuracy,
        dataset_split=dataset_split,
        dataset_system_prompt_column=dataset_system_prompt_column,
        tokenizer_path_override=tokenizer_path_override,
        api_token=api_token,
        total_sample_count=total_sample_count,
        batch_size=batch_size,
        max_generated_tokens=max_generated_tokens,
        max_input_tokens=max_input_tokens,
        fixed_input_length=fixed_input_length,
        output_dir=output_dir,
        flexbench_image=flexbench_image,
        device_type=device_type.value,
        gpu_devices=gpu_device_list,
        gpu_count=gpu_count,
        vllm_repo=vllm_repo,
        vllm_branch=vllm_branch,
        vllm_build_args=vllm_build_args,
        vllm_port=vllm_port,
        vllm_max_model_len=vllm_max_model_len,
        model_cache_dir=model_cache_dir,
        vllm_memory_limit=vllm_memory_limit,
        flexbench_memory_limit=flexbench_memory_limit,
        no_cleanup=no_cleanup,
        no_pull=no_pull,
        no_build=no_build,
        wait_timeout=wait_timeout,
        compose_file=compose_file,
        dry_run=dry_run,
    )
    if vllm_image:
        args_kwargs["vllm_image"] = vllm_image
    log.debug(f"Args passed to config: {args_kwargs}")
    args = Args(**args_kwargs)
    
    # Create benchmark config
    benchmark_config = create_benchmark_config(args)
    
    # Create docker config
    docker_config = DockerConfig(
        vllm_image=vllm_image,
        flexbench_image=flexbench_image,
        device_type=device_type.value,
        gpu_devices=gpu_device_list,
        gpu_count=gpu_count,
        vllm_repo=vllm_repo,
        vllm_branch=vllm_branch,
        vllm_build_args=vllm_build_args,
        vllm_port=vllm_port,
        vllm_max_model_len=vllm_max_model_len,
        model_cache_dir=model_cache_dir,
        results_dir=output_dir,
        vllm_memory_limit=vllm_memory_limit,
        flexbench_memory_limit=flexbench_memory_limit,
    )
    
    # Create complete config
    config = FlexBenchDockerConfig(
        benchmark_config=benchmark_config,
        docker_config=docker_config,
        cleanup=not no_cleanup,
        pull_images=not no_pull,
        build_flexbench=not no_build,
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
