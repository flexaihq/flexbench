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


class Scenario(str, Enum):
    offline = "Offline"
    server = "Server"
    single_stream = "SingleStream"


class Backend(str, Enum):
    loadgen = "loadgen"
    vllm = "vllm"


class Mode(str, Enum):
    performance = "performance"
    accuracy = "accuracy"
    all = "all"


# Create the Docker orchestration CLI app
app = typer.Typer(
    help="[bold blue]FlexBench[/bold blue] - :rocket: Docker orchestration for MLPerf-style text benchmarking",
    epilog="""
**Examples:**

• **Basic benchmark** with auto device detection:
  `flexbench --model-path HuggingFaceTB/SmolLM2-135M-Instruct --dataset-path ctuning/MLPerf-OpenOrca --dataset-input-column question --scenario Server --target-qps 1`

• **GPU configuration** with tensor parallelism:
  `flexbench --model-path meta-llama/Llama-3.2-1B-Instruct --dataset-path ctuning/MLPerf-OpenOrca --dataset-input-column question --scenario Server --target-qps 10 --device-type cuda --gpu-devices "0,1" --tensor-parallel-size 2`

• **Sweep mode** to find optimal QPS:
  `flexbench --model-path microsoft/DialoGPT-medium --dataset-path ctuning/MLPerf-OpenOrca --dataset-input-column question --scenario Server --sweep --num-sweep-points 5`

---
Built by [bold blue]Daniel Altunay[/bold blue] and [bold blue]Grigori Fursin[/bold blue] under [bold yellow]FlexAI[/bold yellow] ([italic]FCS Labs[/italic]) :building_construction:
    """,
    rich_markup_mode="rich",
)


@app.command()
def run(
    # === CORE BENCHMARK CONFIGURATION ===
    model_path: str = typer.Option(
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        help="Model name on HuggingFace or local path",
        rich_help_panel="Core Configuration",
    ),
    dataset_path: str = typer.Option(
        "ctuning/MLPerf-OpenOrca",
        help="Dataset path on HuggingFace or local pickle file",
        rich_help_panel="Core Configuration",
    ),
    dataset_input_column: str = typer.Option(
        "question",
        help="Input text column name in dataset",
        rich_help_panel="Core Configuration",
    ),
    scenario: Scenario = typer.Option(
        Scenario.offline,
        help="MLPerf scenario: Offline (all queries sent at once for max throughput), Server (Poisson distribution mimicking real-world load), or SingleStream (one query at a time for sequential latency)",
        rich_help_panel="Core Configuration",
    ),
    mode: Mode = typer.Option(
        Mode.performance,
        help="Benchmark mode: performance (benchmark only), accuracy (accuracy evaluation only), or all (run performance then accuracy)",
        rich_help_panel="Core Configuration",
    ),
    # === PERFORMANCE TUNING ===
    target_qps: float | None = typer.Option(
        10,
        help="Target queries per second (required unless using sweep mode)",
        rich_help_panel="Performance Tuning",
    ),
    sweep: bool = typer.Option(
        False,
        help="Run sweep mode: find max QPS then sweep different values",
        rich_help_panel="Performance Tuning",
    ),
    num_sweep_points: int = typer.Option(
        10,
        help="Number of QPS points to test in sweep mode",
        rich_help_panel="Performance Tuning",
    ),
    total_sample_count: int | None = typer.Option(
        100,
        help="Number of samples to process",
        rich_help_panel="Performance Tuning",
    ),
    batch_size: int | None = typer.Option(
        None,
        help="Batch size for offline scenario. If not specified, loads the entire dataset at once",
        rich_help_panel="Performance Tuning",
    ),
    # === MODEL CONFIGURATION ===
    max_generated_tokens: int = typer.Option(
        1024,
        help="Maximum tokens to generate",
        rich_help_panel="Model Configuration",
    ),
    max_input_tokens: int | None = typer.Option(
        None,
        help="Maximum input tokens (longer inputs truncated)",
        rich_help_panel="Model Configuration",
    ),
    fixed_input_length: bool = typer.Option(
        False,
        help="Pad inputs to max-input-tokens",
        rich_help_panel="Model Configuration",
    ),
    remote_model_path: str | None = typer.Option(
        None,
        help="Model name for remote endpoint",
        rich_help_panel="Model Configuration",
    ),
    tokenizer_path_override: str | None = typer.Option(
        None,
        help="Custom tokenizer path",
        rich_help_panel="Model Configuration",
    ),
    model_name: str = typer.Option(
        "llama2-70b",
        help="Model name for MLPerf configuration",
        rich_help_panel="Model Configuration",
    ),
    # === DATASET CONFIGURATION ===
    dataset_output_column: str | None = typer.Option(
        "response",
        help="Reference text column (required for accuracy mode)",
        rich_help_panel="Dataset Configuration",
    ),
    dataset_split: str = typer.Option(
        "train",
        help="Dataset split to use",
        rich_help_panel="Dataset Configuration",
    ),
    dataset_system_prompt_column: str | None = typer.Option(
        None,  # use `system_prompt` for MLPerf-OpenOrca
        help="System prompt column name (optional - adds system context to prompts)",
        rich_help_panel="Dataset Configuration",
    ),
    # === HARDWARE CONFIGURATION ===
    device_type: DeviceType = typer.Option(
        DeviceType.auto,
        help="Hardware device type (auto-detects: cuda -> rocm -> arm -> cpu)",
        rich_help_panel="Hardware Configuration",
    ),
    gpu_devices: str | None = typer.Option(
        None,
        help="Comma-separated GPU device IDs (e.g., '0,1,2'). Auto-detects if not specified",
        envvar=["CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"],
        rich_help_panel="Hardware Configuration",
    ),
    tensor_parallel_size: int | None = typer.Option(
        None,
        help="Number of GPUs for tensor parallelism (e.g., 2, 4, 8)",
        rich_help_panel="Hardware Configuration",
    ),
    # === VLLM SERVER CONFIGURATION ===
    port: int = typer.Option(
        8000,
        help="Port to forward vLLM server to on the host (for Docker communication)",
        rich_help_panel="vLLM Server Configuration",
    ),
    vllm_max_model_len: int = typer.Option(
        2048,
        help="Maximum model length",
        rich_help_panel="vLLM Server Configuration",
    ),
    vllm_disable_log_requests: bool = typer.Option(
        True,
        help="Disable vLLM request logging for better performance",
        rich_help_panel="vLLM Server Configuration",
    ),
    vllm_gpu_memory_utilization: float = typer.Option(
        0.9,
        help="GPU memory utilization for vLLM (0.1-1.0)",
        rich_help_panel="vLLM Server Configuration",
    ),
    vllm_server: str | None = typer.Option(
        None,
        help="Existing vLLM server URL (e.g., 'http://localhost:8000'). If specified, FlexBench will use this server instead of creating its own",
        rich_help_panel="vLLM Server Configuration",
    ),
    vllm_server_token: str | None = typer.Option(
        None,
        help="Authentication token for existing vLLM server (if required)",
        rich_help_panel="vLLM Server Configuration",
    ),
    # === DOCKER CONFIGURATION ===
    vllm_image: str | None = typer.Option(
        None,
        help="Full vLLM Docker image name (e.g. 'vllm/vllm-openai:latest', 'public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.9.1', 'rocm/vllm:latest'). Overrides default for device type. Highly recommended for reproducibility",
        rich_help_panel="Docker Configuration",
    ),
    # === RESOURCE LIMITS ===
    vllm_memory_limit: str | None = typer.Option(
        None,
        help="vLLM memory limit (e.g., '8g')",
        rich_help_panel="Resource Limits",
    ),
    flexbench_memory_limit: str | None = typer.Option(
        None,
        help="FlexBench memory limit (e.g., '4g')",
        rich_help_panel="Resource Limits",
    ),
    # === AUTHENTICATION & TOKENS ===
    hf_token: Annotated[
        str | None,
        typer.Option(
            help="HuggingFace authentication token",
            envvar="HF_TOKEN",
            rich_help_panel="Authentication & Tokens",
        ),
    ] = None,
    # === DIRECTORY CONFIGURATION ===
    model_cache_dir: Annotated[
        str,
        typer.Option(
            help="Model cache directory (HuggingFace cache)",
            envvar="HF_HOME",
            rich_help_panel="Directory Configuration",
        ),
    ] = "~/.cache/huggingface",
    output_dir: str | None = typer.Option(
        None,
        help="Directory to store results. If not specified, results will be stored in a timestamped subdirectory of results/",
        rich_help_panel="Directory Configuration",
    ),
    # === ADVANCED CONFIGURATION ===
    backend: Backend = typer.Option(
        Backend.loadgen,
        help="Benchmark backend: loadgen (stable) or vllm (experimental)",
        rich_help_panel="Advanced Configuration",
    ),
    config_path: str = typer.Option(
        "user.conf",
        help="MLPerf configuration file path",
        rich_help_panel="Advanced Configuration",
    ),
    enable_trace: bool = typer.Option(
        False,
        help="Enable MLPerf trace logging",
        rich_help_panel="Advanced Configuration",
    ),
    log_output_to_stdout: bool = typer.Option(
        True,
        help="Log MLPerf output to stdout",
        rich_help_panel="Advanced Configuration",
    ),
    # === BUILD CONFIGURATION ===
    vllm_repo: str = typer.Option(
        "https://github.com/vllm-project/vllm.git",
        help="vLLM repository URL",
        rich_help_panel="Build Configuration",
    ),
    vllm_branch: str = typer.Option(
        "main",
        help="vLLM branch/tag to build",
        rich_help_panel="Build Configuration",
    ),
    # === EXECUTION CONTROL ===
    cleanup: bool = typer.Option(
        True,
        "--cleanup/--no-cleanup",
        help="Clean up containers after run (default: True)",
        rich_help_panel="Execution Control",
    ),
    pull_images: bool = typer.Option(
        True,
        "--pull/--no-pull",
        help="Pull latest Docker images before run (default: True)",
        rich_help_panel="Execution Control",
    ),
    build_flexbench: bool = typer.Option(
        True,
        "--build/--no-build",
        help="Build FlexBench image if needed (default: True)",
        rich_help_panel="Execution Control",
    ),
    wait_timeout: int = typer.Option(
        300,
        help="Container startup timeout (seconds)",
        rich_help_panel="Execution Control",
    ),
    dry_run: bool = typer.Option(
        False,
        help="Show config without running",
        rich_help_panel="Execution Control",
    ),
):
    """
    Run FlexBench benchmarking with Docker orchestration.

    This command orchestrates vLLM and FlexBench containers to run
    MLPerf-style benchmarks on language models with automatic hardware detection and optimization.

    For detailed examples, see the help above.
    """

    # Parse GPU devices - gpu_count is auto-calculated from gpu_devices
    gpu_device_list = None
    if gpu_devices:
        gpu_device_list = [device.strip() for device in gpu_devices.split(",")]

    # Import here to avoid circular imports
    from cli.config import (
        DockerConfig,
        FlexBenchDockerConfig,
    )
    from cli.main import run_benchmark_async
    from flexbench.config import create_benchmark_config

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
        scenario=scenario.value,
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
        backend=backend.value,
        vllm_server_token=vllm_server_token,
        # Dataset configuration
        dataset_output_column=dataset_output_column,
        dataset_split=dataset_split,
        dataset_system_prompt_column=dataset_system_prompt_column,
        # Accuracy and output configuration
        mode=mode.value,
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
        # Device and hardware configuration
        device_type=device_type.value,
        gpu_devices=gpu_device_list,
        tensor_parallel_size=tensor_parallel_size,
        # vLLM build configuration
        vllm_repo=vllm_repo,
        vllm_branch=vllm_branch,
        # vLLM server configuration
        vllm_port=port,
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
