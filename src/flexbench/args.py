"""Shared argument parser for FlexBench module and CLI."""

import argparse
from datetime import datetime


def add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    """Add all core benchmark arguments to the parser."""
    
    # Required arguments
    parser.add_argument(
        "--task",
        choices=["text", "vision"],
        required=True,
        help="Task type (text or vision)",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Model name on HuggingFace or local path",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Dataset path on HuggingFace or local pickle file",
    )
    parser.add_argument(
        "--dataset-input-column",
        required=True,
        help="Input text column name in dataset",
    )
    parser.add_argument(
        "--scenario",
        choices=["Offline", "Server", "SingleStream"],
        required=True,
        help="MLPerf scenario (Offline, Server, or SingleStream)",
    )

    # Optional core arguments
    parser.add_argument(
        "--remote-model-path",
        help="Model name used to serve the model at the remote endpoint",
    )
    parser.add_argument(
        "--target-qps",
        type=float,
        help="Target queries per second",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run sweep mode: first find max QPS, then sweep different QPS values",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=10,
        help="Number of QPS points to test in sweep mode (default: 10)",
    )
    parser.add_argument(
        "--backend",
        choices=["loadgen", "vllm"],
        default="loadgen",
        help="Benchmark backend (default: loadgen)",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset-output-column",
        help="Reference text column name in dataset (required for accuracy mode)",
    )
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="Run accuracy evaluation (default: performance mode)",
    )
    parser.add_argument(
        "--dataset-split",
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--dataset-system-prompt-column",
        help="System prompt column name",
    )
    parser.add_argument(
        "--dataset-image-column",
        help="Image column name (required for vision tasks)",
    )

    # Model and tokenizer arguments
    parser.add_argument(
        "--tokenizer-path-override",
        help="Custom tokenizer path if different from model",
    )
    parser.add_argument(
        "--api-token",
        help="API authentication token",
    )

    # Performance arguments
    parser.add_argument(
        "--total-sample-count",
        type=int,
        help="Number of samples to process",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for offline scenario",
    )
    parser.add_argument(
        "--max-generated-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate (default: 1024)",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        help="Maximum number of tokens for input. Longer inputs will be truncated.",
    )
    parser.add_argument(
        "--fixed-input-length",
        action="store_true",
        help="Pad inputs to reach exactly max-input-tokens (padding on right side)",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to store benchmark results",
    )


def add_module_arguments(parser: argparse.ArgumentParser) -> None:
    """Add module-specific arguments (direct API server connection)."""
    parser.add_argument(
        "--api-server",
        required=True,
        help="vLLM API server URL (default: http://localhost:8000)",
    )


def add_docker_arguments(parser: argparse.ArgumentParser) -> None:
    """Add Docker-specific arguments for CLI."""
    
    docker_group = parser.add_argument_group("Docker Configuration")
    
    docker_group.add_argument(
        "--vllm-image",
        default="vllm/vllm-openai:latest",
        help="vLLM Docker image to use (default: vllm/vllm-openai:latest)",
    )
    docker_group.add_argument(
        "--flexbench-image", 
        default="flexbench:latest",
        help="FlexBench Docker image to use (default: flexbench:latest)",
    )
    
    # GPU configuration
    gpu_group = docker_group.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--gpu-devices",
        help="Comma-separated list of GPU device IDs to use (e.g., '0,1,2')",
    )
    gpu_group.add_argument(
        "--gpu-count",
        type=int,
        help="Number of GPUs to use (will use first N GPUs)",
    )
    
    # vLLM configuration
    docker_group.add_argument(
        "--vllm-port",
        type=int,
        default=8000,
        help="Port for vLLM server (default: 8000)",
    )
    docker_group.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=2048,
        help="Maximum model length for vLLM (default: 2048)",
    )
    
    # Volume mounts
    docker_group.add_argument(
        "--model-cache-dir",
        help="Host directory to mount for model cache (speeds up subsequent runs)",
    )
    
    # Resource limits
    docker_group.add_argument(
        "--vllm-memory-limit",
        help="Memory limit for vLLM container (e.g., '16g')",
    )
    docker_group.add_argument(
        "--flexbench-memory-limit",
        help="Memory limit for FlexBench container (e.g., '4g')",
    )


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Add CLI-specific arguments."""
    
    cli_group = parser.add_argument_group("CLI Configuration")
    
    cli_group.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't clean up containers after benchmark (useful for debugging)",
    )
    cli_group.add_argument(
        "--no-pull",
        action="store_true", 
        help="Don't pull latest Docker images before running",
    )
    cli_group.add_argument(
        "--no-build",
        action="store_true",
        help="Don't build FlexBench image (assume it exists)",
    )
    cli_group.add_argument(
        "--wait-timeout",
        type=int,
        default=300,
        help="Timeout in seconds to wait for containers to start (default: 300)",
    )
    cli_group.add_argument(
        "--compose-file",
        help="Path to custom docker-compose.yml file",
    )
    cli_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually running containers",
    )


def validate_args(args):
    """Validate and transform arguments."""
    
    # Process GPU devices for CLI
    if hasattr(args, 'gpu_devices') and args.gpu_devices:
        args.gpu_devices = [device.strip() for device in args.gpu_devices.split(",")]
        args.gpu_count = len(args.gpu_devices)
    
    # Set default output directory
    if not args.output_dir:
        args.output_dir = f"results/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Set model cache dir default for CLI
    if hasattr(args, 'model_cache_dir') and not args.model_cache_dir:
        import os
        args.model_cache_dir = os.path.expanduser("~/.cache/huggingface")
    
    return args


def create_module_parser() -> argparse.ArgumentParser:
    """Create argument parser for module usage."""
    parser = argparse.ArgumentParser(description="MLPerf Inference Benchmark")
    add_benchmark_arguments(parser)
    add_module_arguments(parser)
    return parser


def create_cli_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI usage."""
    parser = argparse.ArgumentParser(
        description="FlexBench CLI - Containerized benchmarking with vLLM and FlexBench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark with automatic GPU detection
  flexbench --task text --model-path meta-llama/Llama-2-7b-chat-hf \\
            --scenario Server --target-qps 10 \\
            --dataset-path ctuning/MLPerf-OpenOrca \\
            --dataset-input-column question

  # Sweep mode with specific GPUs
  flexbench --task text --model-path microsoft/DialoGPT-medium \\
            --scenario Server --sweep --gpu-devices 0,1 \\
            --dataset-path ctuning/MLPerf-OpenOrca \\
            --dataset-input-column question

  # Custom Docker images and settings
  flexbench --task text --model-path HuggingFaceTB/SmolLM2-135M-Instruct \\
            --scenario Offline --target-qps 5 --batch-size 8 \\
            --dataset-path ctuning/MLPerf-OpenOrca \\
            --dataset-input-column question \\
            --vllm-image custom-vllm:latest \\
            --no-cleanup
        """
    )
    add_benchmark_arguments(parser)
    add_docker_arguments(parser)
    add_cli_arguments(parser)
    return parser
