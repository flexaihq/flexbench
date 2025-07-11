"""Shared argument definitions for FlexBench module and CLI."""

import argparse
from datetime import datetime


def create_module_parser():
    """Create a minimal argument parser for direct module usage."""
    parser = argparse.ArgumentParser(description="FlexBench Core - Direct Python module interface")

    # Required arguments (text-only)
    parser.add_argument(
        "--model-path", required=True, help="Model name on HuggingFace or local path"
    )
    parser.add_argument(
        "--dataset-path", required=True, help="Dataset path on HuggingFace or local pickle file"
    )
    parser.add_argument(
        "--dataset-input-column", required=True, help="Input text column name in dataset"
    )
    parser.add_argument(
        "--scenario",
        required=True,
        choices=["Offline", "Server", "SingleStream"],
        help="MLPerf scenario",
    )

    # Core benchmark options
    parser.add_argument("--api-server", default="http://localhost:8000", help="vLLM API server URL")
    parser.add_argument("--remote-model-path", help="Model name for remote endpoint")
    parser.add_argument("--target-qps", type=float, help="Target queries per second")
    parser.add_argument("--sweep", action="store_true", help="Run sweep mode")
    parser.add_argument(
        "--num-points", type=int, default=10, help="Number of QPS points to test in sweep mode"
    )
    parser.add_argument(
        "--backend", default="loadgen", choices=["loadgen", "vllm"], help="Benchmark backend"
    )

    # Dataset options (text-only)
    parser.add_argument(
        "--dataset-output-column", help="Reference text column (required for accuracy mode)"
    )
    parser.add_argument(
        "--mode",
        default="performance",
        choices=["performance", "accuracy", "all"],
        help="Benchmark mode: performance, accuracy, or all",
    )
    parser.add_argument("--dataset-split", default="train", help="Dataset split to use")
    parser.add_argument("--dataset-system-prompt-column", help="System prompt column name")

    # Model and tokenizer options
    parser.add_argument("--tokenizer-path-override", help="Custom tokenizer path")
    parser.add_argument("--api-token", help="API authentication token")

    # Performance options
    parser.add_argument("--total-sample-count", type=int, help="Number of samples to process")
    parser.add_argument("--batch-size", type=int, help="Batch size for offline scenario")
    parser.add_argument(
        "--max-generated-tokens", type=int, default=1024, help="Maximum tokens to generate"
    )
    parser.add_argument("--max-input-tokens", type=int, help="Maximum input tokens")
    parser.add_argument(
        "--fixed-input-length", action="store_true", help="Pad inputs to max-input-tokens"
    )
    parser.add_argument("--output-dir", help="Directory to store results")

    return parser


def validate_args(args):
    """Validate and transform arguments."""

    # Process GPU devices for CLI
    if hasattr(args, "gpu_devices") and args.gpu_devices:
        args.gpu_devices = [device.strip() for device in args.gpu_devices.split(",")]
        args.gpu_count = len(args.gpu_devices)

    # Set default output directory
    if not args.output_dir:
        args.output_dir = f"results/{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Set model cache dir default for CLI
    if hasattr(args, "model_cache_dir") and not args.model_cache_dir:
        import os

        args.model_cache_dir = os.path.expanduser("~/.cache/huggingface")

    return args
