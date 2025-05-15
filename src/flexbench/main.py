import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from flexbench.dataset.base import DatasetConfig
from flexbench.runners.base import BenchmarkConfig
from flexbench.runners.factory import create_benchmark_runner
from flexbench.utils import get_logger

log = get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="MLPerf Inference Benchmark")

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
        "--api-server",
        required=True,
        help="vLLM API server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--scenario",
        choices=["Offline", "Server", "SingleStream"],
        required=True,
        help="MLPerf scenario (Offline, Server, or SingleStream)",
    )

    # Target QPS configuration (not required for SingleStream)
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

    # Add num_points parameter for sweep mode
    parser.add_argument(
        "--num-points",
        type=int,
        default=10,
        help="Number of QPS points to test in sweep mode (default: 10)",
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
        "--backend",
        choices=["loadgen", "vllm"],
        default="loadgen",
        help="Benchmark backend (default: loadgen)",
    )

    # Optional arguments
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
    parser.add_argument(
        "--tokenizer-path-override",
        help="Custom tokenizer path if different from model",
    )
    parser.add_argument(
        "--api-token",
        help="API authentication token",
    )
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

    args = parser.parse_args()

    return args


async def async_main() -> dict:
    args = get_args()
    log.info(f"Parsed arguments: {args}")

    dataset_config = DatasetConfig(
        path=args.dataset_path,
        input_column=args.dataset_input_column,
        output_column=args.dataset_output_column,
        system_prompt_column=args.dataset_system_prompt_column,
        image_column=args.dataset_image_column,
        split=args.dataset_split,
        accuracy_mode=args.accuracy,
    )

    benchmark_config = BenchmarkConfig(
        task=args.task,
        model_path=args.model_path,
        tokenizer_path_override=args.tokenizer_path_override,
        api_server=args.api_server,
        api_token=args.api_token,
        dataset_config=dataset_config,
        scenario=args.scenario,
        target_qps=args.target_qps,
        sweep_mode=args.sweep,
        num_sweep_points=args.num_points,
        batch_size=args.batch_size,
        max_generated_tokens=args.max_generated_tokens,
        max_input_tokens=args.max_input_tokens,
        fixed_input_length=args.fixed_input_length,
        accuracy=args.accuracy,
        total_sample_count=args.total_sample_count,
        output_dir=args.output_dir,  # Add the output_dir parameter
    )

    runner = create_benchmark_runner(args.backend, benchmark_config)
    result = await runner.run()

    # Save results to file
    # Use the specified output directory if provided
    if args.output_dir:
        results_dir = Path(args.output_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
    else:
        results_dir = runner.results_dir

    results_path = results_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    log.info("Benchmark run completed")
    log.info(f"Results saved to: {results_path.absolute()}")

    # Update result with path for subprocesses to find
    if isinstance(result, dict):
        result["results_path"] = str(results_path.absolute())

    return result


def main():
    try:
        result = asyncio.run(async_main())
        # For subprocess runs, print the path to help parent process find it
        if isinstance(result, dict) and "results_path" in result:
            log.info(f"Results saved to: {result['results_path']}")
        return 0
    except KeyboardInterrupt:
        log.info("Benchmark interrupted by user")
        return 130
    except Exception as e:
        log.error(f"Benchmark failed: {e}", exc_info=True)
        # Make sure we print the error to stdout for parent process to see
        if os.environ.get("LOG_LEVEL", "").upper() == "DEBUG":
            log.error(f"ERROR: {e}", exc_info=True, stack_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
