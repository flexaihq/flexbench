import argparse
import asyncio
import json
import os
import sys

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
        choices=["Offline", "Server"],
        required=True,
        help="MLPerf scenario (Offline or Server)",
    )

    # Target QPS configuration (either --target-qps or --sweep must be provided)
    qps_group = parser.add_mutually_exclusive_group(required=True)
    qps_group.add_argument(
        "--target-qps",
        type=float,
        help="Target queries per second",
    )
    qps_group.add_argument(
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
        "--tokenizer-path",
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
    return parser.parse_args()


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
        tokenizer_path=args.tokenizer_path,
        api_server=args.api_server,
        api_token=args.api_token,
        dataset_config=dataset_config,
        scenario=args.scenario,
        target_qps=args.target_qps,
        sweep_mode=args.sweep,
        num_sweep_points=args.num_points,  # Use the CLI argument value
        batch_size=args.batch_size,
        max_generated_tokens=args.max_generated_tokens,
        accuracy=args.accuracy,
        total_sample_count=args.total_sample_count,
    )

    runner = create_benchmark_runner(args.backend, benchmark_config)
    result = await runner.run()

    # Save results to file
    results_path = runner.results_dir / "benchmark_results.json"
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
            print(f"Results saved to: {result['results_path']}")
        return 0
    except KeyboardInterrupt:
        print("Benchmark interrupted by user")
        return 130
    except Exception as e:
        log.error(f"Benchmark failed: {e}", exc_info=True)
        # Make sure we print the error to stdout for parent process to see
        if os.environ.get("LOG_LEVEL", "").upper() == "DEBUG":
            import traceback

            print(f"ERROR: {e}")
            print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
