import argparse
import json

from flexbench.benchmark_runner import BenchmarkRunner
from flexbench.configs import BenchmarkConfig, LoadgenConfig
from flexbench.dataset.base import DatasetConfig
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
    parser.add_argument(
        "--target-qps",
        type=float,
        required=True,
        help="Target queries per second",
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


def main():
    args = get_args()

    dataset_config = DatasetConfig(
        path=args.dataset_path,
        input_column=args.dataset_input_column,
        output_column=args.dataset_output_column,
        system_prompt_column=args.dataset_system_prompt_column,
        image_column=args.dataset_image_column,
        split=args.dataset_split,
    )

    loadgen_config = LoadgenConfig(
        scenario=args.scenario,
        target_qps=args.target_qps,
        accuracy=args.accuracy,
        total_sample_count=args.total_sample_count,
    )

    benchmark_config = BenchmarkConfig(
        task=args.task,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        api_server=args.api_server,
        api_token=args.api_token,
        dataset_config=dataset_config,
        loadgen_config=loadgen_config,
        batch_size=args.batch_size,
        max_generated_tokens=args.max_generated_tokens,
    )

    runner = BenchmarkRunner(benchmark_config)
    result = runner.run()
    results_path = runner.results_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(result.__dict__, f)
    log.info(f"\nBenchmark Results:\n{result}")
    log.info(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
