"""FlexBench pytest suite."""

import json
import sys

import pytest
import pytest_asyncio

from flexbench.main import async_main  # Import async_main instead of main
from flexbench.tests.configs import (
    BASE_CONFIG,
    TEST_CASES,
    start_vllm_server,
    stop_vllm_server,
)


async def run_benchmark_cli(config):
    """Run benchmark through CLI."""
    # Store original args
    old_args = sys.argv[:]

    # Construct args list
    args = [
        "--task",
        config["task"],
        "--model-path",
        config["model_path"],
        "--api-server",
        config["api_server"],
        "--scenario",
        config["scenario"],
        "--dataset-path",
        config["dataset_path"],
        "--dataset-input-column",
        config["dataset_input_column"],
        "--target-qps",
        str(config["target_qps"]),
        "--backend",
        config["backend"],
    ]

    # Add optional args
    if config["dataset_output_column"]:
        args.extend(["--dataset-output-column", config["dataset_output_column"]])
    if config.get("accuracy"):
        args.append("--accuracy")
    if config.get("batch_size"):
        args.extend(["--batch-size", str(config["batch_size"])])
    if config.get("dataset_system_prompt_column"):
        args.extend(
            ["--dataset-system-prompt-column", config["dataset_system_prompt_column"]]
        )
    if config.get("max_generated_tokens"):
        args.extend(["--max-generated-tokens", str(config["max_generated_tokens"])])
    if config.get("total_sample_count"):
        args.extend(["--total-sample-count", str(config["total_sample_count"])])

    # Set args and run
    sys.argv[1:] = args
    try:
        return await async_main()  # Await the async function directly
    finally:
        # Restore original args
        sys.argv = old_args


@pytest_asyncio.fixture(scope="session", autouse=True)
async def vllm_server():
    """Start vLLM server for all tests."""
    server_info = await start_vllm_server()
    yield server_info
    await stop_vllm_server(server_info)


def pytest_configure(config):
    """Configure pytest."""
    config.option.capture = "no"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend,scenario,accuracy",
    TEST_CASES.values(),
    ids=TEST_CASES.keys(),
)
async def test_benchmark(backend, scenario, accuracy, request):
    """Test benchmark scenarios."""
    test_case_key = request.node.name.split("[")[-1].split("]")[0]
    print(
        f"\n=== [{test_case_key}] Running {backend} {scenario} {'accuracy' if accuracy else 'performance'} test ==="
    )

    config = BASE_CONFIG.copy()
    config.update({
        "backend": backend,
        "scenario": scenario,
        "accuracy": accuracy,
    })

    if scenario == "Server":
        config["batch_size"] = None
        config["target_qps"] = BASE_CONFIG["target_qps"]
    else:
        config["batch_size"] = BASE_CONFIG["batch_size"]
        config["target_qps"] = float('inf')

    try:
        result = await run_benchmark_cli(config)
        
        # Basic validation
        assert result is not None, "Benchmark result is None"
        assert isinstance(result, dict), "Result should be a dictionary"
        assert result.get("scenario") == scenario, f"Expected scenario {scenario}, got {result.get('scenario')}"
        
        # Performance validation
        if not accuracy:
            assert result.get("samples_per_second", 0) > 0, "samples_per_second should be positive"
            assert result.get("tokens_per_second", 0) > 0, "tokens_per_second should be positive"
            
            assert result.get("completed", 0) > 0, "No samples completed"
            
            if scenario == "Server":
                assert result.get("p90_latency_ns", 0) > 0, "p90_latency_ns should be positive"
                assert result.get("mean_latency_ns", 0) > 0, "mean_latency_ns should be positive"

        # Accuracy validation
        else:
            assert result.get("rouge1") is not None, "No ROUGE-1 score in accuracy results"
            assert result.get("rouge2") is not None, "No ROUGE-2 score in accuracy results"
            assert result.get("rougeL") is not None, "No ROUGE-L score in accuracy results"

    except Exception as e:
        print(f"\n[{test_case_key}] Test failed: {str(e)}")
        print(f"[{test_case_key}] Result was: {json.dumps(result, indent=2) if result else None}")
        raise e
