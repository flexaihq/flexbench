"""FlexBench pytest suite."""

import json
import subprocess
import sys

import pytest
import requests

from flexbench.tests.configs import (
    BASE_CONFIG,
    TEST_CASES,
    start_vllm_server,
    stop_vllm_server,
)


def run_benchmark_subprocess(config: dict) -> dict:
    """Run benchmark through subprocess and return parsed results."""
    cmd = [
        sys.executable,
        "-m",
        "flexbench.main",
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

    if config["dataset_output_column"]:
        cmd.extend(["--dataset-output-column", config["dataset_output_column"]])
    if config.get("accuracy"):
        cmd.append("--accuracy")
    if config.get("batch_size"):
        cmd.extend(["--batch-size", str(config["batch_size"])])
    if config.get("dataset_system_prompt_column"):
        cmd.extend(
            ["--dataset-system-prompt-column", config["dataset_system_prompt_column"]]
        )
    if config.get("max_generated_tokens"):
        cmd.extend(["--max-generated-tokens", str(config["max_generated_tokens"])])
    if config.get("total_sample_count"):
        cmd.extend(["--total-sample-count", str(config["total_sample_count"])])

    # Print reproducible command
    print("\nTo reproduce this test, run:")
    print(" ".join(cmd))

    # Run benchmark
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,  # Use text mode for string output
        bufsize=1,  # Line buffering
    )

    # Stream all output in real-time
    results_file = None
    while True:
        line = process.stdout.readline()
        if line:
            line = line.strip()
            print("[FlexBench]", line)
            if "Results saved to:" in line:
                results_file = line.split(": ")[-1]

        if process.poll() is not None:
            # Read any remaining output
            for line in process.stdout:
                line = line.strip()
                print("[FlexBench]", line)
                if "Results saved to:" in line:
                    results_file = line.split(": ")[-1]
            break

    if process.returncode != 0:
        print("\nBenchmark failed with non-zero exit code")
        return None

    if not results_file:
        print("\nCouldn't find results file in output")
        return None

    with open(results_file) as f:
        return json.load(f)


@pytest.fixture(scope="session", autouse=True)
def vllm_server():
    """Start vLLM server for all tests."""
    server_info = start_vllm_server()
    yield server_info
    stop_vllm_server(server_info)


def test_vllm_server_health():
    """Verify vLLM server is healthy before running benchmarks."""
    server_url = BASE_CONFIG["api_server"]
    try:
        response = requests.get(f"{server_url}/health")
        assert (
            response.status_code == 200
        ), f"Server health check failed with status {response.status_code}"
        print(f"\n=== vLLM server at {server_url} is healthy ===")
    except Exception as e:
        pytest.fail(f"Failed to connect to vLLM server at {server_url}: {e}")


@pytest.mark.parametrize(
    "backend,scenario,accuracy",
    TEST_CASES.values(),
    ids=TEST_CASES.keys(),
)
def test_benchmark(backend, scenario, accuracy, request):
    """Test benchmark scenarios."""
    test_case_key = request.node.name.split("[")[-1].split("]")[0]
    print(
        f"\n=== [{test_case_key}] Running {backend} {scenario} {'accuracy' if accuracy else 'performance'} test ==="
    )

    config = BASE_CONFIG.copy()
    config.update(
        {
            "backend": backend,
            "scenario": scenario,
            "accuracy": accuracy,
        }
    )

    if scenario == "Server":
        config["batch_size"] = None
        config["target_qps"] = BASE_CONFIG["target_qps"]
    else:
        config["batch_size"] = BASE_CONFIG["batch_size"]
        config["target_qps"] = float("inf")

    try:
        result = run_benchmark_subprocess(config)

        assert result is not None, "Benchmark result is None"
        assert isinstance(result, dict), "Result should be a dictionary"
        assert (
            result.get("scenario") == scenario
        ), f"Expected scenario {scenario}, got {result.get('scenario')}"

        if not accuracy:
            assert (
                result.get("samples_per_second", 0) > 0
            ), "samples_per_second should be positive"
            assert (
                result.get("tokens_per_second", 0) > 0
            ), "tokens_per_second should be positive"

            assert result.get("completed", 0) > 0, "No samples completed"

            if scenario == "Server":
                assert (
                    result.get("p90_latency_ns", 0) > 0
                ), "p90_latency_ns should be positive"
                assert (
                    result.get("mean_latency_ns", 0) > 0
                ), "mean_latency_ns should be positive"

        else:
            assert (
                result.get("rouge1") is not None
            ), "No ROUGE-1 score in accuracy results"
            assert (
                result.get("rouge2") is not None
            ), "No ROUGE-2 score in accuracy results"
            assert (
                result.get("rougeL") is not None
            ), "No ROUGE-L score in accuracy results"

    except Exception as e:
        print(f"\n[{test_case_key}] Test failed: {str(e)}")
        print(
            f"[{test_case_key}] Result was: {json.dumps(result, indent=2) if result else None}"
        )
        raise e
