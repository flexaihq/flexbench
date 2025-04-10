"""
Test cases for FlexBench benchmarking framework.

Usage:
    python -m pytest

The tests automatically:
1. Start a vLLM server with the test model
2. Run all test cases
3. Shut down the server when done

Example manual test:
    python -m flexbench \
        --task text \
        --model-path HuggingFaceH4/smol_llama_2_135m \
        --api-server http://localhost:8000 \
        --scenario Server \
        --target-qps 1 \
        --dataset-path ctuning/MLPerf-OpenOrca \
        --dataset-input-column question \
        --total-sample-count 10
"""

import json
import subprocess
import sys
import os
import signal
import atexit

import pytest

from flexbench.tests.configs import BASE_CONFIG, TEST_CASES
from flexbench.tests.server import check_server_health, start_server, stop_server


child_pid = None
def kill_child():
    if child_pid is None:
        pass
    else:
        os.kill(child_pid, signal.SIGTERM)
atexit.register(kill_child)


def run_benchmark_subprocess(config: dict) -> dict:
    """Run benchmark through subprocess and return parsed results."""
    # Basic command setup
    cmd = [sys.executable, "-m", "flexbench.main"]

    # Convert all config items to CLI arguments
    for key, value in config.items():
        if value is not None:  # Skip None values
            arg_name = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:  # Only add flag if True
                    cmd.append(arg_name)
            else:
                cmd.extend([arg_name, str(value)])

    # Print reproducible command
    print("\nTo reproduce this test, run:")
    print(" ".join(cmd))

    # Run benchmark and capture output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream output and find results file
    results_file = None
    while True:
        line = process.stdout.readline().strip()
        if not line and process.poll() is not None:
            break
        if line:
            print("[FlexBench]", line)
            if "Results saved to:" in line:
                results_file = line.split(": ")[-1]

    if process.returncode != 0 or not results_file:
        return None
    
    global child_pid
    child_pid = process.pid

    # Load and return results
    with open(results_file) as f:
        results = json.load(f)
        print(f"\n=== Benchmark results saved to {results_file} ===")
        print(json.dumps(results, indent=2))
        return results


@pytest.fixture(scope="session", autouse=True)
def vllm_server():
    """Start vLLM server for all tests."""
    server_info = start_server()
    # Check server health before proceeding
    assert check_server_health(), "vLLM server failed health check"
    yield server_info
    stop_server(server_info)


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

    # Prepare test configuration
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

    # Run benchmark
    result = run_benchmark_subprocess(config)
    assert result is not None, "Benchmark failed to produce results"

    # Validate common fields
    assert isinstance(result, dict)
    assert result["scenario"] == scenario

    # Validate mode-specific metrics
    if not accuracy:
        assert result.get("samples_per_second", 0) > 0
        assert result.get("tokens_per_second", 0) > 0
        assert result.get("completed", 0) > 0
        if scenario == "Server":
            assert result.get("p90_latency_ns", 0) > 0
            assert result.get("mean_latency_ns", 0) > 0
    else:
        assert result.get("rouge1") is not None
        assert result.get("rouge2") is not None
        assert result.get("rougeL") is not None
