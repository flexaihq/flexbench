import asyncio
import os
import subprocess
import time
from pathlib import Path

# Core test configuration
MODEL_PATH = "HuggingFaceTB/SmolLM2-135M"
DATASET_PATH = "ctuning/MLPerf-OpenOrca"
SAMPLE_COUNT = 15  # Small count for quick tests

BASE_CONFIG = {
    "task": "text",
    "model_path": MODEL_PATH,
    "api_server": "http://localhost:1234",
    "dataset_path": DATASET_PATH,
    "dataset_input_column": "question",
    "dataset_output_column": "response",
    "dataset_system_prompt_column": "system_prompt",
    "total_sample_count": SAMPLE_COUNT,
    "max_generated_tokens": 64,  # Smaller for faster tests
    "batch_size": 2,  # Default batch size for offline mode
    "target_qps": 5,  # Default QPS for server mode
}


def make_test_id(backend: str, scenario: str, accuracy: bool) -> str:
    """Generate test ID from scenario parameters."""
    mode = "acc" if accuracy else "perf"
    return f"{backend}-{scenario.lower()}-{mode}"


TEST_CASES = {
    "loadgen-server-perf": ("loadgen", "Server", False),
    "loadgen-offline-perf": ("loadgen", "Offline", False),
    "loadgen-server-accuracy": ("loadgen", "Server", True),
    "loadgen-offline-accuracy": ("loadgen", "Offline", True),
    "vllm-server-perf": ("vllm", "Server", False),
    "vllm-offline-perf": ("vllm", "Offline", False),
}


def log_server_output(process: subprocess.Popen):
    """Log server output in real-time."""
    while True:
        line = process.stderr.readline()
        if not line:
            break
        print(f"[vLLM Server] {line.strip()}")


async def start_vllm_server():
    """Start vLLM server with test model."""
    # Change to flexbench root dir
    original_dir = os.getcwd()
    os.chdir(Path(__file__).parent.parent)
    print(f"Changed directory to {os.getcwd()}")

    process = subprocess.Popen(
        [
            "uv",
            "run",
            "vllm",
            "serve",
            MODEL_PATH,
            "--disable-log-requests",
            "--port",
            "1234",
        ],
        stdout=subprocess.PIPE,  # Capture but forward output
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
    )

    # Forward server output in real-time while we wait for startup
    start_time = time.time()
    while time.time() - start_time < 300:
        line = await asyncio.get_event_loop().run_in_executor(
            None, process.stdout.readline
        )
        if line:
            print("[vLLM Server]", line.decode().strip())
        if process.poll() is not None:
            raise RuntimeError("vLLM server failed to start")
        if b"Application startup complete" in line:
            break
        await asyncio.sleep(0.01)

    # Keep process in root dir for loadgen outputs
    return process, original_dir


async def stop_vllm_server(server_info):
    """Stop vLLM server and restore directory."""
    process, original_dir = server_info
    process.terminate()
    await asyncio.get_event_loop().run_in_executor(None, process.wait)
    os.chdir(original_dir)
