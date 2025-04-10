import os
import subprocess
import signal
import atexit
import sys
import time
from pathlib import Path

import requests

from flexbench.utils import get_logger

log = get_logger(__name__)

# Test model configuration
MODEL_PATH = "HuggingFaceTB/SmolLM2-135M"

child_pid = None
def kill_child():
    if child_pid is None:
        pass
    else:
        os.kill(child_pid, signal.SIGTERM)
atexit.register(kill_child)

def start_server() -> tuple[subprocess.Popen, str]:
    """Start vLLM server with test model."""
    # Change to flexbench root dir to avoid loadgen issues
    original_dir = os.getcwd()
    os.chdir(Path(__file__).parent.parent)

    server_env = os.environ.copy()
    server_env["VLLM_CPU_KVCACHE_SPACE"] = server_env.get("VLLM_CPU_KVCACHE_SPACE", "2")

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
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,  # Use text mode for string output
        bufsize=1,  # Line buffering
        env=server_env,
    )

    # Wait for server to start
    start_time = time.time()
    output_lines = []
    while time.time() - start_time < 300:
        line = process.stdout.readline().strip()
        if line:
            print("[vLLM Server]", line)
            output_lines.append(line)
        if process.poll() is not None:
            # Collect any remaining output
            remaining = process.stdout.read()
            if remaining:
                print("[vLLM Server]", remaining)
                output_lines.append(remaining)
            error_msg = "\n".join(output_lines)
            raise RuntimeError(
                f"vLLM server failed to start. Server output:\n{error_msg}"
            )
        if "Application startup complete" in line:
            break
        time.sleep(0.1)

    global child_pid
    child_pid = process.pid

    return process, original_dir


def stop_server(server_info: tuple[subprocess.Popen, str]) -> None:
    """Stop vLLM server and restore directory."""
    process, original_dir = server_info
    process.terminate()
    process.wait()
    os.chdir(original_dir)


def check_server_health(url: str = "http://localhost:1234") -> bool:
    """Check if vLLM server is healthy."""
    try:
        response = requests.get(f"{url}/health")
        return response.status_code == 200
    except Exception as e:
        log.error(f"Server health check failed: {e}")
        return False
