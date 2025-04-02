import os
import subprocess
import sys
import time
from pathlib import Path

import requests

from flexbench.utils import get_logger

log = get_logger(__name__)

# Test model configuration
MODEL_PATH = "HuggingFaceTB/SmolLM2-135M"


def start_server() -> tuple[subprocess.Popen, str]:
    """Start vLLM server with test model."""
    # Change to flexbench root dir to avoid loadgen issues
    original_dir = os.getcwd()
    os.chdir(Path(__file__).parent.parent)

    # Set environment variables for network interface
    env = os.environ.copy()
    env["GLOO_SOCKET_IFNAME"] = "lo0"  # Use loopback interface on macOS
    if "darwin" not in sys.platform.lower():
        env["GLOO_SOCKET_IFNAME"] = "lo"  # Use loopback interface on Linux

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
            "--enforce-eager",  # fast startup
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,  # Use text mode for string output
        bufsize=1,  # Line buffering
        env=env,  # Add environment variables
    )

    # Wait for server to start
    start_time = time.time()
    while time.time() - start_time < 300:
        line = process.stdout.readline().strip()
        if line:
            print("[vLLM Server]", line)
        if process.poll() is not None:
            raise RuntimeError("vLLM server failed to start")
        if "Application startup complete" in line:
            break
        time.sleep(0.1)

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
