"""Simple utilities for FlexBench CLI."""

import logging
import os
import platform
import subprocess
import sys

log = logging.getLogger(__name__)


def setup_logging():
    """Set up logging configuration."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given name."""
    return logging.getLogger(name)


async def check_docker_available():
    """Check if Docker is available."""
    try:
        result = subprocess.run(["docker", "version"], capture_output=True, timeout=10)
        if result.returncode != 0:
            raise RuntimeError("Docker is not running or not accessible")
        log.info("Docker is available and running")
    except FileNotFoundError:
        raise RuntimeError("Docker is not installed") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError("Docker command timed out") from None


def detect_device_type() -> str:
    """Auto-detect device type: cuda -> rocm -> arm -> cpu."""
    log.info("Auto-detecting device type...")

    if _check_command("nvidia-smi"):
        log.info("Detected NVIDIA GPU - using CUDA")
        return "cuda"

    if _check_command("rocm-smi") or _has_amd_vendor():
        log.info("Detected AMD GPU - using ROCm")
        return "rocm"

    if platform.machine().lower() in ("arm64", "aarch64"):
        log.info("Detected ARM architecture")
        return "arm"

    log.info("No GPU detected - using CPU")
    return "cpu"


def get_available_gpus(device_type: str) -> list[str]:
    """Get available GPU indices for device type."""
    if device_type == "cuda":
        return _get_gpu_count("nvidia-smi", "--list-gpus")
    elif device_type == "rocm":
        return _get_gpu_count("rocm-smi", "--showid")
    return []


def _check_command(command: str) -> bool:
    """Check if command exists and runs successfully."""
    try:
        result = subprocess.run([command], capture_output=True, timeout=30)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _has_amd_vendor() -> bool:
    """Check for AMD GPU via vendor ID."""
    try:
        import glob

        for path in glob.glob("/sys/class/drm/card*/device/vendor"):
            with open(path, "r") as f:
                if f.read().strip() == "0x1002":  # AMD vendor ID
                    return True
    except Exception:
        pass
    return False


def _get_gpu_count(command: str, arg: str) -> list[str]:
    """Get GPU count using specified command."""
    try:
        result = subprocess.run([command, arg], capture_output=True, timeout=10)
        if result.returncode == 0:
            lines = [line.strip() for line in result.stdout.decode().split("\n") if line.strip()]
            count = len([line for line in lines if "GPU" in line])
            return [str(i) for i in range(count)] if count > 0 else ["0"]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        log.warning(f"Command '{command}' failed or timed out")
        pass
    return ["0"]
