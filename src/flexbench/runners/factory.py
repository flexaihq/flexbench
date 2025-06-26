from flexbench.config import BenchmarkConfig
from flexbench.runners.base import BaseRunner
from flexbench.runners.loadgen.runner import LoadGenRunner
from flexbench.runners.vllm.runner import VLLMRunner

RUNNER_REGISTRY = {
    "loadgen": LoadGenRunner,
    "vllm": VLLMRunner,
}


def create_benchmark_runner(backend: str, config: BenchmarkConfig) -> BaseRunner:
    """Create benchmark runner instance based on backend type."""
    if backend not in RUNNER_REGISTRY:
        raise ValueError(f"Unsupported backend: {backend}")

    return RUNNER_REGISTRY[backend](config)
