import json
import typing as tp
from dataclasses import dataclass

import numpy as np

from flexbench.configs import BenchmarkConfig
from flexbench.runners.base import BaseRunner
from flexbench.runners.vllm.backend import RequestOutput, VLLMBackend
from flexbench.utils import get_logger

log = get_logger(__name__)


@dataclass
class VLLMResultBase:
    """Base class for vLLM benchmark results."""

    scenario: str
    mode: str
    valid: bool
    completed: int
    total_samples: int
    batch_size: int  # Added for offline mode

    @classmethod
    def error_result(cls) -> "VLLMResultBase":
        """Create an error result with None values."""
        return cls(**{field: None for field in cls.__dataclass_fields__})

    def __str__(self) -> str:
        """Format result as JSON string."""
        return json.dumps(self.__dict__, indent=2)


@dataclass
class VLLMPerformanceResult(VLLMResultBase):
    """Performance metrics from vLLM benchmark."""

    total_input_tokens: int
    total_output_tokens: int
    samples_per_second: float  # Renamed from request_throughput
    tokens_per_second: float  # Renamed from token_throughput
    mean_first_token_ns: float  # Renamed from mean_ttft_ms
    p50_ttft_ms: float
    p90_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ns: float  # Convert to ns
    p50_tpot_ms: float
    p90_tpot_ms: float
    p99_tpot_ms: float
    mean_latency_ns: float  # Convert to ns instead of ms
    p50_latency_ns: float
    p90_latency_ns: float
    p99_latency_ns: float

    @classmethod
    def from_measurements(cls, outputs: list[RequestOutput], duration: float, config: BenchmarkConfig) -> "VLLMPerformanceResult":
        metrics = calculate_metrics(outputs, duration)
        if "error" in metrics:
            return cls.error_result()

        # Convert ms to ns for consistency with loadgen
        for metric in metrics:
            if metric.endswith('_ms'):
                metrics[metric.replace('_ms', '_ns')] = metrics.pop(metric) * 1_000_000

        return cls(
            scenario=config.benchmarking_config.scenario,  # Updated from loadgen_config
            mode="PerformanceOnly",
            valid=True,
            **metrics,
        )


def calculate_metrics(outputs: list[RequestOutput], duration: float) -> dict[str, tp.Any]:
    """Calculate aggregate metrics from request outputs."""
    successful = [r for r in outputs if r.success]
    if not successful:
        return {"error": "No successful requests"}

    ttfts = [r.ttft for r in successful if r.ttft]
    latencies = [r.latency for r in successful]
    tpots = []

    total_input = 0
    total_output = 0
    total_batches = 0  # Track number of batches for offline mode

    for r in successful:
        total_input += r.prompt_len * r.batch_size  # Multiply by batch size
        total_output += r.output_tokens * r.batch_size
        total_batches += 1
        if r.output_tokens > 1 and r.ttft:
            tpot = (r.latency - r.ttft) / (r.output_tokens - 1)
            tpots.append(tpot)

    metrics = {
        "completed": len(successful),
        "total_samples": len(outputs),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "samples_per_second": len(successful) / duration,  # Renamed
        "tokens_per_second": (total_input + total_output) / duration,  # Renamed
        "batch_size": successful[0].batch_size if successful else 1,
        # Keep timing metrics in ms for now, convert to ns in from_measurements
        "mean_ttft_ms": np.mean(ttfts) * 1000,
        "p50_ttft_ms": np.percentile(ttfts, 50) * 1000,
        "p90_ttft_ms": np.percentile(ttfts, 90) * 1000,
        "p99_ttft_ms": np.percentile(ttfts, 99) * 1000,
        "mean_tpot_ms": np.mean(tpots) * 1000,
        "p50_tpot_ms": np.percentile(tpots, 50) * 1000,
        "p90_tpot_ms": np.percentile(tpots, 90) * 1000,
        "p99_tpot_ms": np.percentile(tpots, 99) * 1000,
        "mean_latency_ms": np.mean(latencies) * 1000,
        "p50_latency_ms": np.percentile(latencies, 50) * 1000,
        "p90_latency_ms": np.percentile(latencies, 90) * 1000,
        "p99_latency_ms": np.percentile(latencies, 99) * 1000,
    }

    if successful[0].batch_size > 1:  # Add offline-specific metrics
        metrics.update({
            "batch_throughput": total_batches / duration,
            "samples_per_second": (total_input + total_output) / (duration * successful[0].batch_size)
        })

    return metrics


class VLLMRunner(BaseRunner):
    """vLLM benchmark runner."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.backend = VLLMBackend(config=config, results_dir=self.results_dir)

    async def run(self) -> dict:
        """Run benchmark and return results."""
        try:
            outputs, duration = await self.backend.run()
            if not outputs:
                return VLLMPerformanceResult.error_result().__dict__

            result = VLLMPerformanceResult.from_measurements(
                outputs=outputs, duration=duration, config=self.config
            )
            return result.__dict__
        finally:
            self.backend.stop()
