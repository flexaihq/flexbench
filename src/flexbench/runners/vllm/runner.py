import typing as tp
from dataclasses import dataclass

import numpy as np

from flexbench.config import BenchmarkConfig
from flexbench.runners.base import BaseRunner
from flexbench.runners.vllm.backend import RequestOutput, VLLMBackend
from flexbench.utils import get_logger

log = get_logger(__name__)


@dataclass
class VLLMResult:
    """vLLM benchmark results."""

    scenario: str
    mode: tp.Literal["PerformanceOnly", "AccuracyOnly"]
    valid: bool
    completed: int
    total_samples: int
    batch_size: int  # For offline mode

    # Performance metrics
    samples_per_second: float
    tokens_per_second: float
    mean_latency_ns: float
    p50_latency_ns: float
    p90_latency_ns: float
    p99_latency_ns: float

    @classmethod
    def from_measurements(
        cls, outputs: list[RequestOutput], duration: float, config: BenchmarkConfig
    ) -> "VLLMResult":
        """Calculate metrics from outputs."""
        successful = [r for r in outputs if r.success]
        if not successful:
            return cls(
                scenario=config.scenario,
                mode="PerformanceOnly",
                valid=False,
                completed=0,
                total_samples=len(outputs),
                batch_size=getattr(config, "batch_size", 1),
                samples_per_second=0,
                tokens_per_second=0,
                mean_latency_ns=0,
                p50_latency_ns=0,
                p90_latency_ns=0,
                p99_latency_ns=0,
            )

        total_tokens = sum(r.prompt_len + r.output_tokens for r in successful)
        latencies = [r.latency * 1e9 for r in successful]

        return cls(
            scenario=config.scenario,
            mode="PerformanceOnly",
            valid=True,
            completed=len(successful),
            total_samples=len(outputs),
            batch_size=successful[0].batch_size,
            samples_per_second=len(successful) / duration,
            tokens_per_second=total_tokens / duration,
            mean_latency_ns=float(np.mean(latencies)),
            p50_latency_ns=float(np.percentile(latencies, 50)),
            p90_latency_ns=float(np.percentile(latencies, 90)),
            p99_latency_ns=float(np.percentile(latencies, 99)),
        )


class VLLMRunner(BaseRunner):
    """vLLM benchmark runner."""

    def __init__(self, config: BenchmarkConfig):
        log.warning("vLLM runner is still in development. Expect bugs.")
        super().__init__(config)
        self.backend = VLLMBackend(config=config, results_dir=self.results_dir)

    async def run(self) -> dict:
        """Run benchmark and return results."""
        try:
            outputs, duration = await self.backend.run()
            if not outputs:
                return {}
            result = VLLMResult.from_measurements(outputs, duration, self.config)
            return result.__dict__
        finally:
            self.backend.stop()
