import re
import typing as tp
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mlperf_loadgen as lg

from flexbench.accuracy_check import run_accuracy_check
from flexbench.runners.base import BaseRunner, BenchmarkConfig
from flexbench.runners.loadgen.backend import LoadGenBackend
from flexbench.utils import get_logger

log = get_logger(__name__)


@dataclass
class LoadgenResult:
    """LoadGen benchmark results."""

    scenario: str
    mode: tp.Literal["PerformanceOnly", "AccuracyOnly"]
    valid: bool
    completed: int
    total_samples: int

    # Performance metrics
    samples_per_second: float | None = None
    tokens_per_second: float | None = None
    mean_latency_ns: float | None = None
    p50_latency_ns: float | None = None
    p90_latency_ns: float | None = None
    p99_latency_ns: float | None = None

    # Accuracy metrics
    rouge1: float | None = None
    rouge2: float | None = None
    rougeL: float | None = None
    gen_len: int | None = None
    tokens_per_sample: float | None = None

    @classmethod
    def from_mlperf_log(
        cls, log_path: Path, config: BenchmarkConfig, mode: str = "PerformanceOnly"
    ) -> "LoadgenResult":
        """Create result from MLPerf logs."""
        if not log_path.exists():
            log.error(f"MLPerf log not found at {log_path}")
            return cls(
                scenario=config.scenario,
                mode=mode,
                valid=False,
                completed=0,
                total_samples=0,
            )

        if mode == "AccuracyOnly":
            metrics = run_accuracy_check(
                model_path=config.model_path,
                dataset_config=config.dataset_config,
                mlperf_accuracy_file=log_path,
            )
            return cls(
                scenario=config.scenario,
                mode=mode,
                valid=True,
                completed=metrics.get("gen_num", 0),
                total_samples=config.total_sample_count or 0,
                rouge1=metrics.get("rouge1"),
                rouge2=metrics.get("rouge2"),
                rougeL=metrics.get("rougeL"),
                gen_len=metrics.get("gen_len"),
                tokens_per_sample=metrics.get("tokens_per_sample"),
            )

        with open(log_path) as f:
            content = f.read()

        def extract_float(pattern: str) -> float | None:
            match = re.search(pattern, content)
            return float(match.group(1)) if match else None

        patterns = {
            "samples_per_second": r"(?:Completed )?[Ss]amples per second\s*:\s*([\d.]+)",
            "tokens_per_second": r"(?:Completed )?[Tt]okens per second\s*:\s*([\d.]+)",
            "mean_latency_ns": r"Mean latency \(ns\)\s*:\s*([\d.]+)",
            "p50_latency_ns": r"50.00 percentile latency \(ns\)\s*:\s*([\d.]+)",
            "p90_latency_ns": r"90.00 percentile latency \(ns\)\s*:\s*([\d.]+)",
            "p99_latency_ns": r"99.00 percentile latency \(ns\)\s*:\s*([\d.]+)",
        }

        metrics = {k: extract_float(v) for k, v in patterns.items()}
        valid = "Result is : VALID" in content
        completed = (
            config.total_sample_count if "Early stopping satisfied" in content else 0
        )

        return cls(
            scenario=config.scenario,
            mode=mode,
            valid=valid,
            completed=completed,
            total_samples=config.total_sample_count or 0,
            **metrics,
        )


class LoadGenRunner(BaseRunner):
    """MLPerf LoadGen benchmark runner."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.backend = LoadGenBackend(config=config, results_dir=self.results_dir)

    async def run(self) -> dict:
        """Run benchmark and return results."""
        try:
            result = self._run_benchmark()
            return result.__dict__ if result else LoadgenResult.error_result().__dict__
        finally:
            self.backend.stop()

    def _run_benchmark(self) -> LoadgenResult:
        test_settings = self._setup_test_settings(self.config)
        log_settings = self._setup_logging(
            self.results_dir,
            self.config.log_output_to_stdout,
            self.config.enable_trace,
        )

        lg.StartTestWithLogSettings(
            self.backend.sut, self.backend.qsl, test_settings, log_settings
        )

        if test_settings.mode == lg.TestMode.PerformanceOnly:
            return LoadgenResult.from_mlperf_log(
                log_path=self.results_dir / "mlperf_log_summary.txt",
                config=self.config,
                mode="PerformanceOnly",
            )
        else:
            return LoadgenResult.from_mlperf_log(
                log_path=self.results_dir / "mlperf_log_accuracy.json",
                config=self.config,
                mode="AccuracyOnly",
            )

    @staticmethod
    def _setup_results_dir() -> Path:
        results_dir = Path("results") / datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    def _setup_test_settings(
        self,
        config: BenchmarkConfig,
    ) -> lg.TestSettings:
        """Setup MLPerf loadgen test settings."""
        test_settings = lg.TestSettings()
        test_settings.scenario = getattr(lg.TestScenario, config.scenario)
        test_settings.FromConfig(
            config.config_path,
            config.model_name,
            config.scenario,
        )
        test_settings.mode = (
            lg.TestMode.AccuracyOnly if config.accuracy else lg.TestMode.PerformanceOnly
        )

        if config.scenario == "Offline":
            test_settings.offline_expected_qps = config.target_qps
        elif config.scenario == "Server":
            test_settings.server_target_qps = config.target_qps

        return test_settings

    @staticmethod
    def _setup_logging(
        output_dir: Path, copy_to_stdout: bool = True, enable_trace: bool = False
    ) -> lg.LogSettings:
        """Setup MLPerf loadgen logging settings."""
        log_settings = lg.LogSettings()
        log_settings.log_output.outdir = str(output_dir)
        log_settings.log_output.copy_summary_to_stdout = copy_to_stdout
        log_settings.enable_trace = enable_trace
        return log_settings
