import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import mlperf_loadgen as lg

from flexbench.accuracy_check import run_accuracy_check
from flexbench.configs import BenchmarkConfig, LoadgenConfig
from flexbench.SUT import SUTOffline, SUTServer
from flexbench.utils import get_logger

log = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Base class for benchmark results."""

    scenario: str
    mode: str
    valid: bool

    @classmethod
    def error_result(cls) -> "BenchmarkResult":
        """Create an error result with None values."""
        return cls(**{field: None for field in cls.__dataclass_fields__})

    def __str__(self) -> str:
        def fmt_val(value: float | None) -> float | str:
            if not isinstance(value, float):
                return value
            if "rouge" in str(field) and value is not None:
                return f"{value:.2f}%"
            return round(value, 2)

        return json.dumps(
            {
                field: fmt_val(getattr(self, field))
                for field in self.__dataclass_fields__
            },
            indent=2,
        )


@dataclass
class PerformanceResult(BenchmarkResult):
    """Results from performance benchmark."""

    samples_per_second: float
    tokens_per_second: float
    mean_latency_ns: float
    mean_first_token_ns: float
    mean_tpot_ns: float
    p50_latency_ns: float
    p90_latency_ns: float
    p99_latency_ns: float

    @classmethod
    def from_mlperf_log(
        cls, log_path: Path, config: BenchmarkConfig
    ) -> "PerformanceResult":
        """Create PerformanceResult from MLPerf summary log."""
        if not log_path.exists():
            log.error(f"MLPerf summary log not found at {log_path}")
            return cls.error_result()

        with open(log_path) as f:
            content = f.read()

        def extract_float(pattern: str) -> float | None:
            match = re.search(pattern, content)
            return float(match.group(1)) if match else None

        return cls(
            scenario=config.loadgen_config.scenario,
            mode="PerformanceOnly",
            valid="INVALID" not in content,
            **{
                field: extract_float(pattern)
                for field, pattern in {
                    "samples_per_second": r"Completed samples per second\s*:\s*([\d.]+)",
                    "tokens_per_second": r"Completed tokens per second\s*:\s*([\d.]+)",
                    "mean_latency_ns": r"Mean latency \(ns\)\s*:\s*([\d.]+)",
                    "mean_first_token_ns": r"Mean First Token latency \(ns\)\s*:\s*([\d.]+)",
                    "mean_tpot_ns": r"Mean Time to Output Token \(ns\)\s*:\s*([\d.]+)",
                    "p50_latency_ns": r"50.00 percentile latency \(ns\)\s*:\s*([\d.]+)",
                    "p90_latency_ns": r"90.00 percentile latency \(ns\)\s*:\s*([\d.]+)",
                    "p99_latency_ns": r"99.00 percentile latency \(ns\)\s*:\s*([\d.]+)",
                }.items()
            },
        )


@dataclass
class AccuracyResult(BenchmarkResult):
    """Results from accuracy benchmark."""

    rouge1: float
    rouge2: float
    rougeL: float
    rougeLsum: float
    gen_len: int
    gen_num: int
    gen_tok_len: int
    tokens_per_sample: float

    @classmethod
    def from_mlperf_log(
        cls, log_path: Path, config: BenchmarkConfig
    ) -> "AccuracyResult":
        """Create AccuracyResult from MLPerf accuracy log."""
        if not log_path.exists():
            log.error(f"MLPerf accuracy log not found at {log_path}")
            return cls.error_result()

        metrics = run_accuracy_check(
            model_path=config.model_path,
            dataset_config=config.dataset_config,
            mlperf_accuracy_file=log_path,
            export_txt=True,
            export_json=True,
        )

        return cls(
            scenario=config.loadgen_config.scenario,
            mode="AccuracyOnly",
            valid=True,
            **metrics,
        )


class BenchmarkRunner:
    """Orchestrates MLPerf benchmark execution"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.sut_class = (
            SUTOffline if config.loadgen_config.scenario == "Offline" else SUTServer
        )
        self.results_dir = self._setup_results_dir()

    def run(self) -> BenchmarkResult:
        """Execute benchmark and return results"""
        test_settings = self._setup_test_settings(self.config.loadgen_config)
        log_settings = self._setup_logging(
            self.results_dir,
            self.config.loadgen_config.log_output_to_stdout,
            self.config.loadgen_config.enable_trace,
        )

        sut = self.sut_class(
            task_type=self.config.task,
            model_path=self.config.model_path,
            tokenizer_path=self.config.tokenizer_path,
            api_server=self.config.api_server,
            api_token=self.config.api_token,
            dataset_config=self.config.dataset_config,
            loadgen_config=self.config.loadgen_config,
            max_generated_tokens=self.config.max_generated_tokens,
            batch_size=getattr(self.config, "batch_size", None),
        )

        try:
            lg.StartTestWithLogSettings(sut.sut, sut.qsl, test_settings, log_settings)
            if test_settings.mode == lg.TestMode.PerformanceOnly:
                return PerformanceResult.from_mlperf_log(
                    log_path=self.results_dir / "mlperf_log_summary.txt",
                    config=self.config,
                )
            else:
                return AccuracyResult.from_mlperf_log(
                    log_path=self.results_dir / "mlperf_log_accuracy.json",
                    config=self.config,
                )
        finally:
            sut.stop()
            lg.DestroySUT(sut.sut)
            lg.DestroyQSL(sut.qsl)

    @staticmethod
    def _setup_results_dir() -> Path:
        results_dir = Path("results") / datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    @staticmethod
    def _setup_test_settings(loadgen_config: LoadgenConfig) -> lg.TestSettings:
        """Setup MLPerf loadgen test settings."""
        test_settings = lg.TestSettings()
        test_settings.scenario = getattr(lg.TestScenario, loadgen_config.scenario)
        test_settings.FromConfig(
            loadgen_config.config_path,
            loadgen_config.model_name,
            loadgen_config.scenario,
        )
        test_settings.mode = (
            lg.TestMode.AccuracyOnly
            if loadgen_config.accuracy
            else lg.TestMode.PerformanceOnly
        )

        if loadgen_config.scenario == "Offline":
            test_settings.offline_expected_qps = loadgen_config.target_qps
        elif loadgen_config.scenario == "Server":
            test_settings.server_target_qps = loadgen_config.target_qps

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
