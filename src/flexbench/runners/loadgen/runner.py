import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mlperf_loadgen as lg

from flexbench.accuracy_check import run_accuracy_check
from flexbench.configs import BenchmarkConfig, BenchmarkingConfig
from flexbench.runners.base import BaseRunner
from flexbench.runners.loadgen.backend import LoadGenBackend
from flexbench.utils import get_logger

log = get_logger(__name__)


@dataclass
class LoadgenResultBase:
    """Base class for loadgen benchmark results."""

    scenario: str
    mode: str
    valid: bool
    completed: int
    total_samples: int  # Added for consistency with vLLM

    @classmethod
    def error_result(cls) -> "LoadgenResultBase":
        """Create an error result with None values."""
        return cls(**{field: None for field in cls.__dataclass_fields__})

    def __str__(self) -> str:
        """Format result as JSON string."""
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_mlperf_log(cls, log_path: Path, config: BenchmarkConfig) -> "LoadgenResultBase":
        """Create PerformanceResult from MLPerf summary log."""
        if not log_path.exists():
            log.error(f"MLPerf summary log not found at {log_path}")
            return cls.error_result()

        with open(log_path) as f:
            content = f.read()

        return cls(
            scenario=config.benchmarking_config.scenario,  # Updated from loadgen_config
            mode="AccuracyOnly",
            valid=True,
            completed=0,  # These will be filled by child classes
            total_samples=config.benchmarking_config.total_sample_count  # Updated from loadgen_config
        )


@dataclass
class LoadgenPerformanceResult(LoadgenResultBase):
    """Performance benchmark results from loadgen."""

    # Performance-specific fields
    samples_per_second: float
    tokens_per_second: float
    mean_latency_ns: float
    p50_latency_ns: float
    p90_latency_ns: float
    p99_latency_ns: float
    mean_first_token_ns: float | None = None  # Optional for offline mode
    mean_tpot_ns: float | None = None  # Optional for offline mode

    @classmethod
    def from_mlperf_log(
        cls, log_path: Path, config: BenchmarkConfig
    ) -> "LoadgenPerformanceResult":
        if not log_path.exists():
            log.error(f"MLPerf summary log not found at {log_path}")
            return cls.error_result()

        with open(log_path) as f:
            content = f.read()

        # Extract only the Results and Additional Stats sections
        results_section = re.search(
            r"MLPerf Results Summary.*?Additional Stats",
            content, 
            re.DOTALL
        )
        stats_section = re.search(
            r"Additional Stats.*?Test Parameters Used",
            content,
            re.DOTALL
        )
        
        if not results_section or not stats_section:
            log.error("Could not find results sections in log")
            return cls.error_result()
            
        relevant_content = results_section.group(0) + stats_section.group(0)

        def extract_float(pattern: str) -> float | None:
            match = re.search(pattern, relevant_content)
            return float(match.group(1)) if match else None

        # Update regex patterns to match only results section
        patterns = {
            "completed": r"Early stopping satisfied:\s*Yes",  # Updated completion check
            "samples_per_second": r"Samples per second:\s*([\d.]+)",
            "tokens_per_second": r"Tokens per second:\s*([\d.]+)",
            "mean_latency_ns": r"Mean latency \(ns\)\s*:\s*([\d.]+)",
            "p50_latency_ns": r"50.00 percentile latency \(ns\)\s*:\s*([\d.]+)",
            "p90_latency_ns": r"90.00 percentile latency \(ns\)\s*:\s*([\d.]+)",
            "p99_latency_ns": r"99.00 percentile latency \(ns\)\s*:\s*([\d.]+)",
        }

        # Extract metrics using patterns
        metrics = {}
        for key, pattern in patterns.items():
            if key == "completed":
                metrics[key] = (
                    config.benchmarking_config.total_sample_count 
                    if re.search(pattern, relevant_content) 
                    else 0
                )
            else:
                metrics[key] = extract_float(pattern)

        valid = "Result is : VALID" in relevant_content

        return cls(
            scenario=config.benchmarking_config.scenario,
            mode="PerformanceOnly",
            valid=valid,
            total_samples=config.benchmarking_config.total_sample_count or 0,
            mean_first_token_ns=None,  # TTFT not available in summary
            mean_tpot_ns=None,  # TPOT not available in summary
            **metrics
        )


@dataclass
class LoadgenAccuracyResult(LoadgenResultBase):
    """Accuracy benchmark results from loadgen."""

    # Accuracy-specific fields
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
        cls, log_path: Path, config: BenchmarkConfig, output_dir: Path = None
    ) -> "LoadgenAccuracyResult":
        """Create AccuracyResult from MLPerf accuracy log."""
        if not log_path.exists():
            log.error(f"MLPerf accuracy log not found at {log_path}")
            return cls.error_result()

        metrics = run_accuracy_check(
            model_path=config.model_path,
            dataset_config=config.dataset_config,
            mlperf_accuracy_file=log_path,
            output_path=output_dir,
            export_txt=True,
            export_json=True,
        )

        if not metrics:  # Handle empty metrics
            log.error("No accuracy metrics computed")
            return cls.error_result()

        return cls(
            scenario=config.benchmarking_config.scenario,
            mode="AccuracyOnly",
            valid=True,
            completed=metrics.get("gen_num", 0),
            total_samples=config.benchmarking_config.total_sample_count or 0,
            **metrics,
        )

    @classmethod
    def error_result(cls) -> "LoadgenAccuracyResult":
        """Create an error result."""
        return cls(
            scenario="unknown",
            mode="AccuracyOnly",
            valid=False,
            completed=0,
            total_samples=0,
            rouge1=0.0,
            rouge2=0.0,
            rougeL=0.0,
            rougeLsum=0.0,
            gen_len=0,
            gen_num=0,
            gen_tok_len=0,
            tokens_per_sample=0.0,
        )


class LoadGenRunner(BaseRunner):
    """MLPerf LoadGen benchmark runner."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.backend = LoadGenBackend(config=config, results_dir=self.results_dir)

    def run(self) -> dict:
        """Run benchmark and return results."""
        try:
            result = self._run_benchmark()
            return (
                result.__dict__
                if result
                else LoadgenPerformanceResult.error_result().__dict__
            )
        finally:
            self.backend.stop()

    def _run_benchmark(self) -> LoadgenResultBase:
        test_settings = self._setup_test_settings(self.config.benchmarking_config)  # Updated from loadgen_config
        log_settings = self._setup_logging(
            self.results_dir,
            self.config.benchmarking_config.log_output_to_stdout,  # Updated from loadgen_config
            self.config.benchmarking_config.enable_trace,  # Updated from loadgen_config
        )

        lg.StartTestWithLogSettings(
            self.backend.sut, self.backend.qsl, test_settings, log_settings
        )

        if test_settings.mode == lg.TestMode.PerformanceOnly:
            return LoadgenPerformanceResult.from_mlperf_log(
                log_path=self.results_dir / "mlperf_log_summary.txt",
                config=self.config,
            )
        else:
            return LoadgenAccuracyResult.from_mlperf_log(
                log_path=self.results_dir / "mlperf_log_accuracy.json",
                config=self.config,
                output_dir=self.results_dir,
            )

    @staticmethod
    def _setup_results_dir() -> Path:
        results_dir = Path("results") / datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    @staticmethod
    def _setup_test_settings(benchmarking_config: BenchmarkingConfig) -> lg.TestSettings:
        """Setup MLPerf loadgen test settings."""
        test_settings = lg.TestSettings()
        test_settings.scenario = getattr(lg.TestScenario, benchmarking_config.scenario)
        test_settings.FromConfig(
            benchmarking_config.config_path,
            benchmarking_config.model_name,
            benchmarking_config.scenario,
        )
        test_settings.mode = (
            lg.TestMode.AccuracyOnly
            if benchmarking_config.accuracy
            else lg.TestMode.PerformanceOnly
        )

        if benchmarking_config.scenario == "Offline":
            test_settings.offline_expected_qps = benchmarking_config.target_qps
        elif benchmarking_config.scenario == "Server":
            test_settings.server_target_qps = benchmarking_config.target_qps

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
