import json
import os
import re
import subprocess
import sys
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

    # First Token Latency metrics
    min_first_token_latency_ns: float | None = None
    max_first_token_latency_ns: float | None = None
    mean_first_token_latency_ns: float | None = None
    p50_first_token_latency_ns: float | None = None
    p90_first_token_latency_ns: float | None = None
    p95_first_token_latency_ns: float | None = None
    p97_first_token_latency_ns: float | None = None
    p99_first_token_latency_ns: float | None = None
    p99_9_first_token_latency_ns: float | None = None

    # Time to Output Token metrics
    min_tpot_ns: float | None = None
    max_tpot_ns: float | None = None
    mean_tpot_ns: float | None = None
    p50_tpot_ns: float | None = None
    p90_tpot_ns: float | None = None
    p95_tpot_ns: float | None = None
    p97_tpot_ns: float | None = None
    p99_tpot_ns: float | None = None
    p99_9_tpot_ns: float | None = None

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
            # Basic performance metrics
            "samples_per_second": r"(?:Completed )?[Ss]amples per second\s*:\s*([\d.]+)",
            "tokens_per_second": r"(?:Completed )?[Tt]okens per second\s*:\s*([\d.]+)",
            "mean_latency_ns": r"Mean latency \(ns\)\s*:\s*([\d.]+)",
            "p50_latency_ns": r"50.00 percentile latency \(ns\)\s*:\s*([\d.]+)",
            "p90_latency_ns": r"90.00 percentile latency \(ns\)\s*:\s*([\d.]+)",
            "p99_latency_ns": r"99.00 percentile latency \(ns\)\s*:\s*([\d.]+)",
            # First Token Latency metrics
            "min_first_token_latency_ns": r"Min First Token latency \(ns\)\s*:\s*([\d.]+)",
            "max_first_token_latency_ns": r"Max First Token latency \(ns\)\s*:\s*([\d.]+)",
            "mean_first_token_latency_ns": r"Mean First Token latency \(ns\)\s*:\s*([\d.]+)",
            "p50_first_token_latency_ns": r"50.00 percentile first token latency \(ns\)\s*:\s*([\d.]+)",
            "p90_first_token_latency_ns": r"90.00 percentile first token latency \(ns\)\s*:\s*([\d.]+)",
            "p95_first_token_latency_ns": r"95.00 percentile first token latency \(ns\)\s*:\s*([\d.]+)",
            "p97_first_token_latency_ns": r"97.00 percentile first token latency \(ns\)\s*:\s*([\d.]+)",
            "p99_first_token_latency_ns": r"99.00 percentile first token latency \(ns\)\s*:\s*([\d.]+)",
            "p99_9_first_token_latency_ns": r"99.90 percentile first token latency \(ns\)\s*:\s*([\d.]+)",
            # Time to Output Token (TPOT) metrics
            "min_tpot_ns": r"Min Time to Output Token \(ns\)\s*:\s*([\d.]+)",
            "max_tpot_ns": r"Max Time to Output Token \(ns\)\s*:\s*([\d.]+)",
            "mean_tpot_ns": r"Mean Time to Output Token \(ns\)\s*:\s*([\d.]+)",
            "p50_tpot_ns": r"50.00 percentile time to output token \(ns\)\s*:\s*([\d.]+)",
            "p90_tpot_ns": r"90.00 percentile time to output token \(ns\)\s*:\s*([\d.]+)",
            "p95_tpot_ns": r"95.00 percentile time to output token \(ns\)\s*:\s*([\d.]+)",
            "p97_tpot_ns": r"97.00 percentile time to output token \(ns\)\s*:\s*([\d.]+)",
            "p99_tpot_ns": r"99.00 percentile time to output token \(ns\)\s*:\s*([\d.]+)",
            "p99_9_tpot_ns": r"99.90 percentile time to output token \(ns\)\s*:\s*([\d.]+)",
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
        self.all_results = []

    async def run(self) -> dict:
        """Run benchmark and return results."""
        try:
            if self.config.sweep_mode:
                result = await self._run_sweep_benchmark()
                return result
            else:
                result = self._run_benchmark()
                self.all_results.append(result)
                return (
                    result.__dict__ if result else LoadgenResult.error_result().__dict__
                )
        finally:
            self.backend.stop()

    async def _run_sweep_benchmark(
        self, initial_qps: float = 1000.0, budget: float = 1.2, num_points: int = None
    ) -> dict:
        """Run sweep benchmark with multiple QPS values using separate processes."""
        log.info("Starting sweep benchmark mode")

        # First, run with high QPS to find max effective throughput in a separate process
        log.info(f"Finding maximum throughput with QPS={initial_qps}")

        # Run max throughput test in a separate process
        max_throughput_result = self._run_single_benchmark_process(initial_qps)

        # Extract the max effective QPS achieved from the result
        max_throughput = max_throughput_result.get("samples_per_second", 0)
        if max_throughput <= 0:
            log.warning(
                "Failed to determine maximum throughput, defaulting to initial value"
            )
            max_throughput = initial_qps / 2

        log.info(f"Maximum throughput detected: {max_throughput:.2f} QPS")

        # Create a range of QPS values to sweep through
        qps_values = []
        max_target_qps = max_throughput * budget
        log.info(f"Running sweep with {num_points} data points")
        num_points = num_points or self.config.num_sweep_points
        for i in range(1, num_points + 1):
            qps = (i * max_target_qps) / num_points
            qps_values.append(round(qps, 2))

        log.info(f"Sweeping through QPS values: {[f'{qps:.2f}' for qps in qps_values]}")

        # Run benchmarks for each QPS value in separate processes
        sweep_results = []
        for qps in qps_values:
            log.info(f"Running benchmark with QPS={qps:.2f}")
            result = self._run_single_benchmark_process(qps)
            sweep_results.append(result)

        # Return a dictionary containing all results
        return {
            "max_throughput": max_throughput,
            "sweep_results": sweep_results,
            "latest_result": sweep_results[-1] if sweep_results else {},
        }

    def _run_single_benchmark_process(self, target_qps: float) -> dict:
        """Run a single benchmark with the specified QPS in a separate process."""
        try:
            # Prepare command to run benchmark in a separate process
            cmd = [
                sys.executable,
                "-m",
                "flexbench",
                "--task",
                self.config.task,
                "--model-path",
                self.config.model_path,
                "--api-server",
                self.config.api_server,
                "--scenario",
                self.config.scenario,
                "--target-qps",
                str(target_qps),
                "--dataset-path",
                self.config.dataset_config.path,
                "--dataset-input-column",
                self.config.dataset_config.input_column,
                "--backend",
                "loadgen",
            ]

            # Add optional arguments
            if self.config.dataset_config.output_column:
                cmd.extend(
                    [
                        "--dataset-output-column",
                        self.config.dataset_config.output_column,
                    ]
                )

            if self.config.dataset_config.system_prompt_column:
                cmd.extend(
                    [
                        "--dataset-system-prompt-column",
                        self.config.dataset_config.system_prompt_column,
                    ]
                )

            if self.config.dataset_config.image_column:
                cmd.extend(
                    ["--dataset-image-column", self.config.dataset_config.image_column]
                )

            if self.config.dataset_config.split:
                cmd.extend(["--dataset-split", self.config.dataset_config.split])

            if self.config.tokenizer_path:
                cmd.extend(["--tokenizer-path", self.config.tokenizer_path])

            if self.config.api_token:
                cmd.extend(["--api-token", self.config.api_token])

            if self.config.total_sample_count:
                cmd.extend(
                    ["--total-sample-count", str(self.config.total_sample_count)]
                )

            if self.config.batch_size:
                cmd.extend(["--batch-size", str(self.config.batch_size)])

            if self.config.max_generated_tokens:
                cmd.extend(
                    ["--max-generated-tokens", str(self.config.max_generated_tokens)]
                )

            if self.config.accuracy:
                cmd.append("--accuracy")

            # If we're running a subprocess for sweep, pass the same num_points
            if (
                self.config.sweep_mode and self.config.num_sweep_points != 10
            ):  # If not default
                cmd.extend(["--num-points", str(self.config.num_sweep_points)])

            # Always use DEBUG log level for child processes during sweep mode
            env = os.environ.copy()
            env["LOG_LEVEL"] = "DEBUG"

            # Print a separator to visually distinguish different runs
            separator = (
                f"\n{'-'*80}\n{' '*30}BENCHMARK RUN: QPS={target_qps}\n{'-'*80}\n"
            )
            print(separator)

            # Run process without capturing output so it appears in real-time
            log.debug(f"Running command: {' '.join(cmd)}")

            # Pass stdout/stderr through to parent process to see output in real-time
            # Wait for process to complete before continuing
            process = subprocess.run(
                cmd,
                check=False,  # Don't raise exception, let us handle errors
                env=env,
                text=True,
                # Don't capture output - let it flow to the console in real time
            )

            if process.returncode != 0:
                log.error(f"Subprocess failed with exit code {process.returncode}")
                return {
                    "error": f"Process exited with code {process.returncode}",
                    "qps": target_qps,
                }

            # Find the results file from our own results directory
            results_dir = self.results_dir / f"sweep_qps_{target_qps:.2f}"
            results_dir.mkdir(parents=True, exist_ok=True)

            # Check the last few directories created in the results folder to find our results
            main_results_dir = Path("results")
            if main_results_dir.exists():
                # Find all directories created in the last 5 minutes
                recent_dirs = []
                now = datetime.now()

                for dir_path in main_results_dir.glob("*"):
                    if dir_path.is_dir():
                        try:
                            # Parse the directory name to get timestamp (format: YYYYMMDD-HHMMSS)
                            dir_time_str = dir_path.name
                            dir_time = datetime.strptime(dir_time_str, "%Y%m%d-%H%M%S")
                            time_diff = (now - dir_time).total_seconds()

                            # If directory was created in the last 5 minutes, consider it
                            if time_diff < 300:  # 5 minutes in seconds
                                recent_dirs.append((dir_path, time_diff))
                        except (ValueError, IndexError):
                            # Skip directories that don't match the expected format
                            continue

                # Sort by time (most recent first)
                recent_dirs.sort(key=lambda x: x[1])

                # Look for benchmark_results.json in the most recent directories
                result_path = None
                for dir_path, _ in recent_dirs[
                    :5
                ]:  # Check the 5 most recent directories
                    candidate = dir_path / "benchmark_results.json"
                    if candidate.exists():
                        result_path = candidate
                        log.debug(f"Found results file in {dir_path}")
                        break

            # If we couldn't find it that way, check if the subprocess printed the path
            if not result_path:
                # Try checking the previous command output logs for the results path
                try:
                    log_file = self.results_dir / "subprocess_logs.txt"
                    if log_file.exists():
                        with open(log_file, "r") as f:
                            content = f.read()
                            for line in content.splitlines():
                                if "Results saved to:" in line:
                                    path_str = line.split("Results saved to:")[
                                        1
                                    ].strip()
                                    path = Path(path_str)
                                    if path.exists():
                                        result_path = path
                                        break
                except Exception as e:
                    log.warning(f"Error while searching for results in logs: {e}")

            if not result_path:
                log.warning(f"Could not find results file for QPS={target_qps}")
                return {"error": "No results found", "qps": target_qps}

            # Load and return results
            try:
                with open(result_path) as f:
                    result = json.load(f)
                    log.debug(f"Successfully loaded results from {result_path}")
                    # Add QPS target for reference
                    result["target_qps"] = target_qps
                    return result
            except (FileNotFoundError, json.JSONDecodeError) as e:
                log.error(f"Error loading result file: {e}")
                return {"error": f"Failed to load results: {str(e)}", "qps": target_qps}

        except Exception as e:
            log.error(f"Error running benchmark process: {e}", exc_info=True)
            return {"error": str(e), "qps": target_qps}

    def _run_benchmark(self) -> LoadgenResult:
        test_settings = self._setup_test_settings(self.config)
        log_settings = self._setup_logging(
            self.results_dir,
            self.config.log_output_to_stdout,
            self.config.enable_trace,
        )
        try:
            lg.StartTestWithLogSettings(
                self.backend.sut, self.backend.qsl, test_settings, log_settings
            )

            result = LoadgenResult.from_mlperf_log(
                log_path=self.results_dir / "mlperf_log_summary.txt",
                config=self.config,
                mode="PerformanceOnly",
            )
            return result
        except subprocess.CalledProcessError as e:
            log.error(f"Benchmark process failed: {e}")
        except Exception as e:
            log.error(f"Error running benchmark process: {e}")
            return {"error": str(e)}

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
