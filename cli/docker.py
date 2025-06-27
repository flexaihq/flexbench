"""Docker orchestration for FlexBench CLI."""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from cli.config import FlexBenchDockerConfig
from cli.utils import get_logger

log = get_logger(__name__)


def _get_available_gpus() -> list[str]:
    """Get list of available GPU device indices."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            gpu_indices = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            log.debug(f"Detected GPU indices: {gpu_indices}")
            return gpu_indices
        else:
            log.warning("Failed to detect GPUs with nvidia-smi, falling back to single GPU")
            return ["0"]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        log.warning("nvidia-smi not available, falling back to single GPU")
        return ["0"]


async def _check_docker_available():
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            raise RuntimeError("Docker is not running or not accessible")
        log.info("Docker is available and running")
    except FileNotFoundError:
        raise RuntimeError("Docker is not installed") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError("Docker command timed out - Docker may not be running") from None


class DockerOrchestrator:
    """Manages Docker containers for FlexBench benchmarking."""

    def __init__(self, config: FlexBenchDockerConfig):
        self.config = config
        self.compose_file: Path | None = None
        self.temp_dir: Path | None = None

    async def run_benchmark(self) -> dict[str, Any]:
        """Run complete benchmark with Docker orchestration."""

        try:
            # Setup
            self._setup_temp_directory()
            self._create_compose_file()

            if self.config.pull_images:
                await self._pull_images()

            if self.config.build_flexbench:
                await self._build_flexbench_image()

            # Run containers
            await self._start_containers()
            await self._wait_for_vllm_ready()

            # Run benchmark
            result = await self._run_flexbench()

            # Collect results
            await self._collect_results()

            return result

        finally:
            if self.config.cleanup:
                await self._cleanup_containers()
            self._cleanup_temp_files()

    def _setup_temp_directory(self):
        """Create temporary directory for Docker files."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="flexbench-"))
        log.info(f"Created temporary directory: {self.temp_dir}")

    def _create_compose_file(self):
        """Generate docker-compose.yml file."""

        # Create results directory
        results_dir_str = self.config.docker_config.results_dir or "results"
        results_dir = Path(results_dir_str).absolute()
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create model cache directory
        cache_dir_str = self.config.docker_config.model_cache_dir or "~/.cache/huggingface"
        cache_dir = Path(cache_dir_str).expanduser().absolute()
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Detect GPU devices first, before creating service configs
        device_type = self.config.docker_config.device_type
        if device_type == "nvidia":
            # Get actual GPU devices instead of using "all"
            gpu_devices = self.config.docker_config.gpu_devices
            if not gpu_devices:
                gpu_devices = _get_available_gpus()
                log.info(f"No GPU devices specified, using detected GPUs: {gpu_devices}")
            
            # Store detected GPU devices for use in service configs
            self._detected_gpu_devices = gpu_devices

        compose_config = {
            "services": {
                "vllm-server": self._get_vllm_service_config(cache_dir),
                "flexbench": self._get_flexbench_service_config(results_dir, cache_dir),
            },
            "networks": {
                self.config.docker_config.network_name: {
                    "driver": "bridge"
                }
            }
        }

        # Add device-specific runtime configuration
        if device_type == "nvidia":
            # NVIDIA GPU configuration - use already detected GPU devices
            compose_config["services"]["vllm-server"]["ipc"] = "host"  # Needed for NVIDIA containers
            compose_config["services"]["vllm-server"]["deploy"] = {
                "resources": {
                    "reservations": {
                        "devices": [{
                            "driver": "nvidia",
                            "capabilities": ["gpu"],
                            "device_ids": self._detected_gpu_devices
                        }]
                    }
                }
            }
        elif device_type == "rocm":
            # AMD ROCm GPU configuration
            compose_config["services"]["vllm-server"]["devices"] = ["/dev/kfd", "/dev/dri"]
            compose_config["services"]["vllm-server"]["group_add"] = ["video", "render"]
        # CPU devices don't need special runtime configuration

        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for CLI functionality. Install with: pip install pyyaml") from None

        if self.temp_dir is None:
            raise RuntimeError("Temp directory not initialized")

        self.compose_file = self.temp_dir / "docker-compose.yml"

        with open(self.compose_file, 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False, allow_unicode=True)

        log.info(f"Created docker-compose.yml at {self.compose_file}")
        log.debug(f"Docker compose configuration:\n{yaml.dump(compose_config, default_flow_style=False, allow_unicode=True)}")

    def _get_vllm_service_config(self, cache_dir: Path) -> dict[str, Any]:
        """
        Get vLLM service configuration for docker-compose.
        This method sets up the vLLM container for different device types (NVIDIA, ROCm, CPU, ARM),
        applying the correct environment variables and command-line arguments as required by vLLM and hardware.
        """
        device_type = self.config.docker_config.device_type
        # Map 'nvidia' to 'cuda' for vLLM compatibility
        vllm_device_type = 'cuda' if device_type == 'nvidia' else device_type

        # --- Base vLLM container config (common to all devices) ---
        config = {
            "image": self.config.docker_config.vllm_image,
            "container_name": "vllm-server",
            "ports": [f"{self.config.docker_config.vllm_port}:8000"],
            "networks": [self.config.docker_config.network_name],
            "volumes": [f"{cache_dir}:/root/.cache/huggingface"],
            "environment": {
                "HF_HOME": "/root/.cache/huggingface",
                "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO")
            },
            "healthcheck": {
                "test": ["CMD", "curl", "-f", "http://127.0.0.1:8000/health"],
                "interval": "30s",
                "timeout": "30s",
                "retries": 15,
                "start_period": "180s"
            }
        }

        # --- Common: Mount /proc/cpuinfo and /proc/meminfo for all devices ---
        config["volumes"].extend([
            "/proc/cpuinfo:/proc/cpuinfo:ro",
            "/proc/meminfo:/proc/meminfo:ro"
        ])

        # --- Command-line arguments for vLLM server (common) ---
        config["command"] = [
            "--model", self.config.benchmark_config.remote_model_path,
            "--host", "0.0.0.0",
            "--port", "8000",
            "--max-model-len", str(self.config.docker_config.vllm_max_model_len),
        ]

        # --- Device-specific configuration ---
        if vllm_device_type == "cuda":
            # CUDA GPU: set CUDA_VISIBLE_DEVICES and tensor parallel if needed
            gpu_devices = getattr(self, '_detected_gpu_devices', self.config.docker_config.gpu_devices or [])
            config["environment"]["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_devices)
            if self.config.docker_config.gpu_count and self.config.docker_config.gpu_count > 1:
                config["command"].extend([
                    "--tensor-parallel-size", str(self.config.docker_config.gpu_count)
                ])
            config["ipc"] = "host"
            config["deploy"] = {
                "resources": {
                    "reservations": {
                        "devices": [{
                            "driver": "nvidia",
                            "capabilities": ["gpu"],
                            "device_ids": gpu_devices
                        }]
                    }
                }
            }
        elif vllm_device_type == "rocm":
            # AMD ROCm GPU: set ROCR_VISIBLE_DEVICES, tensor parallel, and device access
            gpu_devices = getattr(self, '_detected_gpu_devices', self.config.docker_config.gpu_devices or [])
            config["environment"]["ROCR_VISIBLE_DEVICES"] = ",".join(gpu_devices)
            if self.config.docker_config.gpu_count and self.config.docker_config.gpu_count > 1:
                config["command"].extend([
                    "--tensor-parallel-size", str(self.config.docker_config.gpu_count)
                ])
            config["devices"] = ["/dev/kfd", "/dev/dri"]
            config["group_add"] = ["video", "render"]
        elif vllm_device_type in ("cpu", "arm"):
            # CPU/ARM: set required vLLM CPU env vars (see vLLM docs)
            config["environment"]["VLLM_TARGET_DEVICE"] = "cpu"
            config["environment"].setdefault("VLLM_CPU_KVCACHE_SPACE", "4")
            config["environment"].setdefault("VLLM_CPU_OMP_THREADS_BIND", "auto")
            # Add CPU-specific flags to vLLM server command
            config["command"].extend([
                "--enforce-eager",                "--disable-custom-all-reduce"            ])
            # --- Fix: Allow NUMA and thread affinity syscalls in Docker ---
            config["privileged"] = True
            config.setdefault("security_opt", []).append("seccomp:unconfined")
            config.setdefault("cap_add", []).append("SYS_NICE")
        # Set VLLM_TARGET_DEVICE for all devices except nvidia (which infers from CUDA)
        elif vllm_device_type not in ("cuda", "rocm", "cpu", "arm"):
            config["environment"]["VLLM_TARGET_DEVICE"] = vllm_device_type

        # Always set VLLM_LOGGING_LEVEL=DEBUG for easier debugging
        config["environment"]["VLLM_LOGGING_LEVEL"] = os.getenv("VLLM_LOGGING_LEVEL", "DEBUG")

        # --- Optional/advanced settings ---
        if self.config.docker_config.vllm_disable_log_requests:
            config["command"].append("--disable-log-requests")
        if self.config.docker_config.vllm_memory_limit:
            config["mem_limit"] = self.config.docker_config.vllm_memory_limit

        # --- End of vLLM service config ---
        return config

    def _get_flexbench_service_config(self, results_dir: Path, cache_dir: Path) -> dict[str, Any]:
        """Get FlexBench service configuration for docker-compose."""

        # Prepare benchmark arguments
        benchmark_args = self._get_benchmark_command_args()

        config = {
            "image": self.config.docker_config.flexbench_image,
            "container_name": "flexbench-runner",
            # Always use amd64 for mlcommons-loadgen wheel compatibility
            "platform": "linux/amd64",
            "networks": [self.config.docker_config.network_name],
            "volumes": [
                f"{results_dir}:/app/results",
                f"{cache_dir}:/root/.cache/huggingface"
            ],
            "environment": {
                "HF_HOME": "/root/.cache/huggingface",
                "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO")
            },
            "command": benchmark_args,
            "depends_on": {
                "vllm-server": {
                    "condition": "service_healthy"
                }
            }
        }

        if self.config.docker_config.flexbench_memory_limit:
            config["mem_limit"] = self.config.docker_config.flexbench_memory_limit

        return config

    def _get_benchmark_command_args(self) -> list[str]:
        """Get command arguments for FlexBench container."""

        config = self.config.benchmark_config

        args = [
            "python", "-m", "flexbench",
            "--model-path", config.model_path,
            "--api-server", f"http://vllm-server:{self.config.docker_config.vllm_port}",
            "--scenario", config.scenario,
            "--dataset-path", config.dataset_config.path,
            "--dataset-input-column", config.dataset_config.input_column,
            "--backend", "loadgen",  # Always use loadgen in containers
            "--output-dir", "/app/results"
        ]

        # Add optional arguments
        if config.remote_model_path and config.remote_model_path != config.model_path:
            args.extend(["--remote-model-path", config.remote_model_path])

        if config.dataset_config.output_column:
            args.extend(["--dataset-output-column", config.dataset_config.output_column])

        if config.dataset_config.system_prompt_column:
            args.extend(["--dataset-system-prompt-column", config.dataset_config.system_prompt_column])

        if config.dataset_config.split != "train":
            args.extend(["--dataset-split", config.dataset_config.split])

        if config.tokenizer_path_override:
            args.extend(["--tokenizer-path-override", config.tokenizer_path_override])

        if config.api_token:
            args.extend(["--api-token", config.api_token])

        if config.target_qps is not None:
            args.extend(["--target-qps", str(config.target_qps)])

        if config.sweep_mode:
            args.append("--sweep")
            args.extend(["--num-points", str(config.num_sweep_points)])

        if config.batch_size is not None:
            args.extend(["--batch-size", str(config.batch_size)])

        if config.max_generated_tokens is not None:
            args.extend(["--max-generated-tokens", str(config.max_generated_tokens)])

        if config.max_input_tokens is not None:
            args.extend(["--max-input-tokens", str(config.max_input_tokens)])

        if config.fixed_input_length:
            args.append("--fixed-input-length")

        if config.accuracy:
            args.append("--accuracy")

        if config.total_sample_count is not None:
            args.extend(["--total-sample-count", str(config.total_sample_count)])

        return args

    async def _pull_images(self):
        """Pull Docker images or build from source if needed."""
        if self.config.docker_config.needs_build_from_source:
            # Build vLLM from source for non-NVIDIA devices
            await self._build_vllm_from_source()
        else:
            # Pull published vLLM image for NVIDIA devices
            vllm_image = self.config.docker_config.vllm_image
            log.info(f"Pulling Docker image: {vllm_image}")
            result = subprocess.run(
                ["docker", "pull", vllm_image],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                log.warning(f"Failed to pull {vllm_image}: {result.stderr}")

    async def _build_flexbench_image(self):
        """Build FlexBench Docker image."""
        # Find the project root (where Dockerfile is located)
        project_root = Path(__file__).parent.parent
        dockerfile_path = project_root / "Dockerfile"

        if not dockerfile_path.exists():
            raise RuntimeError(f"Dockerfile not found at {dockerfile_path}")

        log.info("Building FlexBench Docker image...")

        # Always use amd64 for mlcommons-loadgen wheel compatibility
        # Enable BuildKit for the --mount functionality
        env = {"DOCKER_BUILDKIT": "1"}
        result = subprocess.run(
            [
                "docker", "build",
                "--platform", "linux/amd64",
                "--ulimit", "nofile=65536:65536",
                "-t", self.config.docker_config.flexbench_image,
                str(project_root)
            ],
            capture_output=True,
            text=True,
            env={**env, **dict(os.environ)},
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to build FlexBench image: {result.stderr}")

        log.info("FlexBench Docker image built successfully")

    async def _build_vllm_from_source(self):
        """Build vLLM Docker image from source for non-NVIDIA devices."""
        device_type = self.config.docker_config.device_type
        log.info(f"Building vLLM from source for {device_type} device...")

        import tempfile

        # Create temporary directory for vLLM source
        with tempfile.TemporaryDirectory() as temp_dir:
            vllm_dir = Path(temp_dir) / "vllm"

            # Clone vLLM repository
            log.info(f"Cloning vLLM repository from {self.config.docker_config.vllm_repo}")
            result = subprocess.run([
                "git", "clone",
                "--branch", self.config.docker_config.vllm_branch,
                "--depth", "1",  # Shallow clone for faster download
                self.config.docker_config.vllm_repo,
                str(vllm_dir)
            ], capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Failed to clone vLLM repository: {result.stderr}")

            # Build Docker image with appropriate Dockerfile (only for ARM)
            if device_type == "arm":
                dockerfile = "Dockerfile.arm"
                dockerfile_path = vllm_dir / "docker" / dockerfile
                if not dockerfile_path.exists():
                    raise RuntimeError(f"Dockerfile {dockerfile} not found in vLLM repository")
                platform_str = "linux/arm64"
                build_command = [
                    "docker", "build",
                    "--platform", platform_str,
                    "-t", self.config.docker_config.custom_vllm_image_name,
                    "-f", str(dockerfile_path),
                    str(vllm_dir)
                ]
            else:
                raise RuntimeError("Building vLLM from source is only supported for ARM architecture. Other architectures should use pre-built images.")

            # Add device-specific build arguments
            if device_type == "rocm":
                build_command.extend(["--target", "final"])
            elif device_type == "cpu":
                build_command.extend(["--target", "vllm-openai"])

            # Add all custom build arguments
            if self.config.docker_config.vllm_build_args:
                for key, value in self.config.docker_config.vllm_build_args.items():
                    build_command.extend(["--build-arg", f"{key}={value}"])

            log.info(f"Building vLLM Docker image: {self.config.docker_config.custom_vllm_image_name}")
            log.info("This may take 10-30 minutes...")

            # Run the build (this can take a while)
            # Enable BuildKit for better build performance and features
            env = {"DOCKER_BUILDKIT": "1"}
            result = subprocess.run(
                build_command,
                capture_output=True,
                text=True,
                env={**env, **dict(os.environ)},
            )

            if result.returncode != 0:
                raise RuntimeError(f"Failed to build vLLM image: {result.stderr}")

            log.info(f"Successfully built vLLM image: {self.config.docker_config.custom_vllm_image_name}")

            # Update the vLLM image name to use the custom built image
            self.config.docker_config.vllm_image = self.config.docker_config.custom_vllm_image_name

    async def _start_containers(self):
        """Start Docker containers using docker-compose."""
        if self.compose_file is None or self.temp_dir is None:
            raise RuntimeError("Docker compose not initialized")

        log.info("Starting Docker containers...")
        log.debug(f"Using compose file: {self.compose_file}")
        log.debug(f"Working directory: {self.temp_dir}")

        log.debug("Running docker compose up command...")
        result = subprocess.run(
            ["docker", "compose", "-f", str(self.compose_file), "up", "-d"],
            capture_output=True,
            text=True,
            cwd=self.temp_dir
        )

        log.debug(f"Docker compose command return code: {result.returncode}")
        log.debug(f"Docker compose stdout: {result.stdout}")
        log.debug(f"Docker compose stderr: {result.stderr}")

        if result.returncode != 0:
            log.error(f"Docker compose failed with return code {result.returncode}")
            log.error(f"stdout: {result.stdout}")
            log.error(f"stderr: {result.stderr}")

            # Get detailed logs from failed containers
            await self._show_container_logs()

            raise RuntimeError(f"Failed to start containers: {result.stderr}")

        log.info("Containers started successfully")

    async def _wait_for_vllm_ready(self):
        """Wait for vLLM server to be ready."""
        log.info("Waiting for vLLM server to be ready...")

        import aiohttp
        import asyncio

        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            while time.time() - start_time < self.config.wait_timeout:
                try:
                    async with session.get(f"http://localhost:{self.config.docker_config.vllm_port}/health") as resp:
                        if resp.status == 200:
                            log.info("vLLM server is ready")
                            return
                except Exception:
                    pass

                await asyncio.sleep(5)

        raise TimeoutError(f"vLLM server not ready after {self.config.wait_timeout} seconds")

    async def _run_flexbench(self) -> dict[str, Any]:
        """Run FlexBench container and wait for completion."""
        if self.compose_file is None or self.temp_dir is None:
            raise RuntimeError("Docker compose not initialized")

        log.info("Running FlexBench benchmark...")

        # Wait for flexbench container to complete
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(self.compose_file),
                "logs",
                "-f",
                "flexbench",
            ],
            cwd=self.temp_dir,
        )

        # Get exit code
        inspect_result = subprocess.run(
            ["docker", "inspect", "flexbench-runner", "--format={{.State.ExitCode}}"],
            capture_output=True,
            text=True
        )

        exit_code = int(inspect_result.stdout.strip())
        if exit_code != 0:
            raise RuntimeError(f"FlexBench container failed with exit code {exit_code}")

        # Load results
        results_dir_str = self.config.docker_config.results_dir or "results"
        results_file = Path(results_dir_str) / "benchmark_results.json"
        if results_file.exists():
            with open(results_file) as f:
                result = json.load(f)
            # Add results path to the result for CLI consistency
            result["results_path"] = str(results_file.absolute())
            return result
        else:
            log.warning("No results file found")
            return {}

    async def _collect_results(self):
        """Collect results from containers."""
        # Results are already in the mounted volume
        results_dir_str = self.config.docker_config.results_dir or "results"
        results_dir = Path(results_dir_str)
        log.info(f"Results collected in: {results_dir.absolute()}")

    async def _cleanup_containers(self):
        """Clean up Docker containers."""
        if self.compose_file is None or self.temp_dir is None:
            return

        log.info("Cleaning up containers...")

        subprocess.run(
            ["docker", "compose", "-f", str(self.compose_file), "down"],
            capture_output=True,
            cwd=self.temp_dir
        )

        log.info("Containers cleaned up")

    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
            log.info("Temporary files cleaned up")

    async def _show_container_logs(self):
        """Show logs from containers to help debug startup failures."""
        if self.compose_file is None or self.temp_dir is None:
            return

        log.info("Getting container logs for debugging...")

        # First check container status
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", "name=vllm-server", "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            log.debug(f"vLLM container status: {result.stdout}")
        except Exception as e:
            log.warning(f"Could not get container status: {e}")

        # Check container health
        try:
            result = subprocess.run(
                ["docker", "inspect", "vllm-server", "--format", "{{.State.Health.Status}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            log.debug(f"vLLM container health: {result.stdout.strip()}")
        except Exception as e:
            log.warning(f"Could not get container health: {e}")

        # Get logs from vllm-server container
        try:
            result = subprocess.run(
                ["docker", "logs", "vllm-server"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.stdout or result.stderr:
                log.error("=== vLLM Server Container Logs ===")
                if result.stdout:
                    log.error(f"stdout: {result.stdout}")
                if result.stderr:
                    log.error(f"stderr: {result.stderr}")
        except Exception as e:
            log.warning(f"Could not get vllm-server logs: {e}")

        # Get logs from flexbench container if it exists
        try:
            result = subprocess.run(
                ["docker", "logs", "flexbench-runner"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.stdout or result.stderr:
                log.error("=== FlexBench Container Logs ===")
                if result.stdout:
                    log.error(f"stdout: {result.stdout}")
                if result.stderr:
                    log.error(f"stderr: {result.stderr}")
        except Exception as e:
            log.debug(f"Could not get flexbench-runner logs (this is normal if it didn't start): {e}")