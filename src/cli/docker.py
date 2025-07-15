"""Docker orchestration for FlexBench CLI."""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import aiohttp

from cli.config import FlexBenchDockerConfig
from cli.utils import get_available_gpus, get_logger

log = get_logger(__name__)


class DockerOrchestrator:
    """Manages Docker containers for FlexBench benchmarking."""

    def __init__(self, config: FlexBenchDockerConfig):
        self.config = config
        self.compose_file: Path | None = None
        self.temp_dir: Path | None = None
        # Generate timestamp for this run to use consistently
        from datetime import datetime

        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    async def run_benchmark(self) -> dict[str, Any]:
        """Run complete benchmark with Docker orchestration."""
        try:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="flexbench-"))
            log.info(f"Created temporary directory: {self.temp_dir}")

            await self._setup_vllm_server()

            if self.config.build_flexbench:
                await self._build_flexbench_image()

            return await self._run_benchmark_modes()

        finally:
            await self._cleanup_resources()

    async def _setup_vllm_server(self):
        """Set up vLLM server (external or internal)."""
        if self.config.docker_config.vllm_server:
            log.info(f"Using external vLLM server: {self.config.docker_config.vllm_server}")
            await self._check_external_vllm_server()
        else:
            # Setup for internal vLLM server
            self._create_compose_file()

            if self.config.pull_images:
                await self._pull_or_build_vllm_image()

            await self._start_vllm_server()
            await self._wait_for_vllm_ready()

    async def _run_benchmark_modes(self) -> dict[str, Any]:
        """Run benchmark for all specified modes."""
        # Determine modes to run
        mode = self.config.benchmark_config.mode
        modes = ["performance", "accuracy"] if mode == "all" else [mode]

        log.info(f"Running modes: {modes}")

        # Run benchmark(s)
        results = {}
        for current_mode in modes:
            log.info(f"Running {current_mode} benchmark...")
            result = await self._run_flexbench(current_mode)
            results[current_mode] = result

        return results if len(modes) > 1 else results[modes[0]]

    async def _cleanup_resources(self):
        """Clean up all resources."""
        if self.config.cleanup and self.compose_file and self.temp_dir:
            log.info("Cleaning up containers...")
            subprocess.run(
                ["docker", "compose", "-f", str(self.compose_file), "down"],
                capture_output=True,
                cwd=self.temp_dir,
            )
            log.info("Containers cleaned up")

        if self.temp_dir and self.temp_dir.exists():
            import shutil

            shutil.rmtree(self.temp_dir)
            log.info("Temporary files cleaned up")

    def _create_compose_file(self):
        """Generate docker-compose.yml file for vLLM server only."""
        # Prepare directories
        results_dir = Path(self.config.docker_config.results_dir or "results").absolute()
        results_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = Path(self.config.docker_config.model_cache_dir).expanduser().absolute()
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect GPU devices if needed
        device_type = self.config.docker_config.device_type
        if (
            device_type in ("cuda", "rocm")
            and not self.config.docker_config.gpu_devices
            and not self.config.docker_config.vllm_server
        ):
            self.config.docker_config.gpu_devices = get_available_gpus(device_type)
            log.info(
                f"Auto-detected {device_type.upper()} GPU devices: {self.config.docker_config.gpu_devices}"
            )

        # Only create vLLM service - FlexBench runs as individual containers
        compose_config = {
            "services": {"vllm-server": self._get_vllm_service_config(cache_dir)},
            "networks": {self.config.docker_config.network_name: {"driver": "bridge"}},
        }

        # Write compose file
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for CLI functionality. Install with: pip install pyyaml"
            ) from None

        if self.temp_dir is None:
            raise RuntimeError("Temp directory not initialized")

        self.compose_file = self.temp_dir / "docker-compose.yml"
        with open(self.compose_file, "w") as f:
            yaml.dump(compose_config, f, default_flow_style=False, allow_unicode=True)

        log.info(f"Created docker-compose.yml at {self.compose_file}")

    def _get_vllm_service_config(self, cache_dir: Path) -> dict[str, Any]:
        """Get vLLM service configuration for docker-compose."""
        config = {
            "image": self.config.docker_config.vllm_image,
            "container_name": "vllm-server",
            "ports": [f"{self.config.docker_config.vllm_port}:8000"],
            "networks": [self.config.docker_config.network_name],
            "volumes": [
                f"{cache_dir}:/root/.cache/huggingface",
                "/proc/cpuinfo:/proc/cpuinfo:ro",
                "/proc/meminfo:/proc/meminfo:ro",
            ],
            "environment": {
                "HF_HOME": "/root/.cache/huggingface",
                "HF_TOKEN": self.config.benchmark_config.hf_token or os.getenv("HF_TOKEN"),
                "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
                "VLLM_LOGGING_LEVEL": os.getenv("VLLM_LOGGING_LEVEL", "DEBUG"),
            },
            "healthcheck": {
                "test": ["CMD", "curl", "-f", "http://127.0.0.1:8000/health"],
                "interval": "30s",
                "timeout": "30s",
                "retries": 15,
                "start_period": "180s",
            },
            "command": [
                "--model",
                self.config.benchmark_config.remote_model_path,
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--max-model-len",
                str(self.config.docker_config.vllm_max_model_len),
                "--gpu-memory-utilization",
                str(self.config.docker_config.vllm_gpu_memory_utilization),
            ],
        }

        # Apply device-specific configuration
        self._apply_device_config(config, self.config.docker_config.device_type)

        # Add optional settings
        if self.config.docker_config.vllm_disable_log_requests:
            config["command"].append("--disable-log-requests")
        if self.config.docker_config.vllm_memory_limit:
            config["mem_limit"] = self.config.docker_config.vllm_memory_limit

        return config

    def _apply_device_config(self, config: dict[str, Any], device_type: str) -> dict[str, Any]:
        """Apply device-specific configuration to vLLM service."""
        config["environment"]["VLLM_TARGET_DEVICE"] = device_type
        gpu_devices = self.config.docker_config.gpu_devices or (
            ["0"] if device_type in ("cuda", "rocm") else []
        )

        if device_type == "cuda":
            config["environment"]["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(i) for i in range(len(gpu_devices))
            )
            config["ipc"] = "host"
            config["deploy"] = {
                "resources": {
                    "reservations": {
                        "devices": [
                            {"driver": "nvidia", "capabilities": ["gpu"], "device_ids": gpu_devices}
                        ]
                    }
                }
            }

        elif device_type == "rocm":
            config["environment"]["ROCR_VISIBLE_DEVICES"] = ",".join(gpu_devices)
            config["devices"] = ["/dev/kfd", "/dev/dri"]
            config["group_add"] = ["video", "render"]

        elif device_type in ("cpu", "arm"):
            config["environment"]["VLLM_TARGET_DEVICE"] = "cpu"
            config["environment"]["VLLM_CPU_KVCACHE_SPACE"] = "4"
            config["privileged"] = True
            config["security_opt"] = ["seccomp:unconfined"]
            config["cap_add"] = ["SYS_NICE"]
            config["command"].extend(["--enforce-eager", "--disable-custom-all-reduce"])

            if device_type == "arm":
                config["environment"]["VLLM_CPU_OMP_THREADS_BIND"] = "0"
                config["environment"]["OMP_NUM_THREADS"] = "1"
            else:  # cpu
                config["environment"]["VLLM_CPU_OMP_THREADS_BIND"] = "auto"

        else:
            raise ValueError(f"Unsupported device type: {device_type}")

        # Add tensor parallel for GPU devices
        if (
            device_type in ("cuda", "rocm")
            and self.config.docker_config.tensor_parallel_size
            and self.config.docker_config.tensor_parallel_size > 1
        ):
            config["command"].extend(
                ["--tensor-parallel-size", str(self.config.docker_config.tensor_parallel_size)]
            )

        return config

    def _get_benchmark_command_args(self, mode: str) -> list[str]:
        """Get command arguments for FlexBench container."""
        config = self.config.benchmark_config

        vllm_server_url = (
            self.config.docker_config.vllm_server
            or "http://vllm-server:8000"  # Internal container port is always 8000
        )

        container_output_dir = f"/app/results/{self.timestamp}/{mode}"

        args = [
            "python",
            "-m",
            "flexbench",
            "--model-path",
            config.model_path,
            "--api-server",
            vllm_server_url,
            "--scenario",
            config.scenario,
            "--dataset-path",
            config.dataset_config.path,
            "--dataset-input-column",
            config.dataset_config.input_column,
            "--backend",
            config.backend,
            "--output-dir",
            container_output_dir,
        ]

        # Add optional arguments
        if config.remote_model_path and config.remote_model_path != config.model_path:
            args.extend(["--remote-model-path", config.remote_model_path])
        if config.dataset_config.output_column:
            args.extend(["--dataset-output-column", config.dataset_config.output_column])
        if config.dataset_config.system_prompt_column:
            args.extend(
                ["--dataset-system-prompt-column", config.dataset_config.system_prompt_column]
            )
        if config.dataset_config.split != "train":
            args.extend(["--dataset-split", config.dataset_config.split])
        if config.tokenizer_path_override:
            args.extend(["--tokenizer-path-override", config.tokenizer_path_override])
        if config.vllm_server_token:
            args.extend(["--api-token", config.vllm_server_token])
        if config.target_qps is not None:
            args.extend(["--target-qps", str(config.target_qps)])
        if config.batch_size is not None:
            args.extend(["--batch-size", str(config.batch_size)])
        if config.max_generated_tokens is not None:
            args.extend(["--max-generated-tokens", str(config.max_generated_tokens)])
        if config.max_input_tokens is not None:
            args.extend(["--max-input-tokens", str(config.max_input_tokens)])
        if config.total_sample_count is not None:
            args.extend(["--total-sample-count", str(config.total_sample_count)])

        # Add boolean flags
        if config.sweep:
            args.extend(["--sweep", "--num-points", str(config.num_sweep_points)])
        if config.fixed_input_length:
            args.append("--fixed-input-length")
        if mode == "accuracy":
            args.extend(["--mode", "accuracy"])

        return args

    async def _pull_or_build_vllm_image(self):
        """Pull Docker image or build from source if needed."""
        device_type = self.config.docker_config.device_type

        if device_type == "arm":
            await self._build_vllm_from_source()
            return

        vllm_image = self.config.docker_config.vllm_image
        if not vllm_image:
            log.warning("No vLLM image specified, skipping pull")
            return

        log.info(f"Pulling Docker image: {vllm_image}")
        result = subprocess.run(["docker", "pull", vllm_image], capture_output=True, text=True)

        if result.returncode != 0:
            log.warning(f"Failed to pull {vllm_image}: {result.stderr}")

    async def _build_vllm_from_source(self):
        """Build vLLM Docker image from source."""
        device_type = self.config.docker_config.device_type
        log.info(f"Building vLLM from source for {device_type} device...")

        with tempfile.TemporaryDirectory() as temp_dir:  # ty: ignore[no-matching-overload]
            # Clone vLLM repository
            vllm_dir = Path(temp_dir) / "vllm"
            log.info(f"Cloning vLLM repository from {self.config.docker_config.vllm_repo}")

            result = subprocess.run(
                [
                    "git",
                    "clone",
                    "--single-branch",
                    "--branch",
                    self.config.docker_config.vllm_branch,
                    "--depth",
                    "1",
                    self.config.docker_config.vllm_repo,
                    str(vllm_dir),
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Failed to clone vLLM repository: {result.stderr}")

            # Get Dockerfile path
            dockerfile_map = {
                "cuda": "Dockerfile",
                "arm": "Dockerfile.arm",
                "rocm": "Dockerfile.rocm",
                "cpu": "Dockerfile.cpu",
            }
            dockerfile_name = dockerfile_map.get(device_type)
            if not dockerfile_name:
                raise RuntimeError(f"No Dockerfile mapping found for device type: {device_type}")

            dockerfile_path = vllm_dir / "docker" / dockerfile_name
            if not dockerfile_path.exists():
                raise RuntimeError(f"Dockerfile {dockerfile_name} not found in vLLM repository")

            # Prepare build command
            platform_str = "linux/arm64" if device_type == "arm" else "linux/amd64"
            build_command = [
                "docker",
                "build",
                "--platform",
                platform_str,
                "-t",
                self.config.docker_config.custom_vllm_image_name,
                "-f",
                str(dockerfile_path),
                str(vllm_dir),
            ]

            # Add device-specific build arguments
            if device_type == "rocm":
                build_command.extend(["--target", "final"])
            elif device_type == "cpu":
                build_command.extend(["--target", "vllm-openai"])

            # Execute build
            log.info(
                f"Building vLLM Docker image: {self.config.docker_config.custom_vllm_image_name}"
            )
            log.info("This may take a few minutes...")

            env = {"DOCKER_BUILDKIT": "1"}
            if device_type == "arm":
                env.update({"VLLM_TARGET_DEVICE": "cpu", "MAX_JOBS": "1"})

            result = subprocess.run(
                build_command, capture_output=True, text=True, env={**env, **dict(os.environ)}
            )

            if result.returncode != 0:
                raise RuntimeError(f"Failed to build vLLM image: {result.stderr}")

            log.info(
                f"Successfully built vLLM image: {self.config.docker_config.custom_vllm_image_name}"
            )
            self.config.docker_config.vllm_image = self.config.docker_config.custom_vllm_image_name

    async def _build_flexbench_image(self):
        """Build FlexBench Docker image."""
        flexbench_root = Path(__file__).parent.parent / "flexbench"
        dockerfile_path = flexbench_root / "Dockerfile"

        if not dockerfile_path.exists():
            raise RuntimeError(f"Dockerfile not found at {dockerfile_path}")

        log.info("Building FlexBench Docker image...")

        env = {"DOCKER_BUILDKIT": "1"}
        result = subprocess.run(
            [
                "docker",
                "build",
                "--platform",
                "linux/amd64",
                "--ulimit",
                "nofile=65536:65536",
                "-t",
                self.config.docker_config.flexbench_image,
                str(flexbench_root),
            ],
            capture_output=True,
            text=True,
            env={**env, **dict(os.environ)},
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to build FlexBench image: {result.stderr}")

        log.info("FlexBench Docker image built successfully")

    async def _start_vllm_server(self):
        """Start vLLM server using docker-compose."""
        if not self.compose_file or not self.temp_dir:
            raise RuntimeError("Docker compose not initialized")

        log.info("Starting vLLM server...")

        result = subprocess.run(
            ["docker", "compose", "-f", str(self.compose_file), "up", "-d", "vllm-server"],
            capture_output=True,
            text=True,
            cwd=self.temp_dir,
        )

        if result.returncode != 0:
            await self._show_container_logs()
            raise RuntimeError(f"Failed to start vLLM server: {result.stderr}")

        log.info("vLLM server started successfully")

    async def _wait_for_vllm_ready(self):
        """Wait for vLLM server to be ready while streaming logs."""
        log.info("Waiting for vLLM server to be ready...")

        # Start background task to stream vLLM logs
        log_task = asyncio.create_task(self._stream_container_logs("vllm-server", "32"))

        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                while time.time() - start_time < self.config.wait_timeout:
                    try:
                        health_url = (
                            f"http://localhost:{self.config.docker_config.vllm_port}/health"
                        )
                        log.debug(f"Checking vLLM health at: {health_url}")
                        timeout = aiohttp.ClientTimeout(total=10)
                        async with session.get(health_url, timeout=timeout) as resp:
                            log.debug(f"Health check response: status={resp.status}")
                            if resp.status == 200:
                                log.info("vLLM server is ready")
                                return
                            log.debug(f"Health check returned non-200 status: {resp.status}")
                    except Exception as e:
                        log.debug(
                            f"Health check failed with error: '{e}'. Retrying in 5 seconds..."
                        )

                    await asyncio.sleep(5)

                raise TimeoutError(
                    f"vLLM server not ready after {self.config.wait_timeout} seconds"
                )
        finally:
            # Always cancel the log streaming task
            log_task.cancel()
            try:
                await log_task
            except asyncio.CancelledError:
                pass

    async def _stream_container_logs(self, container_name: str, color_code: str):
        """Stream container logs in real-time with colored prefix."""
        process = None
        try:
            await asyncio.sleep(2)  # Wait for container to start

            process = await asyncio.create_subprocess_exec(
                "docker",
                "logs",
                "-f",
                container_name,
                stdout=asyncio.subprocess.PIPE,  # ty: ignore[unresolved-attribute]
                stderr=asyncio.subprocess.STDOUT,  # ty: ignore[unresolved-attribute]
                cwd=self.temp_dir,
            )

            prefix = f"\033[{color_code}m{container_name}\033[0m    | "
            if process.stdout:
                async for line in process.stdout:
                    line_str = line.decode().rstrip()
                    if line_str:
                        print(f"{prefix}{line_str}", flush=True)

            await process.wait()

        except asyncio.CancelledError:
            # Clean shutdown when cancelled
            if process:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                except Exception:
                    pass
            raise
        except Exception as e:
            log.debug(f"Error streaming {container_name} logs: {e}")
            if process:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except Exception:
                    pass

    async def _run_flexbench(self, mode: str) -> dict[str, Any]:
        """Run FlexBench container for a specific mode."""
        log.info("Running FlexBench benchmark...")

        # Get command arguments for this mode
        command_args = self._get_benchmark_command_args(mode)

        # Build docker run command
        docker_run_cmd = self._build_docker_run_command(mode, command_args)

        # Run the container with real-time output streaming
        return await self._run_container_with_streaming(docker_run_cmd, mode)

    def _build_docker_run_command(self, mode: str, command_args: list[str]) -> list[str]:
        """Build docker run command for FlexBench container."""
        if not self.temp_dir:
            raise RuntimeError("Temp directory not initialized")

        results_dir = Path(self.config.docker_config.results_dir or "results").absolute()
        cache_dir = Path(self.config.docker_config.model_cache_dir).expanduser().absolute()

        docker_run_cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            f"flexbench-runner-{mode}",
            "-v",
            f"{results_dir}:/app/results",
            "-v",
            f"{cache_dir}:/root/.cache/huggingface",
            "-e",
            "HF_HOME=/root/.cache/huggingface",
            "-e",
            f"HF_TOKEN={self.config.benchmark_config.hf_token or os.getenv('HF_TOKEN', '')}",
            "-e",
            f"LOG_LEVEL={os.getenv('LOG_LEVEL', 'INFO')}",
        ]

        # Add network configuration
        if self.config.docker_config.vllm_server:
            docker_run_cmd.extend(["--network", "host"])
        else:
            compose_network = f"{self.temp_dir.name}_{self.config.docker_config.network_name}"
            docker_run_cmd.extend(["--network", compose_network])

        # Add memory limit if specified
        if self.config.docker_config.flexbench_memory_limit:
            docker_run_cmd.extend(["--memory", self.config.docker_config.flexbench_memory_limit])

        # Add image and command
        docker_run_cmd.extend([self.config.docker_config.flexbench_image] + command_args)
        return docker_run_cmd

    async def _run_container_with_streaming(
        self, docker_run_cmd: list[str], mode: str
    ) -> dict[str, Any]:
        """Run container with real-time log streaming."""
        process = subprocess.Popen(
            docker_run_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=self.temp_dir,
        )

        if not process.stdout:
            raise RuntimeError("Failed to capture process output")

        # Stream output with colored prefix (blue for flexbench-runner)
        prefix = "\033[34mflexbench-runner\033[0m  | "
        for line in iter(process.stdout.readline, ""):  # ty: ignore[possibly-unbound-attribute]
            if line.strip():
                print(f"{prefix}{line.rstrip()}", flush=True)

        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"FlexBench container failed with exit code {process.returncode}")

        return self._collect_results(mode)

    def _collect_results(self, mode: str) -> dict[str, Any]:
        """Collect results from FlexBench run."""
        results_dir = Path(self.config.docker_config.results_dir or "results")
        mode_results_path = results_dir / self.timestamp / mode
        results_file = mode_results_path / "benchmark_results.json"

        if not results_file.exists():
            log.warning(f"No results file found at {results_file}")
            return {}

        with open(results_file) as f:
            result = json.load(f)

        result["results_path"] = str(results_file.absolute())
        log.info(f"Results collected in: {mode_results_path.absolute()}")
        return result

    async def _show_container_logs(self):
        """Show logs from containers to help debug startup failures."""
        if not self.compose_file or not self.temp_dir:
            return

        log.info("Getting container logs for debugging...")
        for container_name in ["vllm-server", "flexbench-runner"]:
            try:
                # Container status
                result = subprocess.run(
                    [
                        "docker",
                        "ps",
                        "-a",
                        "--filter",
                        f"name={container_name}",
                        "--format",
                        "table {{.Names}}\t{{.Status}}\t{{.Ports}}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                log.debug(f"{container_name} status: {result.stdout}")

                # Container health (for vllm-server)
                if container_name == "vllm-server":
                    result = subprocess.run(
                        [
                            "docker",
                            "inspect",
                            container_name,
                            "--format",
                            "{{.State.Health.Status}}",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    log.debug(f"{container_name} health: {result.stdout.strip()}")

                # Container logs
                result = subprocess.run(
                    ["docker", "logs", container_name], capture_output=True, text=True, timeout=30
                )
                if result.stdout or result.stderr:
                    log.error(f"=== {container_name} Container Logs ===")
                    if result.stdout:
                        log.error(f"stdout: {result.stdout}")
                    if result.stderr:
                        log.error(f"stderr: {result.stderr}")
            except Exception as e:
                log.debug(f"Could not get {container_name} logs: {e}")

    async def _check_external_vllm_server(self):
        """Check if external vLLM server is healthy and accessible."""
        vllm_server = self.config.docker_config.vllm_server
        if not vllm_server:
            raise RuntimeError("vLLM server URL is not configured")

        health_url = f"{vllm_server.rstrip('/')}/health"

        log.info(f"Checking external vLLM server health: {health_url}")

        try:
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=10)
                async with session.get(health_url, timeout=timeout) as resp:
                    if resp.status == 200:
                        log.info("External vLLM server is healthy and ready")
                        return
                    else:
                        raise RuntimeError(f"External vLLM server returned status {resp.status}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to external vLLM server {vllm_server}: {e}"
            ) from e
