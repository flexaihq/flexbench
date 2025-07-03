"""Docker orchestration for FlexBench CLI."""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from cli.config import FlexBenchDockerConfig
from cli.utils import get_logger, get_available_gpus

log = get_logger(__name__)


class DockerOrchestrator:
    """Manages Docker containers for FlexBench benchmarking."""

    def __init__(self, config: FlexBenchDockerConfig):
        self.config = config
        self.compose_file: Path | None = None
        self.temp_dir: Path | None = None

    async def run_benchmark(self) -> dict[str, Any]:
        """Run complete benchmark with Docker orchestration."""
        try:
            # Check if using external vLLM server
            if self.config.docker_config.api_server:
                log.info(f"Using external vLLM server: {self.config.docker_config.api_server}")
                await self._check_external_vllm_server()
                
                # Only run FlexBench container
                self._setup_for_external_vllm()
                return await self._run_flexbench_with_external_vllm()
            else:
                # Full Docker orchestration with managed vLLM
                log.info("Creating managed vLLM server")
                self.temp_dir = Path(tempfile.mkdtemp(prefix="flexbench-"))
                log.info(f"Created temporary directory: {self.temp_dir}")
                
                self._create_compose_file()

                if self.config.pull_images:
                    await self._pull_or_build_vllm_image()

                if self.config.build_flexbench:
                    await self._build_flexbench_image()

                # Run containers
                await self._start_containers()
                await self._wait_for_vllm_ready()

                # Run benchmark and return results
                return await self._run_flexbench()

        finally:
            if self.config.cleanup:
                await self._cleanup_containers()
            if self.temp_dir and self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir)
                log.info("Temporary files cleaned up")

    def _create_compose_file(self):
        """Generate docker-compose.yml file."""
        results_dir_str = self.config.docker_config.results_dir or "results"
        results_dir = Path(results_dir_str).absolute()
        results_dir.mkdir(parents=True, exist_ok=True)

        cache_dir = Path(self.config.docker_config.model_cache_dir).expanduser().absolute()
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect GPU devices if not specified
        device_type = self.config.docker_config.device_type
        if device_type in ("cuda", "rocm") and not self.config.docker_config.gpu_devices:
            self.config.docker_config.gpu_devices = get_available_gpus(device_type)
            log.info(f"Auto-detected {device_type.upper()} GPU devices: {self.config.docker_config.gpu_devices}")

        compose_config = {
            "services": {
                "flexbench": self._get_flexbench_service_config(results_dir, cache_dir),
            },
            "networks": {
                self.config.docker_config.network_name: {"driver": "bridge"}
            }
        }

        # Only add vLLM service if not using external server
        if not self.config.docker_config.api_server:
            compose_config["services"]["vllm-server"] = self._get_vllm_service_config(cache_dir)

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

    def _get_vllm_service_config(self, cache_dir: Path) -> dict[str, Any]:
        """Get vLLM service configuration for docker-compose."""
        device_type = self.config.docker_config.device_type

        config = {
            "image": self.config.docker_config.vllm_image,
            "container_name": "vllm-server",
            "ports": [f"{self.config.docker_config.vllm_port}:8000"],
            "networks": [self.config.docker_config.network_name],
            "volumes": [
                f"{cache_dir}:/root/.cache/huggingface",
                "/proc/cpuinfo:/proc/cpuinfo:ro",
                "/proc/meminfo:/proc/meminfo:ro"
            ],
            "environment": {
                "HF_HOME": "/root/.cache/huggingface",
                "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
                "VLLM_LOGGING_LEVEL": os.getenv("VLLM_LOGGING_LEVEL", "DEBUG")
            },
            "healthcheck": {
                "test": ["CMD", "curl", "-f", "http://127.0.0.1:8000/health"],
                "interval": "30s",
                "timeout": "30s",
                "retries": 15,
                "start_period": "180s"
            },
            "command": [
                "--model", self.config.benchmark_config.remote_model_path,
                "--host", "0.0.0.0",
                "--port", "8000",
                "--max-model-len", str(self.config.docker_config.vllm_max_model_len),
            ]
        }

        # Device-specific configuration
        self._apply_device_config(config, device_type)

        # Optional settings
        if self.config.docker_config.vllm_disable_log_requests:
            config["command"].append("--disable-log-requests")
        if self.config.docker_config.vllm_memory_limit:
            config["mem_limit"] = self.config.docker_config.vllm_memory_limit

        return config

    def _apply_device_config(self, config: dict[str, Any], device_type: str) -> None:
        """Apply device-specific configuration to vLLM service."""
        gpu_devices = self.config.docker_config.gpu_devices or ["0"]
        
        if device_type == "cuda":
            config["environment"]["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_devices)
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
        elif device_type == "rocm":
            config["environment"]["ROCR_VISIBLE_DEVICES"] = ",".join(gpu_devices)
            config["devices"] = ["/dev/kfd", "/dev/dri"]
            config["group_add"] = ["video", "render"]
        elif device_type in ("cpu", "arm"):
            config["environment"]["VLLM_TARGET_DEVICE"] = "cpu"
            config["environment"]["VLLM_CPU_KVCACHE_SPACE"] = "4"
            config["environment"]["VLLM_CPU_OMP_THREADS_BIND"] = "auto"
            config["command"].extend(["--enforce-eager", "--disable-custom-all-reduce"])
            config["privileged"] = True
            config["security_opt"] = ["seccomp:unconfined"]
            config["cap_add"] = ["SYS_NICE"]
        else:
            config["environment"]["VLLM_TARGET_DEVICE"] = device_type

        # Add tensor parallel if explicitly specified
        if (device_type in ("cuda", "rocm") and 
            self.config.docker_config.tensor_parallel_size and 
            self.config.docker_config.tensor_parallel_size > 1):
            config["command"].extend([
                "--tensor-parallel-size", str(self.config.docker_config.tensor_parallel_size)
            ])
            config["command"].append("--disable-log-requests")
        if self.config.docker_config.vllm_memory_limit:
            config["mem_limit"] = self.config.docker_config.vllm_memory_limit

        return config

    def _get_flexbench_service_config(self, results_dir: Path, cache_dir: Path) -> dict[str, Any]:
        """Get FlexBench service configuration for docker-compose."""
        config = {
            "image": self.config.docker_config.flexbench_image,
            "container_name": "flexbench-runner",
            "platform": "linux/amd64",  # mlcommons-loadgen wheel compatibility
            "networks": [self.config.docker_config.network_name],
            "volumes": [
                f"{results_dir}:/app/results",
                f"{cache_dir}:/root/.cache/huggingface"
            ],
            "environment": {
                "HF_HOME": "/root/.cache/huggingface",
                "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO")
            },
            "command": self._get_benchmark_command_args()
        }

        # Only add dependency on vLLM service if not using external server
        if not self.config.docker_config.api_server:
            config["depends_on"] = {
                "vllm-server": {
                    "condition": "service_healthy"
                }
            }

        if self.config.docker_config.flexbench_memory_limit:
            config["mem_limit"] = self.config.docker_config.flexbench_memory_limit

        return config

    def _get_benchmark_command_args(self) -> list[str]:
        """Get command arguments for FlexBench container."""
        config = self.config.benchmark_config

        # Use external API server if specified, otherwise use containerized vLLM
        api_server_url = (self.config.docker_config.api_server or 
                         f"http://vllm-server:{self.config.docker_config.vllm_port}")

        args = [
            "python", "-m", "flexbench",
            "--model-path", config.model_path,
            "--api-server", api_server_url,
            "--scenario", config.scenario,
            "--dataset-path", config.dataset_config.path,
            "--dataset-input-column", config.dataset_config.input_column,
            "--backend", "loadgen",
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

    async def _pull_or_build_vllm_image(self):
        """Pull Docker image or build from source if needed."""
        device_type = self.config.docker_config.device_type
        
        # Only ARM builds from source by default (no public image available)
        if device_type == "arm":
            await self._build_vllm_from_source()
        else:
            vllm_image = self.config.docker_config.vllm_image
            log.info(f"Pulling Docker image: {vllm_image}")
            
            result = subprocess.run(
                ["docker", "pull", vllm_image],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                log.warning(f"Failed to pull {vllm_image}: {result.stderr}")

    async def _build_vllm_from_source(self):
        """Build vLLM Docker image from source."""
        device_type = self.config.docker_config.device_type
        log.info(f"Building vLLM from source for {device_type} device...")

        with tempfile.TemporaryDirectory() as temp_dir:
            vllm_dir = Path(temp_dir) / "vllm"

            # Clone vLLM repository
            log.info(f"Cloning vLLM repository from {self.config.docker_config.vllm_repo}")
            result = subprocess.run([
                "git", "clone",
                "--branch", self.config.docker_config.vllm_branch,
                "--depth", "1",
                self.config.docker_config.vllm_repo,
                str(vllm_dir)
            ], capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Failed to clone vLLM repository: {result.stderr}")

            # Get Dockerfile and build command
            dockerfile_map = {
                "cuda": "Dockerfile",
                "arm": "Dockerfile.arm", 
                "rocm": "Dockerfile.rocm",
                "cpu": "Dockerfile.cpu"
            }
            
            dockerfile_name = dockerfile_map.get(device_type)
            if not dockerfile_name:
                raise RuntimeError(f"No Dockerfile mapping found for device type: {device_type}")
                
            dockerfile_path = vllm_dir / "docker" / dockerfile_name
            if not dockerfile_path.exists():
                raise RuntimeError(f"Dockerfile {dockerfile_name} not found in vLLM repository")

            # Build Docker command
            build_command = [
                "docker", "build",
                "-t", self.config.docker_config.custom_vllm_image_name,
                "-f", str(dockerfile_path),
                str(vllm_dir)
            ]
            
            # Add device-specific build options
            if device_type == "arm":
                build_command.insert(2, "--platform")
                build_command.insert(3, "linux/arm64")
            elif device_type == "rocm":
                build_command.insert(2, "--target")
                build_command.insert(3, "final")
            elif device_type == "cpu":
                build_command.insert(2, "--target")
                build_command.insert(3, "vllm-openai")

            # Add custom build arguments if provided
            if self.config.docker_config.vllm_build_args:
                for key, value in self.config.docker_config.vllm_build_args.items():
                    build_command.extend(["--build-arg", f"{key}={value}"])

            log.info(f"Building vLLM Docker image: {self.config.docker_config.custom_vllm_image_name}")
            log.info("This may take 10-30 minutes...")

            # Run the build
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
            self.config.docker_config.vllm_image = self.config.docker_config.custom_vllm_image_name

    async def _build_flexbench_image(self):
        """Build FlexBench Docker image."""
        project_root = Path(__file__).parent.parent
        dockerfile_path = project_root / "Dockerfile"

        if not dockerfile_path.exists():
            raise RuntimeError(f"Dockerfile not found at {dockerfile_path}")

        log.info("Building FlexBench Docker image...")

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

    async def _start_containers(self):
        """Start Docker containers using docker-compose."""
        if self.compose_file is None or self.temp_dir is None:
            raise RuntimeError("Docker compose not initialized")

        log.info("Starting Docker containers...")

        result = subprocess.run(
            ["docker", "compose", "-f", str(self.compose_file), "up", "-d"],
            capture_output=True,
            text=True,
            cwd=self.temp_dir
        )

        if result.returncode != 0:
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

        # Follow logs and wait for completion
        subprocess.run(
            ["docker", "compose", "-f", str(self.compose_file), "logs", "-f", "flexbench"],
            cwd=self.temp_dir,
        )

        # Check exit code
        inspect_result = subprocess.run(
            ["docker", "inspect", "flexbench-runner", "--format={{.State.ExitCode}}"],
            capture_output=True,
            text=True
        )

        exit_code = int(inspect_result.stdout.strip())
        if exit_code != 0:
            raise RuntimeError(f"FlexBench container failed with exit code {exit_code}")

        # Load and return results
        results_dir = Path(self.config.docker_config.results_dir or "results")
        results_file = results_dir / "benchmark_results.json"
        
        if results_file.exists():
            with open(results_file) as f:
                result = json.load(f)
            result["results_path"] = str(results_file.absolute())
            log.info(f"Results collected in: {results_dir.absolute()}")
            return result
        else:
            log.warning("No results file found")
            return {}

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

    async def _show_container_logs(self):
        """Show logs from containers to help debug startup failures."""
        if self.compose_file is None or self.temp_dir is None:
            return

        log.info("Getting container logs for debugging...")

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

    async def _check_external_vllm_server(self):
        """Check if external vLLM server is healthy and accessible."""
        import aiohttp
        
        api_server = self.config.docker_config.api_server
        health_url = f"{api_server.rstrip('/')}/health"
        
        log.info(f"Checking external vLLM server health: {health_url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=10) as resp:
                    if resp.status == 200:
                        log.info("External vLLM server is healthy and ready")
                        return
                    else:
                        raise RuntimeError(f"External vLLM server returned status {resp.status}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to external vLLM server {api_server}: {e}")

    def _setup_for_external_vllm(self):
        """Set up for running with external vLLM server (no temp directory needed)."""
        # Create results directory
        results_dir_str = self.config.docker_config.results_dir or "results"
        results_dir = Path(results_dir_str).absolute()
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cache directory
        cache_dir = Path(self.config.docker_config.model_cache_dir).expanduser().absolute()
        cache_dir.mkdir(parents=True, exist_ok=True)

    async def _run_flexbench_with_external_vllm(self) -> dict[str, Any]:
        """Run FlexBench directly against external vLLM server."""
        # For external vLLM, we run FlexBench directly without Docker orchestration
        log.info("Running FlexBench against external vLLM server...")
        
        # Import and run FlexBench directly
        from cli.main import run_benchmark_async
        
        # Create a minimal config for direct execution
        from cli.config import FlexBenchDockerConfig, DockerConfig
        
        # Update the config to not use Docker for vLLM
        direct_config = FlexBenchDockerConfig(
            benchmark_config=self.config.benchmark_config,
            docker_config=self.config.docker_config,
            cleanup=False,  # No containers to clean up
            pull_images=False,  # No images to pull
            build_flexbench=False,  # No FlexBench container needed
            wait_timeout=self.config.wait_timeout,
        )
        
        # Run benchmark directly
        from flexbench.main import main as flexbench_main
        import sys
        from io import StringIO
        
        # Capture the result
        result = {}
        
        # Build args for direct FlexBench execution
        benchmark_args = self._get_direct_benchmark_args()
        
        # Save original argv and replace
        original_argv = sys.argv
        try:
            sys.argv = ["flexbench"] + benchmark_args
            
            # TODO: This needs to be implemented to run FlexBench directly
            # For now, return a placeholder
            log.warning("Direct FlexBench execution not yet implemented")
            result = {"status": "external_server_mode", "api_server": self.config.docker_config.api_server}
            
        finally:
            sys.argv = original_argv
            
        return result

    def _get_direct_benchmark_args(self) -> list[str]:
        """Get command arguments for direct FlexBench execution."""
        config = self.config.benchmark_config
        
        args = [
            "--model-path", config.model_path,
            "--api-server", self.config.docker_config.api_server,
            "--scenario", config.scenario,
            "--dataset-path", config.dataset_config.path,
            "--dataset-input-column", config.dataset_config.input_column,
            "--backend", "loadgen"
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

        # Set output directory
        results_dir = self.config.docker_config.results_dir or "results"
        args.extend(["--output-dir", results_dir])

        return args