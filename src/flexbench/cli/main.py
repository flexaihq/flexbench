"""Main CLI entry point for FlexBench."""

import asyncio
import sys

from flexbench.cli.args import get_cli_args
from flexbench.cli.config import create_docker_config_from_args
from flexbench.cli.docker import DockerOrchestrator
from flexbench.utils import get_logger

log = get_logger(__name__)


async def async_main() -> int:
    """Main async CLI function."""
    try:
        # Parse and validate arguments
        args = get_cli_args()
        
        log.info("FlexBench CLI starting...")
        log.info(f"Arguments: {vars(args)}")
        
        # Check for dry run
        if getattr(args, 'dry_run', False):
            log.info("Dry run mode - showing configuration without running")
            config = create_docker_config_from_args(args)
            log.info(f"Docker config: {config}")
            return 0
        
        # Create configuration
        config = create_docker_config_from_args(args)
        
        # Check Docker availability
        await _check_docker_available()
        
        # Run benchmark with Docker orchestration
        orchestrator = DockerOrchestrator(config)
        result = await orchestrator.run_benchmark()
        
        log.info("Benchmark completed successfully")
        log.info(f"Results: {result.get('results_path', 'Unknown')}")
        
        return 0
        
    except KeyboardInterrupt:
        log.info("Benchmark interrupted by user")
        return 130
    except Exception as e:
        log.error(f"Benchmark failed: {e}", exc_info=True)
        return 1


def main() -> int:
    """Main CLI entry point."""
    return asyncio.run(async_main())


async def _check_docker_available():
    """Check if Docker and docker-compose are available."""
    import subprocess
    
    # Check Docker
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        log.info(f"Docker version: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "Docker is not available. Please install Docker and ensure it's running."
        ) from None
    
    # Check docker-compose
    try:
        result = subprocess.run(
            ["docker-compose", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        log.info(f"Docker Compose version: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # Try docker compose (newer syntax)
            result = subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True,
                text=True,
                check=True
            )
            log.info(f"Docker Compose version: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "Docker Compose is not available. Please install docker-compose."
            ) from None


if __name__ == "__main__":
    sys.exit(main())
