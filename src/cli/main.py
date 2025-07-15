"""Main CLI entry point for FlexBench."""

import logging
import os
import sys

from cli.args import app
from cli.docker import DockerOrchestrator
from cli.utils import check_docker_available, get_logger, setup_logging

# Set up logging once at module level
setup_logging()
log = get_logger(__name__)


async def run_benchmark_async(config, dry_run: bool = False) -> int:
    """Run the benchmark with the given configuration."""
    try:
        log.info("FlexBench CLI starting...")
        log.info(
            f"Log level set to: {os.getenv('LOG_LEVEL', 'INFO')}. Use LOG_LEVEL environment variable to change it."
        )
        if log.isEnabledFor(logging.DEBUG):
            log.debug("Debug logging enabled")

        # Check for dry run
        if dry_run:
            log.info("Dry run mode - showing configuration without running")
            log.info(f"Docker config: {config}")
            return 0

        # Check Docker availability
        await check_docker_available()

        # Run benchmark with Docker orchestration
        orchestrator = DockerOrchestrator(config)
        result = await orchestrator.run_benchmark()

        if isinstance(result, dict) and all(isinstance(v, dict) for v in result.values()):
            for mode, mode_result in result.items():
                results_path = mode_result.get("results_path", "Unknown")
                log.info(f"Results for {mode}: {results_path}")
        else:
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
    try:
        # Use typer to run the app
        app(prog_name="flexbench")
        return 0
    except SystemExit as e:
        return e.code or 0
    except Exception as e:
        log.error(f"CLI failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
