"""Main CLI entry point for FlexBench."""

import asyncio
import logging
import os
import sys

from cli.args import create_cli_parser, validate_args
from cli.config import create_docker_config_from_args
from cli.docker import DockerOrchestrator, _check_docker_available
from cli.utils import get_logger, setup_logging

# Set up logging once at module level
setup_logging()
log = get_logger(__name__)


async def async_main() -> int:
    """Main async CLI function."""
    try:
        # Parse and validate arguments
        parser = create_cli_parser()
        args = parser.parse_args()
        args = validate_args(args)
        
        log.info("FlexBench CLI starting...")
        log.debug(f"Log level set to: {os.getenv('LOG_LEVEL', 'INFO')}")
        log.info(f"Arguments: {vars(args)}")
        if log.isEnabledFor(logging.DEBUG):
            log.debug("Debug logging enabled")
        
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


if __name__ == "__main__":
    sys.exit(main())