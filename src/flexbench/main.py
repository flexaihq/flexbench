"""Main module entry point for FlexBench.

This module provides the direct Python API for FlexBench.
For CLI usage, use the flexbench command or flexbench.cli.main.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Import lightweight modules only
from flexbench.utils import get_logger

log = get_logger(__name__)


def get_args():
    """Parse command line arguments for module usage."""
    from flexbench.args import create_module_parser, validate_args

    parser = create_module_parser()
    args = parser.parse_args()
    return validate_args(args)


async def async_main(args=None) -> dict:
    """Main async function that can accept args directly or parse from command line."""
    # Lazy imports
    from flexbench.config import create_benchmark_config
    from flexbench.runners.factory import create_benchmark_runner

    if args is None:
        args = get_args()

    log.info(f"Running FlexBench with arguments: {args}")

    # Create configurations using shared builders
    benchmark_config = create_benchmark_config(args)

    runner = create_benchmark_runner(args.backend, benchmark_config)
    result = await runner.run()

    # Save results to file
    # Use the specified output directory if provided
    if args.output_dir:
        results_dir = Path(args.output_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
    else:
        results_dir = runner.results_dir

    results_path = results_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    log.info("Benchmark run completed")
    log.info(f"Results saved to: {results_path.absolute()}")

    # Update result with path for subprocesses to find
    if isinstance(result, dict):
        result["results_path"] = str(results_path.absolute())

    return result


def main():
    try:
        asyncio.run(async_main())
        return 0
    except KeyboardInterrupt:
        log.info("Benchmark interrupted by user")
        return 130
    except Exception as e:
        log.error(f"Benchmark failed: {e}", exc_info=True)
        if os.environ.get("LOG_LEVEL", "").upper() == "DEBUG":
            log.error(f"ERROR: {e}", exc_info=True, stack_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
