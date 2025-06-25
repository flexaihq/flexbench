"""Simple logging utilities for FlexBench CLI."""

import logging
import os
import sys


def setup_logging():
    """Set up logging configuration for the entire CLI."""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    level = getattr(logging, log_level, logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
        force=True  # Override any existing configuration
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given name."""
    return logging.getLogger(name)
