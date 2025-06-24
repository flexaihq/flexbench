"""Simple logging utilities for FlexBench CLI."""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Get a configured logger for the CLI."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Configure handler
        handler = logging.StreamHandler(sys.stdout)
        
        # Set format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Set level from environment or parameter
    import os
    log_level = level or os.getenv('LOG_LEVEL', 'INFO')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    return logger
