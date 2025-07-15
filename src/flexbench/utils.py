import logging
import os


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger.setLevel(log_level)

    if not logger.handlers:
        handler = logging.StreamHandler()  # ty: ignore[no-matching-overload]
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
