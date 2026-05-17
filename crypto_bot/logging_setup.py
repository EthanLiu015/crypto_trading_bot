"""Centralized logging configuration."""

from __future__ import annotations

import logging
import sys

from crypto_bot import config


def setup_logging() -> None:
    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    root = logging.getLogger()
    if root.handlers:
        return
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE),
    ]
    logging.basicConfig(level=level, format=fmt, handlers=handlers)
