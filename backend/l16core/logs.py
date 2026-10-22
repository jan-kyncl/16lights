"""Strukturované logy (JSON v kontejneru, čitelné v terminálu)."""

from __future__ import annotations

import logging
import os
import sys

import structlog


def setup_logging() -> None:
    as_json = os.environ.get("L16_LOG_JSON", "0") == "1" or not sys.stderr.isatty()
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer() if as_json else structlog.dev.ConsoleRenderer()
    )
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        logger_factory=structlog.PrintLoggerFactory(sys.stderr),
    )
