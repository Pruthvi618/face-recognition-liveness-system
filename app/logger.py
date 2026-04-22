"""Central logging configuration for application modules.

Why this file exists:
- Keep logging setup consistent across scripts and services.
- Enable easier observability in production deployments.
"""

from __future__ import annotations

import logging
import os


def configure_logging() -> None:
    """Configure root logging once for the whole process."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger for a given module name."""
    return logging.getLogger(name)
