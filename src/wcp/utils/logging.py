"""Minimal, dependency-free logging setup shared across the package."""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "wcp") -> logging.Logger:
    """Return a package logger, configuring a stderr handler exactly once."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        root = logging.getLogger("wcp")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("wcp") else f"wcp.{name}")
