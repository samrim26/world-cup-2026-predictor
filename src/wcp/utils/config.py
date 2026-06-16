"""Configuration loading and project-path resolution.

All tunable model parameters live in ``config.yaml`` at the project root. This
module loads it once into a light ``Config`` wrapper that supports dotted
access (``cfg.get("xg.base_goal_rate")``) and resolves project-relative paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root (the directory containing ``config.yaml``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config.yaml").exists():
            return parent
    # Fallback: three levels up from src/wcp/utils/config.py
    return here.parents[3]


@dataclass
class Config:
    """Thin wrapper over the parsed YAML config with dotted-key access."""

    data: dict[str, Any]
    root: Path

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def path(self, key: str) -> Path:
        """Resolve a ``paths.*`` entry to an absolute path under the repo root."""
        rel = self.get(f"paths.{key}")
        if rel is None:
            raise KeyError(f"No path configured for '{key}'")
        return (self.root / rel).resolve()


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> Config:
    """Load and cache the project configuration."""
    root = project_root()
    path = Path(config_path) if config_path else root / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data=data, root=root)
