"""Small I/O helpers for the config-driven CSV/parquet data store."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import load_config
from .logging import get_logger

log = get_logger("io")


def _resolve(path_key_or_file: str, subdir: str) -> Path:
    """Resolve a bare filename under a configured data subdir, or a full path."""
    cfg = load_config()
    p = Path(path_key_or_file)
    if p.is_absolute() or p.parent != Path("."):
        return p
    return cfg.path(subdir) / path_key_or_file


def read_table(name: str, subdir: str = "data_raw") -> pd.DataFrame:
    """Read a CSV (or parquet, by extension) from a configured data dir."""
    path = _resolve(name, subdir)
    if not path.exists():
        raise FileNotFoundError(f"Expected data file not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, name: str, subdir: str = "outputs") -> Path:
    """Write a DataFrame to a configured data dir, creating it if needed."""
    cfg = load_config()
    out_dir = cfg.path(subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    log.info("wrote %d rows -> %s", len(df), path)
    return path
