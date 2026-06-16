"""Market-odds loader.

Ships with a sample file; replace with a live provider by subclassing
``OddsLoader`` and pointing ``load`` at your feed. The value engine treats
absent odds as "no market", never as a fabricated price.
"""
from __future__ import annotations

import pandas as pd

from ..utils.io import read_table
from .base import BaseLoader


class OddsLoader(BaseLoader):
    """Prefers a real odds snapshot (``odds_real.csv`` from fetch_real_data.py),
    else falls back to the bundled sample odds."""

    source = "seed:odds_sample.csv (SAMPLE — replace with live feed)"
    data_quality = "sample"

    def __init__(self, filename: str | None = None) -> None:
        self.filename = filename

    def load(self) -> pd.DataFrame:
        candidates = ([self.filename] if self.filename
                      else ["odds_real.csv", "odds_sample.csv"])
        for name in candidates:
            try:
                df = read_table(name, "data_raw")
            except FileNotFoundError:
                continue
            if name == "odds_real.csv":
                self.source = "real:odds_real.csv (market snapshot)"
                self.data_quality = "real"
            return df
        return pd.DataFrame(
            columns=["market", "selection", "odds", "format", "source",
                     "timestamp"])
