"""Recent international results loader (feeds form + Bayesian + ML layers).

Prefers REAL results (written by ``data/fetch_real_data.py``) and degrades to the
synthetic Elo-anchored seed when only that is present. A ``synthetic`` column
(0 = real, 1 = synthetic) drives the reported data quality.
"""
from __future__ import annotations

import pandas as pd

from ..utils.io import read_table
from .base import BaseLoader


class MatchHistoryLoader(BaseLoader):
    source = "seed:match_history.csv"
    data_quality = "synthetic"

    def load(self) -> pd.DataFrame:
        df = read_table("match_history.csv", "data_raw")
        df["date"] = pd.to_datetime(df["date"])
        if "synthetic" in df.columns and not bool(df["synthetic"].max()):
            self.source = "real:match_history.csv (martj42 internationals)"
            self.data_quality = "real"
        return df.sort_values("date").reset_index(drop=True)
