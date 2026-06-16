"""Squad / player loader. Optional — ships as an empty template.

When squad data is absent the strength model falls back to team-level ratings
and the affected teams are tagged with lower data quality.
"""
from __future__ import annotations

import pandas as pd

from ..utils.io import read_table
from .base import BaseLoader


class SquadsLoader(BaseLoader):
    source = "seed:squads_template.csv (empty template)"
    data_quality = "missing"

    def load(self) -> pd.DataFrame:
        try:
            df = read_table("squads_template.csv", "data_raw")
        except FileNotFoundError:
            df = pd.DataFrame()
        if not df.empty:
            self.data_quality = "partial"
        return df
