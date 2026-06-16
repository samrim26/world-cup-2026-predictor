"""Team ratings / rankings loader (Elo, FIFA, squad value, style sub-ratings)."""
from __future__ import annotations

import pandas as pd

from ..utils.io import read_table
from .base import BaseLoader


class TeamsLoader(BaseLoader):
    source = "seed:teams.csv (real Elo/FIFA/value headline, derived sub-ratings)"
    data_quality = "real"

    def load(self) -> pd.DataFrame:
        return read_table("teams.csv", "data_raw")
