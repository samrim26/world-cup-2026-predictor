"""Fixtures + venues loaders."""
from __future__ import annotations

import pandas as pd

from ..utils.io import read_table
from .base import BaseLoader


class FixturesLoader(BaseLoader):
    source = "seed:fixtures.csv (real draw, derived schedule)"
    data_quality = "real"

    def load(self) -> pd.DataFrame:
        df = read_table("fixtures.csv", "data_raw")
        df["date"] = pd.to_datetime(df["date"])
        return df


class VenuesLoader(BaseLoader):
    source = "seed:venues.csv"
    data_quality = "real"

    def load(self) -> pd.DataFrame:
        return read_table("venues.csv", "data_raw")
