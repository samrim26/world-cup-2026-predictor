"""Swappable data-loader interface.

Each concrete loader reads one logical dataset and reports a ``data_quality``
tag describing how complete/real its source is. Swap the seed CSV loaders for
live API/scraper loaders by subclassing ``BaseLoader`` and overriding
``load``; the rest of the engine is agnostic to the source.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseLoader(ABC):
    """Abstract base for all dataset loaders."""

    #: short human label for the source, surfaced in data-quality reporting
    source: str = "unknown"
    #: one of "real", "sample", "synthetic", "partial", "missing"
    data_quality: str = "missing"

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return the dataset as a validated DataFrame."""

    def describe(self) -> dict[str, str]:
        return {"source": self.source, "data_quality": self.data_quality}
