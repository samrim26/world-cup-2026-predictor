"""Assemble all ingested datasets into one validated container."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..utils.logging import get_logger
from ..utils.validation import (validate_fixtures, validate_strength_weights,
                                validate_teams)
from .fixtures import FixturesLoader, VenuesLoader
from .match_history import MatchHistoryLoader
from .odds import OddsLoader
from .rankings import TeamsLoader
from .squads import SquadsLoader

log = get_logger("ingest")


@dataclass
class Dataset:
    teams: pd.DataFrame
    venues: pd.DataFrame
    fixtures: pd.DataFrame
    history: pd.DataFrame
    odds: pd.DataFrame
    squads: pd.DataFrame
    sources: dict[str, dict]
    warnings: list[str]

    @property
    def has_odds(self) -> bool:
        return self.odds is not None and not self.odds.empty


def load_dataset() -> Dataset:
    """Load, merge venue geo onto fixtures, validate, and return the dataset."""
    loaders = {
        "teams": TeamsLoader(), "venues": VenuesLoader(),
        "fixtures": FixturesLoader(), "history": MatchHistoryLoader(),
        "odds": OddsLoader(), "squads": SquadsLoader(),
    }
    data = {k: ld.load() for k, ld in loaders.items()}
    sources = {k: ld.describe() for k, ld in loaders.items()}

    warnings: list[str] = []
    for vr in (validate_teams(data["teams"]),
               validate_fixtures(data["fixtures"], data["teams"]),
               validate_strength_weights()):
        if not vr.ok:
            for e in vr.errors:
                log.error("validation: %s", e)
            raise ValueError(f"Data validation failed: {vr.errors}")
        warnings.extend(vr.warnings)
    for w in warnings:
        log.warning("validation: %s", w)

    return Dataset(
        teams=data["teams"], venues=data["venues"], fixtures=data["fixtures"],
        history=data["history"], odds=data["odds"], squads=data["squads"],
        sources=sources, warnings=warnings,
    )
