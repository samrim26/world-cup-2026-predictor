"""Schema + integrity validation and per-entity data-quality scoring.

Every ingested table is checked for required columns, basic range sanity, and
referential integrity. The output of the engine carries a data-quality score so
no prediction silently pretends to be better-supported than it is.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import load_config


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        return ValidationResult(
            ok=self.ok and other.ok,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
        )


def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> ValidationResult:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return ValidationResult(False, [f"{name}: missing columns {missing}"])
    return ValidationResult(True)


def validate_fixtures(fixtures: pd.DataFrame,
                      teams: pd.DataFrame) -> ValidationResult:
    res = require_columns(
        fixtures,
        ["match_id", "date", "stage", "group", "team_a", "team_b",
         "venue", "city", "country"],
        "fixtures",
    )
    if not res.ok:
        return res
    if fixtures["match_id"].duplicated().any():
        res = res.merge(ValidationResult(False, ["fixtures: duplicate match_id"]))
    # Group-stage referential integrity (knockout slots are placeholders).
    group_rows = fixtures[fixtures["stage"] == "group"]
    known = set(teams["team"])
    unknown = (set(group_rows["team_a"]) | set(group_rows["team_b"])) - known
    if unknown:
        res = res.merge(ValidationResult(
            False, [f"fixtures: group teams not in teams table: {sorted(unknown)}"]))
    n_group = len(group_rows)
    if n_group != 72:
        res = res.merge(ValidationResult(
            True, [], [f"fixtures: expected 72 group matches, found {n_group}"]))
    return res


def validate_teams(teams: pd.DataFrame) -> ValidationResult:
    res = require_columns(
        teams, ["team", "confederation", "fifa_rank", "elo"], "teams")
    if not res.ok:
        return res
    if len(teams) != 48:
        res = res.merge(ValidationResult(
            True, [], [f"teams: expected 48 teams, found {len(teams)}"]))
    if (teams["elo"] <= 0).any():
        res = res.merge(ValidationResult(False, ["teams: non-positive elo"]))
    return res


def validate_strength_weights() -> ValidationResult:
    cfg = load_config()
    weights = cfg.get("strength_weights", {})
    total = sum(weights.values())
    if abs(total - 1.0) > 0.02:
        return ValidationResult(
            True, [], [f"strength_weights sum to {total:.3f}, not ~1.0"])
    return ValidationResult(True)


def data_quality_score(flags: dict[str, bool]) -> tuple[str, float]:
    """Map a dict of availability flags to a (label, score) data-quality tag.

    ``flags`` keys are data facets (e.g. recent_results, squad, odds, venue,
    injuries); ``True`` means present. Score is the weighted fraction available.
    """
    weights = {
        "recent_results": 0.30,
        "ratings": 0.25,
        "venue": 0.15,
        "squad": 0.15,
        "odds": 0.10,
        "injuries": 0.05,
    }
    score = sum(w for k, w in weights.items() if flags.get(k, False))
    cfg = load_config()
    hi = cfg.get("data_quality.high_threshold", 0.8)
    med = cfg.get("data_quality.medium_threshold", 0.5)
    label = "High" if score >= hi else ("Medium" if score >= med else "Low")
    return label, round(score, 3)
