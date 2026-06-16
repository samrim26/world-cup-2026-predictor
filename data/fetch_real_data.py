"""Fetch REAL data to replace the synthetic seeds.

1. Recent international results  — downloaded live from the public martj42
   `international_results` dataset (all internationals since 1872, exact schema).
   Filtered to a recency window and name-normalised to our 48-team naming.
2. Tournament-winner odds        — a dated real-market snapshot (American odds).
   Odds pages are JS-heavy / rate-limited, so the snapshot is embedded here with
   its source + date; refresh it by editing `WINNER_ODDS_SNAPSHOT` or pointing
   `ODDS_URL` at a parseable feed.

Run:  python3 data/fetch_real_data.py
Writes data/raw/match_history.csv (real) and data/raw/odds_real.csv.
Offline / on failure it leaves the existing synthetic seeds in place so the
engine still runs.

The loaders (`ingest/match_history.py`, `ingest/odds.py`) automatically prefer
these real files when present.
"""
from __future__ import annotations

import io
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
sys.path.insert(0, str(ROOT / "data"))
from build_seed_data import GROUPS  # noqa: E402  (the 48-team draw)

RESULTS_URL = ("https://raw.githubusercontent.com/martj42/"
               "international_results/master/results.csv")
RECENT_SINCE = "2022-01-01"          # recency window for relevance

# Map dataset spellings -> our naming. Only Curaçao differs for our 48.
NAME_FIX = {
    "Curaçao": "Curacao",
    "Cape Verde Islands": "Cape Verde",
    "Republic of Ireland": "Ireland",
    "DR Congo": "DR Congo",
    "South Korea": "South Korea",
    "United States": "United States",
}

# Real tournament-winner odds — snapshot. Source: FOX Sports, 2026-06-10.
ODDS_SOURCE = "FOX Sports"
ODDS_DATE = "2026-06-10"
WINNER_ODDS_SNAPSHOT = {       # team -> American odds
    "Spain": 450, "France": 500, "England": 700, "Brazil": 850,
    "Portugal": 850, "Argentina": 1000, "Germany": 1300, "Netherlands": 1600,
    "Belgium": 2200, "Norway": 3300, "Colombia": 4000, "Morocco": 5000,
    "United States": 5000, "Japan": 5000, "Uruguay": 5500, "Mexico": 5500,
    "Switzerland": 6500, "Croatia": 7500,
}


def _our_teams() -> set[str]:
    return {t for teams in GROUPS.values() for t in teams}


def _download(url: str) -> str | None:
    """Download text, tolerating macOS SSL-cert quirks and falling back to curl."""
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except Exception:
        pass
    try:  # macOS Python often lacks a cert bundle -> unverified context
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(url, timeout=30, context=ctx) as resp:
            return resp.read().decode("utf-8")
    except Exception:
        pass
    try:  # last resort: system curl
        out = subprocess.run(["curl", "-sL", url], capture_output=True,
                             timeout=60, check=True)
        return out.stdout.decode("utf-8")
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[warn] could not download {url}: {exc}")
        return None


def fetch_results() -> pd.DataFrame | None:
    raw = _download(RESULTS_URL)
    if raw is None:
        return None

    df = pd.read_csv(io.StringIO(raw))
    df["home_team"] = df["home_team"].replace(NAME_FIX)
    df["away_team"] = df["away_team"].replace(NAME_FIX)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= RECENT_SINCE].copy()

    out = pd.DataFrame({
        "date": df["date"].dt.date.astype(str),
        "home_team": df["home_team"],
        "away_team": df["away_team"],
        "neutral": df["neutral"].astype(int),
        "home_score": df["home_score"].astype("Int64"),
        "away_score": df["away_score"].astype("Int64"),
        "competition": df["tournament"],
        "synthetic": 0,
    }).dropna(subset=["home_score", "away_score"])

    # Keep matches involving at least one of our 48 (rich opponent context).
    ours = _our_teams()
    mask = out["home_team"].isin(ours) | out["away_team"].isin(ours)
    out = out[mask].reset_index(drop=True)
    return out


def build_odds() -> pd.DataFrame:
    rows = [{
        "market": "tournament_winner", "selection": t, "odds": o,
        "format": "american", "source": ODDS_SOURCE, "timestamp": ODDS_DATE,
    } for t, o in WINNER_ODDS_SNAPSHOT.items()]
    return pd.DataFrame(rows)


def main() -> None:
    results = fetch_results()
    if results is not None and len(results) > 200:
        results.to_csv(RAW / "match_history.csv", index=False)
        n_recent = len(results)
        span = f"{results['date'].min()} … {results['date'].max()}"
        participants = (set(results["home_team"]) | set(results["away_team"])) \
            & _our_teams()
        print(f"match_history.csv : {n_recent} REAL results ({span}); "
              f"{len(participants)}/48 participants have data")
    else:
        print("[warn] keeping existing match_history.csv (synthetic fallback)")

    odds = build_odds()
    odds.to_csv(RAW / "odds_real.csv", index=False)
    print(f"odds_real.csv     : {len(odds)} real winner prices "
          f"({ODDS_SOURCE}, {ODDS_DATE})")


if __name__ == "__main__":
    main()
