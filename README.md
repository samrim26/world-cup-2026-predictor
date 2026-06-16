# World Cup 2026 — Quantamental Prediction & Value Engine

A serious, simulation-based model for the 2026 FIFA World Cup (48 teams, 12
groups, 104 matches, Round-of-32 knockouts, final 19 July at MetLife). It rates
every team, predicts every match, simulates the whole tournament hundreds of
thousands of times, and prices the result against the market — with an
explainable, layered design rather than a black box.

> **Philosophy.** Nothing here pretends to certainty. Every output is
> probabilistic, every weight is in `config.yaml`, every prediction carries
> data-quality and confidence, and missing data is flagged — never fabricated.

---

## What it produces

Run one command and you get, in `outputs/`:

| File | Contents |
|------|----------|
| `predictions.csv` | Every group + R32 match: predicted score, xG, 1X2 probs, top-5 scorelines, over/under, BTTS, confidence, data quality, **key model drivers** |
| `match_predictions.md` | Human-readable **all-72 group matches** (+ R32) grouped by group/matchday with scoreline, xG, W/D/L |
| `team_probabilities.csv` | Per team: ratings, projected points/GD, P(R32 → R16 → QF → SF → Final → Champion), fair winner odds |
| `power_index.csv` | **Multi-lens ranking** — each team's rank under model / sim / market / Elo / form, a blended consensus rank, and a disagreement score |
| `group_tables.csv` | Projected group standings & qualification odds |
| `bracket_projection.csv` | Most-likely group winners & runners-up |
| `value_bets.csv` | De-vigged market vs model edge, EV, fractional-Kelly stake, STRONG/MODERATE/THIN/NO-BET flags |
| `power_rankings.csv` | Composite strength + attack/defense/GK decomposition |
| `insight_*.csv` | Upset Radar, Dark-Horse Index, Overrated/Underrated, Path Difficulty, Bracket Leverage, Consensus-vs-Model |
| `report.md` | An investor-style brief tying it all together |

---

## The model, layer by layer

**Layer 1 — Composite team strength** (`model/team_strength.py`, `model/ratings.py`).
A config-weighted blend (defaults below, all in `config.yaml`):

```
0.30 Elo + 0.15 recent form + 0.15 squad value + 0.10 qualification
+ 0.10 attack/defense balance + 0.07 manager stability + 0.05 confederation
+ 0.04 tournament experience + 0.04 GK/penalty
```

Every component is standardised so weights are comparable, and each team's
output retains the **signed contribution of every component** for explainability.

**Layer 2 — Attack/defense decomposition.** From the composite plus style
sub-ratings we derive attack, defense, midfield, GK, set-piece, transition,
possession, pressing, depth, penalty, manager and discipline ratings.

**Layer 3 — Match-specific adjustments** (`features/`). Per fixture: home &
regional advantage (hosts USA/MEX/CAN + CONCACAF/CONMEBOL familiarity),
altitude & climate, rest days, travel distance + timezone shift (great-circle
on venue coordinates), tactical style mismatch, and matchday-3 group incentives.

**Layer 4 — Expected-goals ensemble** (`model/expected_goals.py`,
`ensemble.py`). Three estimators are blended into per-team attack/defense
ratings:
- **Dixon-Coles** mapping from composite strength (always on);
- **Bayesian** — a penalised Poisson MLE with empirical-Bayes shrinkage toward
  the Elo prior (always on); upgrades to a full **PyMC** hierarchical posterior
  with `--pymc`;
- **ML** — a gradient-boosted xG model (`scikit-learn` HistGBR by default,
  **XGBoost** when installed).

The blended rates feed a Dixon-Coles correct-score matrix (low-score correlation
correction, draw inflation) → 1X2, correct scores, over/under, BTTS.
`model/calibration.py` provides isotonic calibration + confidence scoring.

**Layer 5 — Tournament simulation** (`model/simulator.py`, `bracket.py`).
Fully-vectorised Monte Carlo (default 100k, up to 1M), reproducible by seed.
Each sim draws a per-team form shock, plays the groups with the **real FIFA
tiebreakers** (points → GD → GF → tiebreak), selects the **8 best third-placed
teams**, slots them into the Round of 32 via a memoised **combination table**
that guarantees no same-group R32 rematch, and runs knockouts with reduced
extra-time tempo and skill-weighted penalty shootouts. **A knockout can never
end in a draw.**

**Layer 6 — Value / ROI** (`model/value_betting.py`, `utils/odds_math.py`).
Converts American/decimal/fractional odds, removes vig (proportional or Shin),
computes edge / EV / fractional-Kelly, applies an **uncertainty penalty** so
thin or low-confidence edges aren't over-bet, and flags each opportunity.

**Power Index — multi-lens ranking** (`model/power_index.py`). Ranks every team
under five *independent* methods — composite model, Monte-Carlo title odds,
vig-removed market, raw Elo, recent form — and blends them into a consensus rank
with a **disagreement score**. This is where you see *where the methods part
ways* (e.g. a team the model rates #1 but the market rates #6), and you can drop
in your own method by adding `data/external/external_ranking.csv` (a `team`
column + a `score` or `rank` column) to blend an outside opinion into the board.

---

## Quickstart

> On macOS use `python3`/`pip3` (there is usually no bare `python`/`pip`).

```bash
cd world-cup-predictor
pip3 install -e .                     # core engine (numpy/pandas/scipy/sklearn)
# optional heavy layers:
#   pip3 install -e ".[ml]"     # XGBoost / LightGBM
#   pip3 install -e ".[bayes]"  # PyMC full Bayesian posterior
#   pip3 install -e ".[app]"    # Streamlit dashboard

python3 data/build_seed_data.py        # (re)build committed seed CSVs
python3 data/fetch_real_data.py        # pull REAL recent results + winner odds

python3 -m wcp ingest                  # validate data + show quality
python3 -m wcp rate                    # power rankings
python3 -m wcp predict --stage group   # all 72 group match predictions
python3 -m wcp simulate --sims 100000  # full sim -> CSV outputs
python3 -m wcp all --sims 200000       # everything + outputs/report.md
```

`fetch_real_data.py` downloads the public **martj42 international-results**
dataset (real matches, recency- + competition-weighted) and writes a dated
**real market odds** snapshot. With it, data-quality reads **High** and the value
board produces actionable edges; without it the engine falls back to the
synthetic/sample seeds and still runs offline.

After `pip3 install -e .` a `wcp` command is also on your PATH, so you can drop
the `python3 -m` prefix (e.g. `wcp all --sims 200000`).

Scenario analysis:

```bash
python3 -m wcp all --injure Argentina --boost "United States"
python3 -m wcp all --scenario "Brazil:-0.5:-0.3"   # custom attack:defense z deltas
```

Dashboard (needs the `[app]` extra installed):

```bash
pip3 install -e ".[app]"
streamlit run src/wcp/app/streamlit_app.py
```

Tests:

```bash
python3 -m pytest -q                   # 44 tests: odds, scores, group rules,
                                       # third-place, bracket mapping, simulator, value
```

---

## Data

The **real Dec-2025 draw** (all 48 teams, groups A–L), the 104-match schedule,
16 host venues (coordinates + altitude + climate), and **real Elo/FIFA/squad-
value headline ratings** are committed under `data/raw/` (built reproducibly by
`data/build_seed_data.py`). Style sub-ratings are *derived* from those headline
values and clearly labelled as model inputs.

**Real recent results & odds** come from `data/fetch_real_data.py`: ~2,000 real
internationals since 2022 (martj42 dataset, recency- and competition-weighted in
the Bayesian/ML fit) and a dated real tournament-winner odds snapshot. If you
skip that step, the engine falls back to an Elo-anchored **synthetic** results
seed and **sample** odds so it still runs fully offline.

Everything is swappable: each dataset has a loader in `ingest/` implementing
`BaseLoader.load()`. Point a subclass at a live API/scraper (real results, an
odds feed, squad/injury data) and the rest of the engine is unchanged — the
affected predictions automatically gain data-quality.

### Data quality
Each team and match carries a **High / Medium / Low** data-quality tag from
source availability (recent results, ratings, venue, squad, odds, injuries).
Out of the box most teams are **Medium** (rating- and schedule-complete, sample
odds, no squad/injury feed). Wire live feeds to reach **High**.

---

## Layout

```
src/wcp/
  ingest/    fixtures rankings odds squads match_history  (+ base, dataset)
  model/     ratings team_strength expected_goals score_model
             bayesian ml_xg ensemble simulator bracket standings
             value_betting calibration predictor
  features/  venues travel matchup context incentives
  reports/   generate_report insights
  app/       cli  streamlit_app
  utils/     config logging io validation geo odds_math
data/        raw/ processed/ external/  + build_seed_data.py
tests/       7 suites, 44 tests
config.yaml  every weight, sim count, seed, threshold
```

---

## Limitations (read these)

- **Odds are a dated snapshot, not a live feed.** `fetch_real_data.py` embeds a
  real market snapshot; refresh it (or wire a live odds API in `ingest/odds.py`)
  before betting on the value flags. Squad/injury data is still not ingested, so
  personnel adjustments are scenario-driven, not automatic.
- **Bracket slotting** is a *valid, balanced* 2026-format bracket, not a
  byte-for-byte copy of FIFA's published slot letters; swap `R32_MATCHES` to an
  official sheet if needed (downstream is agnostic).
- **MD3 incentives** and in-game game-theory are coarse, transparent
  approximations.
- Monte-Carlo noise remains; raise `--sims` to tighten bands. Confidence is
  computed from entropy + data quality, **never hardcoded**.
