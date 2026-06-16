"""Predicted-vs-actual: score the user's predictions against live results.

Scoring (matches the user's pool rules): +3 correct outcome (W/D/L),
+1 correct home goals, +1 correct away goals (max 5 per match).
"""
from __future__ import annotations

from datetime import date, timedelta

from . import user_picks as up

GROUP_START = date(2026, 6, 11)
_RESULTS_CACHE: dict[str, list[dict]] = {}     # date -> finished matches (final)


def _outcome(hs: int, as_: int) -> str:
    return "H" if hs > as_ else ("A" if as_ > hs else "D")


def finished_results(feed, through: date | None = None) -> list[dict]:
    """All finished group-stage matches from June 11 through ``through`` (today).

    Past dates are cached (final); only the current day is re-fetched live.
    """
    through = through or date.today()
    end = min(through, date(2026, 6, 27))
    out: list[dict] = []
    d = GROUP_START
    while d <= end:
        key = d.strftime("%Y%m%d")
        is_today = (d == date.today())
        if key in _RESULTS_CACHE and not is_today:
            day = _RESULTS_CACHE[key]
        else:
            try:
                day = [m for m in feed.day(key) if m["state"] == "post"]
            except Exception:
                day = []
            if not is_today:            # only cache finalised days
                _RESULTS_CACHE[key] = day
        out.extend(day)
        d += timedelta(days=1)
    return out


def score_predictions(feed, through: date | None = None) -> dict:
    """Return per-match predicted-vs-actual rows + running points total."""
    rows = []
    total = exact = correct_outcome = 0
    for m in finished_results(feed, through):
        try:
            hs, as_ = int(m["home_score"]), int(m["away_score"])
        except (ValueError, TypeError):
            continue
        pred = up.predicted_score_by_abbrs(m["home"], m["away"])
        if pred is None:
            continue
        phome_abbr, phs, pas = pred
        # Align prediction to the actual home/away orientation.
        if phome_abbr != m["home"]:
            phs, pas = pas, phs
        pts = 0
        if _outcome(phs, pas) == _outcome(hs, as_):
            pts += 3; correct_outcome += 1
        if phs == hs:
            pts += 1
        if pas == as_:
            pts += 1
        if phs == hs and pas == as_:
            exact += 1
        total += pts
        rows.append({
            "date": m["date"], "match": f"{m['home']} {hs}-{as_} {m['away']}",
            "predicted": f"{phs}-{pas}", "actual": f"{hs}-{as_}", "points": pts,
        })
    rows.sort(key=lambda r: r["date"], reverse=True)
    return {"rows": rows, "total": total, "n": len(rows),
            "exact": exact, "correct_outcome": correct_outcome,
            "max_possible": len(rows) * 5}


def predicted_group_order() -> dict[str, list[str]]:
    """Compute the user's predicted finishing order (abbrs) for each group."""
    out = {}
    for g, matches in up.GROUP_PREDICTIONS.items():
        teams = {}
        for h, a, hs, as_ in matches:
            for t in (h, a):
                teams.setdefault(t, {"P": 0, "GF": 0, "GA": 0, "Pts": 0})
            teams[h]["GF"] += hs; teams[h]["GA"] += as_
            teams[a]["GF"] += as_; teams[a]["GA"] += hs
            if hs > as_:
                teams[h]["Pts"] += 3
            elif as_ > hs:
                teams[a]["Pts"] += 3
            else:
                teams[h]["Pts"] += 1; teams[a]["Pts"] += 1
        order = sorted(teams, key=lambda t: (teams[t]["Pts"],
                       teams[t]["GF"] - teams[t]["GA"], teams[t]["GF"]),
                       reverse=True)
        out[g] = [up.NAME2ABBR.get(t, t) for t in order]
    return out


def qualifier_accuracy(groups: dict[str, list[dict]]) -> list[dict]:
    """Compare the user's predicted top-2 per group with the live actual top-2."""
    pred = predicted_group_order()
    out = []
    for g, rows in groups.items():
        actual_top2 = [r["abbr"] for r in rows[:2]]
        pred_top2 = pred.get(g, [])[:2]
        hits = len(set(actual_top2) & set(pred_top2))
        out.append({"group": g, "predicted": pred_top2, "actual": actual_top2,
                    "match": f"{hits}/2",
                    "exact_order": pred_top2 == actual_top2})
    return out
