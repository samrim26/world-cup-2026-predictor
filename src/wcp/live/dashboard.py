"""Terminal dashboard: live standings, third-place wildcard race, and
predicted-vs-actual scoring, with desktop notifications when the wildcard cut
line changes."""
from __future__ import annotations

import subprocess
import time
from datetime import datetime

from . import compare
from .bracket_tracker import bracket_status, goal_feed
from .feed import BaseFeed, ESPNFeed
from .tracker import (qualification_flags, third_place_race,
                      tournament_complete_groups)

CLEAR = "\033[2J\033[H"
BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"
GREEN = "\033[32m"; YELLOW = "\033[33m"; RED = "\033[31m"; CYAN = "\033[36m"


def notify(title: str, message: str) -> None:
    """Fire a macOS desktop notification (no-op off macOS)."""
    msg = message.replace('"', "'"); ttl = title.replace('"', "'")
    try:
        subprocess.run(["osascript", "-e",
                        f'display notification "{msg}" with title "{ttl}" sound name "Glass"'],
                       check=False, capture_output=True, timeout=5)
    except Exception:
        pass


def snapshot(feed: BaseFeed) -> dict:
    groups = feed.standings()
    today = feed.today()
    return {
        "groups": groups,
        "today": today,
        "flagged": qualification_flags(groups),
        "thirds": third_place_race(groups),
        "done": tournament_complete_groups(groups),
        "score": compare.score_predictions(feed),
        "qual": compare.qualifier_accuracy(groups),
        "goals": goal_feed(today),
        "bracket": bracket_status(feed, groups),
    }


def _live_line(m: dict) -> str:
    sc = f"{m['home_score']}-{m['away_score']}"
    if m["state"] == "in":
        tag = f"{RED}{BOLD}LIVE {m.get('clock','')}{RESET}"
    elif m["state"] == "post":
        tag = f"{DIM}FT{RESET}"
    else:
        tag = f"{DIM}{m.get('detail','') or 'scheduled'}{RESET}"
    return f"  {m['home']:>3} {BOLD}{sc:^5}{RESET} {m['away']:<3}  {tag}"


def render(snap: dict, feed_src: str) -> str:
    flagged, thirds = snap["flagged"], snap["thirds"]
    out = [f"{BOLD}{CYAN}FIFA WORLD CUP 2026 - LIVE{RESET}   "
           f"{DIM}updated {datetime.now():%H:%M:%S} - {feed_src}{RESET}",
           f"{DIM}groups completed: {snap['done']}/12{RESET}", ""]

    live_now = [m for m in snap["today"] if m["state"] == "in"]
    out.append(f"{BOLD}TODAY{RESET}" + (f"  ({GREEN}{len(live_now)} live{RESET})"
                                        if live_now else ""))
    out += [_live_line(m) for m in snap["today"]] or ["  (no matches today)"]
    out.append("")

    # Goal-by-goal feed.
    if snap["goals"]:
        out.append(f"{BOLD}{GREEN}GOALS{RESET}  {DIM}(latest first){RESET}")
        for g in snap["goals"]:
            live = f"{RED}*{RESET}" if g["live"] else " "
            typ = "" if g["type"] == "Goal" else f" {DIM}({g['type']}){RESET}"
            out.append(f" {live} {g['clock']:>7}  {BOLD}{g['team']:>3}{RESET}  "
                       f"{g['scorer']}{typ}  {DIM}[{g['match']}]{RESET}")
        out.append("")

    out.append(f"{BOLD}{YELLOW}THIRD-PLACE WILDCARD RACE{RESET}  "
               f"{DIM}(top 8 advance){RESET}")
    out.append(f"  {'#':>2}  {'Team':<22}{'Grp':>4}{'P':>3}{'Pts':>4}{'GD':>4}{'GF':>4}  ")
    for t in thirds:
        in_ = t["wildcard_status"] == "IN"
        col = GREEN if in_ else RED
        if t["wildcard_rank"] == 9:
            out.append("  " + DIM + "-" * 44 + " cut line" + RESET)
        out.append(f"  {col}{t['wildcard_rank']:>2}  {t['team']:<22}{t['group']:>4}"
                   f"{t['P']:>3}{t['Pts']:>4}{t['GD']:>+4}{t['GF']:>4}  "
                   f"{'IN' if in_ else 'OUT'}{RESET}")
    out.append("")

    # Predicted vs actual.
    s = snap["score"]
    out.append(f"{BOLD}{CYAN}PREDICTED vs ACTUAL{RESET}  {DIM}(your pool score){RESET}")
    out.append(f"  {BOLD}{s['total']} pts{RESET} from {s['n']} matches  "
               f"{DIM}({s['exact']} exact, {s['correct_outcome']} correct results; "
               f"max {s['max_possible']}){RESET}")
    q_exact = sum(1 for q in snap["qual"] if q["exact_order"])
    q_both = sum(1 for q in snap["qual"] if q["match"] == "2/2")
    out.append(f"  group qualifiers: {GREEN}{q_both}/12{RESET} with both top-2 right "
               f"{DIM}({q_exact}/12 in exact order){RESET}")
    for r in s["rows"][:8]:
        c = GREEN if r["points"] >= 3 else (DIM if r["points"] == 0 else RESET)
        out.append(f"  {c}{r['match']:<20} pred {r['predicted']}  "
                   f"-> {r['points']:+d} pts{RESET}")
    out.append("")

    # Bracket survival.
    b = snap["bracket"]
    champ_col = GREEN if b["champion_alive"] else RED
    out.append(f"{BOLD}{CYAN}YOUR BRACKET - survival{RESET}  "
               f"{DIM}(champion pick: {champ_col}{b['champion']} "
               f"{'ALIVE' if b['champion_alive'] else 'OUT'}{RESET}{DIM}){RESET}")
    for tier in b["tiers"]:
        frac = f"{tier['alive']}/{tier['total']}"
        col = GREEN if tier["alive"] == tier["total"] else (
            YELLOW if tier["alive"] else RED)
        teams = " ".join(
            (f"{GREEN}{t['team']}{RESET}" if t["alive"] else f"{RED}{t['team']}̶{RESET}")
            for t in tier["teams"])
        out.append(f"  {col}{tier['label']:<18} {frac:>6}{RESET}  {teams}")
    out.append("")

    out.append(f"{BOLD}GROUP STANDINGS{RESET}  "
               f"{DIM}(green=top 2, yellow=wildcard spot){RESET}")
    letters = list(flagged)
    for i in range(0, len(letters), 2):
        blocks = []
        for g in letters[i:i + 2]:
            lines = [f"{BOLD}Group {g}{RESET}  {DIM}P W-D-L  GD Pts{RESET}"]
            for t in flagged[g]:
                c = (GREEN if t["live_status"].startswith("advancing")
                     else YELLOW if t["live_status"] == "wildcard IN" else RESET)
                lines.append(f"{c}{t['abbr']:<4}{t['P']} {t['W']}-{t['D']}-{t['L']} "
                             f"{t['GD']:>+3}{t['Pts']:>4}{RESET}")
            blocks.append(lines)
        for r in range(max(len(b) for b in blocks)):
            out.append("  " + "   ".join(f"{(b[r] if r < len(b) else ''):<46}" for b in blocks))
        out.append("")
    return "\n".join(out)


def _check_cutline(prev: set | None, thirds: list[dict]) -> set:
    """Notify on teams crossing the top-8 wildcard cut line."""
    now_in = {t["team"] for t in thirds if t["wildcard_status"] == "IN"}
    if prev is not None and now_in != prev:
        for team in now_in - prev:
            notify("WC wildcard", f"⬆️ {team} moved INTO the top 8")
        for team in prev - now_in:
            notify("WC wildcard", f"⬇️ {team} dropped OUT of the top 8")
    return now_in


def run(refresh: int = 30, once: bool = False, notify_changes: bool = True,
        feed: BaseFeed | None = None) -> None:
    feed = feed or ESPNFeed()
    prev_in = None
    while True:
        try:
            snap = snapshot(feed)
            screen = render(snap, feed.source)
            if notify_changes:
                prev_in = _check_cutline(prev_in, snap["thirds"])
        except Exception as exc:
            screen = f"feed error (will retry): {exc}"
        if once:
            print(screen)
            return
        print(CLEAR + screen)
        print(f"\n{DIM}refreshing every {refresh}s - Ctrl-C to stop{RESET}")
        try:
            time.sleep(refresh)
        except KeyboardInterrupt:
            print("\nstopped.")
            return
