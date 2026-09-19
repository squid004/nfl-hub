"""Home/away average-points model, pulled from a personal Google Sheet.

Public read-only sheet (id configured via ELWAY_SHEET_ID), one row per game for the
current week: home/away team, each side's average points, and each side's win
probability. The sheet's own spread/total columns are intentionally ignored — nfl-hub
derives its own spread (home avg pts vs away avg pts) and total (their sum) instead, so
they can be compared against the real sportsbook lines rather than the sheet author's own
line, per the "ELWAY" feature's purpose.

Soft-fails like every other scrape in this project: any problem raises ElwayUnavailable
and refresh.py treats it as a non-fatal skip.
"""

from __future__ import annotations

import csv
import io
from typing import Any, Optional

import requests

TIMEOUT = 20

# Sheet author's team codes that don't match ESPN's (nfl-hub's game.home/away come from
# ESPN — same kind of mismatch already handled for actionnetwork.py / pickem-edge).
_TEAM_ALIAS = {"WAS": "WSH", "JAC": "JAX", "LA": "LAR"}


class ElwayUnavailable(RuntimeError):
    """Fetch or parse failed — caller should treat this as "no data this run"."""


def _team_abbr(raw: str) -> str:
    code = (raw or "").strip().upper()
    return _TEAM_ALIAS.get(code, code)


def _pct(cell: str) -> Optional[float]:
    try:
        return float(cell.strip().rstrip("%")) / 100.0
    except (ValueError, AttributeError):
        return None


def _num(cell: str) -> Optional[float]:
    try:
        return float(cell.strip())
    except (ValueError, AttributeError):
        return None


def _csv_url(sheet_id: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"


def fetch_week(sheet_id: str, week: int, games: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """game_id -> {home_win_prob, away_win_prob, spread_home, total}, matched against this
    week's nfl-hub games. `spread_home` follows the same sign convention used everywhere
    else in the app (negative = home favored): away_avg_pts - home_avg_pts.
    """
    if not sheet_id:
        raise ElwayUnavailable("no sheet id configured")
    try:
        resp = requests.get(_csv_url(sheet_id), timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ElwayUnavailable(f"sheet request failed: {exc}") from exc

    rows = list(csv.reader(io.StringIO(resp.text)))
    if len(rows) < 2:
        raise ElwayUnavailable("empty response")

    games_by_teams = {(g["home"], g["away"]): g for g in games}
    out: dict[str, dict[str, Any]] = {}
    for row in rows[1:]:  # row 0 is the (multi-line) header
        if len(row) < 7:
            continue
        wk = _num(row[0])
        if wk is None or int(wk) != week:
            continue
        home, away = _team_abbr(row[1]), _team_abbr(row[4])
        home_avg, away_avg = _num(row[2]), _num(row[5])
        if home_avg is None or away_avg is None:
            continue
        g = games_by_teams.get((home, away))
        if not g:
            continue
        out[g["game_id"]] = {
            "game_id": g["game_id"],
            "week": g["week"],
            "home_win_prob": _pct(row[3]),
            "away_win_prob": _pct(row[6]),
            "spread_home": round(away_avg - home_avg, 2),
            "total": round(home_avg + away_avg, 2),
        }
    if not out:
        raise ElwayUnavailable(f"parsed 0 rows for week {week} — sheet layout may have changed")
    return out
