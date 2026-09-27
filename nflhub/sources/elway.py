"""ELWAY (Nate Silver's Silver Bulletin NFL forecasting model), transcribed weekly into a
personal Google Sheet.

ELWAY is a real, professionally maintained model -- team ratings plus its QBERT
quarterback rating, built on every NFL game since 1920, refined for 2026 (see
natesilver.net/i/176207317/2026-changes-to-elway-and-qbert). This module just reads the
transcription: a public read-only sheet (id configured via ELWAY_SHEET_ID), one row per
game: home/away team, each side's average points, and each side's win probability. The
sheet's own spread/total columns are intentionally ignored — nfl-hub derives its own
spread (home avg pts vs away avg pts) and total (their sum) instead, so they can be
compared against the real sportsbook lines rather than restating ELWAY's own line.

The sheet started as one flat tab with a "Wk" column (filtered client-side). As of
2026-09-24 the author switched to one tab per week ("Week 2", "Week 3", ...) — the same
convention edge_sheet.py already handles for the pick'em pool sheet. Google's gviz
endpoint has no way to ask "give me whatever tab is currently first" if a new tab gets
added without becoming first in position, and silently returns the wrong week's data
instead of erroring, so this checks for a "Week {week}" tab by name (via the same
htmlview tab-list scrape edge_sheet.py uses) and prefers it; if no such tab exists yet it
falls back to the original bare fetch + Wk-column filter, so it still works against the
old flat layout or against a week that hasn't gotten its own tab yet.

Also reads a second, unrelated tab: "Current Rankings" (one row per team, not per game),
whose last column names the QB each team is currently being *evaluated* with at QB1 —
see fetch_qb1(). That column already reflects the sheet's own QBERT depth-chart/health
tracking (an injured starter's healthy backup gets promoted there automatically), so the
only gap left to catch is a breaking-news injury since the user's last weekly
transcription — compared client-side against nfl-hub's own live-polled ESPN injury feed.
Fetched by gid (looked up by exact tab name via the htmlview scrape) rather than gviz's
`sheet=<name>` param, which was seen to silently resolve to the wrong tab when the real
name has incidental whitespace ("Current QBERT" vs. the sheet's actual " Current QBERT").

Soft-fails like every other scrape in this project: any problem raises ElwayUnavailable
and refresh.py treats it as a non-fatal skip.
"""

from __future__ import annotations

import csv
import io
import re
from typing import Any, Optional
from urllib.parse import quote

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


def _csv_url(sheet_id: str, tab: Optional[str] = None) -> str:
    base = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"
    return f"{base}&sheet={quote(tab)}" if tab else base


def _tab_gid(sheet_id: str, name: str) -> Optional[str]:
    """gid of the tab named exactly `name`, or None if it can't be confirmed. Fetching by
    gid (rather than gviz's `sheet=<name>` param) is what makes a lookup immune to the
    name-resolution issue described in the module docstring.
    """
    try:
        resp = requests.get(f"https://docs.google.com/spreadsheets/d/{sheet_id}/htmlview", timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException:
        return None
    m = re.search(r'\{name:\s*"' + re.escape(name) + r'",\s*pageUrl:\s*"[^"]*",\s*gid:\s*"(\d+)"', resp.text)
    return m.group(1) if m else None


def _week_tab_exists(sheet_id: str, week: int) -> bool:
    return _tab_gid(sheet_id, f"Week {week}") is not None


def fetch_week(sheet_id: str, week: int, games: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """game_id -> {home_win_prob, away_win_prob, spread_home, total}, matched against this
    week's nfl-hub games. `spread_home` follows the same sign convention used everywhere
    else in the app (negative = home favored): away_avg_pts - home_avg_pts.
    """
    if not sheet_id:
        raise ElwayUnavailable("no sheet id configured")
    tab = f"Week {week}" if _week_tab_exists(sheet_id, week) else None
    try:
        resp = requests.get(_csv_url(sheet_id, tab), timeout=TIMEOUT)
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


def fetch_qb1(sheet_id: str) -> dict[str, dict[str, Any]]:
    """team_abbr -> {"name": <last name>, "qbert": <rating>} for the QB ELWAY is currently
    evaluating that team with at QB1, from the "Current Rankings" tab's last column
    (formatted "Lastname (rating)" per row). See module docstring for why this is fetched
    by gid rather than by tab name.
    """
    if not sheet_id:
        raise ElwayUnavailable("no sheet id configured")
    gid = _tab_gid(sheet_id, "Current Rankings")
    if gid is None:
        raise ElwayUnavailable('no "Current Rankings" tab found')
    try:
        resp = requests.get(
            f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&gid={gid}",
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ElwayUnavailable(f"Current Rankings fetch failed: {exc}") from exc

    rows = list(csv.reader(io.StringIO(resp.text)))
    if len(rows) < 2:
        raise ElwayUnavailable("Current Rankings: empty response")

    out: dict[str, dict[str, Any]] = {}
    for row in rows[1:]:  # row 0 is the (multi-line) header
        if len(row) < 8:
            continue
        m = re.match(r"^(.+?)\s*\(([\d.]+)\)\s*$", row[7].strip())
        if not m:
            continue
        out[_team_abbr(row[0])] = {"name": m.group(1).strip(), "qbert": float(m.group(2))}
    if not out:
        raise ElwayUnavailable("Current Rankings: parsed 0 rows — layout may have changed")
    return out
