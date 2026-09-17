"""Pull opponent picks (and the user's own season standing) from a Google Sheet instead of
(or alongside) the paste box on the page. One tab per week, named "Week N", no header row:

    "{rank ordinal}{name}", weekly_correct, season_correct, pick1, pick2, ..., pickN, tiebreaker

Confirmed layout (2026-09-17, live sheet): column 1 = this week's correct picks, column 2 =
season-to-date correct picks, rank = the ordinal prefix on the name itself ("1st", "25th").
Pick cells and the trailing tiebreaker guess are un-labeled; team cells are told apart from
metadata by whether they normalize to a real team code, not by column position — keeps this
working even if the sheet's metadata-column count changes.

Google's gviz endpoint silently falls back to the sheet's FIRST tab when you ask for a
`sheet=` name that doesn't exist yet (no error), which would otherwise mislabel last week's
picks as this week's. _tab_exists() checks the real tab list (scraped from the public
htmlview page's embedded sheet-switcher JS) before trusting a fetch.

Soft-fails throughout, like the nflpickwatch scraper: any problem raises
EdgeSheetUnavailable and the caller (refresh.py) treats it as a non-fatal skip.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from urllib.parse import quote

import requests

_ORDINAL_RE = re.compile(r"^(\d+)(st|nd|rd|th)\s*", re.IGNORECASE)
_NO_PICK = {"", "-", "--", "n/a", "na", "bye"}
TIMEOUT = 20


class EdgeSheetUnavailable(RuntimeError):
    """Raised on any fetch/parse failure. Soft failure only — never let it propagate
    out of a refresh step."""


@dataclass(frozen=True)
class SheetPick:
    opponent: str
    team_picked: str


@dataclass(frozen=True)
class SheetStanding:
    opponent: str
    rank: int | None
    correct_week: int | None
    correct_season: int | None


@dataclass(frozen=True)
class SheetWeekData:
    picks: list[SheetPick]
    standings: list[SheetStanding]
    pool_size: int


def _csv_url(sheet_id: str, week: int) -> str:
    tab = quote(f"Week {week}")
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab}"


def _htmlview_url(sheet_id: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/htmlview"


def _tab_exists(sheet_id: str, week: int) -> bool:
    try:
        resp = requests.get(_htmlview_url(sheet_id), timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise EdgeSheetUnavailable(f"could not list sheet tabs: {exc}") from exc
    # Tab switcher bootstrap: items.push({name: "Week 1", pageUrl: "..."});
    pattern = re.compile(r'\{name:\s*"Week ' + str(week) + r'",\s*pageUrl:')
    return pattern.search(resp.text) is not None


def _parse_int(cell: str) -> int | None:
    try:
        return int(cell.strip())
    except (ValueError, AttributeError):
        return None


def fetch_week(sheet_id: str, week: int, normalize_team) -> SheetWeekData:
    """`normalize_team(code) -> canonical abbr, raising ValueError on an unknown code` is
    injected so this module doesn't need its own team table (see edge_teams.normalize_team).
    """
    if not sheet_id:
        raise EdgeSheetUnavailable("no sheet id configured")
    if not _tab_exists(sheet_id, week):
        raise EdgeSheetUnavailable(f"no 'Week {week}' tab in the sheet yet")

    try:
        resp = requests.get(_csv_url(sheet_id, week), timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise EdgeSheetUnavailable(f"sheet request failed: {exc}") from exc

    rows = list(csv.reader(io.StringIO(resp.text)))
    if not rows:
        raise EdgeSheetUnavailable(f"empty response for Week {week} tab")

    picks: list[SheetPick] = []
    standings: list[SheetStanding] = []
    for row in rows:
        if len(row) < 2:
            continue
        m = _ORDINAL_RE.match(row[0])
        rank = int(m.group(1)) if m else None
        opponent = _ORDINAL_RE.sub("", row[0]).strip()
        if not opponent:
            continue

        standings.append(SheetStanding(
            opponent=opponent, rank=rank,
            correct_week=_parse_int(row[1]) if len(row) > 1 else None,
            correct_season=_parse_int(row[2]) if len(row) > 2 else None,
        ))
        for cell in row[1:]:
            code = (cell or "").strip()
            if code.lower() in _NO_PICK:
                continue
            try:
                team = normalize_team(code)
            except ValueError:
                continue
            picks.append(SheetPick(opponent=opponent, team_picked=team))

    if not picks:
        raise EdgeSheetUnavailable(f"parsed 0 picks from Week {week} tab — sheet layout may have changed")
    return SheetWeekData(picks=picks, standings=standings, pool_size=len(standings))
