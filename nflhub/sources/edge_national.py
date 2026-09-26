"""National pick percentages from Yahoo's public Pick'em pick-distribution page.

Replaces nflpickwatch.com, which has been reliably 403-blocked from GitHub Actions IPs
since week 1 this season with no sign of resolving. This endpoint needs no login, ignores
its own `gid`/`type` query params (verified by comparing output across different `gid`
values -- identical either way, so it's genuine Yahoo-wide data, not one group's picks),
and supports arbitrary past weeks via `week=N` matching nfl-hub's own week numbering
exactly. Found by the user via their own Yahoo pick'em pool's page.

Soft-fails like every other scrape in this project: any problem raises
NationalPctUnavailable and refresh.py treats it as a non-fatal skip.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import requests

_URL = "https://football.fantasysports.yahoo.com/pickem/pickdistribution"
TIMEOUT = 20

# Yahoo's team-name strings (verified against 3 weeks / 48 games of real 2026 data --
# covers all 32 teams). The two ambiguous cities disambiguate themselves with a
# parenthetical abbreviation Yahoo already includes ("Los Angeles (LAC)").
_NAME_TO_ABBR = {
    "Arizona": "ARI", "Atlanta": "ATL", "Baltimore": "BAL", "Buffalo": "BUF",
    "Carolina": "CAR", "Chicago": "CHI", "Cincinnati": "CIN", "Cleveland": "CLE",
    "Dallas": "DAL", "Denver": "DEN", "Detroit": "DET", "Green Bay": "GB",
    "Houston": "HOU", "Indianapolis": "IND", "Jacksonville": "JAX", "Kansas City": "KC",
    "Las Vegas": "LV", "Los Angeles (LAC)": "LAC", "Los Angeles (LAR)": "LAR",
    "Miami": "MIA", "Minnesota": "MIN", "New England": "NE", "New Orleans": "NO",
    "New York (NYG)": "NYG", "New York (NYJ)": "NYJ", "Philadelphia": "PHI",
    "Pittsburgh": "PIT", "San Francisco": "SF", "Seattle": "SEA", "Tampa Bay": "TB",
    "Tennessee": "TEN", "Washington": "WSH",
}

_GAME_RE = re.compile(
    r'<h4><span>[^<]+</span></h4>.*?'
    r'class="team">@?\s*<a[^>]*>([^<]+)</a>.*?dd class="percent">(\d+)%</dd>\s*</dl>\s*'
    r'<dl class="underdog[^"]*">.*?'
    r'class="team">@?\s*<a[^>]*>([^<]+)</a>.*?dd class="percent">(\d+)%</dd>',
    re.S,
)


class NationalPctUnavailable(RuntimeError):
    """Raised on any request/parse failure. Treat as a soft failure, never let it
    propagate out of a refresh step."""


@dataclass(frozen=True)
class NationalPickRow:
    team: str
    pct: float


def fetch_week(week: int, normalize_team) -> list[NationalPickRow]:
    """`normalize_team(code) -> canonical abbr, raising ValueError on an unknown code` is
    injected so this module doesn't need its own team table beyond the Yahoo-specific
    full-name mapping above."""
    try:
        resp = requests.get(_URL, params={"gid": "", "type": "", "week": week}, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise NationalPctUnavailable(f"yahoo pickdistribution request failed: {exc}") from exc

    rows: list[NationalPickRow] = []
    for fav_name, fav_pct, dog_name, dog_pct in _GAME_RE.findall(resp.text):
        fav_abbr: Optional[str] = _NAME_TO_ABBR.get(fav_name)
        dog_abbr: Optional[str] = _NAME_TO_ABBR.get(dog_name)
        if not fav_abbr or not dog_abbr:
            continue
        try:
            fav_abbr = normalize_team(fav_abbr)
            dog_abbr = normalize_team(dog_abbr)
        except ValueError:
            continue
        rows.append(NationalPickRow(team=fav_abbr, pct=round(float(fav_pct) / 100.0, 4)))
        rows.append(NationalPickRow(team=dog_abbr, pct=round(float(dog_pct) / 100.0, 4)))

    if not rows:
        raise NationalPctUnavailable(f"no games parsed for week {week} — page structure may have changed")
    return rows
