"""Best-effort national pick % scrape from nflpickwatch.com.

Ported from github.com/squid004/pickem-edge (src/pke/picks/nflpickwatch.py), based on the
`thmsdrew/pypicks` scraper as prior art. SPEC.md section 3.2 calls this source "unreliable
by design" — nflpickwatch is known to break scrapers, so every failure here raises
NflPickwatchUnavailable and the caller (refresh.py) treats it as a soft failure: log it,
leave whatever manual/previously-scraped rows already exist untouched, move on. A manual
override always exists (js/edge.js) as the always-available fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import requests

_URL = "http://nflpickwatch.com/"
_TAG_RE = re.compile(r"<[^>]+>")
# "ABC@XYZ ... XYZ ... 68%" — two team codes joined by @, then a consensus team code,
# then a percentage, within one line/row of the (HTML-tag-stripped) page text.
_MATCHUP_RE = re.compile(
    r"([A-Z]{2,3})\s*@\s*([A-Z]{2,3}).{0,40}?\b([A-Z]{2,3})\b.{0,20}?(\d{1,3})\s*%",
    re.DOTALL,
)


class NflPickwatchUnavailable(RuntimeError):
    """Raised on any request/parse failure. Treat as a soft failure, never let it
    propagate out of a refresh step."""


@dataclass(frozen=True)
class NationalPickRow:
    team: str
    pct: float


def fetch_week(normalize_team) -> list[NationalPickRow]:
    """`normalize_team(code) -> canonical abbr or raises KeyError/ValueError` is injected
    so this module doesn't need its own team table (see edge_teams.normalize)."""
    try:
        resp = requests.get(_URL, params={"text": "1"}, timeout=30)
        resp.raise_for_status()
        html = resp.text
    except requests.RequestException as exc:
        raise NflPickwatchUnavailable(f"nflpickwatch request failed: {exc}") from exc

    rows = _parse_consensus_lines(html, normalize_team)
    if not rows:
        raise NflPickwatchUnavailable(
            "no consensus picks parsed from nflpickwatch response — page structure may "
            "have changed; use the manual national-% entry instead"
        )
    return rows


def _parse_consensus_lines(html: str, normalize_team) -> list[NationalPickRow]:
    text = _TAG_RE.sub(" ", html)
    rows: list[NationalPickRow] = []
    for team_a, team_b, consensus_team, pct_str in _MATCHUP_RE.findall(text):
        try:
            consensus = normalize_team(consensus_team)
            other = normalize_team(team_b if consensus == normalize_team(team_a) else team_a)
        except (KeyError, ValueError):
            continue
        pct = float(pct_str) / 100.0
        rows.append(NationalPickRow(team=consensus, pct=pct))
        rows.append(NationalPickRow(team=other, pct=round(1.0 - pct, 4)))
    return rows
