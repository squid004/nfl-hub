"""Per-book spread/moneyline odds, scraped from actionnetwork.com/nfl/odds.

No official API. The page is server-rendered Next.js: the full per-book odds payload
(spread/moneyline/total for DraftKings, FanDuel, BetMGM, etc.) is embedded as JSON in a
`<script id="__NEXT_DATA__">` tag, so a plain GET with a browser User-Agent is enough — no
headless browser, no JS execution, no Cloudflare fight (unlike the unofficial DraftKings
endpoints, which do block CI IPs). This is undocumented internal page data, not a stable
API, so it's treated the same as nflpickwatch/edge_national.py: best-effort, soft-fail,
never raises out of refresh.py.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from typing import Any, Optional

import requests

log = logging.getLogger(__name__)

URL = "https://www.actionnetwork.com/nfl/odds"
TIMEOUT = 15
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)
_NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)

# Action Network's team abbreviations that don't match ESPN's (nfl-hub's game.home/away
# come from ESPN — same JAC/WAS/LA mismatch already handled for pickem-edge's team codes).
_TEAM_ALIAS = {"JAC": "JAX", "WAS": "WSH", "LA": "LAR"}

# Book ids that aren't real shoppable lines — Consensus and the game's Opening line.
_SKIP_BOOK_IDS = {"15", "30"}


class ActionNetworkUnavailable(Exception):
    """Fetch or parse failed — caller should treat this as "no data this run"."""


def _team_abbr(raw: str) -> str:
    return _TEAM_ALIAS.get(raw, raw)


def _fetch_next_data() -> dict[str, Any]:
    try:
        r = requests.get(URL, headers={"User-Agent": _UA}, timeout=TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise ActionNetworkUnavailable(f"fetch failed: {exc}") from exc
    m = _NEXT_DATA_RE.search(r.text)
    if not m:
        raise ActionNetworkUnavailable("__NEXT_DATA__ not found (page layout changed?)")
    try:
        return json.loads(m.group(1))
    except ValueError as exc:
        raise ActionNetworkUnavailable(f"__NEXT_DATA__ not valid JSON: {exc}") from exc


def _side_entry(entries: list[dict[str, Any]], book_id: str, side: str) -> Optional[dict[str, Any]]:
    return next(
        (e for e in entries if str(e.get("book_id")) == book_id and e.get("side") == side), None
    )


def _book_name(all_books: dict[str, Any], book_id: Any) -> str:
    b = all_books.get(str(book_id)) or {}
    return b.get("parent_name") or b.get("display_name") or str(book_id)


def _book_rows(
    spread_entries: list[dict[str, Any]], ml_entries: list[dict[str, Any]], all_books: dict[str, Any],
) -> list[dict[str, Any]]:
    """One row per real sportsbook that quoted this game (spread and/or moneyline)."""
    book_ids = {
        str(e.get("book_id")) for e in (spread_entries + ml_entries)
        if str(e.get("book_id")) not in _SKIP_BOOK_IDS
    }
    rows = []
    for bid in book_ids:
        sh, sa = _side_entry(spread_entries, bid, "home"), _side_entry(spread_entries, bid, "away")
        mh, ma = _side_entry(ml_entries, bid, "home"), _side_entry(ml_entries, bid, "away")
        rows.append({
            "book": _book_name(all_books, bid),
            "spread_home_line": sh["value"] if sh else None,
            "spread_home_price": sh["odds"] if sh else None,
            "spread_away_line": sa["value"] if sa else None,
            "spread_away_price": sa["odds"] if sa else None,
            "ml_home_price": mh["odds"] if mh else None,
            "ml_away_price": ma["odds"] if ma else None,
        })
    return rows


def _best_moneyline(entries: list[dict[str, Any]], side: str) -> Optional[dict[str, Any]]:
    """Highest price (best payout) for one side, across real books only."""
    cands = [e for e in entries if e.get("side") == side and str(e.get("book_id")) not in _SKIP_BOOK_IDS
             and e.get("odds") is not None]
    return max(cands, key=lambda e: e["odds"]) if cands else None


def _best_spread(entries: list[dict[str, Any]], side: str) -> Optional[dict[str, Any]]:
    """Most favorable line for `side` (spread values are signed, so max(value) is always
    best regardless of favorite/underdog), tie-broken by best price."""
    cands = [e for e in entries if e.get("side") == side and str(e.get("book_id")) not in _SKIP_BOOK_IDS
             and e.get("value") is not None and e.get("odds") is not None]
    if not cands:
        return None
    best_value = max(c["value"] for c in cands)
    at_best = [c for c in cands if c["value"] == best_value]
    return max(at_best, key=lambda c: c["odds"])


def _process_game(
    an_game: dict[str, Any], games_by_teams: dict[tuple[str, str], dict[str, Any]],
    all_books: dict[str, Any],
) -> Optional[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Returns (best-price summary row, per-book rows) for one game, or None if it isn't
    upcoming / couldn't be matched to a nfl-hub game."""
    if an_game.get("status") != "scheduled":
        return None  # only upcoming games have a live line worth shopping
    markets = an_game.get("markets")
    if not markets:
        return None
    team_by_id = {t["id"]: t for t in an_game.get("teams", [])}
    home_t = team_by_id.get(an_game.get("home_team_id"))
    away_t = team_by_id.get(an_game.get("away_team_id"))
    if not home_t or not away_t:
        return None
    home_abbr = _team_abbr(home_t.get("abbr", ""))
    away_abbr = _team_abbr(away_t.get("abbr", ""))
    g = games_by_teams.get((home_abbr, away_abbr))
    if not g:
        return None

    ml_entries: list[dict[str, Any]] = []
    spread_entries: list[dict[str, Any]] = []
    for payload in markets.values():
        event = (payload or {}).get("event") or {}
        ml_entries += event.get("moneyline") or []
        spread_entries += event.get("spread") or []

    books = _book_rows(spread_entries, ml_entries, all_books)
    home_lines = [b["spread_home_line"] for b in books if b["spread_home_line"] is not None]
    avg_spread_home = round(statistics.mean(home_lines), 2) if home_lines else None

    best_ml_home = _best_moneyline(ml_entries, "home")
    best_ml_away = _best_moneyline(ml_entries, "away")
    best_spread_home = _best_spread(spread_entries, "home")
    best_spread_away = _best_spread(spread_entries, "away")

    best_row = {
        "game_id": g["game_id"],
        "week": g["week"],
        "avg_spread_home": avg_spread_home,
        "ml_home_book": _book_name(all_books, best_ml_home["book_id"]) if best_ml_home else None,
        "ml_home_price": best_ml_home["odds"] if best_ml_home else None,
        "ml_away_book": _book_name(all_books, best_ml_away["book_id"]) if best_ml_away else None,
        "ml_away_price": best_ml_away["odds"] if best_ml_away else None,
        "spread_home_book": _book_name(all_books, best_spread_home["book_id"]) if best_spread_home else None,
        "spread_home_line": best_spread_home["value"] if best_spread_home else None,
        "spread_home_price": best_spread_home["odds"] if best_spread_home else None,
        "spread_away_book": _book_name(all_books, best_spread_away["book_id"]) if best_spread_away else None,
        "spread_away_line": best_spread_away["value"] if best_spread_away else None,
        "spread_away_price": best_spread_away["odds"] if best_spread_away else None,
    }
    for b in books:
        b["game_id"] = g["game_id"]
        b["week"] = g["week"]
    return best_row, books


def fetch_odds_detail(
    games: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """(game_id -> best-price summary, game_id -> per-book rows) for this week's games
    found on Action Network's page.

    Raises ActionNetworkUnavailable on total failure (page unreachable / layout changed);
    a single bad game within the response is logged and skipped rather than dropping the
    whole batch.
    """
    data = _fetch_next_data()
    try:
        page = data["props"]["pageProps"]
        an_games = page["scoreboardResponse"]["games"]
        all_books = page["allBooks"]
    except (KeyError, TypeError) as exc:
        raise ActionNetworkUnavailable(f"unexpected page shape: {exc}") from exc

    games_by_teams = {(g["home"], g["away"]): g for g in games}
    best: dict[str, dict[str, Any]] = {}
    by_book: dict[str, list[dict[str, Any]]] = {}
    for an_game in an_games:
        try:
            result = _process_game(an_game, games_by_teams, all_books)
        except Exception as exc:  # noqa: BLE001 - one bad game shouldn't drop the rest
            log.warning("actionnetwork: game %s failed: %s", an_game.get("id"), exc)
            continue
        if result:
            best_row, book_rows = result
            best[best_row["game_id"]] = best_row
            by_book[best_row["game_id"]] = book_rows
    return best, by_book
