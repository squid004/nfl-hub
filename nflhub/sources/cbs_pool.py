"""Direct, read-only fetch of the pool's picks from CBS (source "cbs" in pool_live).

Uses the same GraphQL operation the pool's weekly-standings page runs in the browser
(`FootballPickemManagerWeeklyStandingsContent`, document in cbs_pool_query.graphql, taken from CBS's public page
JavaScript): for one pool period it returns every entry's picks (visible once a game locks; your own are always visible when
logged in), tiebreaker answers, and the period's games with their `isLocked` flags. Authenticated with the user's CBS
session cookie (env CBS_COOKIE = the `cookie` request header copied from a logged-in browser request).

READ ONLY: this module only ever sends that one query. It never calls CBS's savePicks mutation and never will.

The response mapping (pick slot -> game, picked item -> team) follows the query's field names (`slotId`/`cbsSlotId` match a
game's `id`/`cbsEventId`; `pickInfo.itemId`/`cbsItemId` match a team's `id`/`cbsTeamId`) and has not yet been run against a live
response; if it can't map any pick it raises with a diagnostic instead of returning nothing. Run
`CBS_COOKIE=... python -m nflhub.sources.cbs_pool <week>` to check it before relying on it.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

from .edge_teams import normalize_team

log = logging.getLogger(__name__)

GRAPHQL_URL = "https://picks.cbssports.com/graphql"
OPERATION = "FootballPickemManagerWeeklyStandingsContent"
QUERY = (Path(__file__).with_name("cbs_pool_query.graphql")).read_text(encoding="utf-8")
TIMEOUT = 25


class CbsPoolUnavailable(RuntimeError):
    """Any problem fetching or mapping CBS data. Soft failure: the caller records it as that source's error."""


def _post(cookie: str, pool_id: str, period_id: str | None) -> dict[str, Any]:
    headers = {
        "content-type": "application/json", "accept": "application/json",
        "origin": "https://picks.cbssports.com",
        "referer": f"https://picks.cbssports.com/football/pickem/pools/{pool_id}/standings/weekly",
        "user-agent": "Mozilla/5.0 (compatible; nfl-hub personal pool reader)",
        "cookie": cookie,
    }
    body = {"operationName": OPERATION, "variables": {"poolId": pool_id, "poolPeriodId": period_id}, "query": QUERY}
    try:
        resp = requests.post(GRAPHQL_URL, json=body, headers=headers, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise CbsPoolUnavailable(f"request failed: {exc}") from exc
    if resp.status_code in (401, 403):
        raise CbsPoolUnavailable(f"CBS rejected the session (HTTP {resp.status_code}) -- cookie expired, or this IP is blocked")
    if resp.status_code != 200:
        raise CbsPoolUnavailable(f"HTTP {resp.status_code}")
    try:
        payload = resp.json()
    except ValueError as exc:
        raise CbsPoolUnavailable("non-JSON response") from exc
    if not payload.get("data") or not (payload["data"].get("commonPool") or {}):
        msg = "; ".join(str(e.get("message")) for e in payload.get("errors") or [])[:200]
        raise CbsPoolUnavailable(f"no pool data returned ({msg or 'unknown reason'}) -- not logged in as a pool member?")
    return payload["data"]["commonPool"]


def _period_for_week(pool: dict[str, Any], week: int) -> dict[str, Any]:
    periods = pool.get("poolPeriods") or []
    for p in periods:
        if p.get("order") == week:
            return p
    for p in periods:
        if re.search(rf"\b{week}\b", str(p.get("description") or "")):
            return p
    raise CbsPoolUnavailable(f"week {week} isn't a pool period CBS lists yet")


def parse(pool: dict[str, Any], week: int) -> tuple[dict[str, list[str]], dict[str, float], dict[str, Any]]:
    """(picks {name: [team,...]}, tiebreaks {name: value}, meta). Only picks CBS has revealed (locked games, plus your own)."""
    period = pool.get("poolPeriod") or {}
    events = period.get("poolEvents") or []
    by_slot: dict[str, dict[str, Any]] = {}
    for ev in events:
        for k in (ev.get("id"), ev.get("cbsEventId")):
            if k is not None:
                by_slot[str(k)] = ev
    entries = (((pool.get("standings") or {}).get("weekly") or {}).get("rankedEntries")) or []
    picks: dict[str, list[str]] = {}
    tiebreaks: dict[str, float] = {}
    mapped = unmapped = 0
    for re_ in entries:
        entry = re_.get("entry") or {}
        member = (entry.get("member") or {}).get("name")
        names = [n for n in dict.fromkeys([member, entry.get("name")]) if n]
        if not names:
            continue
        mine = bool(entry.get("isMine"))
        teams: list[str] = []
        for p in re_.get("picks") or []:
            info = p.get("pickInfo") or {}
            if info.get("itemId") is None and info.get("cbsItemId") is None:
                continue
            ev = by_slot.get(str(p.get("slotId"))) or by_slot.get(str(p.get("cbsSlotId")))
            if ev is None:
                unmapped += 1
                continue
            if not ev.get("isLocked") and not mine:
                continue    # not revealed for others; never infer an open pick
            home, away = ev.get("homeTeam") or {}, ev.get("awayTeam") or {}
            wanted = {str(x) for x in (info.get("itemId"), info.get("cbsItemId")) if x is not None}
            team = None
            for t in (home, away):
                ids = {str(x) for x in (t.get("id"), t.get("cbsTeamId")) if x is not None}
                if t is home and ev.get("homeTeamId") is not None:
                    ids.add(str(ev["homeTeamId"]))
                if ids & wanted:
                    team = t.get("abbrev")
                    break
            if team is None:
                unmapped += 1
                continue
            try:
                teams.append(normalize_team(team))
                mapped += 1
            except ValueError:
                unmapped += 1
        for ta in re_.get("tiebreakerAnswers") or []:
            v = ((ta.get("tiebreakerAnswerInfo") or {}).get("value"))
            try:
                tb = float(v)
            except (TypeError, ValueError):
                continue
            for n in names:
                tiebreaks[n] = tb
        for n in names:
            picks[n] = teams
    meta = {"entries": len(entries), "events": len(events), "locked_events": sum(1 for e in events if e.get("isLocked")),
            "mapped_picks": mapped, "unmapped_picks": unmapped}
    return picks, tiebreaks, meta


def fetch_raw(cookie: str, pool_id: str, week: int, post: Callable[[str, str, str | None], dict[str, Any]] = _post) -> dict[str, Any]:
    pool = post(cookie, pool_id, None)
    period = _period_for_week(pool, week)
    if not period.get("isCurrent") or (pool.get("poolPeriod") or {}).get("id") != period.get("id"):
        pool = post(cookie, pool_id, str(period["id"]))
    return pool


def fetch(cfg, week: int, post: Callable[[str, str, str | None], dict[str, Any]] = _post):
    """-> pool_live.Snapshot for `week` (imported lazily to avoid a circular import)."""
    from .pool_live import Snapshot
    cookie = getattr(cfg.edge, "cbs_cookie", "") or ""
    if not cookie:
        raise CbsPoolUnavailable("CBS_COOKIE not set")
    pool = fetch_raw(cookie, getattr(cfg.edge, "cbs_pool_id", "") or "", week, post)
    picks, tiebreaks, meta = parse(pool, week)
    if meta["mapped_picks"] == 0 and meta["unmapped_picks"] > 0:
        raise CbsPoolUnavailable(f"couldn't map any of {meta['unmapped_picks']} picks to games/teams -- the response shape differs from the "
                                 "query-based assumption (see docs/pool-live-picks.md to capture one for debugging)")
    if not picks:
        raise CbsPoolUnavailable("CBS returned no entries")
    return Snapshot("cbs", datetime.now(timezone.utc), picks, tiebreaks)


if __name__ == "__main__":
    from ..config import get_config
    week = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cfg = get_config()
    cookie = os.environ.get("CBS_COOKIE", "")
    pool = fetch_raw(cookie, cfg.edge.cbs_pool_id, week)
    picks, tiebreaks, meta = parse(pool, week)
    print(meta)
    for n, t in list(picks.items())[:40]:
        print(f"  {n}: {len(t)} picks {t[:6]}  tiebreak {tiebreaks.get(n)}")
