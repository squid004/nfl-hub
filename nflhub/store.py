"""Persistence over the Supabase REST API.

Public function names match the v1 SQLite module so `sources/*`, `deadlines.py` and
`refresh.py` are unchanged. RLS on every table is "anon all" (personal tool), so the anon
key is enough for both reads and writes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import requests

from .config import get_config

log = logging.getLogger(__name__)
TIMEOUT = 20


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _headers(prefer: str | None = None) -> dict[str, str]:
    cfg = get_config()
    if not cfg.supabase.url or not cfg.supabase.anon_key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_ANON_KEY not set")
    h = {
        "apikey": cfg.supabase.anon_key,
        "Authorization": f"Bearer {cfg.supabase.anon_key}",
        "Content-Type": "application/json",
    }
    if prefer:
        h["Prefer"] = prefer
    return h


def _url(table: str) -> str:
    return f"{get_config().supabase.rest}/{table}"


def _get(table: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    r = requests.get(_url(table), headers=_headers(), params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _upsert(table: str, rows: list[dict[str, Any]] | dict[str, Any], on_conflict: str | None = None) -> None:
    params = {"on_conflict": on_conflict} if on_conflict else {}
    r = requests.post(
        _url(table),
        headers=_headers("resolution=merge-duplicates,return=minimal"),
        params=params,
        json=rows,
        timeout=TIMEOUT,
    )
    r.raise_for_status()


def _patch(table: str, params: dict[str, Any], body: dict[str, Any]) -> None:
    r = requests.patch(
        _url(table), headers=_headers("return=minimal"), params=params, json=body, timeout=TIMEOUT
    )
    r.raise_for_status()


def _delete(table: str, params: dict[str, Any]) -> None:
    r = requests.delete(_url(table), headers=_headers("return=minimal"), params=params, timeout=TIMEOUT)
    r.raise_for_status()


def init_db() -> None:
    """Connectivity check; schema itself is applied via supabase/schema.sql."""
    requests.get(_url("kv"), headers=_headers(), params={"limit": 1}, timeout=TIMEOUT).raise_for_status()


# --- key/value -------------------------------------------------------------

def kv_get(key: str) -> Optional[str]:
    rows = _get("kv", {"key": f"eq.{key}", "select": "value", "limit": 1})
    return rows[0]["value"] if rows else None


def kv_set(key: str, value: str) -> None:
    _upsert("kv", {"key": key, "value": value, "updated_at": _now()}, on_conflict="key")


# --- games & odds --------------------------------------------------------

def upsert_game(g: dict[str, Any]) -> None:
    _upsert("game", {**g, "updated_at": _now()}, on_conflict="game_id")


def upsert_odds(o: dict[str, Any]) -> None:
    _upsert("odds", {**o, "updated_at": _now()}, on_conflict="game_id")


def games_for_week(week: int) -> list[dict[str, Any]]:
    return _get("game", {"week": f"eq.{week}", "order": "kickoff"})


def odds_for_week(week: int) -> dict[str, dict[str, Any]]:
    return {r["game_id"]: r for r in _get("odds", {"week": f"eq.{week}"})}


# --- roster snapshots ---------------------------------------------------

def save_roster_snapshot(league: str, week: int, payload: dict[str, Any]) -> None:
    _upsert("roster_snapshot", {
        "league": league, "week": week, "captured_at": _now(), "payload": payload,
    })


def latest_roster_snapshot(league: str) -> Optional[dict[str, Any]]:
    rows = _get("roster_snapshot", {
        "league": f"eq.{league}", "order": "captured_at.desc", "limit": 1,
    })
    return rows[0] if rows else None


# --- news --------------------------------------------------------------

def replace_news(items: Iterable[dict[str, Any]]) -> None:
    _delete("news", {"id": "not.is.null"})
    rows = [{
        "id": i["id"], "published": i.get("published") or None,
        "headline": i.get("headline", ""), "description": i.get("description", ""),
        "players": i.get("players", []), "link": i.get("link", ""), "updated_at": _now(),
    } for i in items]
    if rows:
        _upsert("news", rows, on_conflict="id")


def recent_news(limit: int = 25) -> list[dict[str, Any]]:
    return _get("news", {"order": "published.desc", "limit": limit})


# --- pick'em & survivor ----------------------------------------------

def record_pickem_pick(week: int, game_id: str, pick: str, confidence: Optional[int] = None) -> None:
    _upsert("pickem_pick", {
        "week": week, "game_id": game_id, "pick": pick,
        "confidence": confidence, "created_at": _now(),
    }, on_conflict="week,game_id")


def get_pickem_picks(week: int) -> dict[str, dict[str, Any]]:
    return {r["game_id"]: r for r in _get("pickem_pick", {"week": f"eq.{week}"})}


def record_survivor_pick(week: int, team: str) -> None:
    _upsert("survivor_pick", {"week": week, "team": team, "created_at": _now()}, on_conflict="week")


# --- weekly budget snapshot (end-of-season analysis) ----------------------

def has_budget_snapshot(week: int) -> bool:
    return bool(_get("budget_snapshot", {"week": f"eq.{week}", "limit": 1}))


def upsert_budget_snapshot(week: int, mode: str, rows: list[dict[str, Any]]) -> None:
    payload = [
        {
            "week": week, "mode": mode, "bin": r["bin"],
            "n_games": r["n_games"], "rate": r["rate"], "suggested": r["suggested"],
            "games": r["games"], "captured_at": _now(),
        }
        for r in rows
    ]
    if payload:
        _upsert("budget_snapshot", payload, on_conflict="week,mode,bin")


def get_survivor_picks() -> dict[int, str]:
    return {r["week"]: r["team"] for r in _get("survivor_pick", {"order": "week"})}


# --- reminder dedupe -------------------------------------------------

def reminder_already_sent(kind: str, key: str) -> bool:
    return bool(_get("reminder_log", {"kind": f"eq.{kind}", "key": f"eq.{key}", "limit": 1}))


def mark_reminder_sent(kind: str, key: str) -> None:
    _upsert("reminder_log", {"kind": kind, "key": key, "sent_at": _now()}, on_conflict="kind,key")


# --- refresh-now flag ---------------------------------------------

def request_refresh() -> None:
    _patch("refresh_request", {"id": "eq.1"}, {"requested_at": _now(), "handled_at": None})


def refresh_pending() -> bool:
    rows = _get("refresh_request", {"id": "eq.1", "limit": 1})
    if not rows:
        return False
    req, done = rows[0].get("requested_at"), rows[0].get("handled_at")
    return bool(req) and (not done or done < req)


def mark_refresh_handled() -> None:
    _patch("refresh_request", {"id": "eq.1"}, {"handled_at": _now()})
