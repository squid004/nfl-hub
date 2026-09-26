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


def upsert_best_price_odds(rows: list[dict[str, Any]]) -> None:
    if rows:
        _upsert("best_price_odds", [{**r, "fetched_at": _now()} for r in rows], on_conflict="game_id")


def best_price_odds_for_week(week: int) -> dict[str, dict[str, Any]]:
    return {r["game_id"]: r for r in _get("best_price_odds", {"week": f"eq.{week}"})}


def upsert_book_odds(rows: list[dict[str, Any]]) -> None:
    if rows:
        _upsert("book_odds", [{**r, "fetched_at": _now()} for r in rows], on_conflict="game_id,book")


def book_odds_for_week(week: int) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in _get("book_odds", {"week": f"eq.{week}"}):
        out.setdefault(r["game_id"], []).append(r)
    return out


def upsert_elway_odds(rows: list[dict[str, Any]]) -> None:
    if rows:
        _upsert("elway_odds", [{**r, "fetched_at": _now()} for r in rows], on_conflict="game_id")


def elway_odds_for_week(week: int) -> dict[str, dict[str, Any]]:
    return {r["game_id"]: r for r in _get("elway_odds", {"week": f"eq.{week}"})}


# --- spread movement history ---------------------------------------------

def spread_history_latest_for_games(game_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Most-recent snapshot per game_id, for diffing against this refresh's new spread."""
    if not game_ids:
        return {}
    ids = ",".join(game_ids)
    rows = _get("spread_history", {"game_id": f"in.({ids})", "order": "captured_at.asc"})
    out: dict[str, dict[str, Any]] = {}
    for r in rows:  # ascending order -> last write per key wins -> latest per game
        out[r["game_id"]] = r
    return out


def spread_history_insert(rows: list[dict[str, Any]]) -> None:
    """Append-only: no on_conflict, each (game_id, captured_at) is a new row."""
    if rows:
        _upsert("spread_history", [{**r, "captured_at": _now()} for r in rows])


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


# --- pick'em -----------------------------------------------------------

def record_pickem_pick(week: int, game_id: str, pick: str, confidence: Optional[int] = None) -> None:
    _upsert("pickem_pick", {
        "week": week, "game_id": game_id, "pick": pick,
        "confidence": confidence, "created_at": _now(),
    }, on_conflict="week,game_id")


def get_pickem_picks(week: int) -> dict[str, dict[str, Any]]:
    return {r["game_id"]: r for r in _get("pickem_pick", {"week": f"eq.{week}"})}


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


# --- pickem-edge: national pick %, opponent picks, bias, standing, recs ----

def edge_get_national_pct(season: int, week: int, team: str) -> Optional[float]:
    """Prefers a manual entry over a scraped one for the same team/week."""
    rows = _get("edge_national_pct", {
        "season": f"eq.{season}", "week": f"eq.{week}", "team": f"eq.{team}",
    })
    if not rows:
        return None
    manual = next((r for r in rows if r["source"] == "manual"), None)
    return (manual or rows[0])["pct"]


def edge_set_national_pct(season: int, week: int, team: str, pct: float, source: str = "manual") -> None:
    _upsert("edge_national_pct", {
        "season": season, "week": week, "team": team, "pct": pct,
        "source": source, "fetched_at": _now(), "is_stale": False,
    }, on_conflict="season,week,team,source")


def edge_bulk_set_national_pct(season: int, week: int, rows: list[dict[str, Any]], source: str = "nflpickwatch") -> None:
    payload = [{
        "season": season, "week": week, "team": r["team"], "pct": r["pct"],
        "source": source, "fetched_at": _now(), "is_stale": False,
    } for r in rows]
    if payload:
        _upsert("edge_national_pct", payload, on_conflict="season,week,team,source")


def edge_bias_all() -> list[dict[str, Any]]:
    return _get("edge_bias", {"order": "team"})


def edge_get_bias(team: str) -> tuple[float, int, bool]:
    """Returns (bias_value, n_observations, overridden); (0.0, 0, False) if unseen."""
    rows = _get("edge_bias", {"team": f"eq.{team}", "limit": 1})
    if not rows:
        return 0.0, 0, False
    r = rows[0]
    return r["bias_value"], r["n_observations"], bool(r.get("overridden", False))


def edge_upsert_bias(team: str, bias_value: float, n_observations: int, overridden: bool = False) -> None:
    _upsert("edge_bias", {
        "team": team, "bias_value": bias_value, "n_observations": n_observations,
        "last_updated": _now(), "overridden": overridden,
    }, on_conflict="team")


def edge_set_bias_override(team: str, bias_value: float) -> None:
    _, n, _ = edge_get_bias(team)
    edge_upsert_bias(team, bias_value, n, overridden=True)


def edge_clear_bias_override(team: str) -> None:
    _patch("edge_bias", {"team": f"eq.{team}"}, {"overridden": False})


def edge_opponent_picks_for_week(season: int, week: int) -> list[dict[str, Any]]:
    return _get("edge_opponent_pick", {
        "season": f"eq.{season}", "week": f"eq.{week}", "order": "opponent",
    })


def edge_insert_opponent_picks(rows: list[dict[str, Any]]) -> None:
    payload = [{**r, "imported_at": _now()} for r in rows]
    if payload:
        _upsert("edge_opponent_pick", payload, on_conflict="season,week,opponent,team_picked")


def edge_delete_opponent_pick(season: int, week: int, opponent: str, team_picked: str) -> None:
    _delete("edge_opponent_pick", {
        "season": f"eq.{season}", "week": f"eq.{week}",
        "opponent": f"eq.{opponent}", "team_picked": f"eq.{team_picked}",
    })


def edge_all_opponent_picks() -> list[dict[str, Any]]:
    return _get("edge_opponent_pick", {"select": "season,week,opponent,team_picked"})


def edge_games_for_team(team: str) -> list[dict[str, Any]]:
    return _get("game", {"or": f"(home.eq.{team},away.eq.{team})", "select": "season,week"})


def edge_get_standing(season: int, week: int) -> Optional[dict[str, Any]]:
    rows = _get("edge_season_standing", {"season": f"eq.{season}", "week": f"eq.{week}", "limit": 1})
    return rows[0] if rows else None


def edge_latest_standing_before(season: int, week: int) -> Optional[dict[str, Any]]:
    rows = _get("edge_season_standing", {
        "season": f"eq.{season}", "week": f"lte.{week}", "order": "week.desc", "limit": 1,
    })
    return rows[0] if rows else None


def edge_all_standings(season: int) -> list[dict[str, Any]]:
    return _get("edge_season_standing", {"season": f"eq.{season}", "order": "week"})


def edge_upsert_standing(
    season: int, week: int, standing_bucket: str, pool_size: int,
    correct_picks: Optional[int] = None, total_picks: Optional[int] = None,
    rank: Optional[int] = None,
) -> None:
    _upsert("edge_season_standing", {
        "season": season, "week": week, "standing_bucket": standing_bucket,
        "pool_size": pool_size, "correct_picks": correct_picks, "total_picks": total_picks,
        "rank": rank, "updated_at": _now(),
    }, on_conflict="season,week")


def edge_recommendation_log_for_week(season: int, week: int) -> dict[str, dict[str, Any]]:
    return {r["game_id"]: r for r in _get("edge_recommendation_log", {
        "season": f"eq.{season}", "week": f"eq.{week}",
    })}


def edge_recommendation_log_for_season(season: int) -> list[dict[str, Any]]:
    return _get("edge_recommendation_log", {"season": f"eq.{season}", "order": "week,game_id"})


def edge_has_recommendation_log(season: int, week: int) -> bool:
    return bool(_get("edge_recommendation_log", {
        "season": f"eq.{season}", "week": f"eq.{week}", "limit": 1,
    }))


def edge_upsert_recommendation_log(season: int, week: int, rows: list[dict[str, Any]]) -> None:
    payload = [{
        "season": season, "week": week, "game_id": r["game_id"],
        "favorite_team": r["favorite_team"], "underdog_team": r["underdog_team"],
        "p_favorite": r["p_favorite"], "vig": r.get("vig"), "f_estimate": r.get("f_estimate"),
        "leverage": r.get("leverage"), "eligible": r["eligible"],
        "recommendation": r["recommendation"], "budget_at_time": r.get("budget_at_time"),
        "generated_at": _now(),
    } for r in rows]
    if payload:
        _upsert("edge_recommendation_log", payload, on_conflict="season,week,game_id")


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
