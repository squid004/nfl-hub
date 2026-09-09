"""Pull every source into Supabase. Each step is isolated so one failure doesn't stop the rest."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from . import optimizer, store
from .config import Config, get_config
from .sources import espn_fantasy, fantasypros, nfl_schedule, odds, yahoo_fantasy

log = logging.getLogger(__name__)


def _rostered_names(week: int) -> set[str]:
    names: set[str] = set()
    for league in ("yahoo", "espn"):
        snap = store.latest_roster_snapshot(league)
        if snap and snap.get("week") == week:
            for grp in ("starters", "bench"):
                names.update(p["name"] for p in snap["payload"].get(grp, []))
    return names


def refresh_all(cfg: Config | None = None) -> dict[str, Any]:
    cfg = cfg or get_config()
    store.init_db()
    summary: dict[str, Any] = {"errors": []}

    season, week = nfl_schedule.current_week()
    store.kv_set("season", str(season))
    store.kv_set("week", str(week))
    summary["season"], summary["week"] = season, week

    # 1. schedule + scores
    try:
        games = nfl_schedule.fetch_scoreboard(season, week)
        for g in games:
            store.upsert_game({k: v for k, v in g.items() if not k.startswith("_")})
        summary["games"] = len(games)
    except Exception as exc:  # noqa: BLE001
        log.exception("schedule refresh failed")
        summary["errors"].append(f"schedule: {exc}")
        games = [dict(g) for g in store.games_for_week(week)]

    # 2. odds
    try:
        wk_odds = odds.get_week_odds(games, cfg.odds.provider, cfg.odds.api_key)
        for o in wk_odds.values():
            store.upsert_odds(o)
        summary["odds"] = len(wk_odds)
    except Exception as exc:  # noqa: BLE001
        log.exception("odds refresh failed")
        summary["errors"].append(f"odds: {exc}")

    # 3. fantasy snapshots
    for name, mod in (("yahoo", yahoo_fantasy), ("espn", espn_fantasy)):
        try:
            snap = mod.build_snapshot(cfg, season, week, games)
            summary[name] = "ok" if snap else "skipped"
        except Exception as exc:  # noqa: BLE001
            log.exception("%s fantasy refresh failed", name)
            summary["errors"].append(f"{name}: {exc}")
            summary[name] = "error"

    # 3b. FantasyPros projections/ECR -> optimal lineup per league
    if cfg.fantasypros.enabled and cfg.fantasypros.api_key:
        try:
            _apply_fantasypros(cfg, season, week, summary)
        except fantasypros.FreeTierLimited as exc:
            log.warning("FantasyPros: %s", exc)
            summary["errors"].append(f"fantasypros: {exc}")
            summary["fantasypros"] = "free-tier-limited"
        except Exception as exc:  # noqa: BLE001
            log.exception("FantasyPros step failed")
            summary["errors"].append(f"fantasypros: {exc}")
            summary["fantasypros"] = "error"

    # 4. news + injuries, filtered to players you actually roster
    try:
        rostered = _rostered_names(week)
        injuries = nfl_schedule.fetch_injuries()
        news_items = []
        for art in nfl_schedule.fetch_news():
            hit = [n for n in art["athletes"] if n in rostered]
            blob = f"{art['headline']} {art['description']}"
            hit += [n for n in rostered if n not in hit and n and n in blob]
            if hit or not rostered:
                news_items.append(
                    {
                        "id": art["id"],
                        "published": art["published"],
                        "headline": art["headline"],
                        "description": art["description"],
                        "players": sorted(set(hit)),
                        "link": art["link"],
                    }
                )
        store.replace_news(news_items)
        store.kv_set("injuries", _json(injuries))
        summary["news"] = len(news_items)
    except Exception as exc:  # noqa: BLE001
        log.exception("news refresh failed")
        summary["errors"].append(f"news: {exc}")

    # 5. bookkeeping: clear the "Refresh now" flag, stamp last_refresh
    try:
        if store.refresh_pending():
            store.mark_refresh_handled()
            summary["refresh_request"] = "handled"
        store.kv_set("last_refresh", datetime.now(timezone.utc).isoformat())
    except Exception as exc:  # noqa: BLE001
        log.warning("bookkeeping failed: %s", exc)

    return summary


def _apply_fantasypros(cfg: Config, season: int, week: int, summary: dict[str, Any]) -> None:
    """Enrich each fantasy snapshot with FP projections/ECR and store the optimal lineup."""
    scoring_by_league = {"yahoo": cfg.fantasypros.scoring_yahoo, "espn": cfg.fantasypros.scoring_espn}
    proj_cache: dict[str, dict] = {}
    ecr_cache: dict[str, dict] = {}
    for league, scoring in scoring_by_league.items():
        snap = store.latest_roster_snapshot(league)
        if not snap or snap.get("week") != week:
            continue
        if scoring not in proj_cache:
            proj_cache[scoring] = fantasypros.weekly_projections(cfg, season, week, scoring)
            ecr_cache[scoring] = fantasypros.weekly_ecr(cfg, season, week, scoring)
        payload = snap["payload"]
        matched = fantasypros.enrich(payload, proj_cache[scoring], ecr_cache[scoring])
        payload["fp"] = optimizer.optimize(payload)
        payload["fp_matched"] = matched
        store.save_roster_snapshot(league, week, payload)
        summary[f"{league}_fp"] = f"{matched} matched, delta {payload['fp']['delta'] if payload['fp'] else 'n/a'}"
    summary["fantasypros"] = "ok"


def _json(obj: Any) -> str:
    import json

    return json.dumps(obj)
