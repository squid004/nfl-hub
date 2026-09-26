"""Pull every source into Supabase. Each step is isolated so one failure doesn't stop the rest."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from . import optimizer, store
from .config import Config, get_config
from .sources import (
    actionnetwork,
    edge_bias,
    edge_core,
    edge_national,
    edge_sheet,
    elway,
    espn_fantasy,
    fantasypros,
    history,
    nfl_schedule,
    odds,
    yahoo_fantasy,
)
from .sources.edge_teams import UnknownTeamError, normalize_team

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
    wk_odds: dict[str, Any] = {}
    try:
        wk_odds = odds.get_week_odds(
            games, cfg.odds.provider, cfg.odds.api_key, cfg.odds.espn_game_odds
        )
        for o in wk_odds.values():
            store.upsert_odds(o)
        summary["odds"] = len(wk_odds)
    except Exception as exc:  # noqa: BLE001
        log.exception("odds refresh failed")
        summary["errors"].append(f"odds: {exc}")

    # 2b. best price across real books (Action Network) — best-effort, unofficial page scrape
    best_price: dict[str, Any] = {}  # always defined: step 4c uses its avg_spread_home
    if cfg.odds.action_network:
        try:
            best_price, book_rows = actionnetwork.fetch_odds_detail(games)
            store.upsert_best_price_odds(list(best_price.values()))
            store.upsert_book_odds([row for rows in book_rows.values() for row in rows])
            summary["best_price_odds"] = len(best_price)
        except Exception as exc:  # noqa: BLE001 - unofficial scrape; never block the rest
            log.warning("actionnetwork best-price scrape failed: %s", exc)
            summary["best_price_odds"] = f"skipped ({exc})"

    # 2c. ELWAY (Nate Silver's Silver Bulletin NFL model), transcribed weekly into a
    # personal Google Sheet — best-effort, soft-fail
    elway_rows: dict[str, Any] = {}  # always defined: step 4c uses it for the budget ranking
    if cfg.elway.sheet_id:
        try:
            elway_rows = elway.fetch_week(cfg.elway.sheet_id, week, games)
            store.upsert_elway_odds(list(elway_rows.values()))
            summary["elway"] = len(elway_rows)
        except Exception as exc:  # noqa: BLE001 - personal sheet; never block the rest
            log.warning("elway sheet pull failed: %s", exc)
            summary["elway"] = f"skipped ({exc})"

    # 2d. spread-movement history — append a snapshot only when the line actually moved
    # since the last one stored, so this is a log of real moves, not re-poll noise. Lets
    # the page show "opened X, now Y" and flag a big one-directional move ("steam").
    try:
        game_ids = [g["game_id"] for g in games]
        latest = store.spread_history_latest_for_games(game_ids)
        new_rows = []
        for gid, o in wk_odds.items():
            sp = o.get("spread")
            if sp is None:
                continue
            prev = latest.get(gid)
            if prev is None or abs(prev["spread_home"] - sp) >= 0.5:
                new_rows.append({"game_id": gid, "week": week, "spread_home": sp})
        store.spread_history_insert(new_rows)
        summary["spread_history"] = f"{len(new_rows)} new snapshot(s)"
    except Exception as exc:  # noqa: BLE001 - best-effort analytics; don't fail the run
        log.warning("spread history snapshot failed: %s", exc)
        summary["spread_history"] = f"skipped ({exc})"

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
        store.kv_set("injuries", json.dumps(injuries))
        summary["news"] = len(news_items)
    except Exception as exc:  # noqa: BLE001
        log.exception("news refresh failed")
        summary["errors"].append(f"news: {exc}")

    # 4b. historical favorite-vs-spread distribution (rebuilt once/day)
    try:
        summary["history"] = history.refresh(store)
    except Exception as exc:  # noqa: BLE001
        log.exception("history refresh failed")
        summary["errors"].append(f"history: {exc}")

    # 4c. freeze this week's "take N" suggestion until first kickoff (year-end analysis)
    try:
        dist_raw = store.kv_get("hist_distribution")
        if dist_raw and games:
            dist = json.loads(dist_raw)
            first = None
            for g in games:
                k = g.get("kickoff")
                if not k:
                    continue
                try:
                    dt = datetime.fromisoformat(k)
                except ValueError:
                    continue
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                first = dt if first is None or dt < first else first
            locked = first is not None and first <= datetime.now(timezone.utc)
            if not locked or not store.has_budget_snapshot(week):
                # Prefer the cross-book average spread (Action Network, several real
                # books) over the single DK-via-ESPN line for bucket assignment — more
                # robust to one book's stale/outlier number, especially near a bucket
                # boundary. Falls back to the existing spread for any game Action
                # Network didn't cover (already final, alias mismatch, etc).
                budget_odds: dict[str, Any] = {}
                for gid, o in wk_odds.items():
                    avg_spread = best_price.get(gid, {}).get("avg_spread_home")
                    budget_odds[gid] = {**o, "spread": avg_spread if avg_spread is not None else o.get("spread")}
                for mode, rows in history.week_budget(dist, week, games, budget_odds, elway_rows).items():
                    store.upsert_budget_snapshot(week, mode, rows)
                summary["budget_snapshot"] = "written"
            else:
                summary["budget_snapshot"] = "locked"
    except Exception as exc:  # noqa: BLE001 - best-effort analytics; don't fail the run
        log.warning("budget snapshot skipped: %s", exc)
        summary["budget_snapshot"] = f"skipped ({exc})"

    # 4d. pickem-edge: leverage/fade recommendations for the straight pick'em pool
    try:
        _apply_edge(cfg, season, week, games, wk_odds, summary)
    except Exception as exc:  # noqa: BLE001 - best-effort; a broken step here never
        log.exception("edge step failed")  # blocks fantasy/odds/etc. from refreshing
        summary["edge_recommendation"] = f"error ({exc})"

    # 5. bookkeeping: clear the "Refresh now" flag, stamp last_refresh
    try:
        if store.refresh_pending():
            store.mark_refresh_handled()
            summary["refresh_request"] = "handled"
        store.kv_set("last_refresh", datetime.now(timezone.utc).isoformat())
    except Exception as exc:  # noqa: BLE001
        log.warning("bookkeeping failed: %s", exc)

    return summary


def _week_first_kickoff(games: list[dict[str, Any]]) -> datetime | None:
    first = None
    for g in games:
        k = g.get("kickoff")
        if not k:
            continue
        try:
            dt = datetime.fromisoformat(k)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        first = dt if first is None or dt < first else first
    return first


def _apply_edge(
    cfg: Config, season: int, week: int, games: list[dict[str, Any]], wk_odds: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    """Leverage/fade recommendations for the straight moneyline pick'em pool, ported from
    github.com/squid004/pickem-edge. See SPEC.md there; nflhub/sources/edge_core.py has the
    ported math and the "why" for each formula.
    """
    # opponent picks + your own standing, pulled from your pool's Google Sheet ("Week N"
    # tab). Every run, so a correction in the sheet shows up within one refresh cycle; the
    # paste box on the page writes the same table, so both sources coexist without conflict.
    if cfg.edge.sheet_id:
        try:
            sheet = edge_sheet.fetch_week(cfg.edge.sheet_id, week, normalize_team)
            store.edge_insert_opponent_picks([
                {"season": season, "week": week, "opponent": p.opponent,
                 "team_picked": p.team_picked, "source": "sheet"}
                for p in sheet.picks
            ])
            summary["edge_sheet"] = f"{len(sheet.picks)} picks, {sheet.pool_size} opponents"

            if cfg.edge.my_name:
                mine = next(
                    (s for s in sheet.standings if cfg.edge.my_name.lower() in s.opponent.lower()),
                    None,
                )
                if mine:
                    existing = store.edge_get_standing(season, week)
                    bucket = existing["standing_bucket"] if existing else "MIDDLE"
                    store.edge_upsert_standing(
                        season, week, bucket, sheet.pool_size,
                        correct_picks=mine.correct_week, total_picks=mine.correct_season,
                        rank=mine.rank,
                    )
                    summary["edge_standing_sync"] = f"rank {mine.rank}/{sheet.pool_size}"
                else:
                    summary["edge_standing_sync"] = f"'{cfg.edge.my_name}' not found in sheet"
        except edge_sheet.EdgeSheetUnavailable as exc:
            log.warning("edge sheet pull failed: %s", exc)
            summary["edge_sheet"] = f"unavailable ({exc})"

    # national pick % — best-effort scrape, at most once/day, never overwrites a manual row
    today = datetime.now(timezone.utc).date().isoformat()
    if store.kv_get("edge_national_date") != today:
        try:
            rows = edge_national.fetch_week(week, normalize_team)
            store.edge_bulk_set_national_pct(
                season, week, [{"team": r.team, "pct": r.pct} for r in rows]
            )
            store.kv_set("edge_national_date", today)
            summary["edge_national"] = f"scraped {len(rows)} teams"
        except edge_national.NationalPctUnavailable as exc:
            log.warning("national pick%% scrape failed: %s", exc)
            summary["edge_national"] = f"scrape failed ({exc})"

    # recompute learned bias for every team with opponent picks (cheap; skips overridden)
    edge_bias.recompute_all_bias()

    # freeze recommendations at first kickoff, same lock pattern as budget_snapshot
    first = _week_first_kickoff(games)
    locked = first is not None and first <= datetime.now(timezone.utc)
    if locked and store.edge_has_recommendation_log(season, week):
        summary["edge_recommendation"] = "locked"
        return

    standing_row = store.edge_get_standing(season, week) or store.edge_latest_standing_before(
        season, week
    )
    if not standing_row:
        summary["edge_recommendation"] = "skipped (no season standing set yet)"
        return
    standing = edge_core.Standing(standing_row["standing_bucket"])
    pool_size = standing_row["pool_size"]
    budget_total = edge_core.deviation_budget(standing, pool_size)

    candidates: list[tuple[dict, str, str, edge_core.DevigResult, float]] = []
    no_data: list[tuple[dict, str, str, edge_core.DevigResult]] = []
    for g in games:
        o = wk_odds.get(g["game_id"])
        if not o or o.get("spread") is None:
            continue
        home_fav = o["spread"] <= 0
        fav_team = g["home"] if home_fav else g["away"]
        dog_team = g["away"] if home_fav else g["home"]
        ml_fav = o.get("ml_home") if home_fav else o.get("ml_away")
        ml_dog = o.get("ml_away") if home_fav else o.get("ml_home")
        if ml_fav is None or ml_dog is None:
            continue
        d = edge_core.devig_two_way(int(ml_fav), int(ml_dog))
        try:
            fav_norm = normalize_team(fav_team)
        except UnknownTeamError:
            fav_norm = fav_team
        national = store.edge_get_national_pct(season, week, fav_norm)
        if national is None:
            no_data.append((g, fav_team, dog_team, d))
            continue
        bias_value, _, _ = store.edge_get_bias(fav_norm)
        f = edge_core.pool_popularity(national, bias_value)
        candidates.append((g, fav_team, dog_team, d, f))

    ranked = sorted(
        candidates,
        key=lambda c: edge_core.leverage_score(c[3].p, c[4]) or -1.0,
        reverse=True,
    )
    remaining = budget_total
    rows: list[dict[str, Any]] = []
    for g, fav, dog, d, f in ranked:
        rec = edge_core.recommend(d.p, f, budget_remaining=remaining)
        if rec.recommendation is edge_core.Recommendation.FADE:
            remaining -= 1
        rows.append({
            "game_id": g["game_id"], "favorite_team": fav, "underdog_team": dog,
            "p_favorite": round(d.p, 6), "vig": round(d.vig, 6),
            "f_estimate": round(f, 6),
            "leverage": round(rec.leverage, 6) if rec.leverage is not None else None,
            "eligible": rec.eligible, "recommendation": rec.recommendation.value,
            "budget_at_time": budget_total,
        })
    for g, fav, dog, d in no_data:
        rows.append({
            "game_id": g["game_id"], "favorite_team": fav, "underdog_team": dog,
            "p_favorite": round(d.p, 6), "vig": round(d.vig, 6),
            "f_estimate": None, "leverage": None,
            "eligible": edge_core.is_eligible(d.p), "recommendation": "NO_DATA",
            "budget_at_time": budget_total,
        })

    if rows:
        store.edge_upsert_recommendation_log(season, week, rows)
    summary["edge_recommendation"] = f"{len(rows)} games, budget {budget_total} ({standing.value})"


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
