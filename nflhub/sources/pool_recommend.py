"""Builds the pool Monte Carlo's inputs from the app's own data and stores the recommendation.

Replaces the FADE / CHALK / NO_PLAY tags: every game now gets the engine's pick plus its impact rating -- the drop in
simulated weekly win chance if that one game were on the other side while every other pick stays optimized (nflhub/sources/pool_mc.py has the model and the pool's payout rules).

Inputs, all market-side (never the Power Rankings model):
  * market win probability per game -- de-vigged moneyline from the odds table;
  * the pool: every entrant's season-to-date wins and past-week picks, read from the pool sheet's completed
    "Week N" tabs (their chalk tendency and skill are estimated from those);
  * the national pick % per favorite (edge_national) -- a predictor of what the field will pick;
  * the posted total of the Monday-night game for the tiebreaker.
Skips quietly (soft-fail) when any of that is missing, e.g. week 1 has no pool history yet.

Recomputed at most every RECOMPUTE_MINUTES (the simulation takes a few seconds). Picks lock by game day (all of
Sunday's games at the first Sunday kickoff), so once a day's games have kicked off they are LOCKED: the user's
own pick on them is fixed (from the app's saved picks), finished games enter as known results, and only the
still-open games and the tiebreaker are re-optimized -- the sequential mode, minus the field's revealed picks
the field's revealed picks come from pool_live (the merged live-picks store: the optional "Week N live" sheet tab
and, later, a direct CBS fetch) when a scrape captured them after the day's first kickoff; without it they are predicted). Stored in kv
`pool_recommendation`; stops updating once every game of the week is final.

The user's own picks: the CBS pool is the official record, so the user's pick on a locked game is read from the
live tab (CBS) -- the in-app pick is only a clearly-labelled fallback, never assumed equal to CBS. Any pick that
differs between the app and CBS, is on CBS but not saved here, or is missing from CBS after lock is flagged per game.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any


from .. import store
from . import edge_core, edge_sheet, pool_live, pool_mc
from .edge_teams import UnknownTeamError, normalize_team

log = logging.getLogger(__name__)

POOL_REC_KEY = "pool_recommendation"
SCHEMA_VERSION = 1
RECOMPUTE_MINUTES = 45


def _american_devig_home(ml_home: float, ml_away: float) -> float:
    ph = edge_core.american_to_implied_prob(int(ml_home))
    pa = edge_core.american_to_implied_prob(int(ml_away))
    return ph / (ph + pa)


def _load_stored() -> dict | None:
    raw = store.kv_get(POOL_REC_KEY)
    try:
        return json.loads(raw) if raw else None
    except (TypeError, ValueError):
        return None


def _fetch_history(sheet_id: str, week: int) -> dict[int, dict[str, Any]]:
    """{w: {"standings": {name: (weekly_correct, season_correct)}, "picks": {name: [team, ...]}}} for every
    completed pool tab before `week`. Tabs that don't exist yet are simply absent."""
    out: dict[int, dict[str, Any]] = {}
    for w in range(1, week):
        try:
            data = edge_sheet.fetch_week(sheet_id, w, normalize_team)
        except edge_sheet.EdgeSheetUnavailable:
            continue
        picks: dict[str, list[str]] = {}
        for p in data.picks:
            picks.setdefault(p.opponent, []).append(p.team_picked)
        out[w] = {"standings": {s.opponent: (s.correct_week or 0, s.correct_season or 0) for s in data.standings}, "picks": picks}
    return out


def build(cfg, season: int, week: int, games: list[dict[str, Any]], wk_odds: dict[str, Any], nfl_rows: list[dict[str, str]],
          n_sims: int = pool_mc.N_SIMS) -> dict | None:
    """Returns the recommendation dict, or None (with a reason via the second return of `reason`) if it can't run."""
    history = _fetch_history(cfg.edge.sheet_id, week)
    if not history:
        raise RuntimeError("no completed pool tabs yet")
    last_w = max(history)
    standings = history[last_w]["standings"]
    me_name = next((n for n in standings if cfg.edge.my_name and cfg.edge.my_name.lower() in n.lower()), None)
    if me_name is None:
        raise RuntimeError("your name isn't in the pool sheet")

    rows_by_pair: dict[tuple[int, str, str], dict[str, str]] = {}
    for r in nfl_rows:
        if r.get("game_type") != "REG" or str(r.get("season")) != str(season):
            continue
        try:
            rows_by_pair[(int(r["week"]), normalize_team(r["home_team"]), normalize_team(r["away_team"]))] = r
        except (UnknownTeamError, ValueError, KeyError):
            continue

    def num(r, k):
        try:
            return float(r[k])
        except (KeyError, ValueError, TypeError):
            return None

    # ---- past weeks (for each entrant's chalk tendency) ----
    hist_games: list[tuple[list[pool_mc.PoolGame], dict[str, dict[str, str]]]] = []
    for w, h in history.items():
        pgs = []
        for (rw, home, away), r in rows_by_pair.items():
            if rw != w:
                continue
            mh, ma = num(r, "home_moneyline"), num(r, "away_moneyline")
            if mh is None or ma is None:
                continue
            pg = pool_mc.PoolGame(r["game_id"], home, away, _american_devig_home(mh, ma), r.get("gameday", ""), num(r, "total_line"))
            pg.national_fav_pct = store.edge_get_national_pct(season, w, pg.fav)
            pgs.append(pg)
        by_team = {}
        for pg in pgs:
            by_team[pg.home] = pg.game_id
            by_team[pg.away] = pg.game_id
        picks_by_name = {name: {by_team[t]: t for t in teams if t in by_team} for name, teams in h["picks"].items()}
        hist_games.append((pgs, picks_by_name))
    names = [n for n in standings if n != me_name]
    offsets = pool_mc.estimate_offsets(hist_games, names)

    opponents = [pool_mc.Entrant(n, offset=offsets.get(n, 0.0)) for n in names]
    me = pool_mc.Entrant(me_name)

    # ---- this week's games ----
    pgs: list[pool_mc.PoolGame] = []
    national_missing = 0
    now = datetime.now(timezone.utc)

    def kickoff_dt(g):
        try:
            dt = datetime.fromisoformat(str(g.get("kickoff")))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return None

    games_not_started = sum(1 for g in games if (kickoff_dt(g) or now) > now)
    # a game day locks when its first game kicks off
    first_by_day: dict[str, datetime] = {}
    for g in games:
        r = rows_by_pair.get((week, normalize_team(g["home"]), normalize_team(g["away"])), {}) if g.get("home") and g.get("away") else {}
        k = kickoff_dt(g)
        day = r.get("gameday") or str(g.get("kickoff", ""))[:10]
        if k and (day not in first_by_day or k < first_by_day[day]):
            first_by_day[day] = k
    my_saved = store.get_pickem_picks(week)
    live = pool_live.load(season, week)
    # entrant names can differ in case/spacing between the sheet and CBS: match on a normalized key
    _norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()  # noqa: E731
    live_by_key = {_norm(n): teams for n, teams in (live.picks.items() if live else [])}
    live_tb_by_key = {_norm(n): v for n, v in (live.tiebreaks.items() if live else [])}
    live_me = live_by_key.get(_norm(me_name)) if live else None
    me_locked_picks: dict[str, str | None] = {}
    pick_info: dict[str, dict[str, Any]] = {}
    revealed_for_game: dict[str, dict[str, str | None]] = {}   # game_id -> {opponent name: team or None}
    for g in games:
        o = wk_odds.get(g["game_id"]) or {}
        try:
            home, away = normalize_team(g["home"]), normalize_team(g["away"])
        except UnknownTeamError:
            continue
        r = rows_by_pair.get((week, home, away), {})
        # the live odds row loses its moneylines once a game is final; nflverse keeps the closing ones
        ml_home = o.get("ml_home") if o.get("ml_home") is not None else num(r, "home_moneyline")
        ml_away = o.get("ml_away") if o.get("ml_away") is not None else num(r, "away_moneyline")
        if ml_home is None or ml_away is None:
            continue
        lock_key = r.get("gameday") or str(g.get("kickoff", ""))[:10]
        pg = pool_mc.PoolGame(g["game_id"], g["home"], g["away"], _american_devig_home(ml_home, ml_away), lock_key, num(r, "total_line"))
        pg.locked = lock_key in first_by_day and first_by_day[lock_key] <= now
        if g.get("state") == "post" and g.get("home_score") is not None and g.get("away_score") is not None \
                and g["home_score"] != g["away_score"]:
            pg.home_won = g["home_score"] > g["away_score"]
        # ---- official (CBS) picks from the live tab, when one was captured after this game day locked ----
        day_first = first_by_day.get(lock_key)
        cbs_fresh = bool(live and (live.fresh(day_first) or live.consistent_with(games_not_started)))
        teams_in_game = {pg.home, pg.away}
        app_pick = (my_saved.get(g["game_id"]) or {}).get("pick")
        cbs_pick = next((t for t in (live_me or []) if t in teams_in_game), None) if live_me is not None else None
        flag = None
        if live_me is not None:
            if cbs_pick and app_pick and cbs_pick != app_pick:
                flag = "differs"
            elif cbs_pick and not app_pick:
                flag = "cbs_only"
            elif app_pick and not cbs_pick and cbs_fresh and pg.locked:
                flag = "missing_on_cbs"
        pick_info[g["game_id"]] = {"app_pick": app_pick, "cbs_pick": cbs_pick, "flag": flag,
                                   "source": "cbs" if cbs_pick else ("app" if app_pick else None)}
        if pg.locked:
            if cbs_pick:
                me_locked_picks[g["game_id"]] = cbs_pick
            elif cbs_fresh:
                me_locked_picks[g["game_id"]] = None          # CBS was captured after lock and shows no pick
            else:
                me_locked_picks[g["game_id"]] = app_pick       # unverified fallback (None = no pick known)
            if cbs_fresh:
                revealed_for_game[g["game_id"]] = {k: next((t for t in teams if t in teams_in_game), None) for k, teams in live_by_key.items()}
        pg.national_fav_pct = store.edge_get_national_pct(season, week, normalize_team(pg.fav))
        if pg.national_fav_pct is None:
            national_missing += 1
        pgs.append(pg)
    if not pgs:
        raise RuntimeError("no games with moneylines yet")

    if not [g for g in pgs if g.home_won is None]:
        raise RuntimeError("every game of the week is final")
    me.picks = me_locked_picks
    for gid, by_name in revealed_for_game.items():
        for opp in opponents:
            if _norm(opp.name) in by_name:
                opp.picks[gid] = by_name[_norm(opp.name)]
    res = pool_mc.optimize(pgs, opponents, me, n=n_sims)
    allfav = {g.game_id: g.fav for g in pgs}
    base = pool_mc.evaluate_card(pgs, opponents, me, allfav, round(res.guess), n=n_sims)

    out_games: dict[str, dict[str, Any]] = {}
    for g in pgs:
        d = res.per_game[g.game_id]
        # impact rating: win chance with every other game optimized, minus win chance with just this game on the other side
        impact = (res.p_win - d["p_win_if_flipped"]) * 100
        other = g.home if d["pick"] == g.away else g.away
        out_games[g.game_id] = {
            "pick": d["pick"], "other": other, "favorite": g.fav, "contrarian": d["contrarian"],
            "p_favorite": round(g.p_fav, 4), "pool_favorite_share": round(d["pool_fav_share"], 4),
            "impact_pts": round(impact, 2), "p_win_if_flipped": round(d["p_win_if_flipped"], 4),
            "national_pct": g.national_fav_pct, "locked": g.locked, "final": g.home_won is not None,
            **pick_info.get(g.game_id, {}),
        }
    return {
        "schema": SCHEMA_VERSION, "season": season, "week": week, "generated": datetime.now(timezone.utc).isoformat(),
        "n_sims": n_sims,
        "p_win": round(res.p_win, 4),
        "expected_correct": round(res.expected_correct, 2), "tiebreak_guess": int(round(res.guess)),
        "all_favorites": {"p_win": round(base["p_win"], 4),
                          "expected_correct": round(base["expected_correct"], 2)},
        "contrarian_picks": sum(1 for v in out_games.values() if v["contrarian"] and not v["locked"]),
        "open_games": sum(1 for v in out_games.values() if not v["locked"]),
        "my_tiebreak": live_tb_by_key.get(_norm(me_name)),
        "cbs": {"available": live is not None, "captured_at": live.captured_at.isoformat() if live and live.captured_at else None,
                "sources": live.sources if live else {}, "conflicts": len(live.conflicts) if live else 0,
                "opponents_revealed": sorted(revealed_for_game),
                "entrants_matched": (sum(1 for o in opponents if _norm(o.name) in live_by_key) + (1 if live_me is not None else 0)) if live else 0,
                "entrants": len(opponents) + 1,
                "pick_flags": sum(1 for v in out_games.values() if v.get("flag")),
                "unverified_locked": sum(1 for v in out_games.values() if v["locked"] and v.get("source") != "cbs")},
        "games": out_games,
        "inputs": {"entrants": len(opponents) + 1, "history_weeks": sorted(history), "national_missing": national_missing},
    }


def refresh(cfg, season: int, week: int, games: list[dict[str, Any]], wk_odds: dict[str, Any], first_kickoff: datetime | None,
            fetch_rows, force: bool = False) -> str:
    """Gate + run + store. `fetch_rows()` returns nflverse games.csv rows (only called when a run is due)."""
    stored = _load_stored()
    live_now = pool_live.load(season, week)
    sig = {"finals": sum(1 for g in games if g.get("state") == "post"), "live_updated": live_now.updated if live_now else None}
    same_week = stored and stored.get("season") == season and stored.get("week") == week and stored.get("schema") == SCHEMA_VERSION
    if same_week and not force and stored.get("sig") == sig:
        try:
            if datetime.now(timezone.utc) - datetime.fromisoformat(stored["generated"]) < timedelta(minutes=RECOMPUTE_MINUTES):
                return "cached"
        except (KeyError, ValueError):
            pass
    if not cfg.edge.sheet_id:
        return "skipped (no pool sheet configured)"
    try:
        rec = build(cfg, season, week, games, wk_odds, fetch_rows())
    except RuntimeError as exc:
        return f"skipped ({exc})"
    rec["sig"] = sig
    store.kv_set(POOL_REC_KEY, json.dumps(rec))
    return (f"built (P(win) {rec['p_win']:.1%} vs {rec['all_favorites']['p_win']:.1%} all-favorites, "
            f"{rec['contrarian_picks']} contrarian)")
