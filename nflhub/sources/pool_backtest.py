"""How the pool optimizer would have done against the actual pool, week by week (Model Trends "Pool optimizer vs. the pool").

For every completed week W of the season, rebuild the card the optimizer would have recommended BEFORE the week using only what was
knowable then, and count how many of its picks were right, next to the week winner's correct count and the user's own:

  * Opponent chalk bias. No per-entrant bias is used until week MIN_PRIOR_WEEKS + 1 (week 4): before that every entrant is treated
    as average. From week 4 on each entrant's bias is estimated from the weeks BEFORE W only (never W or later), exactly the
    information the live optimizer has at that point (pool_mc.estimate_offsets, same shrinkage).
  * The pick-share model's own constants (pool_mc.PICK_A/B/C) are the production values. They were fit on weeks 1-5, so weeks 1-5
    carry a small leak in those three pooled numbers (not in any individual's bias); every later week is clean.
  * Market: nflverse's closing moneylines, de-vigged (what the live optimizer sees minutes before kickoff). National pick %: the
    value the app scraped for that week. Tiebreak habits are not used: the card is the static pre-week recommendation.
  * The card is the optimizer's own (greedy single-flip search on P(win the week)); the week winner's and the user's correct counts
    come straight from that week's final pool tab.

A week is computed once its final "Week N" tab exists and is then frozen in kv `pool_backtest` (its inputs never change afterwards).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from .. import store
from . import edge_sheet, pool_mc
from .edge_teams import UnknownTeamError, normalize_team
from .pool_recommend import _american_devig_home

log = logging.getLogger(__name__)

KEY = "pool_backtest"
MIN_PRIOR_WEEKS = 3        # per-entrant bias is used only once this many earlier weeks of picks exist
N_SIMS = 20000


def _load() -> dict | None:
    raw = store.kv_get(KEY)
    try:
        return json.loads(raw) if raw else None
    except (TypeError, ValueError):
        return None


def _tab(sheet_id: str, week: int) -> dict[str, Any] | None:
    try:
        data = edge_sheet.fetch_week(sheet_id, week, normalize_team)
    except edge_sheet.EdgeSheetUnavailable:
        return None
    picks: dict[str, list[str]] = {}
    for p in data.picks:
        picks.setdefault(p.opponent, []).append(p.team_picked)
    return {"correct": {s.opponent: (s.correct_week or 0) for s in data.standings}, "picks": picks}


def _num(r: dict, k: str) -> float | None:
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return None


def build_week(week: int, tabs: dict[int, dict], rows_by_pair: dict, national_map: dict, season: int, my_name_hint: str,
               n_sims: int = N_SIMS) -> dict | None:
    """One week's record, or None when the week can't be built (no tab, no moneylines, not final)."""
    tab = tabs.get(week)
    if not tab:
        return None
    me_name = next((n for n in tab["correct"] if my_name_hint and my_name_hint.lower() in n.lower()), None)
    if me_name is None:
        return None

    def games_for(w: int) -> list[pool_mc.PoolGame] | None:
        out = []
        for (rw, home, away), r in rows_by_pair.items():
            if rw != w:
                continue
            mh, ma = _num(r, "home_moneyline"), _num(r, "away_moneyline")
            if mh is None or ma is None:
                return None
            pg = pool_mc.PoolGame(r["game_id"], home, away, _american_devig_home(mh, ma), r.get("gameday", ""), _num(r, "total_line"))
            pg.national_fav_pct = national_map.get((season, w, pg.fav))
            out.append(pg)
        return out or None

    games = games_for(week)
    if not games:
        return None   # no moneylines
    # Actual results live OUTSIDE the PoolGame objects: setting PoolGame.home_won would tell the simulation the game is already
    # decided and the optimizer would simply pick every winner.
    winner: dict[str, str | None] = {}
    for g in games:
        r = rows_by_pair.get((week, g.home, g.away), {})
        hs, as_ = _num(r, "home_score"), _num(r, "away_score")
        if hs is None or as_ is None:
            return None   # the week isn't completely final yet
        winner[g.game_id] = g.home if hs > as_ else (g.away if as_ > hs else None)   # None = tie, nobody gets it

    names = [n for n in tab["correct"] if n != me_name]
    offsets: dict[str, float] = {}
    prior = [w for w in sorted(tabs) if w < week]
    bias_used = len(prior) >= MIN_PRIOR_WEEKS
    if bias_used:
        history = []
        for w in prior:
            pgs = games_for(w)
            if not pgs:
                continue
            by_team: dict[str, str] = {}
            for pg in pgs:
                by_team[pg.home] = by_team[pg.away] = pg.game_id
            history.append((pgs, {n: {by_team[t]: t for t in teams if t in by_team} for n, teams in tabs[w]["picks"].items()}))
        offsets = pool_mc.estimate_offsets(history, names)
    opponents = [pool_mc.Entrant(n, offset=offsets.get(n, 0.0)) for n in names]
    me = pool_mc.Entrant(me_name)
    res = pool_mc.optimize(games, opponents, me, n=n_sims, seed=week)

    correct = sum(1 for g in games if winner[g.game_id] is not None and res.card.get(g.game_id) == winner[g.game_id])
    scores = sorted(tab["correct"].values(), reverse=True)
    return {
        "week": week,
        "optimizer_correct": correct,
        "winner_correct": scores[0],
        "my_correct": tab["correct"][me_name],
        "field_mean": round(sum(scores) / len(scores), 2),
        "entrants": len(scores),
        "optimizer_rank": 1 + sum(1 for s in scores if s > correct),      # place by correct picks, before the tiebreaker
        "p_win": round(float(res.p_win), 4),
        "bias_used": bias_used, "prior_weeks": len(prior) if bias_used else 0,
        "upset_picks": sum(1 for g in games if res.card.get(g.game_id) not in (None, g.fav)),
        "games": len(games),
    }


def refresh(cfg, season: int, week: int, fetch_rows, force: bool = False) -> str:
    """Compute any completed week not yet in the stored record. `fetch_rows()` returns nflverse games.csv rows (called only when
    there is something to build). Cheap on the usual run: one tab-exists request per week not yet stored."""
    sheet_id = cfg.edge.sheet_id
    if not sheet_id or not cfg.edge.my_name:
        return "skipped (no pool sheet / name configured)"
    stored = _load()
    if not stored or stored.get("season") != season or force:
        stored = {"season": season, "weeks": {}}
    weeks: dict[str, dict] = stored["weeks"]
    todo = [w for w in range(1, week + 1) if str(w) not in weeks and edge_sheet._tab_exists(sheet_id, w)]
    if not todo:
        return "cached"

    tabs = {w: t for w in range(1, week + 1) if (t := _tab(sheet_id, w))}
    rows_by_pair: dict[tuple[int, str, str], dict] = {}
    for r in fetch_rows():
        if r.get("game_type") != "REG" or str(r.get("season")) != str(season):
            continue
        try:
            rows_by_pair[(int(r["week"]), normalize_team(r["home_team"]), normalize_team(r["away_team"]))] = r
        except (UnknownTeamError, ValueError, KeyError):
            continue
    national_map = store.edge_national_pct_map()
    built = []
    for w in todo:
        rec = build_week(w, tabs, rows_by_pair, national_map, season, cfg.edge.my_name)
        if rec:
            weeks[str(w)] = rec
            built.append(w)
    if built:
        stored["generated"] = datetime.now(timezone.utc).isoformat()
        store.kv_set(KEY, json.dumps(stored))
    return f"built weeks {built}" if built else f"nothing buildable yet ({todo})"
