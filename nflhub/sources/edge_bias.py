"""Wires edge_core.compute_bias to the store. Ported from pickem-edge's bias/pipeline.py.

Recomputes all 32 real teams on each refresh (cheap) rather than tracking which teams a
given paste touched — simpler, no staleness bugs. A team with edge_bias.overridden=true is
left alone until the user clears the override in the bias editor.

Deliberately NOT scoped to just teams seen in edge_opponent_pick: a team nobody in the pool
ever picks is a real, meaningful data point (the pool avoids it even more than the national
rate would suggest), not an absence of one. opponent_pool_pct_by_week() already handles
this correctly (0 picks / N opponents = a real 0% week), so the only fix needed was to
actually ask about every team, not just the ones with at least one pick.
"""

from __future__ import annotations

from .. import store
from .edge_core import compute_bias
from .edge_teams import TEAMS


def opponent_pool_pct_by_week(team: str) -> list[tuple[int, int, float]]:
    """(season, week, pool_pct) for every week this team played AND that week had at
    least one opponent pick imported. A week with imports but zero picks for this team
    is a real 0% observation; a week with no imports at all is excluded, not treated as 0%.
    """
    games = store.edge_games_for_team(team)
    picks = store.edge_all_opponent_picks()

    by_week: dict[tuple[int, int], dict] = {}
    for p in picks:
        key = (p["season"], p["week"])
        d = by_week.setdefault(key, {"opponents": set(), "team_count": 0})
        d["opponents"].add(p["opponent"])
        if p["team_picked"] == team:
            d["team_count"] += 1

    out: list[tuple[int, int, float]] = []
    for g in games:
        key = (g["season"], g["week"])
        d = by_week.get(key)
        if not d or not d["opponents"]:
            continue
        out.append((g["season"], g["week"], d["team_count"] / len(d["opponents"])))
    return sorted(out)


def recompute_bias_for_teams(teams: set[str]) -> None:
    for team in sorted(teams):
        _, _, overridden = store.edge_get_bias(team)
        if overridden:
            continue
        weekly = opponent_pool_pct_by_week(team)
        pool_pcts: list[float] = []
        national_pcts: list[float] = []
        for season, week, pct in weekly:
            national = store.edge_get_national_pct(season, week, team)
            if national is not None:
                pool_pcts.append(pct)
                national_pcts.append(national)
        result = compute_bias(pool_pcts, national_pcts)
        store.edge_upsert_bias(team, result.bias_value, result.n_observations, overridden=False)


def recompute_all_bias() -> None:
    recompute_bias_for_teams(TEAMS)
