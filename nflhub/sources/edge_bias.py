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


def recompute_all_bias(force: bool = False) -> str:
    """Recompute every team's learned bias from one read of the picks, games, national % and existing biases (it used to
    re-download the whole picks table 32 times and make ~200 single-row requests EVERY 10-minute run -- most of this
    project's Supabase egress). Skipped entirely unless the picks or national-% row counts changed since the last run."""
    sig = f"{store.edge_picks_count()}:{store.edge_national_pct_count()}"
    if not force and store.kv_get("edge_bias_sig") == sig:
        return "unchanged"
    picks = store.edge_all_opponent_picks()
    national = store.edge_national_pct_map()
    games = store.all_games()
    existing = {r["team"]: r for r in store.edge_bias_all()}

    # per (season, week): which opponents imported picks, and how many took each team
    by_week: dict[tuple[int, int], dict] = {}
    for p in picks:
        d = by_week.setdefault((p["season"], p["week"]), {"opponents": set(), "counts": {}})
        d["opponents"].add(p["opponent"])
        d["counts"][p["team_picked"]] = d["counts"].get(p["team_picked"], 0) + 1
    games_by_team: dict[str, set[tuple[int, int]]] = {}
    for g in games:
        for t in (g.get("home"), g.get("away")):
            if t:
                games_by_team.setdefault(t, set()).add((g["season"], g["week"]))

    changed = []
    for team in sorted(TEAMS):
        row = existing.get(team)
        if row and row.get("overridden"):
            continue
        pool_pcts: list[float] = []
        national_pcts: list[float] = []
        for season, week in sorted(games_by_team.get(team, ())):
            d = by_week.get((season, week))
            if not d or not d["opponents"]:
                continue
            nat = national.get((season, week, team))
            if nat is not None:
                pool_pcts.append(d["counts"].get(team, 0) / len(d["opponents"]))
                national_pcts.append(nat)
        result = compute_bias(pool_pcts, national_pcts)
        if (not row or abs((row.get("bias_value") or 0.0) - result.bias_value) > 1e-6
                or (row.get("n_observations") or 0) != result.n_observations):
            changed.append({"team": team, "bias_value": result.bias_value, "n_observations": result.n_observations, "overridden": False})
    store.edge_upsert_bias_many(changed)
    store.kv_set("edge_bias_sig", sig)
    return f"recomputed ({len(changed)} changed)"
