"""v17: re-run v16's position-group health-gap regression, but weight each Out/Doubtful
player by their own typical SNAP SHARE instead of counting every name equally. v16's raw
headcount found real signal for QB (where "injured" is almost always literally the starter)
but nothing for OL/skill/DL/LB/secondary -- the suspected reason: those groups carry 3-5+
roster spots, so a backup's injury dilutes the count exactly as much as a starter's would.

Verified before building: nflverse publishes a per-player-per-game snap count dataset
(github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.csv,
confirmed live -- offense_snaps/offense_pct/defense_snaps/defense_pct/st_snaps/st_pct per
player per game) starting SEASON 2012 (2010/2011 confirmed 404) -- later than the injury
report's 2009 start, so this version's usable window is 2012-2025, not 2009-2025. No shared
player-ID column with the injury report (injuries key on gsis_id, snap_counts on
pfr_player_id) and nflverse's player-ID crosswalk file is large/slow to fetch for what this
needs -- joined by normalized name + team instead (same convention as this project's own
elwayStaleQb() name matching elsewhere), good enough for an approximate per-player weight,
not exact identity resolution.

Weight = that player's own CUMULATIVE average snap share (offense_pct for offense-side
positions, defense_pct for defense-side) over their own game log STRICTLY BEFORE the target
week (no lookahead, running mean) -- falls back to a position-wide average share (computed
once, pooled) for a player's very first tracked appearance, when no own history exists yet.
A unit's weekly "weighted absence" is the SUM of this weight across every Out/Doubtful
player in that group that week, replacing v16's flat per-player count of 1.0 -- a starting
tackle (~95% snaps) now counts ~9x a rarely-used depth lineman (~10% snaps).

Run: python research/edge_signal_test_v17_unit_health_snapweighted.py
"""
from __future__ import annotations

import csv
import io
import os
import re
import sys
from collections import defaultdict

import numpy as np
import requests

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
from nflhub.sources.edge_teams import UnknownTeamError, normalize_team  # noqa: E402

import importlib.util
spec = importlib.util.spec_from_file_location("v16", os.path.join(HERE, "edge_signal_test_v16_unit_health.py"))
v16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v16)

CACHE_DIR = os.path.join(HERE, "cache")
SNAP_COUNTS_URL = "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.csv"
FIRST_SNAP_SEASON = 2012
SEASONS = range(FIRST_SNAP_SEASON, 2026)
TEST_START_SEASON = 2016

OFFENSE_GROUPS = {"qb", "ol", "skill"}


def _norm_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[.'`]", "", name)
    name = re.sub(r"\s+(jr|sr|ii|iii|iv|v)\.?$", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def load_snap_shares() -> tuple[dict, dict]:
    """Returns (player_games, position_default_share):
    - player_games: {name_norm: [(season, week, side_pct), ...]} sorted chronologically,
      `side_pct` = offense_pct if the player's position that game is offense-side else
      defense_pct.
    - position_default_share: {position: mean side_pct across the whole dataset} -- fallback
      for a player's first-ever appearance, when no own history exists yet."""
    pos_to_group = {}
    for grp, positions in v16.POSITION_GROUPS.items():
        for p in positions:
            pos_to_group[p] = grp

    player_games: dict[str, list] = defaultdict(list)
    pos_sum: dict[str, float] = defaultdict(float)
    pos_n: dict[str, int] = defaultdict(int)

    for season in SEASONS:
        cache_path = os.path.join(CACHE_DIR, f"snap_counts_{season}.csv")
        os.makedirs(CACHE_DIR, exist_ok=True)
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                text = f.read()
        else:
            resp = requests.get(SNAP_COUNTS_URL.format(season=season), timeout=60)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            text = resp.text
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(text)
        for row in csv.DictReader(io.StringIO(text)):
            if row.get("game_type") != "REG":
                continue
            grp = pos_to_group.get(row.get("position"))
            if grp is None:
                continue
            try:
                week = int(row["week"])
                off_pct = float(row["offense_pct"] or 0)
                def_pct = float(row["defense_pct"] or 0)
            except (ValueError, TypeError):
                continue
            side_pct = off_pct if grp in OFFENSE_GROUPS else def_pct
            name = _norm_name(row.get("player", ""))
            if not name:
                continue
            player_games[name].append((season, week, side_pct))
            pos_sum[row["position"]] += side_pct
            pos_n[row["position"]] += 1

    for name in player_games:
        player_games[name].sort()
    position_default_share = {p: pos_sum[p] / pos_n[p] for p in pos_sum if pos_n[p]}
    return player_games, position_default_share


def load_injury_rows() -> list[dict]:
    """Every Out/Doubtful injury-report row in the 6 tracked groups, 2012+ (matches
    snap_counts' own start -- no point loading injury seasons we can't weight)."""
    pos_to_group = {}
    for grp, positions in v16.POSITION_GROUPS.items():
        for p in positions:
            pos_to_group[p] = grp
    out = []
    for season in SEASONS:
        text = v16._cached_fetch_text(v16.INJURIES_URL.format(season=season), f"injuries_{season}.csv")
        for row in csv.DictReader(io.StringIO(text)):
            if row.get("game_type") != "REG":
                continue
            grp = pos_to_group.get(row.get("position"))
            if grp is None or row.get("report_status") not in ("Out", "Doubtful"):
                continue
            try:
                team = normalize_team(row["team"])
                week = int(row["week"])
            except (UnknownTeamError, ValueError, TypeError):
                continue
            out.append({"team": team, "season": season, "week": week, "group": grp,
                        "name": _norm_name(row.get("full_name", ""))})
    return out


def weighted_absence_by_team_week_group(injury_rows: list[dict], player_games: dict, pos_default: dict, pos_to_group: dict) -> dict:
    """{(team, season, week, group): sum of each flagged player's own trailing average snap
    share}. The trailing average only uses games strictly before (season, week) -- binary
    search into that player's sorted game log, same no-lookahead discipline as the rating
    itself."""
    import bisect
    group_default = {}
    for grp, positions in v16.POSITION_GROUPS.items():
        shares = [pos_default[p] for p in positions if p in pos_default]
        group_default[grp] = sum(shares) / len(shares) if shares else 0.2

    out: dict[tuple, float] = defaultdict(float)
    for r in injury_rows:
        games = player_games.get(r["name"])
        weight = None
        if games:
            keys = [(s, w) for s, w, _ in games]
            idx = bisect.bisect_left(keys, (r["season"], r["week"]))
            prior = games[:idx]
            if prior:
                weight = sum(p for _, _, p in prior) / len(prior)
        if weight is None:
            weight = group_default[r["group"]]
        out[(r["team"], r["season"], r["week"], r["group"])] += weight
    return out


def main():
    print("Loading snap counts (2012+)...", file=sys.stderr)
    player_games, pos_default = load_snap_shares()
    print(f"{len(player_games)} distinct tracked players")
    print("Position-average snap share (fallback for a player's first tracked game):")
    for p, v in sorted(pos_default.items()):
        print(f"  {p:4s} {v:.3f}")

    print("\nLoading injury report rows (2012+)...", file=sys.stderr)
    injury_rows = load_injury_rows()
    pos_to_group = {}
    for grp, positions in v16.POSITION_GROUPS.items():
        for p in positions:
            pos_to_group[p] = grp
    weighted = weighted_absence_by_team_week_group(injury_rows, player_games, pos_default, pos_to_group)

    print("\nBuilding per-team-game rating/actual rows (2012+)...", file=sys.stderr)
    # Reuse v16's builder (it already covers 2009-2025 / SEASONS=range(2009,2026)) and just
    # filter to 2012+ here, rather than re-deriving the rating walk a third time -- the
    # rating itself still needs the full 2009+ history to be correctly "warmed up" by 2012,
    # so filtering AFTER building (not narrowing v16.SEASONS) is deliberate.
    all_rows = v16.build_team_game_rows()
    rows = [r for r in all_rows if r["season"] >= FIRST_SNAP_SEASON]
    print(f"{len(rows)} team-game rows (2012-2025)\n")

    team_weeks = sorted({(r["team"], r["season"], r["week"]) for r in rows})
    gaps = v16.build_health_gaps(team_weeks, weighted)
    for r in rows:
        for grp in v16.POSITION_GROUPS:
            r[f"gap_{grp}"] = gaps.get((r["team"], r["season"], r["week"], grp), 0.0)

    print("--- snap-weighted health-gap distribution sanity check ---")
    for grp in v16.POSITION_GROUPS:
        vals = np.array([r[f"gap_{grp}"] for r in rows])
        print(f"  {grp:10s} mean={vals.mean():+.3f} std={vals.std():.3f} "
              f"min={vals.min():+.2f} max={vals.max():+.2f} nonzero={np.mean(vals != 0)*100:.1f}%")

    print("\n=== Step 2 (snap-weighted): does a unit's weighted health gap explain the rating's pregame prediction error? ===")
    for metric, units in v16.METRIC_UNITS.items():
        print(f"\n[{metric}]  residual = actual - pregame_rating  ~  " + " + ".join(f"gap_{u}" for u in units))
        y = np.array([r[f"{metric}_actual"] - r[f"{metric}_pregame"] for r in rows])
        X = np.array([[r[f"gap_{u}"] for u in units] for r in rows])
        v16.ols_with_stats(X, y, [f"gap_{u}" for u in units])


if __name__ == "__main__":
    main()
