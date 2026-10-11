"""v38: NOT SHIPPED -- confirmed null/no better than the already-shipped baseline. Keeping
this script for the record (same convention as v27/v29/v30's own confirmed-null writeups).

The idea: v37 shipped backup_avg_epa as the LIVE estimate for "whoever replaces an Out/
Doubtful starter" -- a real, validated fix over league-wide average, but still a pooled
number standing in for a specific person. The obvious next step was resolving WHO that
person actually is from ESPN's QB depth chart (a prototype `_resolve_live_backup()`, name-
matched by last name, scoped to the team's own current-season roster so a common surname
can't collide) and using THEIR own trailing EPA instead, when they'd built up enough career
sample (tried >= 50 attempts) to trust over the pool average. Verified the resolution
mechanism itself works correctly against the real live case (CHI's depth chart correctly
named Tyson Bagent, 152 attempts, skipping Caleb Williams still listed at index 0).

The backtest below is what killed it: on the 287-game subset where a real identifiable
backup DID have enough sample (66.6% of affected games -- the mechanism would activate
often enough to matter), using their own trailing EPA scored 176/281 (62.63%) vs backup_avg_
epa's own 178/281 (63.35%) -- slightly WORSE, and the paired flip count (5 improved, 7
regressed, z=-0.58) confirms that's just noise, not a real effect either direction. Makes
sense in hindsight: an individual backup's own trailing EPA over 50-150 attempts is a
noisier estimate than the SAME pooled average computed over 46,000 backup attempts, even
though it's more specific to the actual player -- more granular isn't automatically more
accurate once the individual sample is this thin. (This can't be backtested as a true
end-to-end live simulation either way -- ESPN's depth chart has no historical archive, so
this uses the real historical actual-passer identity as an upper-bound stand-in for
"assume live resolution always correctly names the right player.")

Worth another look if ever revisited: shrinking the specific backup's own EPA toward
backup_avg_epa by sample size (empirical-Bayes-style), instead of using their raw number
outright, might recover some real signal without the added variance -- not attempted here.

Run: python research/edge_signal_test_v38_specific_backup_coverage.py
"""
from __future__ import annotations

import csv
import io
import os
import sys
from collections import defaultdict

import pandas as pd
import requests

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
from nflhub.sources import team_ratings as tr  # noqa: E402


def _rebuild_with_attempt_counts(played_games: list[dict], current_season: int):
    """Same chronological walk as qb_quality_gap_data()'s own (and v37's), extended to also
    track each passer's PRIOR attempt count as of right before each game (not just their
    trailing EPA mean) -- the one extra thing this script needs that nothing else exposes."""
    frames = []
    for season in range(tr.FIRST_INJURY_SEASON, current_season + 1):
        try:
            df = tr._passer_game_stats(season, current_season)
        except Exception:
            continue
        df = df.copy()
        df["season"] = season
        frames.append(df)
    passer_df = pd.concat(frames, ignore_index=True)

    games_by_id = {g["game_id"]: g for g in played_games}

    def week_of(gid):
        g = games_by_id.get(gid)
        if g is not None:
            try:
                return int(g["week"])
            except (TypeError, ValueError):
                pass
        try:
            return int(gid.split("_")[1])
        except (IndexError, ValueError):
            return None

    passer_df["week"] = [week_of(gid) for gid in passer_df["game_id"]]
    passer_df = passer_df.dropna(subset=["week"])
    passer_df["week"] = passer_df["week"].astype(int)

    actual_passer: dict[tuple[str, str], str] = {}
    game_team_epa: dict[tuple[str, str], tuple[float, int]] = {}
    for (gid, team), grp in passer_df.groupby(["game_id", "team"]):
        row = grp.loc[grp["attempts"].idxmax()]
        actual_passer[(gid, team)] = row["passer_player_id"]
        game_team_epa[(gid, team)] = (row["epa_sum"], row["attempts"])

    gt_meta = passer_df[["game_id", "team", "season", "week"]].drop_duplicates().sort_values(["season", "week", "game_id", "team"])
    game_order = list(gt_meta.itertuples(index=False, name=None))

    season_attempts: dict[tuple[str, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    last_season_leader: dict[str, str] = {}
    last_season_num: dict[str, int] = {}
    primary_so_far: dict[tuple[str, str], str] = {}
    passer_hist: dict[str, list[float]] = defaultdict(list)
    passer_attempt_total: dict[str, int] = defaultdict(int)
    trailing_epa: dict[tuple[str, str], float | None] = {}
    prior_attempts: dict[tuple[str, str], int] = {}  # (passer_id, game_id) -> attempts BEFORE this game
    league_avg: list[float] = []
    backup_epa_sum, backup_attempts = 0.0, 0
    team_week_to_gid: dict[tuple[str, int, int], str] = {}

    for gid, team, season, week in game_order:
        if last_season_num.get(team) is not None and last_season_num[team] != season and season_attempts[(team, season - 1)]:
            last_season_leader[team] = max(season_attempts[(team, season - 1)].items(), key=lambda kv: kv[1])[0]
        last_season_num[team] = season
        this_season = season_attempts[(team, season)]
        if this_season:
            primary_so_far[(team, gid)] = max(this_season.items(), key=lambda kv: kv[1])[0]
        elif team in last_season_leader:
            primary_so_far[(team, gid)] = last_season_leader[team]
        else:
            primary_so_far[(team, gid)] = actual_passer.get((gid, team))
        team_week_to_gid[(team, season, week)] = gid

        ap = actual_passer.get((gid, team))
        if ap is not None:
            hist = passer_hist[ap]
            prior_attempts[(ap, gid)] = passer_attempt_total[ap]
            trailing_epa[(ap, gid)] = (sum(hist) / len(hist)) if hist else None
            epa_sum, attempts = game_team_epa[(gid, team)]
            if attempts:
                per_play = epa_sum / attempts
                hist.append(per_play)
                passer_attempt_total[ap] += attempts
                league_avg.append(per_play)
                primary_here = primary_so_far[(team, gid)]
                if primary_here is not None and ap != primary_here:
                    backup_epa_sum += epa_sum
                    backup_attempts += attempts
            this_season[ap] += attempts

    league_avg_epa = (sum(league_avg) / len(league_avg)) if league_avg else 0.0
    backup_avg_epa = (backup_epa_sum / backup_attempts) if backup_attempts else league_avg_epa
    return (primary_so_far, trailing_epa, prior_attempts, team_week_to_gid, league_avg_epa,
            backup_avg_epa, actual_passer)


def _score(scatter: list[dict]) -> tuple[int, int]:
    hits = decided = 0
    for r in scatter:
        if r["home_score"] == r["away_score"] or r["delta"] == 0:
            continue
        decided += 1
        actual = "home" if r["home_score"] > r["away_score"] else "away"
        hits += ("home" if r["delta"] > 0 else "away") == actual
    return hits, decided


def main():
    print("Pulling games.csv + building game_log...", file=sys.stderr)
    resp = requests.get(tr.GAMES_URL, timeout=tr.TIMEOUT)
    resp.raise_for_status()
    all_rows = [r for r in csv.DictReader(io.StringIO(resp.text)) if r["game_type"] == "REG"]
    played = [r for r in all_rows if r.get("result") not in ("", "NA", None) and int(r["season"]) >= tr.FIRST_SEASON]
    current_season = max(int(r["season"]) for r in played)

    game_log: list[dict] = []
    tr.compute_ratings(played, current_season, game_log=game_log)

    print("Re-deriving primary/trailing/prior-attempt-count state...", file=sys.stderr)
    (primary_so_far, trailing_epa, prior_attempts, team_week_to_gid, league_avg_epa,
     backup_avg_epa, actual_passer) = _rebuild_with_attempt_counts(played, current_season)

    qb_out_hist = tr._qb_out_doubtful_by_week(current_season)
    skill_out = tr.skill_epa_out_by_team_week(current_season)
    hfa_by_season = tr.home_field_logit_by_season(played)
    streak = tr.team_streak_by_week(played)
    midgame = tr.midgame_qb_injury_by_week(current_season)

    # The v37 "affected" subset: an Out/Doubtful primary that week, real actual passer
    # differs from primary (same actual_passer used to build prior_attempts above, via
    # trailing_epa's own key set -- a team/game pair only gets an entry there once an
    # actual passer played). This constant never shipped to production (see module
    # docstring) -- defined locally rather than on team_ratings, which this script only
    # otherwise reads from, never depends on for anything that changed.
    MIN_N = 50
    sufficient: dict[tuple[str, int], float] = {}   # real backup EPA, where sample is enough
    insufficient_keys: set[tuple[str, int, int]] = set()
    for (team, season, week), is_out in qb_out_hist.items():
        if not is_out:
            continue
        gid = team_week_to_gid.get((team, season, week))
        if gid is None:
            continue
        primary = primary_so_far.get((team, gid))
        if primary is None:
            continue
        actual = actual_passer.get((gid, team))
        if actual is None or actual == primary:
            continue  # healthy week (normal starter played) -- not part of the affected subset
        n = prior_attempts.get((actual, gid), 0)
        key3 = (team, season, week)  # season kept in every key below -- (team, week) alone
        # collides across different seasons for the same franchise.
        if n >= MIN_N:
            real_epa = trailing_epa[(actual, gid)]
            sufficient[key3] = real_epa if real_epa is not None else backup_avg_epa
        else:
            insufficient_keys.add(key3)

    total_affected = len(sufficient) + len(insufficient_keys)
    print(f"\nAffected team-weeks (Out/Doubtful primary, real backup identifiable): {total_affected}")
    print(f"  -- sufficient sample (>= {MIN_N} prior attempts) for the REAL backup: "
          f"{len(sufficient)} ({100*len(sufficient)/total_affected:.1f}% coverage)")
    print(f"  -- insufficient sample (correctly falls back to backup_avg_epa either way): "
          f"{len(insufficient_keys)}")

    # Within the sufficient-sample subset only: backup_avg_epa (today's shipped baseline)
    # vs the real actual passer's own trailing EPA (this feature's target, assuming correct
    # live resolution) -- same qb_diff/delta/compute_backtest_scatter machinery as v37.
    def build_gap(use_real: bool):
        out = {}
        for (team, season, week) in sufficient:
            gid = team_week_to_gid.get((team, season, week))
            if gid is None:
                continue
            primary = primary_so_far.get((team, gid))
            tp = trailing_epa.get((primary, gid))
            tp = tp if tp is not None else league_avg_epa
            out[(team, season, week)] = tp - (sufficient[(team, season, week)] if use_real else backup_avg_epa)
        return out

    gap_pool_avg = build_gap(use_real=False)
    gap_real_backup = build_gap(use_real=True)

    scatters = {
        "backup_avg_epa (today's shipped baseline)": tr.compute_backtest_scatter(
            game_log, qb_out=qb_out_hist, qb_gap=gap_pool_avg, skill_out=skill_out,
            hfa_by_season=hfa_by_season, midgame_qb_injury=midgame, streak=streak),
        "real backup's own trailing EPA (upper bound if resolved correctly)": tr.compute_backtest_scatter(
            game_log, qb_out=qb_out_hist, qb_gap=gap_real_backup, skill_out=skill_out,
            hfa_by_season=hfa_by_season, midgame_qb_injury=midgame, streak=streak),
    }

    sufficient_keys_set = set(sufficient.keys())
    def in_subset(r):
        return (r["home"], r["season"], r["week"]) in sufficient_keys_set or \
               (r["away"], r["season"], r["week"]) in sufficient_keys_set

    print(f"\nOn the {len(sufficient)}-game sufficient-sample subset only:")
    print(f"{'variant':55s} {'hits':>6s} {'n':>6s} {'hit%':>7s}")
    for label, scatter in scatters.items():
        sub = [r for r in scatter if in_subset(r)]
        hits, decided = _score(sub)
        print(f"{label:55s} {hits:6d} {decided:6d} {100*hits/decided:6.2f}%" if decided else f"{label:55s} no decided games")

    # Paired flip analysis, same games, pool-average estimate -> real-backup estimate --
    # is the small hit-rate gap above a real signal or just which few games happened to
    # flip either way (same style as v37's own paired check).
    pool_by_key = {(r["season"], r["week"], r["home"], r["away"]): r
                   for r in scatters["backup_avg_epa (today's shipped baseline)"] if in_subset(r)}
    real_by_key = {(r["season"], r["week"], r["home"], r["away"]): r
                   for r in scatters["real backup's own trailing EPA (upper bound if resolved correctly)"] if in_subset(r)}
    improved = regressed = unchanged = 0
    for k, rp in pool_by_key.items():
        rr = real_by_key.get(k)
        if rr is None or rp["home_score"] == rp["away_score"] or rp["delta"] == 0 or rr["delta"] == 0:
            continue
        actual_w = "home" if rp["home_score"] > rp["away_score"] else "away"
        pool_pick = "home" if rp["delta"] > 0 else "away"
        real_pick = "home" if rr["delta"] > 0 else "away"
        if pool_pick == real_pick:
            unchanged += 1
        elif real_pick == actual_w and pool_pick != actual_w:
            improved += 1
        elif pool_pick == actual_w and real_pick != actual_w:
            regressed += 1
        else:
            unchanged += 1
    print(f"\nPaired flips (pool-average -> real-backup estimate, same games): "
          f"{improved} improved, {regressed} regressed, {unchanged} unchanged pick.")
    n = improved + regressed
    if n >= 10:
        z = (improved - regressed) / (n ** 0.5)
        print(f"Sign test on the {n} flips: z={z:+.2f} ({'favors real-backup' if z > 0 else 'favors pool-average'})")
    elif n:
        print(f"Only {n} flips -- too few for a meaningful sign test.")


if __name__ == "__main__":
    main()
