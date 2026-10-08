"""v37: the LIVE QB-health gate (team_ratings.py refresh()'s qb_gap_upcoming) estimates
"whoever replaces an Out/Doubtful starter" using league-wide average EPA/dropback as the
stand-in for the unknown backup's performance. Checked the actual numbers (same garbage-
time-excluded window every other EPA figure in this project uses):

    league-wide average (all passers): +0.035 EPA/play, n=271,451 attempts
    starters only:                     +0.056 EPA/play, n=225,396 attempts
    backups only:                      -0.070 EPA/play, n= 46,055 attempts

League average is really "average STARTER" (83% of all attempts belong to starters) --
nowhere close to what an actual backup does. A struggling-but-real starter (Caleb Williams,
trailing EPA roughly -0.003 to -0.013 over his career) compared against that inflated
"backup" estimate looks like an UPGRADE when he's hurt, which is backwards.

This does NOT require refitting QB_QUALITY_GAP_WEIGHT (research/edge_signal_test_v26). That
weight was fit against the HISTORICAL qb_gap, which already uses the real backup's own
trailing EPA in hindsight (qb_quality_gap_data()'s `ta`) -- never a guess. The weight's
meaning ("delta-equivalent units per unit of TRUE quality gap") doesn't change based on how
well we can ESTIMATE that gap live, before the backup has actually played. The bug is
specifically the LIVE/upcoming-week estimate, which has no choice but to guess.

So the right test isn't a new joint fit -- it's: for every HISTORICAL game where a primary
was flagged Out/Doubtful that week (same gate the live system uses, _qb_out_doubtful_by_
week), simulate what the LIVE system would have guessed BEFORE knowing who'd actually play,
under the OLD estimate (league average) vs the NEW one (backup average), and compare which
one's resulting `delta` actually predicted the real outcome better -- using the EXISTING,
already-fit QB_QUALITY_GAP_WEIGHT and the exact same compute_backtest_scatter() production
code every other backtest in this project runs through, just with one input swapped. The
real hindsight-`ta` version (today's displayed historical/backtest numbers) is included too,
as a ceiling: it's the best any estimate could possibly do, since it already knows who
actually played and how.

Run: python research/edge_signal_test_v37_qb_backup_baseline.py
"""
from __future__ import annotations

import csv
import io
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
from nflhub.sources import team_ratings as tr  # noqa: E402


def _rebuild_primary_and_trailing(played_games: list[dict], current_season: int):
    """Re-derives primary_so_far {(team, gid): passer_id} and trailing_epa {(passer_id,
    gid): EPA/dropback} -- the two local-only quantities qb_quality_gap_data() computes but
    doesn't return. Deliberately mirrors that function's own loop exactly (same source data,
    same chronological walk) so results here are directly comparable to production, not a
    drifted reimplementation. Also returns backup_avg_epa, the new estimate this test is
    actually about."""
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
    trailing_epa: dict[tuple[str, str], float | None] = {}
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
            trailing_epa[(ap, gid)] = (sum(hist) / len(hist)) if hist else None
            epa_sum, attempts = game_team_epa[(gid, team)]
            if attempts:
                per_play = epa_sum / attempts
                hist.append(per_play)
                league_avg.append(per_play)
                primary_here = primary_so_far[(team, gid)]
                if primary_here is not None and ap != primary_here:
                    backup_epa_sum += epa_sum
                    backup_attempts += attempts
            this_season[ap] += attempts

    league_avg_epa = (sum(league_avg) / len(league_avg)) if league_avg else 0.0
    backup_avg_epa = (backup_epa_sum / backup_attempts) if backup_attempts else league_avg_epa
    return primary_so_far, trailing_epa, team_week_to_gid, league_avg_epa, backup_avg_epa, backup_attempts


def _live_style_gap(qb_out: dict, primary_so_far: dict, trailing_epa: dict,
                     team_week_to_gid: dict, league_avg_epa: float, backup_estimate: float) -> dict:
    """Simulates the LIVE system's own estimate -- tp(primary) - backup_estimate -- for
    every (team, season, week) the historical Out/Doubtful gate actually flagged. Absent
    key = 0.0 gap, same "healthy week" convention as production."""
    out: dict[tuple[str, int, int], float] = {}
    for (team, season, week), is_out in qb_out.items():
        if not is_out:
            continue
        gid = team_week_to_gid.get((team, season, week))
        if gid is None:
            continue
        primary = primary_so_far.get((team, gid))
        if primary is None:
            continue
        tp = trailing_epa.get((primary, gid), league_avg_epa)
        if tp is None:
            tp = league_avg_epa
        out[(team, season, week)] = tp - backup_estimate
    return out


def _score(scatter: list[dict]) -> tuple[int, int]:
    """(hits, decided) over every non-tie, non-pick'em graded game."""
    hits = decided = 0
    for r in scatter:
        if r["home_score"] == r["away_score"]:
            continue
        if r["delta"] == 0:
            continue
        decided += 1
        actual = "home" if r["home_score"] > r["away_score"] else "away"
        picked = "home" if r["delta"] > 0 else "away"
        if picked == actual:
            hits += 1
    return hits, decided


def main():
    print("Pulling games.csv + building game_log (same pipeline as refresh_historical)...", file=sys.stderr)
    resp = requests.get(tr.GAMES_URL, timeout=tr.TIMEOUT)
    resp.raise_for_status()
    all_rows = [r for r in csv.DictReader(io.StringIO(resp.text)) if r["game_type"] == "REG"]
    played = [r for r in all_rows if r.get("result") not in ("", "NA", None) and int(r["season"]) >= tr.FIRST_SEASON]
    current_season = max(int(r["season"]) for r in played)

    game_log: list[dict] = []
    tr.compute_ratings(played, current_season, game_log=game_log)

    print("Re-deriving primary_so_far / trailing_epa (mirrors qb_quality_gap_data's own loop)...", file=sys.stderr)
    primary_so_far, trailing_epa, team_week_to_gid, league_avg_epa, backup_avg_epa, backup_n = \
        _rebuild_primary_and_trailing(played, current_season)
    print(f"league_avg_epa={league_avg_epa:+.4f}  backup_avg_epa={backup_avg_epa:+.4f}  (n={backup_n} backup attempts)")

    print("Building qb_out gate + HFA/skill/streak/midgame context (all production functions)...", file=sys.stderr)
    qb_out_hist = tr._qb_out_doubtful_by_week(current_season)
    gap_hindsight, _current_primary, _current_trailing, _league_avg, _passer_names = tr.qb_quality_gap_data(played, current_season)
    skill_out = tr.skill_epa_out_by_team_week(current_season)
    hfa_by_season = tr.home_field_logit_by_season(played)
    streak = tr.team_streak_by_week(played)
    midgame = tr.midgame_qb_injury_by_week(current_season)

    gap_live_old = _live_style_gap(qb_out_hist, primary_so_far, trailing_epa, team_week_to_gid, league_avg_epa, league_avg_epa)
    gap_live_new = _live_style_gap(qb_out_hist, primary_so_far, trailing_epa, team_week_to_gid, league_avg_epa, backup_avg_epa)

    print(f"\n{sum(1 for v in gap_live_old.values() if v)} team-weeks with a nonzero live-style QB gap "
          f"(an Out/Doubtful primary that week), out of {len(qb_out_hist)} total flagged team-weeks.\n")

    variants = {
        "delta_raw (no QB term at all -- simple baseline)": None,
        "OLD: live estimate = league average": gap_live_old,
        "NEW: live estimate = backup average": gap_live_new,
        "ORACLE: real hindsight backup EPA (today's historical chart)": gap_hindsight,
    }

    scatters = {}
    for label, gap in variants.items():
        scatters[label] = tr.compute_backtest_scatter(
            game_log, qb_out=qb_out_hist, qb_gap=gap, skill_out=skill_out,
            hfa_by_season=hfa_by_season, midgame_qb_injury=midgame, streak=streak,
        )

    print(f"{'variant':62s} {'hits':>6s} {'n':>6s} {'hit%':>7s}")
    for label, scatter in scatters.items():
        if label.startswith("delta_raw"):
            # delta_raw never includes the QB term -- score it on delta_raw directly.
            hits = decided = 0
            for r in scatter:
                if r["home_score"] == r["away_score"] or r["delta_raw"] == 0:
                    continue
                decided += 1
                actual = "home" if r["home_score"] > r["away_score"] else "away"
                picked = "home" if r["delta_raw"] > 0 else "away"
                hits += picked == actual
        else:
            hits, decided = _score(scatter)
        print(f"{label:62s} {hits:6d} {decided:6d} {100*hits/decided:6.2f}%")

    # The subset that actually matters: games where SOMEONE's primary was flagged Out/
    # Doubtful that week -- everywhere else, old and new estimates are identical (both 0).
    affected_keys = {k for k, v in gap_live_old.items() if v} | {k for k, v in gap_live_new.items() if v}
    def affected(r):
        return (r["home"], r["season"], r["week"]) in affected_keys or (r["away"], r["season"], r["week"]) in affected_keys

    print(f"\nSame comparison, restricted to the {sum(1 for r in scatters['OLD: live estimate = league average'] if affected(r))} "
          f"games actually affected by this change:")
    print(f"{'variant':62s} {'hits':>6s} {'n':>6s} {'hit%':>7s}")
    for label, scatter in scatters.items():
        sub = [r for r in scatter if affected(r)]
        if label.startswith("delta_raw"):
            hits = decided = 0
            for r in sub:
                if r["home_score"] == r["away_score"] or r["delta_raw"] == 0:
                    continue
                decided += 1
                actual = "home" if r["home_score"] > r["away_score"] else "away"
                picked = "home" if r["delta_raw"] > 0 else "away"
                hits += picked == actual
        else:
            hits, decided = _score(sub)
        print(f"{label:62s} {hits:6d} {decided:6d} {100*hits/decided:6.2f}%")

    # Paired flip analysis, old estimate -> new estimate, on the affected subset only
    # (same games, same QB_QUALITY_GAP_WEIGHT -- only the backup-quality INPUT differs).
    old_sub = {(r["season"], r["week"], r["home"], r["away"]): r for r in scatters["OLD: live estimate = league average"] if affected(r)}
    new_sub = {(r["season"], r["week"], r["home"], r["away"]): r for r in scatters["NEW: live estimate = backup average"] if affected(r)}
    improved = regressed = unchanged = 0
    for key, r_old in old_sub.items():
        r_new = new_sub.get(key)
        if r_new is None or r_old["home_score"] == r_old["away_score"]:
            continue
        if r_old["delta"] == 0 or r_new["delta"] == 0:
            continue
        actual = "home" if r_old["home_score"] > r_old["away_score"] else "away"
        old_pick = "home" if r_old["delta"] > 0 else "away"
        new_pick = "home" if r_new["delta"] > 0 else "away"
        if old_pick == new_pick:
            unchanged += 1
        elif new_pick == actual and old_pick != actual:
            improved += 1
        elif old_pick == actual and new_pick != actual:
            regressed += 1
        else:
            unchanged += 1  # flipped pick, but neither was right anyway
    print(f"\nPaired flips (old estimate -> new estimate, same games): "
          f"{improved} improved, {regressed} regressed, {unchanged} unchanged pick.")
    if improved + regressed >= 10:
        n = improved + regressed
        z = (improved - regressed) / (n ** 0.5)
        print(f"Sign test on the {n} flips: z={z:+.2f} ({'favors NEW' if z > 0 else 'favors OLD'})")

    # Hit rate alone was never the bar the shipped QB/skill health term was held to --
    # compute_calibration_stats' own docstring: those adjustments were expected to come back
    # statistically flat on hit rate, and were shipped anyway for CALIBRATION (does the
    # adjustment reduce confidence MORE on delta_raw's misses than on its hits?). Same bar,
    # same subset, OLD vs NEW estimate.
    print("\nCalibration check (same bar the shipped QB/skill term was held to): on the "
          "affected subset, does |delta| shrink vs. |delta_raw| more often on delta_raw's "
          "OWN misses than on its own hits?")
    for label in ("OLD: live estimate = league average", "NEW: live estimate = backup average"):
        sub = [r for r in scatters[label] if affected(r) and r["home_score"] != r["away_score"]
               and r["delta_raw"] != 0 and r["delta"] != 0]
        raw_pick = lambda r: "home" if r["delta_raw"] > 0 else "away"
        winner = lambda r: "home" if r["home_score"] > r["away_score"] else "away"
        misses = [r for r in sub if raw_pick(r) != winner(r)]
        hits = [r for r in sub if raw_pick(r) == winner(r)]
        def reduced_rate(rows):
            n = len(rows)
            return (sum(1 for r in rows if abs(r["delta"]) < abs(r["delta_raw"])), n)
        miss_reduced, miss_n = reduced_rate(misses)
        hit_reduced, hit_n = reduced_rate(hits)
        miss_rate = miss_reduced / miss_n if miss_n else float("nan")
        hit_rate = hit_reduced / hit_n if hit_n else float("nan")
        z = p = None
        if miss_n >= 10 and hit_n >= 10:
            p_pool = (miss_reduced + hit_reduced) / (miss_n + hit_n)
            se = (p_pool * (1 - p_pool) * (1 / miss_n + 1 / hit_n)) ** 0.5
            if se > 0:
                z = (miss_rate - hit_rate) / se
        print(f"  {label}:")
        print(f"    reduced on misses: {miss_reduced}/{miss_n} ({100*miss_rate:.1f}%)   "
              f"reduced on hits: {hit_reduced}/{hit_n} ({100*hit_rate:.1f}%)   "
              f"z={z:+.2f}" if z is not None else "    (n too small for a z-test)")


if __name__ == "__main__":
    main()
