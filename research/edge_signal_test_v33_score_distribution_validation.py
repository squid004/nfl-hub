"""v33: large-scale validation of v32's score-distribution methodology across every 2015-2025
game (2767 games, the same modern-PAT-era window the historical score-frequency table itself
covers -- testing against pre-2015 games would be testing a mismatched scoring regime anyway).

Tightens one thing from v32's quick 6-game demo: the favorite/underdog bias there was a
single flat average over ALL data (a small lookahead issue for rigorous validation -- an
early game's "correction" partly came from its own future). Made walk-forward here
(cumulative-to-date average, no shrinkage needed since v32 already found no trend in it,
just noise) -- home/away bias was already no-lookahead (trailing-5-shrunk-toward-to-date).
The historical score-frequency table itself stays the full 2015-2025 window for every test
game (a conscious simplification, not re-derived walk-forward per game) -- it's a large,
slowly-changing reference distribution characterizing "the PAT-distance era," not a fitted
parameter chasing game outcomes, and the shape is stable enough within era that a walk-
forward version would cost a lot of engineering for an expected tiny difference.

Three checks, same rigor as everything else this session:
  1. point accuracy: MAE/bias of the distribution's mean vs. actual score (should roughly
     match or improve on the known ~7.7 baseline, not regress it).
  2. does Step 2 (historical reweighting) actually improve the distribution, not just make it
     "look like real football scores" -- compares average log-likelihood of the ACTUAL score
     under Step-1-only (raw Normal) vs. the full Step-1+Step-2 pipeline.
  3. win-probability calibration at scale (should be ~exact by construction; confirms the
     bisection step holds up across thousands of games, not just the 6-game demo).

Run: python research/edge_signal_test_v33_score_distribution_validation.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import edge_signal_test_v9_points_prediction as v9  # noqa: E402
import edge_signal_test_v32_score_distribution as v32  # noqa: E402

sys.path.insert(0, os.path.join(HERE, ".."))
from nflhub import config, store  # noqa: E402
from nflhub.sources import team_ratings  # noqa: E402


def fav_dog_bias_walk_forward(by_game: dict) -> dict[str, float]:
    """{game_id: bias correction for the MODEL-favored side that game} computed cumulative-
    to-date (no lookahead) -- the complementary {game_id: dog_bias} is just its negative
    (v32 found fav_bias == -dog_bias to 3 decimals, a symmetric split of a zero-sum
    residual, so one running average suffices for both)."""
    # process games in chronological order
    ordered = sorted(
        ((sides["home"]["season"], sides["home"]["week"], gid, sides) for gid, sides in by_game.items()
         if "home" in sides and "away" in sides),
        key=lambda t: (t[0], t[1])
    )
    fav_sum, fav_n = 0.0, 0
    out: dict[str, float] = {}
    for season, week, gid, sides in ordered:
        out[gid] = (fav_sum / fav_n) if fav_n else 0.0
        h, a = sides["home"], sides["away"]
        if h["pred"] > a["pred"]:
            fav_sum += h["resid"]; fav_n += 1
        elif a["pred"] > h["pred"]:
            fav_sum += a["resid"]; fav_n += 1
    return out


def main():
    print("Building per-side points dataset...", file=sys.stderr)
    rows = v9.build_dataset()
    for r in rows:
        r["pred"] = v32.POINTS_INTERCEPT + float(np.dot(v32.POINTS_WEIGHTS, r["features"]))
        r["resid"] = r["points"] - r["pred"]

    by_game = defaultdict(dict)
    for r in rows:
        by_game[r["game_id"]][r["side"]] = r

    print("Computing walk-forward bias corrections...", file=sys.stderr)
    hfa_bias = v32.home_away_bias_by_season(rows)
    fav_bias_wf = fav_dog_bias_walk_forward(by_game)

    sd_home = float(np.std([r["resid"] for r in rows if r["side"] == "home"]))
    sd_away = float(np.std([r["resid"] for r in rows if r["side"] == "away"]))

    print("Building historical score-frequency table (2015-2025, full window)...", file=sys.stderr)
    hist_freq, n_hist = v32.historical_score_frequency(rows, first_season=2015)

    print("Loading production win-probabilities...", file=sys.stderr)
    config.get_config()
    data = json.loads(store.kv_get("historical_power_rankings"))
    scatter = data["backtest_scatter"]
    wp_lookup = {(r["home"], r["away"], r["season"], r["week"]): r for r in scatter}

    print("Indexing win-probability rows by (season, week) for fast lookup...", file=sys.stderr)
    by_season_week: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for (hk, ak, sk, wk), r in wp_lookup.items():
        by_season_week[(sk, wk)].append(r)

    print("Scoring every 2015-2025 game...", file=sys.stderr)
    n = 0
    abs_errs = []
    wp_errs = []
    ll_step1, ll_step2 = [], []
    skipped_no_wp = 0

    for gid, sides in by_game.items():
        if "home" not in sides or "away" not in sides:
            continue
        h, a = sides["home"], sides["away"]
        season, week = h["season"], h["week"]
        if season < 2015:
            continue
        wp_row = None
        # look up by matching season/week and team substrings in game_id (teams normalized
        # in backtest_scatter, games.csv-era codes in game_id -- same mismatch v32 hit)
        for r in by_season_week.get((season, week), []):
            if r["home"] in gid and r["away"] in gid:
                wp_row = r
                break
        if wp_row is None or wp_row["delta"] == 0:
            skipped_no_wp += 1
            continue

        hb = hfa_bias.get(season, hfa_bias[max(hfa_bias)])
        fb = fav_bias_wf.get(gid, 0.0)
        mean_home = h["pred"] + hb["home"] + (fb if h["pred"] > a["pred"] else -fb)
        mean_away = a["pred"] + hb["away"] + (fb if a["pred"] > h["pred"] else -fb)

        target_p = team_ratings.delta_win_prob(wp_row["delta"]) if wp_row["delta"] > 0 else 1 - team_ratings.delta_win_prob(wp_row["delta"])

        delta_shift, h_dist, a_dist, achieved_p = v32.calibrate_to_win_prob(
            mean_home, mean_away, sd_home, sd_away, hist_freq, target_p)

        h_raw = v32.raw_distribution(mean_home + delta_shift, sd_home)
        a_raw = v32.raw_distribution(mean_away - delta_shift, sd_away)

        actual_h, actual_a = int(round(h["points"])), int(round(a["points"]))
        actual_h = min(actual_h, v32.MAX_SCORE)
        actual_a = min(actual_a, v32.MAX_SCORE)

        mean_pred_h = float(np.dot(np.arange(len(h_dist)), h_dist))
        mean_pred_a = float(np.dot(np.arange(len(a_dist)), a_dist))
        abs_errs.append(abs(mean_pred_h - h["points"]))
        abs_errs.append(abs(mean_pred_a - a["points"]))

        wp_errs.append(abs(achieved_p - target_p))

        eps = 1e-9
        ll_step1.append(np.log(h_raw[actual_h] + eps) + np.log(a_raw[actual_a] + eps))
        ll_step2.append(np.log(h_dist[actual_h] + eps) + np.log(a_dist[actual_a] + eps))
        n += 1

    print(f"\n{n} games scored, {skipped_no_wp} skipped (no matching win-prob row)\n")
    print("=== point accuracy ===")
    print(f"  MAE: {np.mean(abs_errs):.3f}  (baseline uncorrected MAE was ~7.70)")
    print("\n=== does Step 2 (historical reweighting) improve the distribution, not just its look? ===")
    print(f"  avg log-likelihood, Step 1 only (raw Normal):      {np.mean(ll_step1):.4f}")
    print(f"  avg log-likelihood, Step 1 + Step 2 (reweighted):  {np.mean(ll_step2):.4f}")
    print(f"  improvement: {np.mean(ll_step2) - np.mean(ll_step1):+.4f} nats/game-side "
          f"({'better' if np.mean(ll_step2) > np.mean(ll_step1) else 'worse'})")
    print("\n=== win-probability calibration at scale ===")
    print(f"  mean |achieved - target|: {np.mean(wp_errs)*100:.4f}pts  max: {np.max(wp_errs)*100:.4f}pts")


if __name__ == "__main__":
    main()
