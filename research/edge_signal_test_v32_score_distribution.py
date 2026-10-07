"""v32: per-team predictive SCORE DISTRIBUTIONS (not just a point estimate), two-step as
specified:
  Step 1: a raw continuous curve from the model itself -- Normal(bias-corrected predicted
    points, empirical residual sd), discretized to integers via the standard continuity
    correction (P(s) = Phi(s+0.5) - Phi(s-0.5)).
  Step 2: reweight that curve by the EMPIRICAL historical frequency of each exact final
    score value since the extra-point distance increased (2015 season on -- verified live:
    NFL owners approved moving PAT to the 15-yard line in May 2015, first used that season),
    so the output respects real football scoring patterns (14 is a common score, 15 isn't)
    instead of spreading probability mass smoothly across every integer.

Bias correction (both folded into Step 1's mean, per explicit direction):
  - home/away: real, DECLINING over time (same underlying shift as the win-probability HFA
    work) -- same trailing-5-season-shrunk-toward-long-run estimator reused here, just in
    raw points instead of logit units.
  - favorite/underdog: redefined by the MODEL's own predicted margin (not the market spread,
    so predict_points() stays self-contained) -- this shrinks the effect from +/-1.35 (when
    measured against the market) to +/-0.29 (the market was doing most of that work, not a
    real bias in the model's own world), and shows no clear trend over time, so a flat
    historical average is used rather than a trailing/recency estimator.

Reconciliation with the model's own win probability (explicit requirement: "even the
underdog should be expected to score more points some of the time," consistent with the
already-established delta_win_prob()): treats the two final per-team distributions as
INDEPENDENT (a real simplification, flagged here and in the output -- true home/away scores
in one game aren't fully independent, e.g. shared pace/garbage-time effects, but a full joint
model is a much bigger lift than this pass), then solves for a symmetric mean-shift delta
(add to home, subtract from away) via bisection so the IMPLIED P(home scores more) from the
two independent distributions matches delta_win_prob() exactly -- so the final output is
fully consistent with the model's own already-validated, more-accurate win-probability
estimate, not a second, potentially-contradictory one.

Run: python research/edge_signal_test_v32_score_distribution.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import edge_signal_test_v9_points_prediction as v9  # noqa: E402

sys.path.insert(0, os.path.join(HERE, ".."))
from nflhub import config, store  # noqa: E402
from nflhub.sources import team_ratings  # noqa: E402

POINTS_INTERCEPT = 7.21333
POINTS_WEIGHTS = np.array([2.472559, 6.259427, 0.371737, -0.0, 2.470530, 0.666745, 0.317523, -0.0])
MAX_SCORE = 60  # integer score bins 0..MAX_SCORE; negligible mass beyond this in the NFL
HFA_SHRINKAGE_K = 100  # same convention as team_ratings.HFA_SHRINKAGE_K, reused here in points units


def home_away_bias_by_season(rows: list[dict]) -> dict[int, dict[str, float]]:
    """Trailing-5-season home/away scoring bias, shrunk toward the all-time-to-date average
    (same structure as team_ratings.home_field_logit_by_season(), just in raw points instead
    of logit units) -- the bias is real and DECLINING (verified above), so a flat average
    would overstate today's effect same as it would have for win probability."""
    by_season: dict[int, dict[str, list]] = defaultdict(lambda: {"home": [], "away": []})
    for r in rows:
        by_season[r["season"]][r["side"]].append(r["resid"])
    seasons = sorted(by_season)
    out: dict[int, dict[str, float]] = {}
    for i, season in enumerate(seasons):
        prior = seasons[:i]
        if not prior:
            out[season] = {"home": 0.0, "away": 0.0}
            continue
        window = prior[-5:]
        result = {}
        for side in ("home", "away"):
            w_vals = [v for s in window for v in by_season[s][side]]
            a_vals = [v for s in prior for v in by_season[s][side]]
            rate5 = np.mean(w_vals) if w_vals else 0.0
            n5 = len(w_vals)
            rate_all = np.mean(a_vals) if a_vals else 0.0
            n_all = len(a_vals)
            result[side] = (n5 * rate5 + HFA_SHRINKAGE_K * rate_all) / (n5 + HFA_SHRINKAGE_K)
        out[season] = result
    return out


def historical_score_frequency(games_rows: list[dict], first_season: int = 2015) -> dict[int, float]:
    """P(a team's final score == s) for s in 0..MAX_SCORE, from every team-game's own final
    score since `first_season` (2015 -- the modern PAT-distance era). Both home and away
    scores counted (one row per team per game, same as the residual dataset)."""
    counts = defaultdict(int)
    n = 0
    for r in games_rows:
        if r["season"] < first_season:
            continue
        s = int(round(r["points"]))
        if 0 <= s <= MAX_SCORE:
            counts[s] += 1
            n += 1
    return {s: counts.get(s, 0) / n for s in range(MAX_SCORE + 1)}, n


def raw_distribution(mean: float, sd: float) -> np.ndarray:
    """Step 1: Normal(mean, sd) discretized to integers 0..MAX_SCORE via the continuity
    correction, mass below 0 folded into bin 0 (a team can't score negative points) and
    renormalized."""
    edges = np.arange(-0.5, MAX_SCORE + 1.5, 1.0)
    cdf = norm.cdf(edges, loc=mean, scale=sd)
    probs = np.diff(cdf)
    # fold sub-zero mass (and any beyond MAX_SCORE) into the boundary bins
    probs[0] += norm.cdf(-0.5, loc=mean, scale=sd)
    probs[-1] += 1 - norm.cdf(MAX_SCORE + 0.5, loc=mean, scale=sd)
    return probs / probs.sum()


def reweight_by_history(raw: np.ndarray, hist_freq: dict[int, float]) -> np.ndarray:
    """Step 2: multiply the raw curve by the empirical historical frequency of each exact
    score, renormalize. Scores with ZERO historical precedent (e.g. 1 point) get a tiny
    floor instead of an outright zero, so an unprecedented-but-not-impossible context
    (a future rule change, say) doesn't produce a hard zero that can't recover."""
    hist = np.array([hist_freq.get(s, 0.0) for s in range(len(raw))])
    floor = 1e-6
    weighted = raw * (hist + floor)
    return weighted / weighted.sum()


def implied_home_win_prob(home_dist: np.ndarray, away_dist: np.ndarray) -> float:
    """P(home score > away score) treating the two distributions as INDEPENDENT -- a real
    simplification (see module docstring), not a true joint model."""
    # P(home > away) = sum_h home[h] * sum_{a<h} away[a]
    away_cdf_below = np.cumsum(away_dist) - away_dist  # P(away < h) for each h
    return float(np.sum(home_dist * away_cdf_below))


def calibrate_to_win_prob(mean_home: float, mean_away: float, sd_home: float, sd_away: float,
                           hist_freq: dict[int, float], target_home_win_prob: float,
                           tol: float = 0.0005, max_iter: int = 40):
    """Bisection on a symmetric mean-shift delta (home += delta, away -= delta) so the
    IMPLIED win probability from the two final (post-reweight) distributions matches
    `target_home_win_prob` (the model's own, already-validated delta_win_prob()) --
    reconciles the score distributions with the win-probability model instead of leaving
    them potentially inconsistent."""
    lo, hi = -21.0, 21.0  # generous bracket; a 3-score swing in mean is already extreme
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        h_dist = reweight_by_history(raw_distribution(mean_home + mid, sd_home), hist_freq)
        a_dist = reweight_by_history(raw_distribution(mean_away - mid, sd_away), hist_freq)
        p = implied_home_win_prob(h_dist, a_dist)
        if abs(p - target_home_win_prob) < tol:
            break
        if p < target_home_win_prob:
            lo = mid
        else:
            hi = mid
    return mid, h_dist, a_dist, p


def to_whole_percentages(dist: np.ndarray) -> dict[int, int]:
    """Largest-remainder (Hamilton) apportionment -- integer percentages that sum to EXACTLY
    100, instead of naive rounding (which can land on 99 or 101)."""
    raw_pct = dist * 100
    floors = np.floor(raw_pct).astype(int)
    remainder = 100 - floors.sum()
    order = np.argsort(-(raw_pct - floors))  # largest fractional remainder first
    out = floors.copy()
    for i in range(remainder):
        out[order[i]] += 1
    return {s: int(out[s]) for s in range(len(out)) if out[s] > 0}


def main():
    print("Building per-side points dataset...", file=sys.stderr)
    rows = v9.build_dataset()
    for r in rows:
        r["pred"] = POINTS_INTERCEPT + float(np.dot(POINTS_WEIGHTS, r["features"]))
        r["resid"] = r["points"] - r["pred"]

    print("Computing bias corrections...", file=sys.stderr)
    hfa_bias = home_away_bias_by_season(rows)
    last_season = max(hfa_bias)
    home_bias, away_bias = hfa_bias[last_season]["home"], hfa_bias[last_season]["away"]
    print(f"  home/away bias (trailing, as of {last_season}): home={home_bias:+.3f} away={away_bias:+.3f}")

    by_game = defaultdict(dict)
    for r in rows:
        by_game[r["game_id"]][r["side"]] = r
    fav_resid, dog_resid = [], []
    for sides in by_game.values():
        if "home" not in sides or "away" not in sides:
            continue
        h, a = sides["home"], sides["away"]
        if h["pred"] > a["pred"]:
            fav_resid.append(h["resid"]); dog_resid.append(a["resid"])
        elif a["pred"] > h["pred"]:
            fav_resid.append(a["resid"]); dog_resid.append(h["resid"])
    fav_bias, dog_bias = float(np.mean(fav_resid)), float(np.mean(dog_resid))
    print(f"  favorite/underdog bias (flat, model's own margin): fav={fav_bias:+.3f} dog={dog_bias:+.3f}")

    home_resid = np.array([r["resid"] for r in rows if r["side"] == "home"])
    away_resid = np.array([r["resid"] for r in rows if r["side"] == "away"])
    sd_home, sd_away = float(home_resid.std()), float(away_resid.std())
    print(f"  residual sd: home={sd_home:.3f} away={sd_away:.3f}")

    print("\nBuilding historical score-frequency table (2015+)...", file=sys.stderr)
    hist_freq, n_hist = historical_score_frequency(rows, first_season=2015)
    n_hist_games = n_hist // 2
    print(f"  {n_hist} team-game final scores since 2015 ({n_hist_games} games)")
    top10 = sorted(hist_freq.items(), key=lambda kv: -kv[1])[:10]
    print("  most common scores:", [(s, f"{p*100:.1f}%") for s, p in top10])

    print("\n=== validation: a handful of real recent games ===")
    config.get_config()
    data = json.loads(store.kv_get("historical_power_rankings"))
    scatter = data["backtest_scatter"]
    recent = [r for r in scatter if r["season"] == 2025 and r["delta"] != 0][:6]

    total_abs_err, total_games, win_prob_errs = 0.0, 0, []
    for r in recent:
        # find matching game row by season/week/home/away
        match = None
        for sides in by_game.values():
            if "home" in sides and "away" in sides:
                h = sides["home"]
                if h["season"] == r["season"] and h["week"] == r["week"]:
                    # confirm team identity via home/away game_id convention (team codes embedded)
                    if r["home"] in h["game_id"] and r["away"] in h["game_id"]:
                        match = sides
                        break
        if not match:
            continue
        h, a = match["home"], match["away"]
        season = h["season"]
        hb, ab = hfa_bias.get(season, hfa_bias[last_season])["home"], hfa_bias.get(season, hfa_bias[last_season])["away"]
        mean_home = h["pred"] + hb + fav_bias if h["pred"] > a["pred"] else h["pred"] + hb + dog_bias
        mean_away = a["pred"] + ab + fav_bias if a["pred"] > h["pred"] else a["pred"] + ab + dog_bias

        target_p = team_ratings.delta_win_prob(r["delta"]) if r["delta"] > 0 else 1 - team_ratings.delta_win_prob(r["delta"])
        delta_shift, h_dist, a_dist, achieved_p = calibrate_to_win_prob(
            mean_home, mean_away, sd_home, sd_away, hist_freq, target_p)

        h_pct = to_whole_percentages(h_dist)
        a_pct = to_whole_percentages(a_dist)
        h_top = sorted(h_pct.items(), key=lambda kv: -kv[1])[:6]
        a_top = sorted(a_pct.items(), key=lambda kv: -kv[1])[:6]
        print(f"\n  {r['away']} @ {r['home']}  (actual: {r['away_score']}-{r['home_score']}, model target home-win-prob={target_p*100:.1f}%, achieved={achieved_p*100:.1f}%)")
        print(f"    {r['home']} (home) top scores: {h_top}")
        print(f"    {r['away']} (away) top scores: {a_top}")
        print(f"    {r['home']} mean={np.dot(list(range(MAX_SCORE+1)), h_dist):.1f}  {r['away']} mean={np.dot(list(range(MAX_SCORE+1)), a_dist):.1f}")

        total_abs_err += abs(np.dot(list(range(MAX_SCORE+1)), h_dist) - r["home_score"])
        total_abs_err += abs(np.dot(list(range(MAX_SCORE+1)), a_dist) - r["away_score"])
        total_games += 1
        win_prob_errs.append(abs(achieved_p - target_p))

    if total_games:
        print(f"\n  avg |mean - actual| over these {total_games} games: {total_abs_err/(2*total_games):.2f}")
        print(f"  avg win-prob calibration error: {np.mean(win_prob_errs)*100:.3f}pts (should be ~0 by construction)")


if __name__ == "__main__":
    main()
