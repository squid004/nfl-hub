"""v41: SUPERSEDED by v42 -- do not use this script's printed weight. It refit against v40's
ORIGINAL decay/reinsertion mechanism, which had a real bug (the reinsertion boost was gated
on streak_n>0, but streak_n reset to 0 on the very first healthy week after an absence, so
the boost only ever applied for ONE week instead of decaying over several like the penalty
itself does). After fixing that bug in v40.build_qb_gap (and in the production port) and
re-running the sweep, window5+decay no longer ties/beats the static baseline -- AUC drops to
0.6718 vs window5+static's 0.6739 (identical to production). v42 refits using ONLY the
window5 recency fix with the plain static gap, which is what actually shipped. Kept for the
record of what was tried and why it didn't hold up once corrected.

Original docstring, for context (window5+decay/reinsertion validated in v40 (walk-forward:
ties/slightly beats the cumulative+static production baseline, AUC +0.0003-0.0004, while
fixing the real labeling bug the Atlanta/BAL case surfaced -- a team's "primary" starter
should flip within a trailing 5-game window once a new starter has a plurality of recent
starts, instead of waiting on a season-cumulative-attempts crossover an early-season injury
can delay for a long time) -- this result is what turned out to be bug-driven, see above):

Re-runs v26's EXACT final deploy step (one joint logistic regression on ALL data, not
walk-forward -- same "fit on everything, we're deploying this now" convention as every other
production weight here) with the SAME skill_epa_out_diff term unchanged, swapping only
qb_gap_diff for the new window5+decay/reinsertion construction. Produces the new
QB_QUALITY_GAP_WEIGHT (SKILL_EPA_OUT_WEIGHT is refit alongside it, same joint fit, since the
two were always fit together -- but is expected to barely move, nothing about the skill-EPA
signal changed).

Run: python research/edge_signal_test_v41_qb_window5_decay_refit.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

import importlib.util
def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

v14 = _load("v14", "edge_signal_test_v14_qb_injury_dropout.py")
v21 = _load("v21", "edge_signal_test_v21_skill_touches_weighted.py")
v26 = _load("v26", "edge_signal_test_v26_production_weights.py")
v40 = _load("v40", "edge_signal_test_v40_qb_recency_decay.py")

FIRST_SEASON = 2009
PRIMARY_WINDOW = "window5"


def fit_logistic(X: np.ndarray, y: np.ndarray):
    n, p = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    def nll_grad(w):
        z = Xb @ w
        p_hat = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-12
        nll = -np.sum(y * np.log(p_hat + eps) + (1 - y) * np.log(1 - p_hat + eps))
        grad = Xb.T @ (p_hat - y)
        return nll, grad
    res = minimize(nll_grad, x0=np.zeros(p + 1), jac=True, method="L-BFGS-B")
    w = res.x
    z = Xb @ w
    p_hat = 1.0 / (1.0 + np.exp(-z))
    W = p_hat * (1 - p_hat)
    cov = np.linalg.inv((Xb * W[:, None]).T @ Xb)
    se = np.sqrt(np.diag(cov))
    return w, se


def main():
    print("Rebuilding window5+decay/reinsertion QB gap (v40's machinery)...", file=sys.stderr)
    passer_df = v40.load_passer_df()
    actual_passer, game_team_epa, game_order = v40.build_game_order(passer_df)
    primary_so_far = v40.build_primary_so_far(game_order, actual_passer, game_team_epa, PRIMARY_WINDOW)
    trailing_epa, league_avg_epa = v40.build_trailing_epa(game_order, actual_passer, game_team_epa)
    gap = v40.build_qb_gap(game_order, actual_passer, primary_so_far, trailing_epa, league_avg_epa, dynamic=True)

    print("Building skill EPA-out signal (unchanged from production)...", file=sys.stderr)
    skill_player_games, skill_fallback = v26.load_skill_epa_history()
    skill_injury_rows = v21.load_skill_injury_rows()
    skill_epa_out = v26.skill_epa_out_by_team_week(skill_injury_rows, skill_player_games, skill_fallback)

    print("Building game dataset + production delta...", file=sys.stderr)
    rows_all = v14.build_dataset()
    cols = [f"{m}_diff" for m in v14.RATING_METRICS]
    X_all = np.array([[r[c] for c in cols] for r in rows_all])
    mu_all, sd_all = X_all.mean(axis=0), X_all.std(axis=0)
    sd_all[sd_all == 0] = 1.0
    PW = v14.PRODUCTION_WEIGHTS
    for i, r in enumerate(rows_all):
        x = X_all[i]
        r["delta"] = sum(PW[m] * ((x[j] - mu_all[j]) / sd_all[j]) for j, m in enumerate(v14.RATING_METRICS))
    games = [r for r in rows_all if r["season"] >= FIRST_SEASON and r["delta"] != 0]

    for g in games:
        g["qb_gap_diff"] = gap.get((g["home"], g["game_id"]), 0.0) - gap.get((g["away"], g["game_id"]), 0.0)
        g["skill_out_diff"] = (skill_epa_out.get((g["home"], g["season"], g["week"]), 0.0) -
                                skill_epa_out.get((g["away"], g["season"], g["week"]), 0.0))

    y = np.array([g["home_win"] for g in games], dtype=float)
    X = np.array([[g["delta"], g["qb_gap_diff"], g["skill_out_diff"]] for g in games])

    print(f"\n{len(games)} games, {FIRST_SEASON}-2025 -- fitting FINAL joint model (raw units, all data)...")
    w, se = fit_logistic(X, y)
    names = ["intercept", "delta", "qb_gap_diff", "skill_out_diff"]
    print(f"{'term':16s} {'coef':>10s} {'se':>9s} {'z':>7s}")
    for nm, wi, sei in zip(names, w, se):
        print(f"{nm:16s} {wi:10.4f} {sei:9.4f} {wi/sei:7.2f}")

    b_delta, b_qb, b_skill = w[1], w[2], w[3]
    print(f"\nDelta-equivalent conversion (coef / delta's own coef):")
    print(f"  QB_QUALITY_GAP_WEIGHT   = {b_qb/b_delta:+.4f}   (production: -0.5903)")
    print(f"  SKILL_EPA_OUT_WEIGHT    = {b_skill/b_delta:+.4f}   (production: -0.0362)")

    for key, label in (("qb_gap_diff", "qb_gap_diff"), ("skill_out_diff", "skill_out_diff")):
        vals = np.array([g[key] for g in games])
        print(f"  {label}: mean={vals.mean():+.4f} std={vals.std():.4f} nonzero={np.mean(vals!=0)*100:.1f}%")


if __name__ == "__main__":
    main()
