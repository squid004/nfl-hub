"""v47: final production refit for the decay-only, rate=0.4 QB-gap construction validated in
v44 (decomposition: penalty-decay alone beats static, reinsertion alone/combined hurts) and
v45/v46 (rate sweep + chronological select/confirm: rate=0.4 selected on 2014-2020, replicated
on the held-out 2021-2025 confirm set, AUC +0.0018; a 30-repeat random split agreed in
direction but was noisier, 70% of repeats favorable). Window5 primary (already shipped) held
fixed. Reuses v26's exact final-deploy recipe: one joint logistic regression on ALL data
(delta + qb_gap_diff + skill_out_diff, raw units), same as every other production weight here.

Run: python research/edge_signal_test_v47_qb_decay_rate04_refit.py
"""
from __future__ import annotations

import os
import sys

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
v45 = _load("v45", "edge_signal_test_v45_qb_decay_rate_sweep.py")

FIRST_SEASON = 2009
PRIMARY_MODE = "window5"
DECAY_RATE = 0.4


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
    print("Rebuilding window5 + decay-only (rate=0.4) QB gap...", file=sys.stderr)
    passer_df = v40.load_passer_df()
    actual_passer, game_team_epa, game_order = v40.build_game_order(passer_df)
    primary_so_far = v40.build_primary_so_far(game_order, actual_passer, game_team_epa, PRIMARY_MODE)
    trailing_epa, league_avg_epa = v40.build_trailing_epa(game_order, actual_passer, game_team_epa)

    backup_epa_sum = backup_attempts = 0.0
    for gid, team, season, week in game_order:
        primary = primary_so_far.get((team, gid))
        ap = actual_passer.get((gid, team))
        if primary is not None and ap is not None and ap != primary:
            epa_sum, attempts = game_team_epa[(gid, team)]
            backup_epa_sum += epa_sum
            backup_attempts += attempts
    backup_avg_epa = (backup_epa_sum / backup_attempts) if backup_attempts else league_avg_epa

    gap = v45.build_qb_gap_decay_rate(game_order, actual_passer, primary_so_far, trailing_epa,
                                       league_avg_epa, backup_avg_epa, DECAY_RATE)

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
    print(f"  QB_QUALITY_GAP_WEIGHT   = {b_qb/b_delta:+.4f}   (previous production: -0.5202)")
    print(f"  SKILL_EPA_OUT_WEIGHT    = {b_skill/b_delta:+.4f}   (previous production: -0.0360)")

    for key, label in (("qb_gap_diff", "qb_gap_diff"), ("skill_out_diff", "skill_out_diff")):
        vals = np.array([g[key] for g in games])
        print(f"  {label}: mean={vals.mean():+.4f} std={vals.std():.4f} nonzero={np.mean(vals!=0)*100:.1f}%")


if __name__ == "__main__":
    main()
