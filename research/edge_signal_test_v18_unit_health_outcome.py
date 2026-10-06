"""v18: v16/v17 tested each position group's health gap against its OWN phase EPA residual,
one metric at a time -- a strict test, and one where a real-but-small, noisy-per-play effect
could easily fail to clear significance alone even if it's genuinely informative once
combined with everything else. This script instead asks the more forgiving, more relevant
question directly: holding the existing composite model's own prediction (`delta`, exactly
as compute_backtest_scatter() computes it) fixed, do the 5 non-QB units' health gaps jointly
add anything to predicting the actual GAME WINNER?

Game-level dataset (home-minus-away), 2012-2025 (bounded by snap_counts' 2012 start, see
v17's docstring for that verification). For each unit, `gap_X_diff` = home team's health gap
minus the away team's (same per-player snap-share-weighted gap as v17). Two models:
  reduced: home_win ~ delta
  full:    home_win ~ delta + gap_qb_diff + gap_ol_diff + gap_skill_diff + gap_dl_diff +
                       gap_lb_diff + gap_secondary_diff
Reports each term's own Wald p-value in the full model, AND a likelihood-ratio test for the
5 non-QB terms jointly (QB's own coefficient partialled out either way, since it's already
a known, validated effect -- the real question is whether the OTHER 5 add anything together
even if none clears significance alone).

Run: python research/edge_signal_test_v18_unit_health_outcome.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, norm

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

import importlib.util
spec14 = importlib.util.spec_from_file_location("v14", os.path.join(HERE, "edge_signal_test_v14_qb_injury_dropout.py"))
v14 = importlib.util.module_from_spec(spec14)
spec14.loader.exec_module(v14)
spec16 = importlib.util.spec_from_file_location("v16", os.path.join(HERE, "edge_signal_test_v16_unit_health.py"))
v16 = importlib.util.module_from_spec(spec16)
spec16.loader.exec_module(v16)
spec17 = importlib.util.spec_from_file_location("v17", os.path.join(HERE, "edge_signal_test_v17_unit_health_snapweighted.py"))
v17 = importlib.util.module_from_spec(spec17)
spec17.loader.exec_module(v17)

FIRST_SNAP_SEASON = 2012
GROUPS = list(v16.POSITION_GROUPS.keys())  # qb, ol, skill, dl, lb, secondary


def fit_logistic(X: np.ndarray, y: np.ndarray):
    """Plain unconstrained logistic fit. Returns (weights, standard errors, SUM log-
    likelihood at the fit) -- sum (not mean) so log-likelihoods from different models are
    directly comparable for a likelihood-ratio test."""
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
    hessian = (Xb * W[:, None]).T @ Xb
    cov = np.linalg.inv(hessian)
    se = np.sqrt(np.diag(cov))
    ll = -res.fun
    return w, se, ll


def main():
    print("Building full-history game dataset (for production-matching standardization)...", file=sys.stderr)
    rows_all = v14.build_dataset()
    cols = [f"{m}_diff" for m in v14.RATING_METRICS]
    X_all = np.array([[r[c] for c in cols] for r in rows_all])
    mu_all, sd_all = X_all.mean(axis=0), X_all.std(axis=0)
    sd_all[sd_all == 0] = 1.0
    PW = v14.PRODUCTION_WEIGHTS
    for i, r in enumerate(rows_all):
        x = X_all[i]
        r["delta"] = sum(PW[m] * ((x[j] - mu_all[j]) / sd_all[j]) for j, m in enumerate(v14.RATING_METRICS))

    rows = [r for r in rows_all if r["season"] >= FIRST_SNAP_SEASON]
    print(f"{len(rows)} games, {FIRST_SNAP_SEASON}-2025\n")

    print("Loading snap-weighted injury gaps (same as v17)...", file=sys.stderr)
    player_games, pos_default = v17.load_snap_shares()
    injury_rows = v17.load_injury_rows()
    weighted = v17.weighted_absence_by_team_week_group(injury_rows, player_games, pos_default, None)
    team_weeks = sorted({(r[side], r["season"], r["week"]) for r in rows for side in ("home", "away")})
    gaps = v16.build_health_gaps(team_weeks, weighted)

    for r in rows:
        for grp in GROUPS:
            home_gap = gaps.get((r["home"], r["season"], r["week"], grp), 0.0)
            away_gap = gaps.get((r["away"], r["season"], r["week"], grp), 0.0)
            r[f"gap_{grp}_diff"] = home_gap - away_gap

    y = np.array([r["home_win"] for r in rows], dtype=float)
    delta = np.array([r["delta"] for r in rows])

    gap_cols = [f"gap_{g}_diff" for g in GROUPS]
    gap_raw = np.array([[r[c] for c in gap_cols] for r in rows])
    gap_mu, gap_sd = gap_raw.mean(axis=0), gap_raw.std(axis=0)
    gap_sd[gap_sd == 0] = 1.0
    gap_z = (gap_raw - gap_mu) / gap_sd

    print("--- reduced model: home_win ~ delta ---")
    X_reduced = delta.reshape(-1, 1)
    w_r, se_r, ll_r = fit_logistic(X_reduced, y)
    print(f"  intercept {w_r[0]:+.4f} (se {se_r[0]:.4f})   delta {w_r[1]:+.4f} (se {se_r[1]:.4f})   loglik={ll_r:.2f}")

    print("\n--- full model: home_win ~ delta + all 6 unit health gaps (standardized) ---")
    X_full = np.hstack([X_reduced, gap_z])
    w_f, se_f, ll_f = fit_logistic(X_full, y)
    names = ["intercept", "delta"] + gap_cols
    print(f"  {'term':18s} {'coef':>9s} {'se':>8s} {'z':>7s} {'p':>8s}")
    for nm, wi, sei in zip(names, w_f, se_f):
        z = wi / sei
        p = 2 * (1 - norm.cdf(abs(z)))
        flag = "  <-- QB (positive control)" if nm == "gap_qb_diff" else ""
        print(f"  {nm:18s} {wi:9.4f} {sei:8.4f} {z:7.2f} {p:8.4f}{flag}")
    print(f"  loglik={ll_f:.2f}")

    lr_all = 2 * (ll_f - ll_r)
    p_all = 1 - chi2.cdf(lr_all, df=6)
    print(f"\nLR test, all 6 gap terms jointly vs. delta alone: LR={lr_all:.2f}, df=6, p={p_all:.4f}")

    print("\n--- model with delta + QB gap only (QB's own effect partialled in) ---")
    X_qb = np.hstack([X_reduced, gap_z[:, [GROUPS.index("qb")]]])
    w_qb, se_qb, ll_qb = fit_logistic(X_qb, y)
    print(f"  loglik={ll_qb:.2f}")

    lr_nonqb = 2 * (ll_f - ll_qb)
    p_nonqb = 1 - chi2.cdf(lr_nonqb, df=5)
    print(f"\nLR test, the 5 NON-QB gap terms jointly (QB already in both models): "
          f"LR={lr_nonqb:.2f}, df=5, p={p_nonqb:.4f}")


if __name__ == "__main__":
    main()
