"""v15: calibrate the power-ranking model's composite |delta| into a win probability, for a
new "Power Model" pick on the Moneyline Pick'em tab (alongside ELWAY and History). The
exploratory 12-bin chart from this session's earlier analysis showed accuracy climbing
smoothly and roughly monotonically with |delta| (52% near zero up to ~77% at the high end,
backup-QB games excluded) -- this script fits that relationship as a continuous function
instead of discrete bins: P(model favorite wins) = sigmoid(a + b * |delta|), a plain
unconstrained 1-feature logistic regression (2 parameters, huge sample -- no regularization
needed), walk-forward validated the same way as every other fit in this project before
locking in a final "fit on all data" version.

delta itself is computed exactly like compute_backtest_scatter() in production: current
POWER_WEIGHTS applied to each game's pre-game rating diffs, standardized ONCE across the
whole dataset. Fit on the CLEAN (non-backup-QB) games only, per this session's finding that
the production weights aren't meaningfully biased by backup-QB games either way -- but the
calibration curve itself should still describe "how good is the model when it has normal
information," not be diluted by the QB-blind-spot games this exact tab will also show
separately.

Run: python research/edge_signal_test_v15_delta_calibration.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score

HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("v14", os.path.join(HERE, "edge_signal_test_v14_qb_injury_dropout.py"))
v14 = importlib.util.module_from_spec(spec)
sys.path.insert(0, os.path.join(HERE, ".."))
spec.loader.exec_module(v14)

TEST_START_SEASON = 2011  # matches v13's own walk-forward convention


def fit_logistic_1d(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Plain unconstrained 2-parameter logistic fit (intercept + slope on `x`) -- no L2, no
    non-negativity constraint (unlike the 8-stat composite fit): 1 feature, thousands of
    games, nothing here is going to overfit or sign-flip."""
    def nll_grad(w):
        a, b = w
        z = a + b * x
        p = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-12
        nll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        grad_a = np.mean(p - y)
        grad_b = np.mean((p - y) * x)
        return nll, np.array([grad_a, grad_b])

    res = minimize(nll_grad, x0=np.array([0.0, 1.0]), jac=True, method="L-BFGS-B")
    return float(res.x[0]), float(res.x[1])


def main():
    print("Building dataset...", file=sys.stderr)
    rows = v14.build_dataset()
    print("Loading QB-injury report...", file=sys.stderr)
    qb_out = v14.qb_out_doubtful_by_week()
    for r in rows:
        r["backup_flag"] = qb_out.get((r["home"], r["season"], r["week"]), False) or \
                            qb_out.get((r["away"], r["season"], r["week"]), False)

    cols = [f"{m}_diff" for m in v14.RATING_METRICS]
    X_all = np.array([[r[c] for c in cols] for r in rows])
    mu_all, sd_all = X_all.mean(axis=0), X_all.std(axis=0)
    sd_all[sd_all == 0] = 1.0
    PW = v14.PRODUCTION_WEIGHTS

    for i, r in enumerate(rows):
        x = X_all[i]
        r["delta"] = sum(PW[m] * ((x[j] - mu_all[j]) / sd_all[j]) for j, m in enumerate(v14.RATING_METRICS))

    clean = [r for r in rows if not r["backup_flag"] and r["delta"] != 0]
    print(f"{len(clean)} clean decided games (backup-QB games excluded)\n")

    abs_delta = np.array([abs(r["delta"]) for r in clean])
    correct = np.array([(1 if r["delta"] > 0 else 0) == r["home_win"] for r in clean]).astype(float)
    seasons = np.array([r["season"] for r in clean])

    print("--- walk-forward validation (train on seasons < test, test on test season) ---")
    print(f"{'season':>6s} {'n_test':>7s} {'logloss':>8s} {'logloss(baseline .629)':>22s} {'auc':>6s}")
    for test_season in sorted(set(seasons)):
        if test_season < TEST_START_SEASON:
            continue
        train_mask = seasons < test_season
        test_mask = seasons == test_season
        if train_mask.sum() < 200 or test_mask.sum() == 0:
            continue
        a, b = fit_logistic_1d(abs_delta[train_mask], correct[train_mask])
        p_test = 1 / (1 + np.exp(-(a + b * abs_delta[test_mask])))
        y_test = correct[test_mask]
        ll = log_loss(y_test, p_test, labels=[0, 1])
        baseline_ll = log_loss(y_test, np.full_like(p_test, 0.629), labels=[0, 1])
        auc = roc_auc_score(y_test, abs_delta[test_mask]) if len(set(y_test)) > 1 else float("nan")
        print(f"{test_season:6d} {test_mask.sum():7d} {ll:8.4f} {baseline_ll:22.4f} {auc:6.3f}")

    # --- final fit on ALL clean data ---
    a_final, b_final = fit_logistic_1d(abs_delta, correct)
    print(f"\nFinal calibration (all {len(clean)} clean games):")
    print(f"  intercept = {a_final:+.4f}")
    print(f"  slope     = {b_final:+.4f}")
    print(f"  P(correct | delta=0)    = {1/(1+np.exp(-a_final)):.3f}")
    print(f"  P(correct | |delta|=0.3) = {1/(1+np.exp(-(a_final+b_final*0.3))):.3f}")
    print(f"  P(correct | |delta|=0.6) = {1/(1+np.exp(-(a_final+b_final*0.6))):.3f}")
    print(f"  P(correct | |delta|=1.0) = {1/(1+np.exp(-(a_final+b_final*1.0))):.3f}")

    # Sanity check against the 12-bin table from the chart shown this session.
    print("\n--- calibration curve vs. the 12-bin empirical table (sanity check) ---")
    edges = np.quantile(abs_delta, np.linspace(0, 1, 13))
    edges = np.unique(edges)
    bin_idx = np.clip(np.digitize(abs_delta, edges[1:-1], right=True), 0, len(edges) - 2)
    for bidx in range(len(edges) - 1):
        mask = bin_idx == bidx
        if mask.sum() == 0:
            continue
        mid = abs_delta[mask].mean()
        emp = correct[mask].mean()
        pred = 1 / (1 + np.exp(-(a_final + b_final * mid)))
        print(f"  |delta|~{mid:.3f}: empirical {emp*100:5.1f}%  vs  curve {pred*100:5.1f}%")


if __name__ == "__main__":
    main()
