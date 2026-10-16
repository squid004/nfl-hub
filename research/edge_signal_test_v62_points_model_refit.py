"""v62 points-model check: does POINTS_WEIGHTS need a refit for the retuned ratings?

The points regression (POINTS_INTERCEPT / POINTS_WEIGHTS: own offense ratings vs the opponent's
defense ratings -> points scored) was fit on alpha-0.2 ratings. The retuned ratings (alpha 0.08,
bias-corrected, prior 8) are smoother, so the same coefficients can be mis-scaled. This compares, on
the NEW ratings, walk-forward (test seasons 2011+): (a) the shipped weights as-is, (b) weights refit
walk-forward with the same recipe (non-negative, L2 = 0.01 linear regression on actual team points),
and sanity-checks the recipe by refitting on the OLD ratings and comparing to the shipped numbers.
Metric: team-points MAE and the implied-margin (home pred - away pred) AUC for the home win.

Pre-registered rule (same as v57/v61): refit weights are adopted only if MAE better in BOTH halves
(<=2018 / >2018) and paired-bootstrap P(dMAE <= 0) <= 0.01 and better in >= 60% of test seasons;
otherwise the shipped weights stay.

Run: PYTHONPATH=. python research/edge_signal_test_v62_points_model_refit.py
"""
from __future__ import annotations

import csv
import io
import os
import sys

import numpy as np
import requests
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import nflhub.sources.team_ratings as tr  # noqa: E402

L2 = 0.01
SELECT_MAX = 2018
TEST_START = 2011
N_BOOT = 2000
KEYS = list(tr.POINTS_WEIGHTS)  # own_* then opp_*
SHIPPED_W = np.array([tr.POINTS_WEIGHTS[k] for k in KEYS])
SHIPPED_B = tr.POINTS_INTERCEPT


def collect(played, current, alpha, prior, bias):
    """One row per (game, side): features, points, season, home flag, game index."""
    tr.EWMA_ALPHA, tr.SEASON_PRIOR_GAMES, tr.EWMA_BIAS_CORRECT = alpha, prior, bias
    tr.EWMA_ALPHA_BY_METRIC = {}
    orig = tr.predict_points
    calls: list[tuple[dict, dict]] = []

    def spy(own, opp, weather=None):
        calls.append((dict(own), dict(opp)))
        return orig(own, opp, weather)

    tr.predict_points = spy
    log: list[dict] = []
    try:
        tr.compute_ratings(played, current, game_log=log)
    finally:
        tr.predict_points = orig
    assert len(calls) == 2 * len(log), (len(calls), len(log))
    X, pts, season, home, gid = [], [], [], [], []
    for i, r in enumerate(log):
        for side, (own, opp) in (("home", calls[2 * i]), ("away", calls[2 * i + 1])):
            feats = []
            for k in KEYS:
                s, m = k.split("_", 1)
                feats.append((own if s == "own" else opp)[m])
            X.append(feats)
            pts.append(r["home_score"] if side == "home" else r["away_score"])
            season.append(r["season"])
            home.append(side == "home")
            gid.append(i)
    return np.array(X), np.array(pts, float), np.array(season), np.array(home), np.array(gid)


def fit_nonneg_linear(X, y, l2):
    n, p = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])

    def f(w):
        r = Xb @ w - y
        return 0.5 * np.mean(r ** 2) + 0.5 * l2 * np.sum(w[1:] ** 2), Xb.T @ r / n + np.concatenate([[0.0], l2 * w[1:]])

    w0 = np.concatenate([[y.mean()], np.zeros(p)])
    return minimize(f, w0, jac=True, method="L-BFGS-B", bounds=[(None, None)] + [(0, None)] * p).x


def walk_forward_pred(X, y, season):
    pred = np.full(len(y), np.nan)
    for ts in sorted(set(season)):
        if ts < TEST_START:
            continue
        w = fit_nonneg_linear(X[season < ts], y[season < ts], L2)
        te = season == ts
        pred[te] = w[0] + X[te] @ w[1:]
    return pred


def margin_auc(pred, y, home, gid, mask_games):
    ph, pa, yh, ya = {}, {}, {}, {}
    for p, t, h, g in zip(pred, y, home, gid):
        (ph if h else pa)[g] = p
        (yh if h else ya)[g] = t
    gs = [g for g in ph if g in mask_games and yh[g] != ya[g]]
    return roc_auc_score([yh[g] > ya[g] for g in gs], [ph[g] - pa[g] for g in gs])


def main():
    resp = requests.get(tr.GAMES_URL, timeout=tr.TIMEOUT)
    resp.raise_for_status()
    rows = [r for r in csv.DictReader(io.StringIO(resp.text)) if r["game_type"] == "REG"]
    played = [r for r in rows if r.get("result") not in ("", "NA", None) and int(r["season"]) >= tr.FIRST_SEASON]
    current = max(int(r["season"]) for r in played)

    print("=== recipe check: refit on OLD ratings (alpha 0.2 classic) vs shipped weights ===")
    X, y, s, h, g = collect(played, current, 0.2, 5, False)
    w = fit_nonneg_linear(X[s >= 2007], y, L2)
    print("  keys:    " + ", ".join(KEYS))
    print("  shipped: " + ", ".join(f"{v:.3f}" for v in SHIPPED_W) + f" | intercept {SHIPPED_B:.3f}")
    print("  refit:   " + ", ".join(f"{v:.3f}" for v in w[1:]) + f" | intercept {w[0]:.3f}")

    print("\n=== NEW ratings (alpha 0.08, bias-corrected, prior 8) ===")
    X, y, s, h, g = collect(played, current, 0.08, 8, True)
    test = s >= TEST_START
    pred_ship = SHIPPED_B + X @ SHIPPED_W
    pred_new = walk_forward_pred(X, y, s)
    for lab, mask in (("all", test), ("select <=2018", test & (s <= SELECT_MAX)), ("confirm >2018", s > SELECT_MAX)):
        games = set(g[mask])
        print(f"  {lab:14s}: MAE shipped {np.mean(np.abs(pred_ship[mask] - y[mask])):.4f}  refit {np.mean(np.abs(pred_new[mask] - y[mask])):.4f} | "
              f"margin AUC shipped {margin_auc(pred_ship, y, h, g, games):.4f}  refit {margin_auc(pred_new, y, h, g, games):.4f} | "
              f"mean pred shipped {pred_ship[mask].mean():.2f} refit {pred_new[mask].mean():.2f} actual {y[mask].mean():.2f}")
    err_s, err_n = np.abs(pred_ship - y), np.abs(pred_new - y)
    rng = np.random.default_rng(62)
    idx = np.where(test)[0]
    d = err_s[idx] - err_n[idx]
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
    p_le0 = float(np.mean(np.array(boots) <= 0))
    wins = np.mean([err_n[test & (s == yr)].mean() < err_s[test & (s == yr)].mean() for yr in sorted(set(s[test]))])
    halves = all(err_n[test & m].mean() < err_s[test & m].mean() for m in (s <= SELECT_MAX, s > SELECT_MAX))
    ok = halves and p_le0 <= 0.01 and wins >= 0.6
    print(f"\n  dMAE (shipped - refit) {d.mean():+.4f}; P(dMAE<=0)={p_le0:.3f}; season-win={wins:.2f}; both halves={halves} -> {'ADOPT REFIT' if ok else 'KEEP SHIPPED'}")

    wf = fit_nonneg_linear(X, y, L2)
    print("\n=== all-data refit on the NEW ratings (the constants to ship if ADOPT) ===")
    print(f"  POINTS_INTERCEPT = {wf[0]:.5f}")
    for k, v in zip(KEYS, wf[1:]):
        print(f'    "{k}": {v:.6f},')


if __name__ == "__main__":
    main()
