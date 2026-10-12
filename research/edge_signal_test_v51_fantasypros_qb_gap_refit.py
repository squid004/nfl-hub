"""v51: final production refit adding QB_FANTASY_GAP_WEIGHT alongside the existing
QB_QUALITY_GAP_WEIGHT, validated additive (not a replacement) in v49/v50. Same joint-fit
recipe as v26/v42/v47 (fit on ALL data, not walk-forward -- we're deploying this now, same
convention as every other production weight here): home_win ~ delta + qb_gap_diff +
fantasy_gap_diff + skill_out_diff, all four terms in RAW native units (not standardized) --
the ratio of each gap's own coefficient to delta's own IS directly "how many delta-equivalent
units this gap is worth," which is what actually gets added to delta in production.

fantasy_gap_diff only has real coverage 2017+ (FantasyPros' own history, see nflhub.sources.
fantasypros.FIRST_SEASON) -- rows before that get fantasy_gap_diff=0 (no adjustment, same as
"no QB change this week"), so they still inform delta/qb_gap_diff/skill_out_diff's own
weights exactly as before, contributing zero gradient to fantasy_gap_diff's coefficient
specifically (standard, not a fabricated signal).

Run: python research/edge_signal_test_v51_fantasypros_qb_gap_refit.py
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

FIRST_SEASON = 2009


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
    print("Loading production team_ratings module for the real, shipped mechanisms...", file=sys.stderr)
    import nflhub.sources.team_ratings as tr
    import csv
    import io
    import requests

    resp = requests.get(tr.GAMES_URL, timeout=tr.TIMEOUT)
    resp.raise_for_status()
    all_rows = [r for r in csv.DictReader(io.StringIO(resp.text)) if r["game_type"] == "REG"]
    played = [r for r in all_rows if r.get("result") not in ("", "NA", None) and int(r["season"]) >= tr.FIRST_SEASON]
    current_season = max(int(r["season"]) for r in played)

    print("Building the shipped EPA gap, fantasy gap, and skill-out signals...", file=sys.stderr)
    qb_gap, _p, _t, _la, _n, _ba, _sn = tr.qb_quality_gap_data(played, current_season)
    fantasy_gap, _tfp, _lafp, _bafp = tr.qb_fantasy_gap_data(played, current_season, 1)
    skill_out = tr.skill_epa_out_by_team_week(current_season)

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
        g["qb_gap_diff"] = qb_gap.get((g["home"], g["season"], g["week"]), 0.0) - qb_gap.get((g["away"], g["season"], g["week"]), 0.0)
        g["fantasy_gap_diff"] = fantasy_gap.get((g["home"], g["season"], g["week"]), 0.0) - fantasy_gap.get((g["away"], g["season"], g["week"]), 0.0)
        g["skill_out_diff"] = (skill_out.get((g["home"], g["season"], g["week"]), 0.0) -
                                skill_out.get((g["away"], g["season"], g["week"]), 0.0))

    y = np.array([g["home_win"] for g in games], dtype=float)
    X = np.array([[g["delta"], g["qb_gap_diff"], g["fantasy_gap_diff"], g["skill_out_diff"]] for g in games])

    print(f"\n{len(games)} games, {FIRST_SEASON}-{current_season} -- fitting FINAL joint model (raw units, all data)...")
    w, se = fit_logistic(X, y)
    names = ["intercept", "delta", "qb_gap_diff", "fantasy_gap_diff", "skill_out_diff"]
    print(f"{'term':16s} {'coef':>10s} {'se':>9s} {'z':>7s}")
    for nm, wi, sei in zip(names, w, se):
        print(f"{nm:16s} {wi:10.4f} {sei:9.4f} {wi/sei:7.2f}")

    b_delta, b_qb, b_fantasy, b_skill = w[1], w[2], w[3], w[4]
    print(f"\nDelta-equivalent conversion (coef / delta's own coef):")
    print(f"  QB_QUALITY_GAP_WEIGHT   = {b_qb/b_delta:+.4f}   (previous production: -1.0556)")
    print(f"  QB_FANTASY_GAP_WEIGHT   = {b_fantasy/b_delta:+.4f}   (new)")
    print(f"  SKILL_EPA_OUT_WEIGHT    = {b_skill/b_delta:+.4f}   (previous production: -0.0364)")

    for key, label in (("qb_gap_diff", "qb_gap_diff"), ("fantasy_gap_diff", "fantasy_gap_diff"), ("skill_out_diff", "skill_out_diff")):
        vals = np.array([g[key] for g in games])
        print(f"  {label}: mean={vals.mean():+.4f} std={vals.std():.4f} nonzero={np.mean(vals!=0)*100:.1f}%")


if __name__ == "__main__":
    main()
