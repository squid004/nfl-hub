"""v23: the actual backtest -- does adding QB health (already live as its own dashboard
filter, but never folded into a PREDICTION) and the newly-validated skill-position
short-handed signal (v22: p70 threshold, touches-weighted, confirmed train-early/test-late)
to the composite `delta` change which side the model predicts, and does that help?

Walk-forward, same discipline as every other fit in this project (research/edge_signal_
test_v13_new_window_weights.py etc.): for each test season, fit a joint logistic model
  home_win ~ delta + qb_flag_diff + skill_short_diff
on every season BEFORE it only, then apply that exact fit to the test season -- including
recomputing the skill threshold's numeric value from train-only data each fold (not the
full-dataset percentile), so nothing about the test season leaks into its own prediction.
`delta` itself is untouched (still today's production POWER_WEIGHTS/standardization) --
this ADDS two terms on top, it doesn't refit the 8-stat composite itself.

qb_flag: home team's QB listed Out/Doubtful that week minus away team's -- reuses
nflhub.sources.team_ratings._qb_out_doubtful_by_week(), the EXACT function the live
dashboard filter already calls, so this can't drift from what's actually shipped.
skill_short: home team's touches-weighted skill-group severity >= that fold's own p70
threshold, minus the same for away -- reuses v21's severity computation unchanged.

For every game where the adjusted model's predicted winner DIFFERS from delta-alone's own
pick, exactly one of them must be right (only two teams), so every flip is either an
improvement (adjusted model newly correct) or a regression (newly wrong) -- reports both
counts, the net hit-rate change, and a McNemar exact test on the flips (same test used for
the backup-QB dropout check earlier this session) to judge whether the net change is real
or could be noise.

Run: python research/edge_signal_test_v23_full_backtest.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import binomtest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
from nflhub.sources import team_ratings  # noqa: E402

import importlib.util
def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

v14 = _load("v14", "edge_signal_test_v14_qb_injury_dropout.py")
v21 = _load("v21", "edge_signal_test_v21_skill_touches_weighted.py")

FIRST_SEASON = 2009
TEST_START_SEASON = 2014  # needs enough train games to fit a 4-parameter logistic reliably
SKILL_PERCENTILE = 70     # v22's finding: p70, not p85/p95


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
    return res.x


def main():
    print("Building game dataset + production delta (2009-2025)...", file=sys.stderr)
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
    print(f"{len(games)} decided games, {FIRST_SEASON}-{max(g['season'] for g in games)}\n")

    print("Loading QB Out/Doubtful flags (same function the live dashboard filter uses)...", file=sys.stderr)
    last_season = max(g["season"] for g in games)
    qb_out = team_ratings._qb_out_doubtful_by_week(last_season)

    print("Loading touches-weighted skill-group severity (v21)...", file=sys.stderr)
    player_games, fallback = v21.load_skill_usage()
    skill_injury_rows = v21.load_skill_injury_rows()
    sev_skill = v21.weighted_severity(skill_injury_rows, player_games, fallback, "touches")

    def qb_diff(g):
        h = 1 if qb_out.get((g["home"], g["season"], g["week"]), False) else 0
        a = 1 if qb_out.get((g["away"], g["season"], g["week"]), False) else 0
        return h - a

    def skill_diff(g, threshold_val):
        h = 1 if sev_skill.get((g["home"], g["season"], g["week"]), 0.0) >= threshold_val else 0
        a = 1 if sev_skill.get((g["away"], g["season"], g["week"]), 0.0) >= threshold_val else 0
        return h - a

    seasons = sorted({g["season"] for g in games})
    all_old_hits = all_new_hits = all_n = 0
    improvements = regressions = 0
    flip_examples = []
    season_rows = []
    coef_history = []

    for test_season in seasons:
        if test_season < TEST_START_SEASON:
            continue
        train = [g for g in games if g["season"] < test_season]
        test = [g for g in games if g["season"] == test_season]
        if len(train) < 300 or not test:
            continue

        train_team_weeks = {(g[s], g["season"], g["week"]) for g in train for s in ("home", "away")}
        train_sev_vals = np.array([sev_skill.get(tw, 0.0) for tw in train_team_weeks])
        p70_val = np.percentile(train_sev_vals, SKILL_PERCENTILE)

        X_train = np.array([[g["delta"], qb_diff(g), skill_diff(g, p70_val)] for g in train])
        y_train = np.array([g["home_win"] for g in train], dtype=float)
        w = fit_logistic(X_train, y_train)
        coef_history.append((test_season, *w))

        season_old = season_new = season_n = 0
        for g in test:
            old_pred_home = g["delta"] > 0
            z = w[0] + w[1] * g["delta"] + w[2] * qb_diff(g) + w[3] * skill_diff(g, p70_val)
            new_pred_home = z > 0
            actual_home = g["home_win"] == 1

            old_hit = old_pred_home == actual_home
            new_hit = new_pred_home == actual_home
            season_old += old_hit
            season_new += new_hit
            season_n += 1
            all_old_hits += old_hit
            all_new_hits += new_hit
            all_n += 1

            if old_pred_home != new_pred_home:
                if new_hit and not old_hit:
                    improvements += 1
                    if len(flip_examples) < 40:
                        flip_examples.append((g, "IMPROVEMENT", old_pred_home, new_pred_home, actual_home))
                elif old_hit and not new_hit:
                    regressions += 1
                    if len(flip_examples) < 40:
                        flip_examples.append((g, "REGRESSION", old_pred_home, new_pred_home, actual_home))

        season_rows.append((test_season, season_n, season_old, season_new))

    print(f"{'season':>6s} {'n':>5s} {'old hit%':>9s} {'new hit%':>9s} {'delta':>7s}")
    for s, n, o, nw in season_rows:
        print(f"{s:6d} {n:5d} {100*o/n:8.1f}% {100*nw/n:8.1f}% {100*(nw-o)/n:+6.1f}pt")

    print(f"\n=== overall, {TEST_START_SEASON}-{seasons[-1]} walk-forward, {all_n} games ===")
    print(f"old (delta alone):        {all_old_hits}/{all_n} = {100*all_old_hits/all_n:.2f}%")
    print(f"new (delta+QB+skill):     {all_new_hits}/{all_n} = {100*all_new_hits/all_n:.2f}%")
    print(f"net change: {all_new_hits-all_old_hits:+d} games ({100*(all_new_hits-all_old_hits)/all_n:+.2f}pt)")
    print(f"\ntotal flips: {improvements+regressions}  (improvements={improvements}, regressions={regressions})")
    n_disc = improvements + regressions
    if n_disc >= 10:
        p = binomtest(improvements, n_disc, 0.5).pvalue
        print(f"McNemar exact test on {n_disc} flips: p={p:.4f}")

    print("\n=== fitted coefficients by fold (sanity check -- are they stable?) ===")
    print(f"{'season':>6s} {'intercept':>10s} {'delta':>8s} {'qb':>8s} {'skill':>8s}")
    for s, b0, b1, b2, b3 in coef_history:
        print(f"{s:6d} {b0:10.3f} {b1:8.3f} {b2:8.3f} {b3:8.3f}")

    print("\n=== sample flipped games ===")
    for g, kind, old_h, new_h, actual_h in flip_examples[:20]:
        old_pick = g["home"] if old_h else g["away"]
        new_pick = g["home"] if new_h else g["away"]
        actual = g["home"] if actual_h else g["away"]
        print(f"  {kind:11s} {g['season']} wk{g['week']:>2d}  {g['away']}@{g['home']}  "
              f"old picked {old_pick} -> new picked {new_pick}  (actual winner: {actual})")


if __name__ == "__main__":
    main()
