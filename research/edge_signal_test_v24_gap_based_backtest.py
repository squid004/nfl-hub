"""v24: fixes v23's flat-flag problem by switching BOTH QB and skill-position severity to
the EWMA-GAP formulation (v16's original framing, validated repeatedly as the positive
control through v16-v20) instead of a flat "out this week" / "above a fixed percentile"
flag. The diagnosis from v23: `delta` itself already partially absorbs a long-running
absence through its own EWMA (a QB or skill corps that's been thin for 6 straight weeks has
already dragged the team's actual EPA/points down into the rating) -- a FLAT penalty on top
double-counts that and can overcorrect. The GAP (this week's severity minus the EWMA of
recent severity, same alpha=0.2, reset each season) is designed to cancel to ~0 exactly once
the rating has caught up, and only bite hard in the fresh-injury (or fresh-return) window
where the rating is still wrong -- which is the window an adjustment should actually target.

QB: same binary Out/Doubtful signal as the live dashboard filter (nflhub.sources.
team_ratings._qb_out_doubtful_by_week()), just gap-transformed instead of used flat.
Skill: the touches-weighted severity from v21 (more players to track than QB -- summed
across every flagged RB/WR/TE/FB each week, weighted by their own trailing touches/game),
also gap-transformed the same way, instead of v22/v23's binary p70-threshold flag.

Same walk-forward joint-logistic backtest structure as v23 (train on seasons before the
test season only, including re-fitting both gap-column standardization AND the logistic
coefficients from train data alone each fold) -- only the two input signals changed.

Run: python research/edge_signal_test_v24_gap_based_backtest.py
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
TEST_START_SEASON = 2014
EWMA_ALPHA = 0.2  # matches the rating's own half-life -- same reasoning as v16


def build_gap(team_weeks: list[tuple[str, int, int]], severity: dict) -> dict:
    """{(team, season, week): gap}, gap = this week's severity - the EWMA of severity
    strictly BEFORE this week (reset to None at every season boundary). Identical logic to
    v16.build_health_gaps(), generalized to a single flat severity series instead of
    v16's (team,season,week,group)-keyed multi-group dict."""
    ewma: dict[str, float | None] = {}
    last_season: dict[str, int] = {}
    gaps: dict[tuple[str, int, int], float] = {}
    for team, season, week in team_weeks:
        if last_season.get(team) is not None and last_season[team] != season:
            ewma[team] = None
        last_season[team] = season
        val = severity.get((team, season, week), 0.0)
        prev = ewma.get(team)
        gaps[(team, season, week)] = 0.0 if prev is None else val - prev
        ewma[team] = val if prev is None else (1 - EWMA_ALPHA) * prev + EWMA_ALPHA * val
    return gaps


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
    last_season = max(g["season"] for g in games)
    print(f"{len(games)} decided games, {FIRST_SEASON}-{last_season}\n")

    print("Loading QB Out/Doubtful flags...", file=sys.stderr)
    qb_out = team_ratings._qb_out_doubtful_by_week(last_season)
    qb_severity = {k: 1.0 for k, v in qb_out.items() if v}

    print("Loading touches-weighted skill-group severity (v21)...", file=sys.stderr)
    player_games, fallback = v21.load_skill_usage()
    skill_injury_rows = v21.load_skill_injury_rows()
    sev_skill = v21.weighted_severity(skill_injury_rows, player_games, fallback, "touches")

    team_weeks = sorted({(g[s], g["season"], g["week"]) for g in games for s in ("home", "away")})
    gap_qb = build_gap(team_weeks, qb_severity)
    gap_skill = build_gap(team_weeks, sev_skill)

    def gap_diff(g, gap_dict):
        h = gap_dict.get((g["home"], g["season"], g["week"]), 0.0)
        a = gap_dict.get((g["away"], g["season"], g["week"]), 0.0)
        return h - a

    for g in games:
        g["gap_qb_diff"] = gap_diff(g, gap_qb)
        g["gap_skill_diff"] = gap_diff(g, gap_skill)

    print("--- gap distribution sanity check (all games) ---")
    for key in ("gap_qb_diff", "gap_skill_diff"):
        vals = np.array([g[key] for g in games])
        print(f"  {key:16s} mean={vals.mean():+.4f} std={vals.std():.4f} nonzero={np.mean(vals!=0)*100:.1f}%")

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

        # standardize the two gap columns from TRAIN only, apply same transform to test
        qb_train = np.array([g["gap_qb_diff"] for g in train])
        skill_train = np.array([g["gap_skill_diff"] for g in train])
        qb_mu, qb_sd = qb_train.mean(), (qb_train.std() or 1.0)
        skill_mu, skill_sd = skill_train.mean(), (skill_train.std() or 1.0)

        def feat(g):
            return [g["delta"], (g["gap_qb_diff"] - qb_mu) / qb_sd, (g["gap_skill_diff"] - skill_mu) / skill_sd]

        X_train = np.array([feat(g) for g in train])
        y_train = np.array([g["home_win"] for g in train], dtype=float)
        w = fit_logistic(X_train, y_train)
        coef_history.append((test_season, *w))

        season_old = season_new = season_n = 0
        for g in test:
            old_pred_home = g["delta"] > 0
            f = feat(g)
            z = w[0] + w[1] * f[0] + w[2] * f[1] + w[3] * f[2]
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

    print(f"\n{'season':>6s} {'n':>5s} {'old hit%':>9s} {'new hit%':>9s} {'delta':>7s}")
    for s, n, o, nw in season_rows:
        print(f"{s:6d} {n:5d} {100*o/n:8.1f}% {100*nw/n:8.1f}% {100*(nw-o)/n:+6.1f}pt")

    print(f"\n=== overall, {TEST_START_SEASON}-{seasons[-1]} walk-forward, {all_n} games ===")
    print(f"old (delta alone):             {all_old_hits}/{all_n} = {100*all_old_hits/all_n:.2f}%")
    print(f"new (delta + QB gap + skill gap): {all_new_hits}/{all_n} = {100*all_new_hits/all_n:.2f}%")
    print(f"net change: {all_new_hits-all_old_hits:+d} games ({100*(all_new_hits-all_old_hits)/all_n:+.2f}pt)")
    print(f"\ntotal flips: {improvements+regressions}  (improvements={improvements}, regressions={regressions})")
    n_disc = improvements + regressions
    if n_disc >= 10:
        p = binomtest(improvements, n_disc, 0.5).pvalue
        print(f"McNemar exact test on {n_disc} flips: p={p:.4f}")

    print("\n=== fitted coefficients by fold ===")
    print(f"{'season':>6s} {'intercept':>10s} {'delta':>8s} {'qb_gap':>8s} {'skill_gap':>10s}")
    for s, b0, b1, b2, b3 in coef_history:
        print(f"{s:6d} {b0:10.3f} {b1:8.3f} {b2:8.3f} {b3:10.3f}")

    print("\n=== sample flipped games ===")
    for g, kind, old_h, new_h, actual_h in flip_examples[:20]:
        old_pick = g["home"] if old_h else g["away"]
        new_pick = g["home"] if new_h else g["away"]
        actual = g["home"] if actual_h else g["away"]
        print(f"  {kind:11s} {g['season']} wk{g['week']:>2d}  {g['away']}@{g['home']}  "
              f"old picked {old_pick} -> new picked {new_pick}  (actual winner: {actual})")


if __name__ == "__main__":
    main()
