"""v48: does FantasyPros' weekly fantasy-football projections carry any real incremental
signal for the Power Model, on top of the existing 8-stat EPA composite (`delta`, frozen at
PRODUCTION_WEIGHTS)?

Data source: ff-draft-edge's FantasyPros API client (premium tier, verified live) +
nflverse's own weekly box scores for the correct historical team-per-player (FantasyPros'
own `team_id` field is a CURRENT-roster join regardless of season/week requested -- verified
live: Tom Brady's team_id reads "FA" on every season 2018-2022 we pulled, including seasons
he was an active starter, because that's his CURRENT 2023-retirement status, not the
historical reality). Checked for lookahead contamination before trusting this for a
backtest: Puka Nacua's week-1 2023 projection (an unproven 5th-round rookie, before his real
breakout) was 4.67 half-PPR points, ranked 84th among WRs that week -- a genuinely clueless
pre-game number, not hindsight-informed. See ff-draft-edge/scripts/
pull_team_week_projections.py for the full pull/aggregation (run separately; this script
only reads its output, research/cache/fantasypros_team_week.parquet).

Three candidate signals, all built from the SAME two team-week numbers (proj_off_points =
summed QB/RB/WR/TE projected half-PPR points; proj_def_points_allowed = that team's own DST
projected points allowed):
  - proj_margin_diff: (home_off - away_def_allowed) - (away_off - home_def_allowed) -- a
    matchup-adjusted projected scoring margin, same phase-paired structure as the existing
    8-stat composite (each side's offense against the OTHER side's defense).
  - proj_off_only_diff: home_off - away_off -- pure offense signal, ignoring defense.
  - proj_def_only_diff: away_def_allowed - home_def_allowed -- pure defense signal (higher
    value = away's defense projected weaker = favors home).

Each tested via the same walk-forward joint-logistic recipe as every other additive-
correction test in this project (v25/v26/v37/etc.): fit home_win ~ delta + candidate_diff
(candidate z-scored on the training fold), compare AUC/accuracy/logloss against delta alone.
Only seasons FantasyPros projection history actually covers (2017+) are in scope -- far
narrower than this project's usual 2007+ walk-forward, so results here carry a smaller,
noisier sample than most of this project's other backtests; read the numbers with that in
mind.

Run: python research/edge_signal_test_v48_fantasypros_projections.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

import importlib.util
def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

v14 = _load("v14", "edge_signal_test_v14_qb_injury_dropout.py")

FP_PATH = os.path.join(HERE, "cache", "fantasypros_team_week.parquet")
TEST_START_SEASON = 2020  # leaves 2017-2019 as walk-forward training seed within FP's own range


def fit_logistic(X, y):
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


def evaluate(games, key=None):
    """key=None fits delta ALONE (1 feature) -- the properly walk-forward-calibrated
    baseline to compare every candidate against, not just delta's raw-sign accuracy.
    Returns per-season breakdown too (season -> (n, acc)) so a candidate's improvement can
    be checked for consistency across years, not just trusted as one aggregate number."""
    seasons = sorted({g["season"] for g in games})
    all_y, all_p = [], []
    by_season: dict[int, list] = {}
    for test_season in seasons:
        if test_season < TEST_START_SEASON:
            continue
        train = [g for g in games if g["season"] < test_season]
        test = [g for g in games if g["season"] == test_season]
        if len(train) < 150 or not test:
            continue
        if key is None:
            def feat(g):
                return [g["delta"]]
        else:
            vals = np.array([g[key] for g in train])
            mu, sd = vals.mean(), (vals.std() or 1.0)
            def feat(g, mu=mu, sd=sd):
                return [g["delta"], (g[key] - mu) / sd]
        X_train = np.array([feat(g) for g in train])
        y_train = np.array([g["home_win"] for g in train], dtype=float)
        w = fit_logistic(X_train, y_train)
        season_y, season_p = [], []
        for g in test:
            f = feat(g)
            z = w[0] + sum(wi * fi for wi, fi in zip(w[1:], f))
            p_hat = 1.0 / (1.0 + np.exp(-z))
            all_y.append(g["home_win"])
            all_p.append(p_hat)
            season_y.append(g["home_win"])
            season_p.append(p_hat)
        sy, sp = np.array(season_y), np.array(season_p)
        by_season[test_season] = (len(sy), float(np.mean((sp > 0.5).astype(int) == sy)))
    all_y = np.array(all_y)
    n = len(all_y)
    acc = float(np.mean((np.array(all_p) > 0.5).astype(int) == all_y))
    auc = roc_auc_score(all_y, all_p)
    ll = log_loss(all_y, all_p)
    return n, acc, auc, ll, by_season


def evaluate_market_controlled(games, key=None):
    """delta + spread_line (+ optionally `key`) -- the real test of whether a candidate adds
    anything BEYOND what the market line itself already provides. Comparing a candidate
    against delta-ALONE (no market) is the wrong baseline whenever the candidate turns out
    to correlate strongly with the market itself (checked directly: proj_def_only_diff
    correlates with -spread_line at 0.956, and delta+spread_line alone reproduces almost
    its exact AUC/accuracy) -- that comparison just re-discovers "the market beats the
    model," which this project has always known and deliberately keeps OUT of delta by
    design, not a new finding."""
    seasons = sorted({g["season"] for g in games})
    all_y, all_p = [], []
    for test_season in seasons:
        if test_season < TEST_START_SEASON:
            continue
        train = [g for g in games if g["season"] < test_season]
        test = [g for g in games if g["season"] == test_season]
        if len(train) < 150 or not test:
            continue
        sp_vals = np.array([-g["spread_line"] for g in train])
        sp_mu, sp_sd = sp_vals.mean(), (sp_vals.std() or 1.0)
        if key is not None:
            vals = np.array([g[key] for g in train])
            mu, sd = vals.mean(), (vals.std() or 1.0)
        def feat(g):
            f = [g["delta"], (-g["spread_line"] - sp_mu) / sp_sd]
            if key is not None:
                f.append((g[key] - mu) / sd)
            return f
        X_train = np.array([feat(g) for g in train])
        y_train = np.array([g["home_win"] for g in train], dtype=float)
        w = fit_logistic(X_train, y_train)
        for g in test:
            f = feat(g)
            z = w[0] + sum(wi * fi for wi, fi in zip(w[1:], f))
            p_hat = 1.0 / (1.0 + np.exp(-z))
            all_y.append(g["home_win"])
            all_p.append(p_hat)
    all_y = np.array(all_y)
    n = len(all_y)
    acc = float(np.mean((np.array(all_p) > 0.5).astype(int) == all_y))
    auc = roc_auc_score(all_y, all_p)
    ll = log_loss(all_y, all_p)
    return n, acc, auc, ll


def main():
    if not os.path.exists(FP_PATH):
        print(f"Missing {FP_PATH} -- run ff-draft-edge/scripts/pull_team_week_projections.py first.")
        return

    fp = pd.read_parquet(FP_PATH)
    print(f"Loaded {len(fp)} FantasyPros team-week rows, seasons {fp.season.min()}-{fp.season.max()}")
    fp_lookup = {(r.season, r.week, r.team): r for r in fp.itertuples(index=False)}

    print("Building game dataset + frozen production delta...", file=sys.stderr)
    rows_all = v14.build_dataset()
    cols = [f"{m}_diff" for m in v14.RATING_METRICS]
    X_all = np.array([[r[c] for c in cols] for r in rows_all])
    mu_all, sd_all = X_all.mean(axis=0), X_all.std(axis=0)
    sd_all[sd_all == 0] = 1.0
    PW = v14.PRODUCTION_WEIGHTS
    for i, r in enumerate(rows_all):
        x = X_all[i]
        r["delta"] = sum(PW[m] * ((x[j] - mu_all[j]) / sd_all[j]) for j, m in enumerate(v14.RATING_METRICS))

    games = []
    missing = 0
    for r in rows_all:
        if r["delta"] == 0 or r["season"] < 2017:
            continue
        hr = fp_lookup.get((r["season"], r["week"], r["home"]))
        ar = fp_lookup.get((r["season"], r["week"], r["away"]))
        # pandas represents a missing float as NaN, not None -- `x is None` silently let NaN
        # rows through on the first run, which blew up roc_auc_score ("Input contains NaN")
        # several stages downstream instead of failing here where the actual problem is.
        if hr is None or ar is None or pd.isna(hr.proj_off_points) or pd.isna(ar.proj_off_points) \
           or pd.isna(hr.proj_def_points_allowed) or pd.isna(ar.proj_def_points_allowed):
            missing += 1
            continue
        r["proj_margin_diff"] = (hr.proj_off_points - ar.proj_def_points_allowed) - (ar.proj_off_points - hr.proj_def_points_allowed)
        r["proj_off_only_diff"] = hr.proj_off_points - ar.proj_off_points
        r["proj_def_only_diff"] = ar.proj_def_points_allowed - hr.proj_def_points_allowed
        games.append(r)

    print(f"{len(games)} games with both delta and FantasyPros projections (2017+), {missing} skipped for missing projection data\n")

    # Descriptive: how good are these projections on their own terms? (not the main
    # hypothesis test, just useful context for interpreting what follows)
    proj_off = np.array([fp_lookup[(r["season"], r["week"], r["home"])].proj_off_points for r in games])
    actual_off = np.array([fp_lookup[(r["season"], r["week"], r["home"])].actual_off_points for r in games])
    valid = ~np.isnan(actual_off)
    corr = np.corrcoef(proj_off[valid], actual_off[valid])[0, 1]
    print(f"Descriptive: corr(projected home-offense points, actual home-offense points) = {corr:.3f} (n={valid.sum()})\n")

    n0, acc0, auc0, ll0, by_season0 = evaluate(games, key=None)
    print(f"=== walk-forward comparison, {TEST_START_SEASON}-2025, delta + candidate (home-away) ===")
    print(f"{'candidate':20s} {'n':>5s} {'acc':>7s} {'AUC':>7s} {'logloss':>8s}")
    print(f"{'delta alone':20s} {n0:5d} {100*acc0:6.2f}% {auc0:7.4f} {ll0:8.4f}")
    results = {}
    for key in ("proj_margin_diff", "proj_off_only_diff", "proj_def_only_diff"):
        n, acc, auc, ll, by_season = evaluate(games, key)
        print(f"{key:20s} {n:5d} {100*acc:6.2f}% {auc:7.4f} {ll:8.4f}")
        results[key] = by_season

    print(f"\n=== per-season accuracy, delta alone vs each candidate (consistency check) ===")
    seasons_tested = sorted(by_season0)
    header = f"{'season':>6s} {'n':>4s} {'delta':>7s} " + " ".join(f"{k.replace('proj_','').replace('_diff',''):>12s}" for k in results)
    print(header)
    for s in seasons_tested:
        n_s, acc_s = by_season0[s]
        row = f"{s:6d} {n_s:4d} {100*acc_s:6.2f}% "
        row += " ".join(f"{100*results[k][s][1]:11.2f}%" for k in results)
        print(row)

    # The real test: does each candidate add anything once the market line is ALREADY in
    # the model? Comparing against delta-alone (above) rewards anything correlated with the
    # market, which both proj_margin_diff and especially proj_def_only_diff turn out to be
    # (checked directly: corr with -spread_line is -0.49 and -0.96 respectively) -- this
    # project already knows the market beats the Power Model, so "beats delta-alone" can
    # just mean "leaked a worse copy of the market," not "found something new."
    mkt_games = [g for g in games if g.get("spread_line") is not None]
    n_m, acc_m, auc_m, ll_m = evaluate_market_controlled(mkt_games, key=None)
    print(f"\n=== market-controlled: delta + spread_line (+ candidate), {TEST_START_SEASON}-2025 ===")
    print(f"{'candidate':20s} {'n':>5s} {'acc':>7s} {'AUC':>7s} {'logloss':>8s}")
    print(f"{'(none)':20s} {n_m:5d} {100*acc_m:6.2f}% {auc_m:7.4f} {ll_m:8.4f}")
    for key in ("proj_margin_diff", "proj_off_only_diff", "proj_def_only_diff"):
        n, acc, auc, ll = evaluate_market_controlled(mkt_games, key)
        delta_auc = auc - auc_m
        print(f"{key:20s} {n:5d} {100*acc:6.2f}% {auc:7.4f} {ll:8.4f}  (AUC vs market-only: {delta_auc:+.4f})")


if __name__ == "__main__":
    main()
