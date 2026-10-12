"""v49: v48 found that TEAM-TOTAL fantasy projections add nothing once the market line is
in the model -- they're mostly just a noisy copy of "who's favored," which this project
already knows beats the EPA composite (that's why the market is kept OUT of delta by
design, not fed into it). This tests a narrower, different hypothesis: does FantasyPros'
WEEKLY player-level projection make a better (or complementary) INJURY-GAP measurement than
trailing EPA/dropback, for the QB-health correction specifically?

Why this might escape v48's market-redundancy problem: a team-total projection (used in v48)
directly encodes "how good is this team expected to be this week," which is close to
restating the market line. A within-team GAP (this team's own healthy-QB baseline vs this
team's own actual-QB-this-week reality) is a narrower quantity -- it's not "who wins," it's
"how much is a specific backup costing THIS team," which the market line doesn't isolate on
its own. Checked directly below (corr with -spread_line) to confirm this holds, not just
assumed.

Production's existing mechanism (team_ratings.qb_quality_gap_data, window5 primary + decay
rate 0.4): gap = primary's trailing EPA/dropback (decayed by backup tenure) minus actual
passer's trailing EPA/dropback -- a backward-looking per-play efficiency measure, sourced
from nflverse PBP.

Candidate: fantasy_gap = primary's own trailing AVERAGE weekly fantasy-points projection
(from games where he was actually the one projected to start, no lookahead) minus the actual
passer's CURRENT week projection -- a forward-looking, WEEKLY-REFRESHED measure that can
react to real-time news (new OC, opponent matchup, weather) the EWMA-only EPA number can't
see until it's folded into next week's rolling average.

Tests three things against the SAME walk-forward joint-logistic recipe as every other
additive-correction test here: fantasy_gap ALONE vs delta (replaces the EPA gap entirely),
epa_gap + fantasy_gap together vs epa_gap alone (does fantasy add anything on top of the
already-shipped mechanism), and the market-spread correlation check to confirm this isn't
just v48's problem again.

Run: python research/edge_signal_test_v49_fantasypros_qb_gap.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

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
v40 = _load("v40", "edge_signal_test_v40_qb_recency_decay.py")
v45 = _load("v45", "edge_signal_test_v45_qb_decay_rate_sweep.py")

FIRST_SEASON = 2009
TEST_START_SEASON = 2020  # bounded by FantasyPros' own coverage (2017+), same as v48
PRIMARY_MODE = "window5"
DECAY_RATE = 0.4  # matches shipped QB_GAP_DECAY_RATE


def build_trailing_fantasy_proj(game_order, actual_passer, gsis_to_fpid, player_proj):
    """Mirrors v40.build_trailing_epa's exact no-lookahead structure, but trailing per-passer
    AVERAGE is built from FantasyPros' own weekly projected points instead of PBP EPA/play."""
    passer_hist: dict[str, list[float]] = defaultdict(list)
    trailing_proj: dict[tuple[str, str], float | None] = {}
    league_avg: list[float] = []
    for gid, team, season, week in game_order:
        ap = actual_passer.get((gid, team))
        if ap is None:
            continue
        hist = passer_hist[ap]
        trailing_proj[(ap, gid)] = (sum(hist) / len(hist)) if hist else None
        fpid = gsis_to_fpid.get(ap)
        proj = player_proj.get((season, week, fpid)) if fpid is not None else None
        if proj is not None:
            hist.append(proj)
            league_avg.append(proj)
    league_avg_proj = (sum(league_avg) / len(league_avg)) if league_avg else 0.0
    return trailing_proj, league_avg_proj


def build_fantasy_gap(game_order, actual_passer, primary_so_far, trailing_proj, league_avg_proj,
                       gsis_to_fpid, player_proj):
    """Same decay-by-tenure structure as the shipped EPA gap (QB_GAP_DECAY_RATE=0.4), just
    built from fantasy-projection trailing averages instead of EPA. actual passer's CURRENT
    week value uses his real projection directly (not a trailing average) when available --
    that's the whole point: a fresh, this-week number, not a backward-looking one."""
    gap: dict[tuple[str, str], float] = {}
    streak_n: dict[str, int] = defaultdict(int)
    for gid, team, season, week in game_order:
        primary = primary_so_far.get((team, gid))
        actual = actual_passer.get((gid, team))
        if primary is None or actual is None:
            gap[(team, gid)] = 0.0
            continue
        tp = trailing_proj.get((primary, gid))
        tp = tp if tp is not None else league_avg_proj
        fpid = gsis_to_fpid.get(actual)
        ta = player_proj.get((season, week, fpid)) if fpid is not None else None
        if ta is None:
            ta = trailing_proj.get((actual, gid))
        ta = ta if ta is not None else league_avg_proj
        raw_gap = tp - ta

        if actual != primary:
            streak_n[team] += 1
            decay_factor = (1 - DECAY_RATE) ** (streak_n[team] - 1)
            gap[(team, gid)] = raw_gap * decay_factor
        else:
            streak_n[team] = 0
            gap[(team, gid)] = 0.0
    return gap


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


def evaluate(games, feature_keys, restrict_ids=None):
    seasons = sorted({g["season"] for g in games})
    all_y, all_p, all_old = [], [], []
    for test_season in seasons:
        if test_season < TEST_START_SEASON:
            continue
        train = [g for g in games if g["season"] < test_season]
        test = [g for g in games if g["season"] == test_season]
        if restrict_ids is not None:
            test = [g for g in test if g["game_id"] in restrict_ids]
        if len(train) < 150 or not test:
            continue
        stats = {}
        for k in feature_keys:
            vals = np.array([g[k] for g in train])
            stats[k] = (vals.mean(), vals.std() or 1.0)
        def feat(g):
            return [g["delta"]] + [(g[k] - stats[k][0]) / stats[k][1] for k in feature_keys]
        X_train = np.array([feat(g) for g in train])
        y_train = np.array([g["home_win"] for g in train], dtype=float)
        w = fit_logistic(X_train, y_train)
        for g in test:
            f = feat(g)
            z = w[0] + sum(wi * fi for wi, fi in zip(w[1:], f))
            p_hat = 1.0 / (1.0 + np.exp(-z))
            all_y.append(g["home_win"])
            all_p.append(p_hat)
            all_old.append(1 if g["delta"] > 0 else 0)
    all_y = np.array(all_y)
    n = len(all_y)
    acc_old = float(np.mean(np.array(all_old) == all_y))
    acc_new = float(np.mean((np.array(all_p) > 0.5).astype(int) == all_y))
    auc = roc_auc_score(all_y, all_p)
    ll = log_loss(all_y, all_p)
    return n, acc_old, acc_new, auc, ll


def main():
    print("Building window5 game order + id crosswalk + player projections...", file=sys.stderr)
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

    ids = pd.read_parquet(os.path.join(HERE, "cache", "fantasypros_id_map.parquet"))
    ids = ids.dropna(subset=["fantasypros_id", "gsis_id"]).copy()
    ids["fantasypros_id"] = ids["fantasypros_id"].astype(int)
    gsis_to_fpid = dict(zip(ids["gsis_id"], ids["fantasypros_id"]))

    pw = pd.read_parquet(os.path.join(HERE, "cache", "fantasypros_player_week.parquet"))
    pw_qb = pw[pw.position == "QB"]
    player_proj = {(r.season, r.week, r.fpid): r.proj_points for r in pw_qb.itertuples(index=False)}

    trailing_proj, league_avg_proj = build_trailing_fantasy_proj(game_order, actual_passer, gsis_to_fpid, player_proj)
    epa_gap = v45.build_qb_gap_decay_rate(game_order, actual_passer, primary_so_far, trailing_epa, league_avg_epa, backup_avg_epa, DECAY_RATE)
    fantasy_gap = build_fantasy_gap(game_order, actual_passer, primary_so_far, trailing_proj, league_avg_proj, gsis_to_fpid, player_proj)

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
    for r in rows_all:
        if r["delta"] == 0 or r["season"] < 2017:
            continue
        r["epa_gap_diff"] = epa_gap.get((r["home"], r["game_id"]), 0.0) - epa_gap.get((r["away"], r["game_id"]), 0.0)
        r["fantasy_gap_diff"] = fantasy_gap.get((r["home"], r["game_id"]), 0.0) - fantasy_gap.get((r["away"], r["game_id"]), 0.0)
        games.append(r)
    print(f"{len(games)} games, 2017-2025\n")

    nonzero_epa = np.mean([g["epa_gap_diff"] != 0 for g in games])
    nonzero_fantasy = np.mean([g["fantasy_gap_diff"] != 0 for g in games])
    print(f"epa_gap_diff nonzero: {100*nonzero_epa:.1f}%, fantasy_gap_diff nonzero: {100*nonzero_fantasy:.1f}%")
    both = [g for g in games if g.get("spread_line") is not None]
    sp = np.array([-g["spread_line"] for g in both])
    print(f"corr(epa_gap_diff, -spread_line) = {np.corrcoef([g['epa_gap_diff'] for g in both], sp)[0,1]:.3f}")
    print(f"corr(fantasy_gap_diff, -spread_line) = {np.corrcoef([g['fantasy_gap_diff'] for g in both], sp)[0,1]:.3f}\n")

    print(f"=== walk-forward, {TEST_START_SEASON}-2025, delta + feature(s), FULL population ===")
    print(f"{'features':30s} {'n':>5s} {'acc(delta)':>11s} {'acc(+feat)':>11s} {'AUC':>7s} {'logloss':>8s}")
    feature_sets = [
        ("epa_gap (shipped)", ["epa_gap_diff"]),
        ("fantasy_gap (replace)", ["fantasy_gap_diff"]),
        ("epa_gap + fantasy_gap", ["epa_gap_diff", "fantasy_gap_diff"]),
    ]
    for label, keys in feature_sets:
        n, acc_old, acc_new, auc, ll = evaluate(games, keys)
        print(f"{label:30s} {n:5d} {100*acc_old:10.2f}% {100*acc_new:10.2f}% {auc:7.4f} {ll:8.4f}")

    # Restricted to games where EITHER gap is actually nonzero (a real QB change happened) --
    # same lens that clarified the decay-rate signal earlier: ~65-70% of games have NEITHER
    # gap active and dilute the aggregate otherwise.
    test_games = [g for g in games if g["season"] >= TEST_START_SEASON]
    affected_ids = {g["game_id"] for g in test_games if g["epa_gap_diff"] != 0 or g["fantasy_gap_diff"] != 0}
    print(f"\n=== walk-forward, RESTRICTED to {len(affected_ids)} games with an active QB change ===")
    print(f"{'features':30s} {'n':>5s} {'acc(delta)':>11s} {'acc(+feat)':>11s} {'AUC':>7s} {'logloss':>8s}")
    for label, keys in feature_sets:
        n, acc_old, acc_new, auc, ll = evaluate(games, keys, restrict_ids=affected_ids)
        print(f"{label:30s} {n:5d} {100*acc_old:10.2f}% {100*acc_new:10.2f}% {auc:7.4f} {ll:8.4f}")


if __name__ == "__main__":
    main()
