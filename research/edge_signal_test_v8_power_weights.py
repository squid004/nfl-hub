"""v8: jointly optimize the power-ranking composite weights across all 8 stats, instead of
v7's 8 independent univariate fits.

Why not just run one ordinary joint logistic regression? Already tried (see the "Power-
ranking composite weights" section of v7's output) -- it sign-flips weak/collinear stats
(turnovers committed came out positively weighted) because points, EPA, and turnovers
overlap heavily and an unconstrained fit can't tell "genuinely bad predictor" from "redundant
with a stronger correlated predictor" apart from a sign flip.

Fix: orient every stat so higher-is-always-better (flip sign on allowed/committed stats),
then fit ONE L2-regularized logistic regression with coefficients constrained >= 0. A
non-negativity constraint can only shrink a weak/redundant stat toward zero, never flip its
sign -- structurally rules out the v7 pathology. Regularization strength is chosen by the
same walk-forward validation used throughout this project (out-of-sample AUC), not by
in-sample fit quality, so "best fits predictive outcomes" means best OUT of sample.

Run: python research/edge_signal_test_v8_power_weights.py
"""
from __future__ import annotations

import csv
import io
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
SEASONS = range(2007, 2026)
TEST_START_SEASON = 2011
EWMA_ALPHA = 0.2
CARRYOVER = 0.65
WP_LO, WP_HI = 0.05, 0.95
RATING_METRICS = (
    "rush_off_epa", "pass_off_epa", "rush_def_epa_allowed", "pass_def_epa_allowed",
    "points_off", "points_def_allowed", "turnovers_off", "turnovers_def_forced",
)
# +1 if higher is already better, -1 if the raw stat needs flipping to be "higher=better"
ORIENTATION = {
    "rush_off_epa": 1, "pass_off_epa": 1, "rush_def_epa_allowed": -1, "pass_def_epa_allowed": -1,
    "points_off": 1, "points_def_allowed": -1, "turnovers_off": -1, "turnovers_def_forced": 1,
}
PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "epa", "wp", "interception", "fumble_lost"]


def _cached_fetch_text(url: str, cache_name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, cache_name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(path, "w", encoding="utf-8") as f:
        f.write(resp.text)
    return resp.text


def load_games() -> dict[str, dict]:
    text = _cached_fetch_text(GAMES_URL, "games.csv")
    games = {}
    for r in csv.DictReader(io.StringIO(text)):
        if r["game_type"] != "REG":
            continue
        try:
            season = int(r["season"])
        except ValueError:
            continue
        if season in SEASONS:
            games[r["game_id"]] = r
    return games


def team_game_phase_stats(season: int) -> pd.DataFrame:
    cache_path = os.path.join(CACHE_DIR, f"team_game_phase_turnovers_{season}.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"  downloading/parsing pbp {season}...", file=sys.stderr)
    df = pd.read_csv(PBP_URL.format(season=season), compression="gzip", usecols=PBP_COLS, low_memory=False)
    df = df[(df["season_type"] == "REG") & (df["play_type"].isin(["run", "pass"]))]
    df = df.dropna(subset=["wp", "epa", "posteam"])
    df = df[(df["wp"] >= WP_LO) & (df["wp"] <= WP_HI)]
    df["turnover"] = df["interception"].fillna(0) + df["fumble_lost"].fillna(0)

    off = df.groupby(["game_id", "posteam", "play_type"])["epa"].agg(["size", "sum"]).reset_index()
    off = off.pivot_table(index=["game_id", "posteam"], columns="play_type", values=["size", "sum"])
    off.columns = [f"{a}_{b}" for a, b in off.columns]
    off = off.reset_index().rename(columns={"posteam": "team"})
    for col in ("size_run", "size_pass", "sum_run", "sum_pass"):
        if col not in off.columns:
            off[col] = 0.0
    off["rush_off_epa"] = off["sum_run"] / off["size_run"].replace(0, np.nan)
    off["pass_off_epa"] = off["sum_pass"] / off["size_pass"].replace(0, np.nan)
    tov = df.groupby(["game_id", "posteam"])["turnover"].sum().reset_index().rename(columns={"posteam": "team", "turnover": "turnovers_committed"})
    out = off[["game_id", "team", "rush_off_epa", "pass_off_epa"]].merge(tov, on=["game_id", "team"], how="left")
    out["turnovers_committed"] = out["turnovers_committed"].fillna(0.0)
    out = out.dropna(subset=["rush_off_epa", "pass_off_epa"])
    out.to_parquet(cache_path)
    return out


class RunningMean:
    def __init__(self):
        self.n, self.mean = 0, 0.0

    def update(self, x):
        self.n += 1
        self.mean += (x - self.mean) / self.n


def build_dataset() -> list[dict]:
    by_game: dict[str, dict[str, dict]] = defaultdict(dict)
    for season in SEASONS:
        for row in team_game_phase_stats(season).to_dict("records"):
            by_game[row["game_id"]][row["team"]] = row

    games = load_games()
    running_mean = {"rush_off_epa": RunningMean(), "pass_off_epa": RunningMean(),
                     "points_off": RunningMean(), "turnovers_off": RunningMean()}

    def league_mean(metric: str) -> float:
        base = metric.replace("_def_epa_allowed", "_off_epa").replace("_def_allowed", "_off").replace("_def_forced", "_off")
        return running_mean[base].mean

    rating: dict[str, dict] = defaultdict(lambda: {m: None for m in RATING_METRICS})
    last_season: dict[str, int] = {}

    rows_out = []
    for gid, teams in sorted(by_game.items()):
        g = games.get(gid)
        if not g or len(teams) != 2:
            continue
        season = int(g["season"])
        home, away = g["home_team"], g["away_team"]
        if home not in teams or away not in teams:
            continue
        try:
            home_pts, away_pts = float(g["home_score"]), float(g["away_score"])
        except (ValueError, TypeError):
            continue

        home_raw, away_raw = dict(teams[home]), dict(teams[away])
        home_raw["rush_def_epa_allowed"] = away_raw["rush_off_epa"]
        away_raw["rush_def_epa_allowed"] = home_raw["rush_off_epa"]
        home_raw["pass_def_epa_allowed"] = away_raw["pass_off_epa"]
        away_raw["pass_def_epa_allowed"] = home_raw["pass_off_epa"]
        home_raw["points_off"], away_raw["points_off"] = home_pts, away_pts
        home_raw["points_def_allowed"], away_raw["points_def_allowed"] = away_pts, home_pts
        home_raw["turnovers_off"] = home_raw.pop("turnovers_committed")
        away_raw["turnovers_off"] = away_raw.pop("turnovers_committed")
        home_raw["turnovers_def_forced"] = away_raw["turnovers_off"]
        away_raw["turnovers_def_forced"] = home_raw["turnovers_off"]

        for team in (home, away):
            if last_season.get(team) is not None and last_season[team] != season:
                for m in RATING_METRICS:
                    r = rating[team][m]
                    if r is not None:
                        lm = league_mean(m)
                        rating[team][m] = lm + CARRYOVER * (r - lm)
            last_season[team] = season

        ok = all(rating[home][m] is not None and rating[away][m] is not None for m in RATING_METRICS)
        feat_row = {"season": season, "week": int(g["week"]), "game_id": gid}
        if ok:
            for m in RATING_METRICS:
                # oriented so the diff is always "higher = better for home" before any weighting
                feat_row[f"{m}_diff"] = ORIENTATION[m] * (rating[home][m] - rating[away][m])
            try:
                feat_row["spread_line"] = float(g["spread_line"])
            except (ValueError, TypeError):
                ok = False
        if ok:
            try:
                res = float(g["result"])
            except (ValueError, TypeError):
                res = home_pts - away_pts
            feat_row["home_win"] = 1 if res > 0 else 0
            rows_out.append(feat_row)

        for m in ("rush_off_epa", "pass_off_epa", "points_off", "turnovers_off"):
            running_mean[m].update(home_raw[m])
            running_mean[m].update(away_raw[m])
        for team, raw in ((home, home_raw), (away, away_raw)):
            for m in RATING_METRICS:
                prev = rating[team][m]
                rating[team][m] = raw[m] if prev is None else (1 - EWMA_ALPHA) * prev + EWMA_ALPHA * raw[m]

    return rows_out


def fit_nonneg_logistic(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    """Logistic regression with coefficients constrained >= 0 (intercept unconstrained),
    L2-penalized (not on intercept). Solved via L-BFGS-B with an analytic gradient."""
    n, p = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])  # column 0 = intercept

    def loss_grad(w):
        z = Xb @ w
        p_hat = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-12
        nll = -np.mean(y * np.log(p_hat + eps) + (1 - y) * np.log(1 - p_hat + eps))
        reg = 0.5 * l2 * np.sum(w[1:] ** 2)
        grad = Xb.T @ (p_hat - y) / n
        grad[1:] += l2 * w[1:]
        return nll + reg, grad

    bounds = [(None, None)] + [(0, None)] * p
    w0 = np.zeros(p + 1)
    res = minimize(loss_grad, w0, jac=True, method="L-BFGS-B", bounds=bounds)
    return res.x  # [intercept, w_1..w_p]


def walk_forward_auc_nonneg(rows: list[dict], cols: list[str], l2: float) -> dict:
    seasons = sorted({r["season"] for r in rows})
    preds, actuals = [], []
    for test_season in seasons:
        if test_season < TEST_START_SEASON:
            continue
        train = [r for r in rows if r["season"] < test_season]
        test = [r for r in rows if r["season"] == test_season]
        if len(train) < 100 or not test:
            continue
        X_train = np.array([[r[c] for c in cols] for r in train])
        y_train = np.array([r["home_win"] for r in train])
        X_test = np.array([[r[c] for c in cols] for r in test])
        y_test = [r["home_win"] for r in test]
        mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
        sd[sd == 0] = 1.0
        Xs_train, Xs_test = (X_train - mu) / sd, (X_test - mu) / sd
        w = fit_nonneg_logistic(Xs_train, y_train, l2)
        z = w[0] + Xs_test @ w[1:]
        p = 1 / (1 + np.exp(-z))
        preds.extend(p)
        actuals.extend(y_test)
    return {
        "n": len(actuals),
        "auc": round(roc_auc_score(actuals, preds), 4),
        "log_loss": round(log_loss(actuals, preds), 4),
        "accuracy": round(accuracy_score(actuals, [1 if x > 0.5 else 0 for x in preds]), 4),
    }


def main():
    print("Building dataset (oriented higher-is-better diffs)...", file=sys.stderr)
    rows = build_dataset()
    cols = [f"{m}_diff" for m in RATING_METRICS]
    print(f"{len(rows)} games ({SEASONS.start}-{SEASONS.stop - 1})\n")

    print("--- walk-forward AUC by L2 strength (non-negative joint fit) ---")
    print(f"{'l2':>8s} {'n':>6s} {'AUC':>7s} {'logloss':>8s} {'acc':>6s}")
    best_l2, best_auc = None, -1
    for l2 in (0.3, 1, 3, 10, 30, 100, 300):
        r = walk_forward_auc_nonneg(rows, cols, l2)
        flag = ""
        if r["auc"] > best_auc:
            best_auc, best_l2, flag = r["auc"], l2, "  <- best so far"
        print(f"{l2:8.1f} {r['n']:6d} {r['auc']:7.4f} {r['log_loss']:8.4f} {r['accuracy']:6.4f}{flag}")

    print(f"\nBest L2 = {best_l2} (walk-forward AUC {best_auc:.4f})")

    # reference points already established in v7
    X_all = np.array([[r[c] for c in cols] for r in rows])
    y_all = np.array([r["home_win"] for r in rows])
    print("\n(for comparison, from v7: 4 EPA alone 0.673-0.676, all 8 univariate-weighted 0.682-0.683,")
    print(" market alone 0.725-0.726, all 8 + market 0.722)")

    # final production weights: fit on ALL data at the chosen L2
    mu, sd = X_all.mean(axis=0), X_all.std(axis=0)
    w = fit_nonneg_logistic((X_all - mu) / sd, y_all, best_l2)
    print(f"\nFinal non-negative joint weights (L2={best_l2}, all {len(rows)} games):")
    print(f"  {'intercept':22s} {w[0]:+.4f}")
    for m, wi in zip(RATING_METRICS, w[1:]):
        print(f"  {m:22s} {wi:+.4f}  (orientation {ORIENTATION[m]:+d})")

    # also test combining with market spread to see if the optimized composite adds anything
    cols_plus_market = cols + ["spread_line"]
    print("\n--- with market spread added (still non-negative on the 8 stats; spread unconstrained) ---")
    seasons = sorted({r["season"] for r in rows})
    preds, actuals = [], []
    for test_season in seasons:
        if test_season < TEST_START_SEASON:
            continue
        train = [r for r in rows if r["season"] < test_season]
        test = [r for r in rows if r["season"] == test_season]
        if len(train) < 100 or not test:
            continue
        X_train = np.array([[r[c] for c in cols_plus_market] for r in train])
        y_train = np.array([r["home_win"] for r in train])
        X_test = np.array([[r[c] for c in cols_plus_market] for r in test])
        y_test = [r["home_win"] for r in test]
        mu2, sd2 = X_train.mean(axis=0), X_train.std(axis=0)
        sd2[sd2 == 0] = 1.0
        Xs_train, Xs_test = (X_train - mu2) / sd2, (X_test - mu2) / sd2
        n_, p_ = Xs_train.shape
        Xb = np.hstack([np.ones((n_, 1)), Xs_train])

        def loss_grad(w_, Xb=Xb, y=y_train, l2=best_l2, p_=p_):
            z = Xb @ w_
            p_hat = 1.0 / (1.0 + np.exp(-z))
            eps = 1e-12
            nll = -np.mean(y * np.log(p_hat + eps) + (1 - y) * np.log(1 - p_hat + eps))
            reg = 0.5 * l2 * np.sum(w_[1:-1] ** 2)  # no penalty on intercept or spread
            grad = Xb.T @ (p_hat - y) / n_
            grad[1:-1] += l2 * w_[1:-1]
            return nll + reg, grad

        bounds = [(None, None)] + [(0, None)] * (p_ - 1) + [(None, None)]  # last col = spread, unconstrained
        res = minimize(loss_grad, np.zeros(p_ + 1), jac=True, method="L-BFGS-B", bounds=bounds)
        w_ = res.x
        z = w_[0] + Xs_test @ w_[1:]
        p_pred = 1 / (1 + np.exp(-z))
        preds.extend(p_pred)
        actuals.extend(y_test)
    print(f"all 8 (non-neg, L2={best_l2}) + market spread: n={len(actuals)} AUC={roc_auc_score(actuals, preds):.4f} logloss={log_loss(actuals, preds):.4f}")


if __name__ == "__main__":
    main()
