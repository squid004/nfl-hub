"""v13: re-optimize the power-ranking composite weights (v8's methodology, unchanged) against
the NEW rating windowing nflhub/sources/team_ratings.py's compute_ratings() now uses, instead
of v8's old CARRYOVER=0.65-forever mechanism.

Why re-run at all: v8's weights were fit on ratings that let a team's rating from a decade
ago retain nonzero influence on today's rating (geometric decay, never literally zero). The
live rating now works differently per explicit user direction: a new season starts from last
season's own ending rating, fading to ZERO weight (not just small) by SEASON_PRIOR_GAMES games
into the new season -- a game from two or more seasons back has no path into the current
rating at all. That's a different predictor than what v8 optimized against, so the weights
need re-fitting against THIS windowing, not reused from the old one.

Deliberately NOT using compute_historical_season_averages()'s full-season averages here --
those are for the retrospective Historical Power tab (comparing whole, already-known seasons
across eras), a different question from "best in-season predictor of this week's winner, built
from only what was known before kickoff." This script's rating windowing matches the LIVE
compute_ratings() exactly (same season_prior/games_played/fade mechanics, same join-bug fix),
just reimplemented standalone here (same convention as every other research/edge_signal_test_
v*.py -- self-contained, not importing from nflhub so a change to production code can't
silently change what a past backtest run reported).

Everything else is v8's methodology, verbatim: oriented higher-is-better home-minus-away diffs,
L2-regularized logistic regression with coefficients constrained >= 0 (orientation already
rules out sign flips; non-negativity stops a redundant/weak stat from flipping anyway), L2
strength chosen by walk-forward out-of-sample AUC (not in-sample fit), tested against a
scoreboard of (a) the 8 stats alone, (b) the closing market spread alone, (c) both together.

Run: python research/edge_signal_test_v13_new_window_weights.py
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
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nflhub.sources.edge_teams import UnknownTeamError, normalize_team  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
SEASONS = range(2007, 2026)
TEST_START_SEASON = 2011
EWMA_ALPHA = 0.2
SEASON_PRIOR_GAMES = 5  # matches nflhub/sources/team_ratings.py exactly
WP_LO, WP_HI = 0.05, 0.95
RATING_METRICS = (
    "rush_off_epa", "pass_off_epa", "rush_def_epa_allowed", "pass_def_epa_allowed",
    "points_off", "points_def_allowed", "turnovers_off", "turnovers_def_forced",
)
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


def build_dataset() -> list[dict]:
    """Same per-game feature-row shape as v8 (oriented home-minus-away diffs, read BEFORE
    this game updates anything -- no lookahead), but `rating[team][m]` is now the
    season_prior/season_ewma blend described in the module docstring, not an EWMA with
    perpetual cross-season carryover."""
    by_game: dict[str, dict[str, dict]] = defaultdict(dict)
    for season in SEASONS:
        for row in team_game_phase_stats(season).to_dict("records"):
            try:
                team_key = normalize_team(row["team"])
            except UnknownTeamError:
                continue
            by_game[row["game_id"]][team_key] = row

    games = load_games()

    rating: dict[str, dict] = defaultdict(lambda: {m: None for m in RATING_METRICS})
    season_ewma: dict[str, dict] = defaultdict(lambda: {m: None for m in RATING_METRICS})
    season_prior: dict[str, dict] = defaultdict(lambda: {m: None for m in RATING_METRICS})
    games_played: dict[str, int] = defaultdict(int)
    last_season: dict[str, int] = {}

    rows_out = []
    for gid, teams in sorted(by_game.items()):
        g = games.get(gid)
        if not g or len(teams) != 2:
            continue
        season = int(g["season"])
        try:
            home, away = normalize_team(g["home_team"]), normalize_team(g["away_team"])
        except UnknownTeamError:
            continue
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

        prior_w: dict[str, float] = {}
        for team in (home, away):
            if last_season.get(team) is not None and last_season[team] != season:
                season_prior[team] = dict(season_ewma[team])
                season_ewma[team] = {m: None for m in RATING_METRICS}
                games_played[team] = 0
            last_season[team] = season
            games_played[team] += 1
            prior_w[team] = max(0.0, (SEASON_PRIOR_GAMES - games_played[team]) / SEASON_PRIOR_GAMES)

        ok = all(rating[home][m] is not None and rating[away][m] is not None for m in RATING_METRICS)
        feat_row = {"season": season, "week": int(g["week"]), "game_id": gid}
        if ok:
            for m in RATING_METRICS:
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

        for team, raw in ((home, home_raw), (away, away_raw)):
            for m in RATING_METRICS:
                prev = season_ewma[team][m]
                season_ewma[team][m] = raw[m] if prev is None else (1 - EWMA_ALPHA) * prev + EWMA_ALPHA * raw[m]
                prior_val = season_prior[team].get(m)
                rating[team][m] = (
                    prior_w[team] * prior_val + (1 - prior_w[team]) * season_ewma[team][m]
                    if prior_val is not None else season_ewma[team][m]
                )

    return rows_out


def fit_nonneg_logistic(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    n, p = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])

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
    return res.x


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


def walk_forward_market_auc(rows: list[dict]) -> dict:
    """Market-alone baseline: the closing spread_line (oriented so higher = home more
    favored, matching nflverse's own positive-means-home-favored convention) as the sole
    predictor, same walk-forward split as everything else here -- apples-to-apples against
    the composite score above, not a different evaluation window."""
    seasons = sorted({r["season"] for r in rows})
    preds, actuals = [], []
    for test_season in seasons:
        if test_season < TEST_START_SEASON:
            continue
        train = [r for r in rows if r["season"] < test_season]
        test = [r for r in rows if r["season"] == test_season]
        if len(train) < 100 or not test:
            continue
        X_train = np.array([[r["spread_line"]] for r in train])
        y_train = np.array([r["home_win"] for r in train])
        X_test = np.array([[r["spread_line"]] for r in test])
        y_test = [r["home_win"] for r in test]
        mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
        sd[sd == 0] = 1.0
        Xs_train, Xs_test = (X_train - mu) / sd, (X_test - mu) / sd
        w = fit_nonneg_logistic(Xs_train, y_train, 0.0001)  # ~unregularized, 1 feature
        z = w[0] + Xs_test @ w[1:]
        p = 1 / (1 + np.exp(-z))
        preds.extend(p)
        actuals.extend(y_test)
    return {
        "n": len(actuals),
        "auc": round(roc_auc_score(actuals, preds), 4),
        "log_loss": round(log_loss(actuals, preds), 4),
    }


def main():
    print("Building dataset (NEW season-prior-fade windowing, oriented higher-is-better diffs)...", file=sys.stderr)
    rows = build_dataset()
    cols = [f"{m}_diff" for m in RATING_METRICS]
    print(f"{len(rows)} games ({SEASONS.start}-{SEASONS.stop - 1})\n")

    print("--- walk-forward AUC by L2 strength (non-negative joint fit, NEW windowing) ---")
    print(f"{'l2':>8s} {'n':>6s} {'AUC':>7s} {'logloss':>8s} {'acc':>6s}")
    best_l2, best_auc = None, -1
    for l2 in (0.3, 1, 3, 10, 30, 100, 300):
        r = walk_forward_auc_nonneg(rows, cols, l2)
        flag = ""
        if r["auc"] > best_auc:
            best_auc, best_l2, flag = r["auc"], l2, "  <- best so far"
        print(f"{l2:8.1f} {r['n']:6d} {r['auc']:7.4f} {r['log_loss']:8.4f} {r['accuracy']:6.4f}{flag}")

    print(f"\nBest L2 = {best_l2} (walk-forward AUC {best_auc:.4f})")

    market = walk_forward_market_auc(rows)
    print(f"\nMarket spread alone: n={market['n']} AUC={market['auc']:.4f} logloss={market['log_loss']:.4f}")
    print("(v8/old-windowing reference: 8-stat EWMA-forever 0.682-0.683, market alone 0.725-0.726)")

    X_all = np.array([[r[c] for c in cols] for r in rows])
    y_all = np.array([r["home_win"] for r in rows])
    mu, sd = X_all.mean(axis=0), X_all.std(axis=0)
    w = fit_nonneg_logistic((X_all - mu) / sd, y_all, best_l2)
    print(f"\nFinal non-negative joint weights (L2={best_l2}, all {len(rows)} games, NEW windowing):")
    print(f"  {'intercept':22s} {w[0]:+.4f}")
    for m, wi in zip(RATING_METRICS, w[1:]):
        print(f"  {m:22s} {wi:+.4f}  (orientation {ORIENTATION[m]:+d})")

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
            reg = 0.5 * l2 * np.sum(w_[1:-1] ** 2)
            grad = Xb.T @ (p_hat - y) / n_
            grad[1:-1] += l2 * w_[1:-1]
            return nll + reg, grad

        bounds = [(None, None)] + [(0, None)] * (p_ - 1) + [(None, None)]
        res = minimize(loss_grad, np.zeros(p_ + 1), jac=True, method="L-BFGS-B", bounds=bounds)
        w_ = res.x
        z = w_[0] + Xs_test @ w_[1:]
        p_pred = 1 / (1 + np.exp(-z))
        preds.extend(p_pred)
        actuals.extend(y_test)
    print(f"all 8 (non-neg, L2={best_l2}) + market spread: n={len(actuals)} AUC={roc_auc_score(actuals, preds):.4f} logloss={log_loss(actuals, preds):.4f}")


if __name__ == "__main__":
    main()
