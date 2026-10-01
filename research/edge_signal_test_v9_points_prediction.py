"""v9: predict each team's actual POINTS in a game (continuous regression), not just win/loss,
using the same rolling ratings as the power ranking.

One row per (team, game): features = that team's own 4 offensive ratings (rush/pass off EPA,
points_off, turnovers_off) + the OPPONENT's 4 defensive ratings (rush/pass def EPA allowed,
points_def_allowed, turnovers_def_forced) -- "my offense vs their defense". Target = points
actually scored. Ridge regression (small L2 for stability), walk-forward validated exactly
like every classification version in this project.

From the two teams' predicted points in a game: predicted_margin = predicted_home - predicted_away
gives an implied spread, directly comparable to the market's spread_line -- same "does this add
anything over the market" question, just for point margins instead of win probability.

Run: python research/edge_signal_test_v9_points_prediction.py
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score

# +1 if higher is already better for the team's own scoring, -1 if it needs flipping
# (turnovers committed/forced hurt scoring, but raw regression gave them the WRONG sign --
# same multicollinearity issue the power ranking had; fixed the same way: orient, then
# constrain >= 0).
ORIENTATION = {
    "rush_off_epa": 1, "pass_off_epa": 1, "points_off": 1, "turnovers_off": -1,
    "rush_def_epa_allowed": 1, "pass_def_epa_allowed": 1, "points_def_allowed": 1, "turnovers_def_forced": -1,
}

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
OWN_OFF = ("rush_off_epa", "pass_off_epa", "points_off", "turnovers_off")
OPP_DEF = ("rush_def_epa_allowed", "pass_def_epa_allowed", "points_def_allowed", "turnovers_def_forced")
PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "epa", "wp", "interception", "fumble_lost"]
RIDGE_ALPHA = 5.0


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
    """Two rows per game: one from the home team's perspective, one from away's."""
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
        if ok:
            try:
                spread_line = float(g["spread_line"])
            except (ValueError, TypeError):
                ok = False
        if ok:
            # one row per side: features = own-offense (4) + opponent-defense (4)
            rows_out.append({
                "season": season, "week": int(g["week"]), "game_id": gid, "side": "home",
                "features": [rating[home][m] for m in OWN_OFF] + [rating[away][m] for m in OPP_DEF],
                "points": home_pts, "opp_points": away_pts, "spread_line": spread_line,
            })
            rows_out.append({
                "season": season, "week": int(g["week"]), "game_id": gid, "side": "away",
                "features": [rating[away][m] for m in OWN_OFF] + [rating[home][m] for m in OPP_DEF],
                "points": away_pts, "opp_points": home_pts, "spread_line": spread_line,
            })

        for m in ("rush_off_epa", "pass_off_epa", "points_off", "turnovers_off"):
            running_mean[m].update(home_raw[m])
            running_mean[m].update(away_raw[m])
        for team, raw in ((home, home_raw), (away, away_raw)):
            for m in RATING_METRICS:
                prev = rating[team][m]
                rating[team][m] = raw[m] if prev is None else (1 - EWMA_ALPHA) * prev + EWMA_ALPHA * raw[m]

    return rows_out


def walk_forward_points(rows: list[dict]) -> dict:
    """Trains a Ridge points-predictor walk-forward, then evaluates: points MAE, and the
    implied-margin's accuracy at predicting the game winner + MAE vs actual margin, alongside
    the market spread doing the same job."""
    seasons = sorted({r["season"] for r in rows})
    abs_errs = []
    by_game: dict[str, dict] = {}

    for test_season in seasons:
        if test_season < TEST_START_SEASON:
            continue
        train = [r for r in rows if r["season"] < test_season]
        test = [r for r in rows if r["season"] == test_season]
        if len(train) < 200 or not test:
            continue
        X_train = np.array([r["features"] for r in train])
        y_train = np.array([r["points"] for r in train])
        X_test = np.array([r["features"] for r in test])
        mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
        sd[sd == 0] = 1.0
        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit((X_train - mu) / sd, y_train)
        preds = model.predict((X_test - mu) / sd)

        for r, pred in zip(test, preds):
            abs_errs.append(abs(pred - r["points"]))
            slot = by_game.setdefault(r["game_id"], {"spread_line": r["spread_line"]})
            slot[r["side"]] = pred
            slot[f"{r['side']}_actual"] = r["points"]

    margin_errs, spread_errs = [], []
    y_true, pred_margin_sign, market_margin_sign = [], [], []
    for gid, slot in by_game.items():
        if "home" not in slot or "away" not in slot:
            continue
        pred_margin = slot["home"] - slot["away"]
        actual_margin = slot["home_actual"] - slot["away_actual"]
        margin_errs.append(abs(pred_margin - actual_margin))
        spread_errs.append(abs(slot["spread_line"] - actual_margin))
        y_true.append(1 if actual_margin > 0 else 0)
        pred_margin_sign.append(pred_margin)
        market_margin_sign.append(slot["spread_line"])

    return {
        "n_side_rows": len(abs_errs),
        "points_mae": round(float(np.mean(abs_errs)), 3),
        "n_games": len(margin_errs),
        "model_margin_mae": round(float(np.mean(margin_errs)), 3),
        "market_margin_mae": round(float(np.mean(spread_errs)), 3),
        "model_margin_auc": round(roc_auc_score(y_true, pred_margin_sign), 4),
        "market_spread_auc": round(roc_auc_score(y_true, market_margin_sign), 4),
    }


ORIENT_VEC = np.array([ORIENTATION[m] for m in OWN_OFF] + [ORIENTATION[m] for m in OPP_DEF])


def fit_nonneg_ridge(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    """Squared-error loss + L2 penalty, coefficients (all but intercept) constrained >= 0."""
    n, p = X.shape
    def loss_grad(w):
        pred = w[0] + X @ w[1:]
        resid = pred - y
        mse = np.mean(resid ** 2)
        reg = l2 * np.sum(w[1:] ** 2)
        grad = np.zeros_like(w)
        grad[0] = 2 * np.mean(resid)
        grad[1:] = 2 * (X.T @ resid) / n + 2 * l2 * w[1:]
        return mse + reg, grad
    bounds = [(None, None)] + [(0, None)] * p
    res = minimize(loss_grad, np.zeros(p + 1), jac=True, method="L-BFGS-B", bounds=bounds)
    return res.x


def walk_forward_points_nonneg(rows: list[dict], l2: float) -> dict:
    seasons = sorted({r["season"] for r in rows})
    abs_errs = []
    by_game: dict[str, dict] = {}
    for test_season in seasons:
        if test_season < TEST_START_SEASON:
            continue
        train = [r for r in rows if r["season"] < test_season]
        test = [r for r in rows if r["season"] == test_season]
        if len(train) < 200 or not test:
            continue
        X_train = np.array([r["features"] for r in train]) * ORIENT_VEC
        y_train = np.array([r["points"] for r in train])
        X_test = np.array([r["features"] for r in test]) * ORIENT_VEC
        mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
        sd[sd == 0] = 1.0
        w = fit_nonneg_ridge((X_train - mu) / sd, y_train, l2)
        preds = w[0] + ((X_test - mu) / sd) @ w[1:]
        for r_, pred in zip(test, preds):
            abs_errs.append(abs(pred - r_["points"]))
            slot = by_game.setdefault(r_["game_id"], {"spread_line": r_["spread_line"]})
            slot[r_["side"]] = pred
            slot[f"{r_['side']}_actual"] = r_["points"]

    margin_errs, spread_errs, y_true, pred_sign, market_sign = [], [], [], [], []
    for slot in by_game.values():
        if "home" not in slot or "away" not in slot:
            continue
        pm = slot["home"] - slot["away"]
        am = slot["home_actual"] - slot["away_actual"]
        margin_errs.append(abs(pm - am))
        spread_errs.append(abs(slot["spread_line"] - am))
        y_true.append(1 if am > 0 else 0)
        pred_sign.append(pm)
        market_sign.append(slot["spread_line"])
    return {
        "points_mae": round(float(np.mean(abs_errs)), 3),
        "n_games": len(margin_errs),
        "model_margin_mae": round(float(np.mean(margin_errs)), 3),
        "market_margin_mae": round(float(np.mean(spread_errs)), 3),
        "model_margin_auc": round(roc_auc_score(y_true, pred_sign), 4),
        "market_spread_auc": round(roc_auc_score(y_true, market_sign), 4),
    }


def main():
    print("Building points-prediction dataset...", file=sys.stderr)
    rows = build_dataset()
    print(f"{len(rows)} team-game rows ({len(rows)//2} games)\n")

    r = walk_forward_points(rows)
    print("--- unconstrained Ridge, walk-forward (2011-2025) ---")
    print(f"Points MAE: {r['points_mae']}   model margin MAE: {r['model_margin_mae']} (market {r['market_margin_mae']})")
    print(f"Winner AUC from margin sign -- model: {r['model_margin_auc']}   market: {r['market_spread_auc']}")

    print("\n--- non-negative (oriented) Ridge, walk-forward, by L2 ---")
    print(f"{'l2':>8s} {'points_mae':>11s} {'margin_mae':>11s} {'margin_auc':>11s}")
    best_l2, best_mae = None, 1e9
    for l2 in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        rr = walk_forward_points_nonneg(rows, l2)
        flag = ""
        if rr["points_mae"] < best_mae:
            best_mae, best_l2, flag = rr["points_mae"], l2, "  <- best so far"
        print(f"{l2:8.2f} {rr['points_mae']:11.3f} {rr['model_margin_mae']:11.3f} {rr['model_margin_auc']:11.4f}{flag}")

    print(f"\nBest L2 = {best_l2}")

    # final production fit: all data, raw-feature-space coefficients (fold orientation + mu/sd in)
    X_raw = np.array([r["features"] for r in rows])
    y = np.array([r["points"] for r in rows])
    X = X_raw * ORIENT_VEC
    mu, sd = X.mean(axis=0), X.std(axis=0)
    w = fit_nonneg_ridge((X - mu) / sd, y, best_l2)
    raw_weights = (w[1:] / sd) * ORIENT_VEC  # fold the orientation flip back into the raw-space weight's sign
    raw_intercept = w[0] - float(np.sum(w[1:] * mu / sd))

    names = list(OWN_OFF) + ["opp_" + m for m in OPP_DEF]
    print(f"\nFinal production weights (raw feature space, L2={best_l2}, all {len(rows)} rows):")
    print(f"  intercept: {raw_intercept:+.5f}")
    for n, rw in zip(names, raw_weights):
        print(f"  {n:24s} {rw:+.6f}")

    pred_check = raw_intercept + X_raw[:3] @ raw_weights
    print("\nsanity check preds:", pred_check, " actual:", y[:3])


if __name__ == "__main__":
    main()
