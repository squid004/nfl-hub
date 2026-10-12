"""v39: points_off/points_def_allowed are the two heaviest-weighted stats in the whole
8-stat composite (POWER_WEIGHTS: points_off=0.1587, points_def_allowed=0.1006 -- #1 and #3
of 8, ~41% of total weight combined) -- and they're also the only two of the 8 that are NOT
garbage-time-filtered. Every other stat drops plays outside the WP_LO-WP_HI (0.05-0.95) win-
probability band before aggregating; points_off/points_def_allowed are literally the final
box score, blowout garbage-time scoring included.

This rebuilds points_off/points_def_allowed as a garbage-time-filtered stat (same WP band,
applied per-play via the score-delta method below) and refits the ENTIRE 8-stat composite
against it with v13's own methodology unchanged (non-negative L2-regularized joint logistic
regression, same walk-forward validation, same L2 search) -- then compares walk-forward AUC/
logloss/accuracy against the existing (raw-points) production weights, run side by side in
the same script so both numbers come from the identical train/test split and can't drift
apart from incidental differences in methodology.

Garbage-time-filtered points, computed per play (not just run/pass -- field goals, PATs,
2-point conversions, kickoff/punt returns, and defensive/special-teams TDs all score too):
for every play with wp inside the competitive band, credit posteam with
(posteam_score_post - posteam_score) and defteam with (defteam_score_post - defteam_score).
Summing both across a game gives each team's own GT-filtered points scored; the opponent's
own total in that same game is their GT-filtered points allowed (same relationship the raw
version already uses). A team's TRUE final score/win-loss outcome is untouched throughout --
only the points_off/points_def_allowed INPUT FEATURE changes, never the y-target the
regression is fit against.

Run: python research/edge_signal_test_v39_points_garbage_time.py
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
# Needs every play type (not just run/pass, unlike the EPA/turnovers read) plus the four
# pre/post score-state columns the GT-points computation is built on.
PBP_COLS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "epa", "wp",
    "interception", "fumble_lost", "posteam_score", "posteam_score_post", "defteam_score", "defteam_score_post",
]


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


def team_game_phase_and_points(season: int) -> pd.DataFrame:
    """One row per (game_id, team): rush/pass offensive EPA/play + turnovers committed (run/
    pass plays only, same as v13) PLUS points_off_gt/points_def_allowed_gt (ALL play types,
    same WP-band filter). One PBP read per season serves both, instead of downloading twice."""
    cache_path = os.path.join(CACHE_DIR, f"team_game_phase_gt_{season}.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"  downloading/parsing pbp {season}...", file=sys.stderr)
    raw = pd.read_csv(PBP_URL.format(season=season), compression="gzip", usecols=PBP_COLS, low_memory=False)
    raw = raw[raw["season_type"] == "REG"]

    # --- rush/pass EPA + turnovers committed: unchanged from v13, run/pass plays only ---
    df = raw[raw["play_type"].isin(["run", "pass"])].dropna(subset=["wp", "epa", "posteam"])
    df = df[(df["wp"] >= WP_LO) & (df["wp"] <= WP_HI)]
    df = df.copy()
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

    # --- garbage-time-filtered points: EVERY play type, same WP band, score-state deltas ---
    gt = raw.dropna(subset=["wp", "posteam", "defteam"]).copy()
    gt = gt[(gt["wp"] >= WP_LO) & (gt["wp"] <= WP_HI)]
    gt["off_pts"] = (gt["posteam_score_post"] - gt["posteam_score"]).fillna(0)
    gt["def_pts"] = (gt["defteam_score_post"] - gt["defteam_score"]).fillna(0)
    off_pts = gt.groupby(["game_id", "posteam"])["off_pts"].sum().reset_index().rename(columns={"posteam": "team", "off_pts": "pts"})
    def_pts = gt.groupby(["game_id", "defteam"])["def_pts"].sum().reset_index().rename(columns={"defteam": "team", "def_pts": "pts"})
    pts_gt = pd.concat([off_pts, def_pts]).groupby(["game_id", "team"])["pts"].sum().reset_index()
    pts_gt = pts_gt.rename(columns={"pts": "points_off_gt"})

    out = out.merge(pts_gt, on=["game_id", "team"], how="left")
    out["points_off_gt"] = out["points_off_gt"].fillna(0.0)
    out.to_parquet(cache_path)
    return out


def build_dataset(use_gt_points: bool) -> list[dict]:
    """Same EWMA/season-prior rating walk as v13's build_dataset(), unchanged -- the ONLY
    difference is which points_off/points_def_allowed source feeds home_raw/away_raw when
    use_gt_points is True. home_win (the y-target) and every other stat are identical either
    way; only this one feature's INPUT changes."""
    by_game: dict[str, dict[str, dict]] = defaultdict(dict)
    for season in SEASONS:
        for row in team_game_phase_and_points(season).to_dict("records"):
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
        if use_gt_points:
            home_pts_feat, away_pts_feat = home_raw["points_off_gt"], away_raw["points_off_gt"]
        else:
            home_pts_feat, away_pts_feat = home_pts, away_pts
        home_raw["points_off"], away_raw["points_off"] = home_pts_feat, away_pts_feat
        home_raw["points_def_allowed"], away_raw["points_def_allowed"] = away_pts_feat, home_pts_feat
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
            # TRUE outcome, always the real final score -- never the GT-filtered points,
            # even when use_gt_points swapped the FEATURE above.
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


def run_scoreboard(label: str, rows: list[dict], cols: list[str]) -> tuple[float, dict]:
    print(f"\n=== {label}: {len(rows)} games ({SEASONS.start}-{SEASONS.stop - 1}) ===")
    print(f"{'l2':>8s} {'n':>6s} {'AUC':>7s} {'logloss':>8s} {'acc':>6s}")
    best_l2, best_auc, best_r = None, -1, None
    for l2 in (0.3, 1, 3, 10, 30, 100, 300):
        r = walk_forward_auc_nonneg(rows, cols, l2)
        flag = ""
        if r["auc"] > best_auc:
            best_auc, best_l2, best_r, flag = r["auc"], l2, r, "  <- best so far"
        print(f"{l2:8.1f} {r['n']:6d} {r['auc']:7.4f} {r['log_loss']:8.4f} {r['accuracy']:6.4f}{flag}")
    print(f"Best L2 = {best_l2} (walk-forward AUC {best_auc:.4f})")

    X_all = np.array([[r[c] for c in cols] for r in rows])
    y_all = np.array([r["home_win"] for r in rows])
    mu, sd = X_all.mean(axis=0), X_all.std(axis=0)
    sd[sd == 0] = 1.0
    w = fit_nonneg_logistic((X_all - mu) / sd, y_all, best_l2)
    print(f"Final non-negative joint weights (L2={best_l2}, all {len(rows)} games):")
    print(f"  {'intercept':22s} {w[0]:+.4f}")
    for m, wi in zip(RATING_METRICS, w[1:]):
        print(f"  {m:22s} {wi:+.4f}  (orientation {ORIENTATION[m]:+d})")
    return best_l2, best_r


def main():
    cols = [f"{m}_diff" for m in RATING_METRICS]

    print("Building RAW-points dataset (today's production baseline)...", file=sys.stderr)
    rows_raw = build_dataset(use_gt_points=False)
    print("Building GARBAGE-TIME-FILTERED-points dataset...", file=sys.stderr)
    rows_gt = build_dataset(use_gt_points=True)

    _, raw_best = run_scoreboard("BASELINE: raw points_off/points_def_allowed (today's production)", rows_raw, cols)
    _, gt_best = run_scoreboard("TREATMENT: garbage-time-filtered points_off/points_def_allowed", rows_gt, cols)

    print("\n=== Head-to-head (same walk-forward split, same L2 search protocol) ===")
    print(f"{'':30s} {'AUC':>7s} {'logloss':>8s} {'acc':>6s}")
    print(f"{'Baseline (raw points)':30s} {raw_best['auc']:7.4f} {raw_best['log_loss']:8.4f} {raw_best['accuracy']:6.4f}")
    print(f"{'Treatment (GT-filtered points)':30s} {gt_best['auc']:7.4f} {gt_best['log_loss']:8.4f} {gt_best['accuracy']:6.4f}")
    d_auc = gt_best['auc'] - raw_best['auc']
    print(f"\nDelta AUC (treatment - baseline): {d_auc:+.4f}")


if __name__ == "__main__":
    main()
