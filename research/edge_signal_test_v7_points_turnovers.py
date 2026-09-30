"""v7: extends v4's rolling-rating methodology (garbage-time-excluded EPA, EWMA + season
carryover -- the simplest approach, and still tied for best of everything tried) with 4 more
raw counting stats: points scored, points allowed, turnovers committed, turnovers forced.

Points come straight from games.csv final scores (not garbage-time filtered -- a final score
IS the whole-game outcome, unlike a mid-game EPA snapshot, so filtering it would fight the
stat's own meaning). Turnovers reuse the same garbage-time-filtered play population as EPA,
for consistency with the rest of the pipeline.

Same walk-forward discipline as v2-v6. This run is explicitly NOT chasing market-beating
edge (that question's been answered several ways already) -- it's characterizing these 4
stats' standalone predictive value as candidate inputs to a power-ranking composite, per
request.

Run: python research/edge_signal_test_v7_points_turnovers.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
SEASONS = range(2007, 2026)  # through the now-complete 2025 season
TEST_START_SEASON = 2011
EWMA_ALPHA = 0.2
CARRYOVER = 0.65
WP_LO, WP_HI = 0.05, 0.95
RATING_METRICS = (
    "rush_off_epa", "pass_off_epa", "rush_def_epa_allowed", "pass_def_epa_allowed",
    "points_off", "points_def_allowed", "turnovers_off", "turnovers_def_forced",
)
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
    """Adds turnovers_committed to v4's rush/pass EPA aggregation (same garbage-time filter)."""
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
        if gid not in games or len(teams) != 2:
            continue
        g = games[gid]
        try:
            season, week = int(g["season"]), int(g["week"])
        except ValueError:
            continue
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
        feat_row = {"season": season, "week": week, "game_id": gid}
        if ok:
            for m in RATING_METRICS:
                feat_row[f"{m}_diff"] = rating[home][m] - rating[away][m]
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


def walk_forward_auc(rows: list[dict], feature_cols: list[str]) -> dict:
    seasons = sorted({r["season"] for r in rows})
    preds, actuals = [], []
    for test_season in seasons:
        if test_season < TEST_START_SEASON:
            continue
        train = [r for r in rows if r["season"] < test_season]
        test = [r for r in rows if r["season"] == test_season]
        if len(train) < 100 or not test:
            continue
        X_train = np.array([[r[c] for c in feature_cols] for r in train])
        y_train = np.array([r["home_win"] for r in train])
        X_test = np.array([[r[c] for c in feature_cols] for r in test])
        y_test = [r["home_win"] for r in test]
        mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
        sd[sd == 0] = 1.0
        clf = LogisticRegression()
        clf.fit((X_train - mu) / sd, y_train)
        p = clf.predict_proba((X_test - mu) / sd)[:, 1]
        preds.extend(p)
        actuals.extend(y_test)
    return {
        "n": len(actuals),
        "auc": round(roc_auc_score(actuals, preds), 4),
        "log_loss": round(log_loss(actuals, preds), 4),
        "accuracy": round(accuracy_score(actuals, [1 if x > 0.5 else 0 for x in preds]), 4),
    }


def main():
    print("Building points/turnovers-extended dataset...", file=sys.stderr)
    rows = build_dataset()
    print(f"{len(rows)} games with an established rating for both teams ({SEASONS.start}-{SEASONS.stop - 1})\n")

    all_cols = [f"{m}_diff" for m in RATING_METRICS]
    epa_cols = [c for c in all_cols if "epa" in c]
    feature_sets = {
        "points_off_diff alone": ["points_off_diff"],
        "points_def_allowed_diff alone": ["points_def_allowed_diff"],
        "turnovers_off_diff alone": ["turnovers_off_diff"],
        "turnovers_def_forced_diff alone": ["turnovers_def_forced_diff"],
        "points + turnovers (4 new stats combined)": ["points_off_diff", "points_def_allowed_diff", "turnovers_off_diff", "turnovers_def_forced_diff"],
        "4 EPA stats alone (v4 baseline)": epa_cols,
        "all 8 stats combined (v7)": all_cols,
        "market spread_line alone": ["spread_line"],
        "all 8 stats + market spread_line": all_cols + ["spread_line"],
    }

    print(f"{'feature set':42s} {'n':>6s} {'AUC':>7s} {'logloss':>8s} {'acc':>6s}")
    for name, cols in feature_sets.items():
        r = walk_forward_auc(rows, cols)
        print(f"{name:42s} {r['n']:6d} {r['auc']:7.4f} {r['log_loss']:8.4f} {r['accuracy']:6.4f}")

    # final composite weights for the power ranking: fit on ALL historical games (not
    # walk-forward split -- this is for a production "current" weighting, not an
    # out-of-sample accuracy claim, that question is already answered by the table above).
    print("\nPower-ranking composite weights (standardized logistic regression, all games):")
    X = np.array([[r[c] for c in all_cols] for r in rows])
    y = np.array([r["home_win"] for r in rows])
    mu, sd = X.mean(axis=0), X.std(axis=0)
    clf = LogisticRegression()
    clf.fit((X - mu) / sd, y)
    for m, coef in zip(RATING_METRICS, clf.coef_[0]):
        print(f"  {m:22s} {coef:+.4f}")
    print(f"  {'intercept':22s} {clf.intercept_[0]:+.4f}")


if __name__ == "__main__":
    main()
