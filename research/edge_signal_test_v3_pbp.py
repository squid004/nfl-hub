"""v3: same walk-forward question as edge_signal_test.py, but features are built from full
play-by-play instead of the aggregated weekly box score, so two known contaminants in v1/v2
can be removed:

1. Garbage time. v1/v2's EPA was a raw sum over every play in the game, including plays
   after the outcome was no longer in doubt (running clock, backups in, playing not to lose).
   Sites that publish "real" EPA (rbsdm.com, nfelo) filter this out. Here: drop any play
   where the possession team's win probability (`wp`) is outside [0.05, 0.95].
2. Box-score success rate wasn't available at all in v1/v2 (it needs play-by-play down/
   distance context). nflverse's own precomputed `success` column (EPA>0-based, the
   standard definition) is used directly here -- generally considered more stable than
   EPA/play at a single-game sample size, since it doesn't let one 60-yard touchdown
   dominate a whole game's efficiency number the way raw EPA can.

Same leakage discipline as v2 (EWMA rolling rating with season-boundary carryover, no same-
game data in the predictive features) and the same walk-forward-by-season evaluation.

Run: python research/edge_signal_test_v3_pbp.py
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
SEASONS = range(2007, 2025)
TEST_START_SEASON = 2011
EWMA_ALPHA = 0.2
CARRYOVER = 0.65
WP_LO, WP_HI = 0.05, 0.95  # garbage-time filter: drop plays outside this win-prob band
EXPLOSIVE_YARDS = 20  # single threshold for both run and pass, using real per-play yardage
RATING_METRICS = ("off_epa_per_play", "def_epa_per_play_allowed", "off_success_rate",
                   "def_success_rate_allowed", "explosive_rate", "turnover_margin")

PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "play_type",
            "epa", "success", "yards_gained", "interception", "fumble_lost", "wp"]


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


def team_game_stats_for_season(season: int) -> pd.DataFrame:
    """One row per (game_id, team): garbage-time-filtered per-play offensive stats, built
    straight from play-by-play. Cached as a small parquet so re-runs don't re-download and
    re-crunch the full (300+ column, 40-50k row) raw pbp file every time."""
    cache_path = os.path.join(CACHE_DIR, f"team_game_pbp_{season}.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"  downloading/parsing pbp {season}...", file=sys.stderr)
    df = pd.read_csv(PBP_URL.format(season=season), compression="gzip", usecols=PBP_COLS, low_memory=False)
    df = df[(df["season_type"] == "REG") & (df["play_type"].isin(["run", "pass"]))]
    df = df.dropna(subset=["wp", "epa", "success", "posteam"])
    df = df[(df["wp"] >= WP_LO) & (df["wp"] <= WP_HI)]  # garbage-time filter

    df["explosive"] = (df["yards_gained"] >= EXPLOSIVE_YARDS).astype(float)
    df["turnover"] = df["interception"].fillna(0) + df["fumble_lost"].fillna(0)

    off = df.groupby(["game_id", "posteam"]).agg(
        plays=("epa", "size"), epa_sum=("epa", "sum"), success_sum=("success", "sum"),
        explosive_sum=("explosive", "sum"), turnovers=("turnover", "sum"),
    ).reset_index().rename(columns={"posteam": "team"})
    off["off_epa_per_play"] = off["epa_sum"] / off["plays"]
    off["off_success_rate"] = off["success_sum"] / off["plays"]
    off["explosive_rate"] = off["explosive_sum"] / off["plays"]
    out = off[["game_id", "team", "plays", "off_epa_per_play", "off_success_rate", "explosive_rate", "turnovers"]]
    out.to_parquet(cache_path)
    return out


def build_dataset() -> list[dict]:
    by_game: dict[str, dict[str, dict]] = defaultdict(dict)
    for season in SEASONS:
        tg = team_game_stats_for_season(season)
        for row in tg.to_dict("records"):
            by_game[row["game_id"]][row["team"]] = row

    class RunningMean:
        def __init__(self):
            self.n, self.mean = 0, 0.0

        def update(self, x):
            self.n += 1
            self.mean += (x - self.mean) / self.n

    running_mean = {"off_epa_per_play": RunningMean(), "off_success_rate": RunningMean(), "explosive_rate": RunningMean()}

    def league_mean(metric: str) -> float:
        if metric == "turnover_margin":
            return 0.0
        base = metric.replace("def_", "off_").replace("_allowed", "")
        return running_mean[base].mean

    rating: dict[str, dict] = defaultdict(lambda: {m: None for m in RATING_METRICS})
    last_season: dict[str, int] = {}
    last_qb: dict[str, str] = {}

    rows_out = []
    games = load_games()
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

        home_raw, away_raw = dict(teams[home]), dict(teams[away])
        home_raw["def_epa_per_play_allowed"] = away_raw["off_epa_per_play"]
        away_raw["def_epa_per_play_allowed"] = home_raw["off_epa_per_play"]
        home_raw["def_success_rate_allowed"] = away_raw["off_success_rate"]
        away_raw["def_success_rate_allowed"] = home_raw["off_success_rate"]
        home_raw["turnover_margin"] = away_raw["turnovers"] - home_raw["turnovers"]
        away_raw["turnover_margin"] = home_raw["turnovers"] - away_raw["turnovers"]

        for team in (home, away):
            if last_season.get(team) is not None and last_season[team] != season:
                for m in RATING_METRICS:
                    r = rating[team][m]
                    if r is not None:
                        lm = league_mean(m)
                        rating[team][m] = lm + CARRYOVER * (r - lm)
            last_season[team] = season

        feat_row = {"season": season, "week": week, "game_id": gid}
        for metric in ("off_epa_per_play", "off_success_rate", "explosive_rate", "turnover_margin"):
            feat_row[f"{metric}_diff_concurrent"] = home_raw[metric] - away_raw[metric]

        ok = all(rating[home][m] is not None and rating[away][m] is not None for m in RATING_METRICS)
        if ok:
            for m in RATING_METRICS:
                feat_row[f"{m}_diff"] = rating[home][m] - rating[away][m]
            try:
                feat_row["rest_diff"] = float(g.get("home_rest") or 0) - float(g.get("away_rest") or 0)
                feat_row["home_qb_changed"] = 1.0 if last_qb.get(home) not in (None, g.get("home_qb_id")) else 0.0
                feat_row["away_qb_changed"] = 1.0 if last_qb.get(away) not in (None, g.get("away_qb_id")) else 0.0
                feat_row["spread_line"] = float(g["spread_line"])
            except (ValueError, TypeError):
                ok = False
        if ok:
            try:
                res = float(g["result"])
            except (ValueError, TypeError):
                res = float(g.get("home_score") or 0) - float(g.get("away_score") or 0)
            feat_row["home_win"] = 1 if res > 0 else 0
            rows_out.append(feat_row)

        last_qb[home] = g.get("home_qb_id")
        last_qb[away] = g.get("away_qb_id")

        for m in ("off_epa_per_play", "off_success_rate", "explosive_rate"):
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
    print("Building garbage-time-filtered pbp dataset (first run downloads ~19MB/season, cached after)...", file=sys.stderr)
    rows = build_dataset()
    print(f"{len(rows)} games with an established rating for both teams ({SEASONS.start}-{SEASONS.stop - 1})\n")

    v3_cols = [f"{m}_diff" for m in RATING_METRICS]
    situational = ["rest_diff", "home_qb_changed", "away_qb_changed"]
    feature_sets = {
        "off_epa_per_play_diff (v3, gt-filtered)": ["off_epa_per_play_diff"],
        "off_success_rate_diff (v3, gt-filtered)": ["off_success_rate_diff"],
        "def_epa_per_play_allowed_diff (v3)": ["def_epa_per_play_allowed_diff"],
        "def_success_rate_allowed_diff (v3)": ["def_success_rate_allowed_diff"],
        "explosive_rate_diff (v3, gt-filtered)": ["explosive_rate_diff"],
        "turnover_margin_diff (v3)": ["turnover_margin_diff"],
        "all 6 v3 ratings combined": v3_cols,
        "all 6 v3 ratings + rest + QB-change": v3_cols + situational,
        "market spread_line alone": ["spread_line"],
        "all v3 features + market spread_line": v3_cols + situational + ["spread_line"],
        "--- same-game (circular, illustrative only) ---": [],
        "off_epa_per_play_diff (concurrent, gt-filtered)": ["off_epa_per_play_diff_concurrent"],
        "off_success_rate_diff (concurrent, gt-filtered)": ["off_success_rate_diff_concurrent"],
        "turnover_margin_diff (concurrent)": ["turnover_margin_diff_concurrent"],
    }

    print(f"{'feature set':45s} {'n':>6s} {'AUC':>7s} {'logloss':>8s} {'acc':>6s}")
    for name, cols in feature_sets.items():
        if not cols:
            print(name)
            continue
        r = walk_forward_auc(rows, cols)
        print(f"{name:45s} {r['n']:6d} {r['auc']:7.4f} {r['log_loss']:8.4f} {r['accuracy']:6.4f}")


if __name__ == "__main__":
    main()
