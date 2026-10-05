"""v14: does an explicit matchup interaction term (home's pass offense rating * away's pass
defense-allowed rating, and the mirror; same for rush) add anything over the plain linear
power-rank diffs, under the CURRENT (v13) season-prior-fade windowing? v4_matchup.py already
asked this exact question under the OLD forever-decaying windowing and found no -- linear
diffs alone (0.6730 AUC) beat linear+matchup-products (0.6724) and matchup-products-alone
(0.5768) by a wide margin. Re-running under the new windowing to check that conclusion still
holds now that the underlying ratings are computed differently, not reusing the stale result.

The model being asked about: POWER_WEIGHTS (nflhub/sources/team_ratings.py) combines each
team's OWN rush/pass offense rating and OWN rush/pass defense-allowed rating as two separate,
symmetric linear comparisons (my offense vs your offense; my defense vs your defense) -- it
never multiplies "my offense" by "your defense" together. A product term is the literal
mathematical test of whether that specific combination (a genuinely great passing offense
meeting a genuinely bad pass defense) predicts a bigger winner-margin than the SUM of "my
offense edge" + "their defense edge" would already imply on its own.

Run: python research/edge_signal_test_v14_matchup_interaction.py
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nflhub.sources.edge_teams import UnknownTeamError, normalize_team  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
SEASONS = range(2007, 2026)
TEST_START_SEASON = 2011
EWMA_ALPHA = 0.2
SEASON_PRIOR_GAMES = 5
WP_LO, WP_HI = 0.05, 0.95
RATING_METRICS = ("rush_off_epa", "pass_off_epa", "rush_def_epa_allowed", "pass_def_epa_allowed")
PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "epa", "wp"]


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
    cache_path = os.path.join(CACHE_DIR, f"team_game_phase_{season}.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"  downloading/parsing pbp {season}...", file=sys.stderr)
    df = pd.read_csv(PBP_URL.format(season=season), compression="gzip", usecols=PBP_COLS, low_memory=False)
    df = df[(df["season_type"] == "REG") & (df["play_type"].isin(["run", "pass"]))]
    df = df.dropna(subset=["wp", "epa", "posteam"])
    df = df[(df["wp"] >= WP_LO) & (df["wp"] <= WP_HI)]
    off = df.groupby(["game_id", "posteam", "play_type"])["epa"].agg(["size", "sum"]).reset_index()
    off = off.pivot_table(index=["game_id", "posteam"], columns="play_type", values=["size", "sum"])
    off.columns = [f"{a}_{b}" for a, b in off.columns]
    off = off.reset_index().rename(columns={"posteam": "team"})
    for col in ("size_run", "size_pass", "sum_run", "sum_pass"):
        if col not in off.columns:
            off[col] = 0.0
    off["rush_off_epa"] = off["sum_run"] / off["size_run"].replace(0, np.nan)
    off["pass_off_epa"] = off["sum_pass"] / off["size_pass"].replace(0, np.nan)
    out = off[["game_id", "team", "rush_off_epa", "pass_off_epa"]].dropna()
    out.to_parquet(cache_path)
    return out


def build_dataset() -> list[dict]:
    """Same season_prior/games_played/fade windowing as v13 (and live production), 4 EPA
    ratings only (no points/turnovers -- not needed for the interaction question). Feature
    rows carry both the plain linear diffs AND the 4 explicit matchup product terms, read
    BEFORE this game updates anything."""
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

        home_raw, away_raw = dict(teams[home]), dict(teams[away])
        home_raw["rush_def_epa_allowed"] = away_raw["rush_off_epa"]
        away_raw["rush_def_epa_allowed"] = home_raw["rush_off_epa"]
        home_raw["pass_def_epa_allowed"] = away_raw["pass_off_epa"]
        away_raw["pass_def_epa_allowed"] = home_raw["pass_off_epa"]

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
            hr, ar = rating[home], rating[away]
            for m in RATING_METRICS:
                feat_row[f"{m}_diff"] = hr[m] - ar[m]
            feat_row["rush_matchup_home"] = hr["rush_off_epa"] * ar["rush_def_epa_allowed"]
            feat_row["rush_matchup_away"] = ar["rush_off_epa"] * hr["rush_def_epa_allowed"]
            feat_row["pass_matchup_home"] = hr["pass_off_epa"] * ar["pass_def_epa_allowed"]
            feat_row["pass_matchup_away"] = ar["pass_off_epa"] * hr["pass_def_epa_allowed"]
            try:
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
    print("Building dataset (new season-prior-fade windowing, with matchup product terms)...", file=sys.stderr)
    rows = build_dataset()
    print(f"{len(rows)} games ({SEASONS.start}-{SEASONS.stop - 1})\n")

    linear_cols = [f"{m}_diff" for m in RATING_METRICS]
    matchup_cols = ["rush_matchup_home", "rush_matchup_away", "pass_matchup_home", "pass_matchup_away"]
    feature_sets = {
        "4 linear diffs alone (rush+pass, off+def)": linear_cols,
        "4 linear diffs + explicit matchup products": linear_cols + matchup_cols,
        "explicit matchup products alone (no linear diffs)": matchup_cols,
        "market spread_line alone": ["spread_line"],
        "4 linear diffs + matchups + spread_line": linear_cols + matchup_cols + ["spread_line"],
    }

    print(f"{'feature set':50s} {'n':>6s} {'AUC':>7s} {'logloss':>8s} {'acc':>6s}")
    for name, cols in feature_sets.items():
        r = walk_forward_auc(rows, cols)
        print(f"{name:50s} {r['n']:6d} {r['auc']:7.4f} {r['log_loss']:8.4f} {r['accuracy']:6.4f}")


if __name__ == "__main__":
    main()
