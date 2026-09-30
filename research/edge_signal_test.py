"""One-off exploration: does pre-game team-performance data predict who wins, and does it
add anything beyond what the betting market already prices in?

Data: nflverse `stats_team_week_{season}.csv` (per-team-per-game box score + EPA) joined to
nflverse `games.csv` (schedule/result/market spread). Both are the same sources nfl-hub
already uses (see nflhub/sources/history.py) and share game_id format, so no team-code
reconciliation was needed (verified for 2024; spot-checked 2007-2025 all resolve).

Leakage discipline: every team-performance feature is a PRIOR-games-this-season rolling
average, never the game's own box score (correlating a game's own turnover differential
with its own outcome is close to circular and answers a different, uninteresting question).
Evaluation is walk-forward by season (train on strictly earlier seasons, predict the next),
mirroring the methodology already validated in ff-draft-edge.

Run: python research/edge_signal_test.py
"""
from __future__ import annotations

import csv
import io
import os
import sys
from collections import defaultdict

import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_week_{season}.csv"
SEASONS = range(2007, 2025)  # full completed regular seasons; 2025 excluded (in progress)
TEST_START_SEASON = 2011  # first season predicted (needs a few prior seasons to train on)

# v2 rolling-rating knobs. EWMA_ALPHA: weight on each new game (0.2 ~ 4-game half-life) --
# picked to be responsive within a 17-game season, not tuned/cross-validated. CARRYOVER:
# fraction of a team's rating (relative to the current league mean) kept across the
# season boundary; 1-CARRYOVER is regressed to the mean instead of dropping the team's
# whole history at week 1 (the previous version required 3 fresh games/season, throwing
# away ~650 early-season games every year). Same value nfl-hub's ELWAY_BLEND_WEIGHT uses
# elsewhere for an external-model blend -- reused here for consistency, not because 0.65
# was independently found optimal for this.
EWMA_ALPHA = 0.2
CARRYOVER = 0.65
RATING_METRICS = ("off_epa_per_play", "def_epa_per_play_allowed", "yards_per_play", "turnover_margin", "explosive_rate")


def _cached_fetch(url: str, cache_name: str) -> str:
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
    text = _cached_fetch(GAMES_URL, "games.csv")
    games = {}
    for r in csv.DictReader(io.StringIO(text)):
        if r["game_type"] != "REG":
            continue
        try:
            season = int(r["season"])
        except ValueError:
            continue
        if season not in SEASONS:
            continue
        games[r["game_id"]] = r
    return games


def load_team_week(season: int) -> list[dict]:
    text = _cached_fetch(STATS_URL.format(season=season), f"stats_team_week_{season}.csv")
    return [r for r in csv.DictReader(io.StringIO(text)) if r["season_type"] == "REG"]


def f(row: dict, key: str) -> float:
    v = row.get(key, "")
    return float(v) if v not in ("", "NA", None) else 0.0


def team_game_features(row: dict) -> dict:
    """Raw (same-game, NOT lagged) per-team box score features. Used both as the concurrent
    sanity-check target and as the raw material for rolling prior-game averages."""
    plays = f(row, "attempts") + f(row, "carries")
    off_epa = f(row, "passing_epa") + f(row, "rushing_epa")
    off_yards = f(row, "passing_yards") + f(row, "rushing_yards")
    turnovers_committed = f(row, "passing_interceptions") + f(row, "fumbles_lost_total")
    explosive = f(row, "rushing_20") + f(row, "rushing_40") + f(row, "passing_20") + f(row, "passing_40")
    return {
        "plays": plays,
        "epa_per_play": off_epa / plays if plays else 0.0,
        "yards_per_play": off_yards / plays if plays else 0.0,
        "turnovers_committed": turnovers_committed,
        "explosive_rate": explosive / plays if plays else 0.0,
    }


class RunningMean:
    """Online mean, updated only with data seen so far -- safe to use as the season-boundary
    regression target without look-ahead (unlike computing a season's mean from all its games)."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        self.mean += (x - self.mean) / self.n


def league_mean_for(metric: str, running: dict[str, RunningMean]) -> float:
    if metric == "turnover_margin":
        return 0.0  # zero-sum across the two teams in a game, by construction
    if metric == "def_epa_per_play_allowed":
        metric = "off_epa_per_play"  # same pooled distribution, relabeled per side
    return running[metric].mean


def build_dataset() -> list[dict]:
    """One row per REG game: home/away rolling PRIOR-game team ratings + market spread +
    rest/QB-change + outcome. Ratings are an EWMA over a team's own game history (see
    EWMA_ALPHA), regressed toward the current league mean at each season boundary (CARRYOVER)
    instead of being reset to "unknown" for the first few games of every season."""
    by_game: dict[str, dict[str, dict]] = defaultdict(dict)  # game_id -> team -> raw row
    for season in SEASONS:
        for row in load_team_week(season):
            by_game[row["game_id"]][row["team"]] = row

    running_mean: dict[str, RunningMean] = {
        "off_epa_per_play": RunningMean(), "yards_per_play": RunningMean(), "explosive_rate": RunningMean(),
    }
    rating: dict[str, dict[str, float | None]] = defaultdict(lambda: {m: None for m in RATING_METRICS})
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

        home_raw = team_game_features(teams[home])
        away_raw = team_game_features(teams[away])
        home_raw["off_epa_per_play"] = home_raw.pop("epa_per_play")
        away_raw["off_epa_per_play"] = away_raw.pop("epa_per_play")
        # a team's defense allowed exactly what the opponent's offense produced this game
        home_raw["def_epa_per_play_allowed"] = away_raw["off_epa_per_play"]
        away_raw["def_epa_per_play_allowed"] = home_raw["off_epa_per_play"]
        home_raw["turnover_margin"] = f(teams[away], "passing_interceptions") + f(teams[away], "fumbles_lost_total") - home_raw["turnovers_committed"]
        away_raw["turnover_margin"] = f(teams[home], "passing_interceptions") + f(teams[home], "fumbles_lost_total") - away_raw["turnovers_committed"]

        # season-boundary carryover: regress each side's rating toward the CURRENT league
        # mean (computed from data seen so far only) before this game is used for anything.
        for team in (home, away):
            if last_season.get(team) is not None and last_season[team] != season:
                for m in RATING_METRICS:
                    r = rating[team][m]
                    if r is not None:
                        lm = league_mean_for(m, running_mean)
                        rating[team][m] = lm + CARRYOVER * (r - lm)
            last_season[team] = season

        feat_row = {"season": season, "week": week, "game_id": gid}
        # concurrent (same-game, NOT lagged) diffs -- circular by construction, kept only as
        # an illustration of why "what correlates with this game's own result" is the wrong
        # question to ask when the goal is prediction.
        for metric in ("off_epa_per_play", "yards_per_play", "turnover_margin", "explosive_rate"):
            feat_row[f"{metric}_diff_concurrent"] = home_raw[metric] - away_raw[metric]

        ok = all(rating[home][m] is not None and rating[away][m] is not None for m in RATING_METRICS)
        if ok:
            for m in RATING_METRICS:
                feat_row[f"{m}_diff"] = rating[home][m] - rating[away][m]
            try:
                feat_row["rest_diff"] = f(g, "home_rest") - f(g, "away_rest")
                feat_row["home_qb_changed"] = 1.0 if last_qb.get(home) not in (None, g.get("home_qb_id")) else 0.0
                feat_row["away_qb_changed"] = 1.0 if last_qb.get(away) not in (None, g.get("away_qb_id")) else 0.0
                feat_row["spread_line"] = float(g["spread_line"])
            except (ValueError, TypeError):
                ok = False
        if ok:
            try:
                res = float(g["result"])
            except (ValueError, TypeError):
                res = f(g, "home_score") - f(g, "away_score")
            feat_row["home_win"] = 1 if res > 0 else 0
            rows_out.append(feat_row)

        last_qb[home] = g.get("home_qb_id")
        last_qb[away] = g.get("away_qb_id")

        # update ratings and league means AFTER this game's features are locked in
        for m in ("off_epa_per_play", "yards_per_play", "explosive_rate"):
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
    print("Downloading / loading cached nflverse data...", file=sys.stderr)
    rows = build_dataset()
    print(f"{len(rows)} games with an established rating for both teams ({SEASONS.start}-{SEASONS.stop - 1}, EWMA alpha={EWMA_ALPHA}, carryover={CARRYOVER})\n")

    v2_cols = [f"{m}_diff" for m in RATING_METRICS]
    situational_cols = ["rest_diff", "home_qb_changed", "away_qb_changed"]
    feature_sets = {
        "off_epa_per_play_diff (v2 lagged)": ["off_epa_per_play_diff"],
        "def_epa_per_play_allowed_diff (v2 lagged)": ["def_epa_per_play_allowed_diff"],
        "yards_per_play_diff (v2 lagged)": ["yards_per_play_diff"],
        "turnover_margin_diff (v2 lagged)": ["turnover_margin_diff"],
        "explosive_rate_diff (v2 lagged)": ["explosive_rate_diff"],
        "all 5 v2 ratings combined": v2_cols,
        "all 5 v2 ratings + rest + QB-change": v2_cols + situational_cols,
        "market spread_line alone": ["spread_line"],
        "all v2 features + market spread_line": v2_cols + situational_cols + ["spread_line"],
        "--- same-game (circular, illustrative only) ---": [],
        "off_epa_per_play_diff (concurrent)": ["off_epa_per_play_diff_concurrent"],
        "turnover_margin_diff (concurrent)": ["turnover_margin_diff_concurrent"],
        "yards_per_play_diff (concurrent)": ["yards_per_play_diff_concurrent"],
        "explosive_rate_diff (concurrent)": ["explosive_rate_diff_concurrent"],
    }

    print(f"{'feature set':40s} {'n':>6s} {'AUC':>7s} {'logloss':>8s} {'acc':>6s}")
    for name, cols in feature_sets.items():
        if not cols:
            print(name)
            continue
        r = walk_forward_auc(rows, cols)
        print(f"{name:40s} {r['n']:6d} {r['auc']:7.4f} {r['log_loss']:8.4f} {r['accuracy']:6.4f}")


if __name__ == "__main__":
    main()
