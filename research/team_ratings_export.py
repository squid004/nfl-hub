"""Exports, for every matchup, each team's 4 rolling ratings: rush-offense EPA/play,
pass-offense EPA/play, rush-defense EPA/play allowed, pass-defense EPA/play allowed.

Same methodology as edge_signal_test_v4_matchup.py (garbage-time excluded via the win-
probability band, EWMA rolling with season-boundary carryover, PRIOR-games-only so every
rating is genuinely pre-game) -- this script just exports the ratings themselves instead of
only the win/loss AUC test.

Two outputs:
- research/output/team_ratings.csv: one row per played game (2007-present), both teams'
  4 ratings as they stood immediately before that game, plus market spread and result.
- console: the same 4 ratings for every team's UPCOMING (not yet played) game this season,
  using each team's current (latest) rating -- the part actually usable for a real pick.

Run: python research/team_ratings_export.py
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

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
CURRENT_SEASON = 2026
SEASONS = range(2007, CURRENT_SEASON + 1)
EWMA_ALPHA = 0.2
CARRYOVER = 0.65
WP_LO, WP_HI = 0.05, 0.95
RATING_METRICS = ("rush_off_epa", "pass_off_epa", "rush_def_epa_allowed", "pass_def_epa_allowed")
PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "epa", "wp"]


def _cached_fetch_text(url: str, cache_name: str, force: bool = False) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, cache_name)
    if os.path.exists(path) and not force:
        with open(path, encoding="utf-8") as f:
            return f.read()
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(path, "w", encoding="utf-8") as f:
        f.write(resp.text)
    return resp.text


def load_games() -> list[dict]:
    # always re-fetch (not cached) so the current season's schedule/results stay fresh
    text = _cached_fetch_text(GAMES_URL, "games.csv", force=True)
    return [r for r in csv.DictReader(io.StringIO(text)) if r["game_type"] == "REG"]


def team_game_phase_stats(season: int) -> pd.DataFrame:
    cache_path = os.path.join(CACHE_DIR, f"team_game_phase_{season}.parquet")
    force = season == CURRENT_SEASON  # the in-progress season's pbp grows week to week
    if os.path.exists(cache_path) and not force:
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


class RunningMean:
    def __init__(self):
        self.n, self.mean = 0, 0.0

    def update(self, x):
        self.n += 1
        self.mean += (x - self.mean) / self.n


def main():
    print("Loading play-by-play (cached seasons load instantly; current season re-fetched)...", file=sys.stderr)
    by_game: dict[str, dict[str, dict]] = defaultdict(dict)
    for season in SEASONS:
        for row in team_game_phase_stats(season).to_dict("records"):
            by_game[row["game_id"]][row["team"]] = row

    games_list = load_games()
    games = {g["game_id"]: g for g in games_list if int(g["season"]) in SEASONS}

    running_mean = {"rush_off_epa": RunningMean(), "pass_off_epa": RunningMean()}

    def league_mean(metric: str) -> float:
        return running_mean[metric.replace("_def_epa_allowed", "_off_epa")].mean

    rating: dict[str, dict] = defaultdict(lambda: {m: None for m in RATING_METRICS})
    last_season: dict[str, int] = {}
    export_rows = []

    for gid, teams in sorted(by_game.items()):
        if gid not in games or len(teams) != 2:
            continue
        g = games[gid]
        season, week = int(g["season"]), int(g["week"])
        home, away = g["home_team"], g["away_team"]
        if home not in teams or away not in teams:
            continue

        home_raw, away_raw = dict(teams[home]), dict(teams[away])
        home_raw["rush_def_epa_allowed"] = away_raw["rush_off_epa"]
        away_raw["rush_def_epa_allowed"] = home_raw["rush_off_epa"]
        home_raw["pass_def_epa_allowed"] = away_raw["pass_off_epa"]
        away_raw["pass_def_epa_allowed"] = home_raw["pass_off_epa"]

        for team in (home, away):
            if last_season.get(team) is not None and last_season[team] != season:
                for m in RATING_METRICS:
                    r = rating[team][m]
                    if r is not None:
                        lm = league_mean(m)
                        rating[team][m] = lm + CARRYOVER * (r - lm)
            last_season[team] = season

        row = {"game_id": gid, "season": season, "week": week, "home_team": home, "away_team": away,
               "spread_line": g.get("spread_line"), "result": g.get("result")}
        for side, team in (("home", home), ("away", away)):
            for m in RATING_METRICS:
                v = rating[team][m]
                row[f"{side}_{m}"] = round(v, 4) if v is not None else None
        export_rows.append(row)

        for m in ("rush_off_epa", "pass_off_epa"):
            running_mean[m].update(home_raw[m])
            running_mean[m].update(away_raw[m])
        for team, raw in ((home, home_raw), (away, away_raw)):
            for m in RATING_METRICS:
                prev = rating[team][m]
                rating[team][m] = raw[m] if prev is None else (1 - EWMA_ALPHA) * prev + EWMA_ALPHA * raw[m]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "team_ratings.csv")
    pd.DataFrame(export_rows).to_csv(out_path, index=False)
    print(f"\nWrote {len(export_rows)} played games to {out_path}\n")

    # upcoming: current-season games.csv rows with no result yet, using each team's LATEST
    # rating (the state after every played game above has already updated it).
    upcoming = [g for g in games_list if int(g["season"]) == CURRENT_SEASON and g.get("result") in ("", "NA", None)]
    upcoming.sort(key=lambda g: (int(g["week"]), g["game_id"]))
    if not upcoming:
        print("No upcoming games found for", CURRENT_SEASON)
        return
    next_week = int(upcoming[0]["week"])
    print(f"Week {next_week}, {CURRENT_SEASON} -- current ratings entering this matchup:\n")
    hdr = f"{'matchup':16s} {'spread':>7s} | {'rush_off':>9s} {'pass_off':>9s} {'rush_def':>9s} {'pass_def':>9s}"
    for g in upcoming:
        if int(g["week"]) != next_week:
            continue
        print(f"{g['away_team']} @ {g['home_team']}  (spread_line={g.get('spread_line')})")
        print(hdr.replace("matchup", "team").replace("spread", ""))
        for team in (g["home_team"], g["away_team"]):
            r = rating.get(team, {})
            vals = [r.get(m) for m in RATING_METRICS]
            vals_s = " ".join(f"{v:9.4f}" if v is not None else f"{'n/a':>9s}" for v in vals)
            print(f"  {team:14s} {'':7s} | {vals_s}")
        print()


if __name__ == "__main__":
    main()
