"""Research model R feature extraction (play-by-play only, REG season, 2007+, garbage time excluded like production).

Per (game_id, team = the OFFENSE) pass-protection / pass-game / turnover-quality rates. A defense's version of each stat is the opponent
offense's value in the same game, exactly as production derives rush/pass defense from the opponent's offense. Cached per season in
research/cache/v78_features_{season}.parquet (gitignored).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
from nflhub.sources.edge_teams import UnknownTeamError, normalize_team  # noqa: E402

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
CACHE = os.path.join(HERE, "cache")
SEASONS = list(range(2007, 2026))
WP_LO, WP_HI = 0.05, 0.95
WANT = {"game_id", "season_type", "posteam", "play_type", "epa", "wp", "sack", "qb_hit", "pass_attempt", "complete_pass",
        "air_yards", "yards_after_catch", "interception", "fumble", "fumble_lost", "rush_attempt", "play_deleted"}


def extract_season(season: int) -> pd.DataFrame:
    path = os.path.join(CACHE, f"v78_features_{season}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    for attempt in range(6):          # GitHub release downloads time out now and then (504); the cache means only the failed season repeats
        try:
            df = pd.read_csv(PBP_URL.format(season=season), compression="gzip", low_memory=False, usecols=lambda c: c in WANT)
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == 5:
                raise
            print(f"    {season}: download failed ({exc}); retry {attempt + 1}/5", file=sys.stderr)
            time.sleep(15 * (attempt + 1))
    for c in WANT - set(df.columns):
        df[c] = np.nan
    df = df[(df["season_type"] == "REG") & (df["play_deleted"].fillna(0) != 1)]
    df = df[df["play_type"].isin(["run", "pass"])].dropna(subset=["wp", "epa", "posteam"])
    df = df[(df["wp"] >= WP_LO) & (df["wp"] <= WP_HI)].copy()
    for c in ("sack", "qb_hit", "pass_attempt", "complete_pass", "interception", "fumble", "fumble_lost"):
        df[c] = df[c].fillna(0)
    K = ["game_id", "posteam"]
    pas = df[df["play_type"] == "pass"]
    att = pas[pas["pass_attempt"] == 1]
    comp = att[att["complete_pass"] == 1]
    g = pd.DataFrame({
        "dropbacks": pas.groupby(K).size(),
        "sack_rate": pas.groupby(K)["sack"].mean(),
        "hit_rate": pas.groupby(K)["qb_hit"].mean(),          # includes sacks
        "comp_rate": att.groupby(K)["complete_pass"].mean(),
        "air_yards": att.groupby(K)["air_yards"].mean(),
        "yac": comp.groupby(K)["yards_after_catch"].mean(),
        "int_rate": att.groupby(K)["interception"].mean(),
        "epa_nosack": pas[pas["sack"] == 0].groupby(K)["epa"].mean(),
        "epa_sack": pas[pas["sack"] == 1].groupby(K)["epa"].mean(),
    })
    allp = df.groupby(K)
    g["fumble_rate"] = allp["fumble"].sum() / allp.size()
    g["fumbles_lost"] = allp["fumble_lost"].sum()
    g["fumbles"] = allp["fumble"].sum()
    out = g.reset_index().rename(columns={"posteam": "team"})
    out["season"] = season
    os.makedirs(CACHE, exist_ok=True)
    out.to_parquet(path)
    return out


def build() -> pd.DataFrame:
    parts = []
    for s in SEASONS:
        print(f"  extracting v78 features {s}...", file=sys.stderr)
        parts.append(extract_season(s))
    f = pd.concat(parts, ignore_index=True)

    def norm(code):
        try:
            return normalize_team(code)
        except UnknownTeamError:
            return None
    cmap = {c: norm(c) for c in set(f["team"])}
    f["team"] = f["team"].map(cmap)
    return f.dropna(subset=["team"]).reset_index(drop=True)
