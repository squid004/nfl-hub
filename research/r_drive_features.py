"""Research model R experimental metric: expected points per drive (EPD), offense and defense. Research only -- production and the EPA stats are untouched.

EPD of a drive = the sum of the EPA of every play of the drive (rushes, passes, penalty no-plays, kneels and spikes, the closing punt or field goal; kickoffs
and extra points excluded), i.e. the points the drive added or cost compared with what a typical drive from that starting spot is worth, with turnovers,
field position, penalties and the kick that ends the drive all inside one number. Per (game, offense) it is the mean over that team's eligible drives.

Eligibility (owner direction 2026-11-05):
  * garbage time: if ANY play of the drive is snapped with the offense's win probability under 5% or over 95%, the whole drive is discarded
  * two sets by the clock when the drive BEGINS: the regular set (more than 3:00 left in the half) and the 3-minute set (3:00 or less left in the half)
A defense's version is the opponent offense's value in the same game. Cached per season in research/cache/v83_epd_{season}.parquet (gitignored).
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
THREE_MIN = 180
PLAYS = ["run", "pass", "no_play", "punt", "field_goal", "qb_kneel", "qb_spike"]
WANT = {"game_id", "play_id", "fixed_drive", "posteam", "defteam", "play_type", "epa", "wp", "half_seconds_remaining", "season_type", "play_deleted"}


def drive_table(season: int, use_cache: bool = True) -> pd.DataFrame:
    """One row per ELIGIBLE drive (garbage drives already discarded): game_id, team (offense), def_team, epd, start (seconds left in the half when the
    drive began), three_min. The current season is never cached."""
    path = os.path.join(CACHE, f"v83_drives_{season}.parquet")
    if use_cache and os.path.exists(path):
        return pd.read_parquet(path)
    for attempt in range(6):
        try:
            df = pd.read_csv(PBP_URL.format(season=season), compression="gzip", low_memory=False, usecols=lambda c: c in WANT)
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == 5:
                raise
            print(f"    {season}: download failed ({exc}); retry {attempt + 1}/5", file=sys.stderr)
            time.sleep(15 * (attempt + 1))
    df = df[(df["season_type"] == "REG") & (df["play_deleted"].fillna(0) != 1)]
    df = df[df["play_type"].isin(PLAYS) & df["posteam"].notna() & df["fixed_drive"].notna()].sort_values(["game_id", "play_id"])
    d = df.groupby(["game_id", "fixed_drive"]).agg(
        team=("posteam", "first"), def_team=("defteam", "first"), n_off=("posteam", "nunique"), epd=("epa", "sum"), start=("half_seconds_remaining", "first"),
        wp_min=("wp", "min"), wp_max=("wp", "max")).reset_index()
    d = d[(d["n_off"] == 1) & d["epd"].notna()]
    # garbage time: any play in the drive outside the win-probability band discards the whole drive
    d = d[~((d["wp_min"] < WP_LO) | (d["wp_max"] > WP_HI))]
    d["three_min"] = d["start"] <= THREE_MIN
    d = d[["game_id", "team", "def_team", "epd", "start", "three_min"]].reset_index(drop=True)
    if use_cache:
        os.makedirs(CACHE, exist_ok=True)
        d.to_parquet(path)
    return d


def extract_season(season: int) -> pd.DataFrame:
    path = os.path.join(CACHE, f"v83_epd_{season}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    d = drive_table(season)
    g = d.groupby(["game_id", "team"]).apply(lambda x: pd.Series({
        "epd_reg": x.loc[~x.three_min, "epd"].mean(), "n_reg": float((~x.three_min).sum()),
        "epd_3min": x.loc[x.three_min, "epd"].mean(), "n_3min": float(x.three_min.sum()),
        "epd_all": x["epd"].mean(), "n_all": float(len(x))}), include_groups=False).reset_index()
    g["season"] = season
    os.makedirs(CACHE, exist_ok=True)
    g.to_parquet(path)
    return g


def build() -> pd.DataFrame:
    parts = []
    for s in SEASONS:
        print(f"  extracting v83 drive features {s}...", file=sys.stderr)
        parts.append(extract_season(s))
    f = pd.concat(parts, ignore_index=True)

    def norm(code):
        try:
            return normalize_team(code)
        except UnknownTeamError:
            return None
    f["team"] = f["team"].map({c: norm(c) for c in set(f["team"])})
    return f.dropna(subset=["team"]).reset_index(drop=True)
