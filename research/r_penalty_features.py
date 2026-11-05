"""Research model R: penalty-inclusive EPA (play-by-play, REG, 2007+, garbage time excluded like production).

Production's four EPA stats keep only plays labeled run/pass, but nflverse labels most penalty plays `no_play` (defensive pass interference 100%,
false start 100%, defensive holding 76%, offensive holding 64%) while still giving them an EPA (DPI averages +1.76 for the offense). Owner direction
2026-11-05: a spot foul such as pass interference is, for football purposes, the same outcome as a completed pass, so penalties belong in EPA.

Per (game_id, team = the OFFENSE):
  rush_epa_pen / pass_epa_pen   mean EPA over run (pass) plays PLUS the no_play penalty plays whose play was a run (pass) (nflverse `rush`/`pass` flags)
  live_committed / live_drawn   summed EPA of live-ball penalty plays (the play had a run or pass) committed by this offense / by the defense
  presnap_committed / presnap_drawn   the same for penalties before the snap (false start, delay of game, offside, encroachment, neutral zone...)
  rush_epa_allpen / pass_epa_allpen   the same, ALSO counting pre-snap penalties: each pre-snap penalty's EPA goes to rush or pass according to the
                                      type of the next scrimmage play by the same offense in that game (the replayed down), else to the more common
                                      type in that game. This is research model R's definition of rush/pass EPA (owner direction 2026-11-05).
A defense's version of each stat is the opponent offense's value in the same game, as everywhere else. Cached per season in
research/cache/v82_pen_{season}.parquet (gitignored).
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
WANT = {"game_id", "play_id", "season_type", "play_type", "epa", "wp", "pass", "rush", "penalty", "penalty_team", "posteam", "defteam", "play_deleted"}


def extract_season(season: int) -> pd.DataFrame:
    path = os.path.join(CACHE, f"v82_pen_{season}.parquet")
    if os.path.exists(path):
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
    for c in WANT - set(df.columns):
        df[c] = np.nan
    df = df[(df["season_type"] == "REG") & (df["play_deleted"].fillna(0) != 1)].sort_values(["game_id", "play_id"]).copy()
    # type and offense of the NEXT scrimmage play in the game (before any filtering), for assigning pre-snap penalties
    scrim = df["play_type"].isin(["run", "pass"])
    nxt_type = df["play_type"].where(scrim).groupby(df["game_id"]).shift(-1).groupby(df["game_id"]).bfill()
    nxt_pos = df["posteam"].where(scrim).groupby(df["game_id"]).shift(-1).groupby(df["game_id"]).bfill()
    df["next_type"] = nxt_type.where(nxt_pos == df["posteam"])
    df = df.dropna(subset=["wp", "epa", "posteam"])
    df = df[(df["wp"] >= WP_LO) & (df["wp"] <= WP_HI)].copy()
    pen_np = df[(df["play_type"] == "no_play") & (df["penalty"] == 1)].copy()
    pen_np["kind"] = np.where(pen_np["pass"] == 1, "pass", np.where(pen_np["rush"] == 1, "rush", "presnap"))
    K = ["game_id", "posteam"]
    run_all = pd.concat([df[df["play_type"] == "run"][K + ["epa"]], pen_np[pen_np["kind"] == "rush"][K + ["epa"]]])
    pass_all = pd.concat([df[df["play_type"] == "pass"][K + ["epa"]], pen_np[pen_np["kind"] == "pass"][K + ["epa"]]])
    live = pen_np[pen_np["kind"] != "presnap"]
    pre = pen_np[pen_np["kind"] == "presnap"].copy()
    mix = df[df["play_type"].isin(["run", "pass"])].groupby(K)["play_type"].apply(lambda x: "pass" if (x == "pass").mean() >= 0.5 else "run")
    pre = pre.join(mix.rename("mix_type"), on=K)
    pre["to"] = pre["next_type"].fillna(pre["mix_type"]).fillna("pass")
    run_all2 = pd.concat([run_all, pre[pre["to"] == "run"][K + ["epa"]]])
    pass_all2 = pd.concat([pass_all, pre[pre["to"] == "pass"][K + ["epa"]]])
    g = pd.DataFrame({
        "rush_epa_pen": run_all.groupby(K)["epa"].mean(),
        "pass_epa_pen": pass_all.groupby(K)["epa"].mean(),
        "rush_epa_allpen": run_all2.groupby(K)["epa"].mean(),
        "pass_epa_allpen": pass_all2.groupby(K)["epa"].mean(),
        "live_committed": live[live["penalty_team"] == live["posteam"]].groupby(K)["epa"].sum(),
        "live_drawn": live[live["penalty_team"] == live["defteam"]].groupby(K)["epa"].sum(),
        "presnap_committed": pre[pre["penalty_team"] == pre["posteam"]].groupby(K)["epa"].sum(),
        "presnap_drawn": pre[pre["penalty_team"] == pre["defteam"]].groupby(K)["epa"].sum(),
    })
    out = g.reset_index().rename(columns={"posteam": "team"})
    for c in ("live_committed", "live_drawn", "presnap_committed", "presnap_drawn"):
        out[c] = out[c].fillna(0.0)
    out["season"] = season
    os.makedirs(CACHE, exist_ok=True)
    out.to_parquet(path)
    return out


def build() -> pd.DataFrame:
    parts = []
    for s in SEASONS:
        print(f"  extracting v81 penalty features {s}...", file=sys.stderr)
        parts.append(extract_season(s))
    f = pd.concat(parts, ignore_index=True)

    def norm(code):
        try:
            return normalize_team(code)
        except UnknownTeamError:
            return None
    f["team"] = f["team"].map({c: norm(c) for c in set(f["team"])})
    return f.dropna(subset=["team"]).reset_index(drop=True)
