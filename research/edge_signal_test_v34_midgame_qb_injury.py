"""v34: how many of the model's misses could be explained by a QB going down MID-GAME --
i.e. healthy enough to start, but hurt before the final whistle -- rather than the PRE-game
QB-health signal already in `delta` (which only ever sees Out/Doubtful status as of that
week's injury report, filed before kickoff)? User's own framing: "does nflverse have
mid-game injury info" -- answer, verified live below: not a dedicated feed, but play-by-play
text explicitly logs it ("BUF-7-Ta.Johnson was injured during the play"), and the passer
sequence itself shows a real QB change when it's severe enough to end their day. QB only,
per explicit user direction ("good place to start").

Two independent signals, cross-checked against each other rather than trusted alone:
  1. STRUCTURAL: within one game, team's passer_player_id changes mid-stream (not just a
     single gadget-play cameo -- the new passer needs >=3 attempts the rest of the way), AND
     the win probability at the moment of the switch was still competitive (WP_LO/WP_HI =
     0.05/0.95, same band this whole project already uses for garbage-time exclusion) --
     a switch AFTER the game was already decided is far more likely late-game rest for a
     backup than an injury, so those are excluded from the "candidate" bucket entirely.
  2. TEXTUAL: nflverse's own play description for ANY play in that game explicitly says the
     departing passer "was injured during the play" (regex on "{TEAM}-{jersey}-{Name} was
     injured", built from that passer's own passer_player_name/jersey so it can't cross-match
     a different injured player in the same game).

A game only counts as a confirmed candidate if BOTH fire. Joined against the live production
`backtest_scatter` (home/away/season/week/delta/outcome) pulled straight from Supabase
(same `historical_power_rankings` kv blob v29/v30/v32/v33 already read), so "miss" here is
the model's actual production grading, not a recomputation.

Run: python research/edge_signal_test_v34_midgame_qb_injury.py
"""
from __future__ import annotations

import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(__file__)
CACHE_DIR = os.path.join(HERE, "cache")
sys.path.insert(0, os.path.join(HERE, ".."))
from nflhub import config, store  # noqa: E402
from nflhub.sources.edge_teams import UnknownTeamError, normalize_team  # noqa: E402

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
WP_LO, WP_HI = 0.05, 0.95  # same garbage-time band as nflhub/sources/team_ratings.py
MIN_BACKUP_ATTEMPTS = 3    # filters out one-off gadget/trick-play passer changes
FIRST_SEASON = 2007
LAST_SEASON = 2025  # last fully-completed season as of this run; current season excluded
                     # (no cache, and incomplete -- would need live refetch every run)

PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "home_team", "away_team",
            "play_type", "wp", "passer_player_id", "passer_player_name",
            "game_seconds_remaining", "desc"]


def _cached_pbp(season: int) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"qb_plays_{season}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    print(f"  downloading pbp {season}...", file=sys.stderr)
    df = pd.read_csv(PBP_URL.format(season=season), compression="gzip", usecols=PBP_COLS, low_memory=False)
    df = df[df["season_type"] == "REG"].copy()
    df.to_parquet(path)
    return df


def _norm(team: str) -> str | None:
    try:
        return normalize_team(team)
    except UnknownTeamError:
        return None


def find_midgame_qb_changes(df: pd.DataFrame) -> list[dict]:
    """One row per (game_id, team) where the passer changed mid-game outside garbage time,
    with a textual injury-confirmation flag. `df` is one season's full REG-season PBP."""
    pass_df = df[(df["play_type"] == "pass") & df["passer_player_id"].notna()].copy()
    pass_df = pass_df.sort_values(["game_id", "posteam", "game_seconds_remaining"], ascending=[True, True, False])

    out = []
    for (gid, team_raw), grp in pass_df.groupby(["game_id", "posteam"], sort=False):
        passers = grp["passer_player_id"].tolist()
        starter_id = passers[0]
        switch_idx = next((i for i, p in enumerate(passers) if p != starter_id), None)
        if switch_idx is None:
            continue
        new_id = passers[switch_idx]
        backup_attempts = sum(1 for p in passers[switch_idx:] if p == new_id)
        if backup_attempts < MIN_BACKUP_ATTEMPTS:
            continue
        wp_at_switch = grp["wp"].iloc[switch_idx - 1] if switch_idx > 0 else grp["wp"].iloc[0]
        if pd.isna(wp_at_switch) or not (WP_LO <= wp_at_switch <= WP_HI):
            continue  # game already decided when the switch happened -- not our candidate bucket

        team = _norm(team_raw)
        home = _norm(grp["home_team"].iloc[0])
        away = _norm(grp["away_team"].iloc[0])
        if team is None or home is None or away is None:
            continue

        starter_name = grp["passer_player_name"].iloc[0]
        game_desc = df.loc[df["game_id"] == gid, "desc"].dropna()
        name_pat = re.escape(str(starter_name))
        injured_text = bool(game_desc.str.contains(
            rf"{re.escape(str(team_raw))}-\d+-{name_pat} was injured", regex=True, na=False
        ).any())

        out.append({
            "game_id": gid, "team": team, "home": home, "away": away,
            "starter_name": starter_name, "wp_at_switch": round(float(wp_at_switch), 3),
            "backup_attempts": int(backup_attempts), "injured_text_confirmed": injured_text,
        })
    return out


def main():
    print("Loading production backtest_scatter...", file=sys.stderr)
    config.get_config()
    data = json.loads(store.kv_get("historical_power_rankings"))
    scatter = data["backtest_scatter"]
    by_key = {(r["season"], r["week"], r["home"], r["away"]): r for r in scatter}
    print(f"{len(scatter)} backtest rows")

    all_changes: list[dict] = []
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        df = _cached_pbp(season)
        changes = find_midgame_qb_changes(df)
        for c in changes:
            c["season"] = season
            c["week"] = int(df.loc[df["game_id"] == c["game_id"], "week"].iloc[0])
        all_changes.extend(changes)
        print(f"  {season}: {len(changes)} candidate mid-game QB changes", file=sys.stderr)

    confirmed = [c for c in all_changes if c["injured_text_confirmed"]]
    print(f"\n{len(all_changes)} structural candidates, {len(confirmed)} with explicit "
          f"'was injured' text confirmation ({len(confirmed) / len(all_changes) * 100:.0f}%)")

    # Join each CONFIRMED change to its graded backtest row.
    joined = []
    for c in confirmed:
        r = by_key.get((c["season"], c["week"], c["home"], c["away"]))
        if r is None or r["home_score"] == r["away_score"]:
            continue
        favorite = "home" if r["delta"] > 0 else "away" if r["delta"] < 0 else None
        actual_winner = "home" if r["home_score"] > r["away_score"] else "away"
        affected_side = "home" if c["team"] == r["home"] else "away"
        joined.append({**c, "outcome": r["outcome"], "favorite": favorite,
                        "actual_winner": actual_winner, "affected_side": affected_side,
                        "affected_was_favorite": affected_side == favorite})

    graded_games = {(r["season"], r["week"], r["home"], r["away"]) for r in scatter
                    if r["home_score"] != r["away_score"] and r["outcome"] in ("hit", "miss")}
    misses = {k for k in graded_games if by_key[k]["outcome"] == "miss"}
    hits = graded_games - misses

    affected_games = {(c["season"], c["week"], c["home"], c["away"]) for c in joined}
    affected_misses = affected_games & misses
    affected_hits = affected_games & hits

    print(f"\n{len(graded_games)} graded decided games total: {len(misses)} misses, {len(hits)} hits")
    print(f"Confirmed mid-game QB-injury games: {len(affected_games)} total "
          f"({len(affected_games) / len(graded_games) * 100:.1f}% of all graded games)")
    print(f"  -> among misses: {len(affected_misses)}/{len(misses)} = {len(affected_misses) / len(misses) * 100:.2f}%")
    print(f"  -> among hits:   {len(affected_hits)}/{len(hits)} = {len(affected_hits) / len(hits) * 100:.2f}%")

    fav_affected_misses = [c for c in joined if c["affected_was_favorite"] and c["outcome"] == "miss"]
    print(f"\nOf the {len(affected_misses)} missed games with a confirmed mid-game QB injury, "
          f"{len(fav_affected_misses)} involved the MODEL'S OWN FAVORITE's QB going down "
          f"(the mechanistically plausible explanation for the miss).")

    # The directionally-relevant cut: an UNDERDOG'S QB getting hurt mid-game should, if
    # anything, reinforce a HIT (makes the favorite's win more likely), not cause a miss --
    # diluting that into the blanket "either side" rate above understates the real signal.
    # So compare rates using only "the model's OWN FAVORITE got hurt" as the candidate flag.
    fav_hurt_games = {(c["season"], c["week"], c["home"], c["away"]) for c in joined if c["affected_was_favorite"]}
    fav_hurt_misses = fav_hurt_games & misses
    fav_hurt_hits = fav_hurt_games & hits
    n1, n2 = len(misses), len(hits)
    x1, x2 = len(fav_hurt_misses), len(fav_hurt_hits)
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se else 0.0
    p_val = 2 * (1 - norm.cdf(abs(z)))
    print(f"\nFavorite's-QB-hurt-mid-game rate: {x1}/{n1} = {p1 * 100:.2f}% of misses vs "
          f"{x2}/{n2} = {p2 * 100:.2f}% of hits -- z={z:.2f}, p={p_val:.5f}")

    print("\nExample misses with a confirmed mid-game QB injury (favorite's side affected):")
    shown = 0
    for c in joined:
        if c["outcome"] != "miss" or not c["affected_was_favorite"]:
            continue
        print(f"  {c['season']} wk{c['week']} {c['away']}@{c['home']}: {c['starter_name']} "
              f"({c['team']}) hurt at wp={c['wp_at_switch']}, backup threw {c['backup_attempts']} "
              f"more attempts. Model favored {c['favorite']}, {c['actual_winner']} won.")
        shown += 1
        if shown >= 15:
            break

    print(f"\n{'=' * 70}\nSTRUCTURAL-ONLY (no text confirmation required) for comparison:")
    struct_affected = {(c["season"], c["week"], c["home"], c["away"]) for c in all_changes
                        if (c["season"], c["week"], c["home"], c["away"]) in graded_games}
    print(f"  {len(struct_affected)} games ({len(struct_affected) / len(graded_games) * 100:.1f}%), "
          f"of which {len(struct_affected & misses)} are misses "
          f"({len(struct_affected & misses) / len(misses) * 100:.2f}% of all misses) vs "
          f"{len(struct_affected & hits) / len(hits) * 100:.2f}% of hits")


if __name__ == "__main__":
    main()
