"""Research model R: offensive and defensive expected points per drive (EPD) for every team, one season at a time (research only).

Offense EPD = mean EPD of the drives the team's offense ran; defense EPD = mean EPD of the drives its defense faced (what it ALLOWED, lower is better).
Net = offense EPD minus defense EPD allowed. Regular set = eligible drives beginning with more than 3:00 left in the half; 3-minute set = eligible drives
beginning with 3:00 or less (garbage-time drives are discarded in both: any play snapped under 5% or over 95% win probability). Drive-weighted season averages
(each drive counts once), unlike the game-mean inputs the models use.

Drives per game (offense: drives the team ran, defense: drives it faced) count EVERY drive: no garbage-time and no 3-minute exclusion.

Run: PYTHONPATH=. python research/r_epd_team_table.py [season ...]
Writes research/output/r_epd_team_table_{season}.csv.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import r_drive_features as rd  # noqa: E402
from nflhub.sources.edge_teams import UnknownTeamError, normalize_team  # noqa: E402


def norm(code):
    try:
        return normalize_team(code)
    except UnknownTeamError:
        return None


def team_table(season: int, current: bool) -> pd.DataFrame:
    d = rd.drive_table(season, use_cache=not current)
    d["team"], d["def_team"] = d["team"].map(norm), d["def_team"].map(norm)
    d = d.dropna(subset=["team", "def_team"])
    out = {}
    for label, sel in (("reg", ~d.three_min), ("3min", d.three_min)):
        x = d[sel]
        off = x.groupby("team")["epd"].agg(["mean", "count"])
        de = x.groupby("def_team")["epd"].agg(["mean", "count"])
        out[f"off_{label}"], out[f"n_off_{label}"] = off["mean"], off["count"]
        out[f"def_{label}"], out[f"n_def_{label}"] = de["mean"], de["count"]
    t = pd.DataFrame(out)
    # drives per game: EVERY drive, no garbage-time or 3-minute exclusion
    a = rd.all_drives(season, use_cache=not current)
    a["team"], a["def_team"] = a["team"].map(norm), a["def_team"].map(norm)
    a = a.dropna(subset=["team", "def_team"])
    games = pd.concat([a[["game_id", "team"]], a[["game_id", "def_team"]].rename(columns={"def_team": "team"})]).drop_duplicates().groupby("team").size()
    t["games"] = games
    t["drives_pg_off"] = a.groupby("team").size() / games
    t["drives_pg_def"] = a.groupby("def_team").size() / games
    t["eligible_share_off"] = t["n_off_reg"] / a.groupby("team").size()      # share of the offense's drives that are in the regular EPD set
    t["net_reg"] = t["off_reg"] - t["def_reg"]
    t["rank_off"] = t["off_reg"].rank(ascending=False).astype(int)
    t["rank_def"] = t["def_reg"].rank(ascending=True).astype(int)
    t["rank_net"] = t["net_reg"].rank(ascending=False).astype(int)
    return t.sort_values("net_reg", ascending=False)


def main() -> None:
    pd.set_option("display.width", 250)
    seasons = [int(a) for a in sys.argv[1:]] or [2026]
    for s in seasons:
        t = team_table(s, current=(s >= 2026))
        path = os.path.join(os.path.dirname(__file__), "output", f"r_epd_team_table_{s}.csv")
        t.round(4).to_csv(path)
        print(f"=== {s}: EPD per drive by team (offense = ran, defense = allowed; regular set; 3-minute set shown for reference) ===")
        show = t[["off_reg", "def_reg", "net_reg", "rank_off", "rank_def", "rank_net", "games", "drives_pg_off", "drives_pg_def", "eligible_share_off",
                  "off_3min", "def_3min", "n_off_3min"]]
        print(show.round(3).to_string())
        print(f"  correlation of offense EPD with defense EPD allowed across teams: {t['off_reg'].corr(t['def_reg']):+.2f}   (saved {path})\n")


if __name__ == "__main__":
    main()
