"""Pulls the pool data the v65/v66 Monte Carlo research needs into research/cache/pool_weeks.json
(gitignored -- it contains the pool's entrant names and picks).

Per week: every entrant's weekly/season correct counts, picks (team codes, sheet column order),
and Monday-night tiebreaker guess; the week's games (nflverse games.csv: closing moneylines, spread,
posted total, scores, kickoff day/time); the national pick % per team the app scraped; and the app's
own FADE/CHALK/NO_PLAY recommendation log for that week.

Read-only against the sheet and Supabase. Run: PYTHONPATH=. python research/pool_data_extract.py
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nflhub import store  # noqa: E402
from nflhub.config import get_config  # noqa: E402
from nflhub.sources import edge_sheet  # noqa: E402
from nflhub.sources.edge_teams import TEAMS, normalize_team  # noqa: E402

SEASON = 2026
OUT = os.path.join(os.path.dirname(__file__), "cache", "pool_weeks.json")
ORD = re.compile(r"^(\d+)(st|nd|rd|th)\s*", re.IGNORECASE)
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_tab(sheet_id, week):
    if not edge_sheet._tab_exists(sheet_id, week):
        return None
    resp = requests.get(edge_sheet._csv_url(sheet_id, week), timeout=edge_sheet.TIMEOUT)
    resp.raise_for_status()
    entrants = []
    for row in csv.reader(io.StringIO(resp.text)):
        if len(row) < 3:
            continue
        m = ORD.match(row[0])
        name = ORD.sub("", row[0]).strip()
        if not name:
            continue
        picks, tiebreak = [], None
        for cell in row[3:]:
            c = (cell or "").strip()
            if not c or c.lower() in edge_sheet._NO_PICK:
                continue
            try:
                picks.append(normalize_team(c))
            except ValueError:
                v = num(c)
                if v is not None:
                    tiebreak = v
        entrants.append({"name": name, "rank": int(m.group(1)) if m else None,
                         "weekly_correct": edge_sheet._parse_int(row[1]), "season_correct": edge_sheet._parse_int(row[2]),
                         "picks": picks, "tiebreak": tiebreak})
    return entrants


def main():
    cfg = get_config()
    games_rows = [r for r in csv.DictReader(io.StringIO(requests.get(GAMES_URL, timeout=30).text))
                  if r["game_type"] == "REG" and int(r["season"]) == SEASON]
    weeks = {}
    for w in range(1, 19):
        entrants = parse_tab(cfg.edge.sheet_id, w)
        if not entrants:
            continue
        games = []
        for r in games_rows:
            if int(r["week"]) != w:
                continue
            games.append({
                "game_id": r["game_id"], "home": normalize_team(r["home_team"]), "away": normalize_team(r["away_team"]),
                "gameday": r["gameday"], "gametime": r["gametime"], "weekday": r["weekday"],
                "home_score": num(r["home_score"]), "away_score": num(r["away_score"]),
                "ml_home": num(r["home_moneyline"]), "ml_away": num(r["away_moneyline"]),
                "spread_line": num(r["spread_line"]), "total_line": num(r["total_line"]),
            })
        national = {t: store.edge_get_national_pct(SEASON, w, t) for t in TEAMS}
        recs = store.edge_recommendation_log_for_week(SEASON, w)
        weeks[str(w)] = {"entrants": entrants, "games": games, "national_pct": national,
                         "recommendation_log": recs, "my_name": cfg.edge.my_name}
        print(f"week {w}: {len(entrants)} entrants, {len(games)} games, rec log {len(recs)}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(weeks, f)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
