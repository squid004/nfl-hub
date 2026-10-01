"""Does weather (temperature, rain, snow, wind) predict scoring ABOVE AND BEYOND what the
existing team-quality model already explains? Naive "average points on cold days" would
confound weather with which teams happen to play in cold cities -- this instead looks at the
RESIDUAL (actual points - this project's own predict_points() projection, which already
accounts for both teams' offense/defense quality) against weather, isolating the weather
effect specifically.

Data:
- roof/location from nflverse games.csv (already used throughout this project) -- filtered to
  roof in (outdoors, open) and location == 'Home' (excludes domes/closed roofs and neutral-site
  international games, which don't have a fixed home-city climate).
- Daily temperature/precipitation/snowfall/wind from Open-Meteo's free historical archive API
  (same source already used in the nest-home project), one request per stadium's coordinates
  covering its full date range -- NOT per-game, to keep this to ~25 requests instead of
  thousands. Caveat: this is a DAILY aggregate (whole-day precipitation/snow/wind-max), not
  specifically during the 3-hour game window -- a reasonable free-data proxy, not exact.

Run: python research/weather_scoring_analysis.py
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys

import time

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
import edge_signal_test_v9_points_prediction as v9  # reuses the rating computation + rebuilds predict_points

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE, END_DATE = "2006-06-01", "2025-12-31"

# One outdoor-era coordinate per team CODE as it appears in games.csv home_team. Teams whose
# only home venue is a fixed dome (ATL/DAL/DET/HOU/IND/LV/MIN/NO -- current roof) are omitted
# on purpose EXCEPT the retractable-roof ones, which can show roof=='open' for some games.
STADIUM_COORDS = {
    "ARI": (33.5276, -112.2626),   # State Farm Stadium, Glendale (retractable)
    "ATL": (33.7554, -84.4009),    # Mercedes-Benz Stadium, Atlanta (retractable)
    "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7870),
    "CAR": (35.2258, -80.8528),
    "CHI": (41.8623, -87.6167),
    "CIN": (39.0954, -84.5160),
    "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0945),    # AT&T Stadium, Arlington (retractable)
    "DEN": (39.7439, -105.0201),
    "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107),    # NRG Stadium (retractable)
    "IND": (39.7601, -86.1639),    # Lucas Oil Stadium (retractable)
    "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839),
    "LA": (34.0141, -118.2879),    # LA Memorial Coliseum era (2016-2019); SoFi (2020+) is a fixed dome, excluded by roof filter
    "LAC": (33.8644, -118.2611),   # StubHub/Dignity Health Sports Park era (2017-2019); SoFi is a fixed dome
    "MIA": (25.9580, -80.2389),
    "MIN": (44.9764, -93.2244),    # TCF Bank Stadium, outdoor 2014-2015 while the dome was rebuilt
    "NE": (42.0909, -71.2643),
    "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745),
    "OAK": (37.7516, -122.2005),
    "PHI": (39.9008, -75.1675),
    "PIT": (40.4468, -80.0158),
    "SD": (32.7831, -117.1196),
    "SEA": (47.5952, -122.3316),
    "SF": (37.4032, -121.9698),
    "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713),
    "WAS": (38.9076, -76.8645),
}


def _cached_fetch_text(url: str, cache_name: str, params: dict | None = None) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, cache_name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    for attempt in range(10):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            wait = 20 * (attempt + 1)
            print(f"    rate limited, waiting {wait}s...", file=sys.stderr)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        with open(path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        time.sleep(5)  # be polite to the free API between calls
        return resp.text
    resp.raise_for_status()
    return resp.text


def load_games() -> list[dict]:
    text = _cached_fetch_text(GAMES_URL, "games.csv")
    rows = [r for r in csv.DictReader(io.StringIO(text))
            if r["game_type"] == "REG" and 2007 <= int(r["season"]) <= 2025]
    return rows


def fetch_stadium_weather(team: str, lat: float, lon: float) -> dict[str, dict]:
    """date (YYYY-MM-DD) -> {temp_max_c, precip_mm, snowfall_cm, wind_kmh}."""
    cache_name = f"weather_{team}.json"
    text = _cached_fetch_text(WEATHER_URL, cache_name, params={
        "latitude": lat, "longitude": lon, "start_date": START_DATE, "end_date": END_DATE,
        "daily": "temperature_2m_max,precipitation_sum,snowfall_sum,windspeed_10m_max",
        "timezone": "America/New_York",
    })
    data = json.loads(text)
    daily = data["daily"]
    out = {}
    for i, d in enumerate(daily["time"]):
        out[d] = {
            "temp_max_c": daily["temperature_2m_max"][i],
            "precip_mm": daily["precipitation_sum"][i],
            "snowfall_cm": daily["snowfall_sum"][i],
            "wind_kmh": daily["windspeed_10m_max"][i],
        }
    return out


def main():
    print("Loading games + fetching weather per stadium...", file=sys.stderr)
    games = load_games()
    outdoor = [g for g in games if g["roof"] in ("outdoors", "open") and g["location"] == "Home"
               and g["home_team"] in STADIUM_COORDS and g["temp"] not in ("", "NA")]
    print(f"{len(outdoor)} outdoor home-city games with a known stadium")

    weather_by_team = {}
    for team in sorted({g["home_team"] for g in outdoor}):
        lat, lon = STADIUM_COORDS[team]
        print(f"  fetching weather for {team}...", file=sys.stderr)
        weather_by_team[team] = fetch_stadium_weather(team, lat, lon)

    matched = 0
    for g in outdoor:
        wx = weather_by_team.get(g["home_team"], {}).get(g["gameday"])
        if wx:
            g["_wx"] = wx
            matched += 1
    print(f"{matched}/{len(outdoor)} games matched to a weather day\n")

    # build the team-quality model's predicted points for each matched game, so we can look at
    # the RESIDUAL (actual - predicted) against weather, not raw points against weather.
    print("Building team ratings for residual computation...", file=sys.stderr)
    rating_rows = v9.build_dataset()  # reuse: gives per-side predicted-points-ready features
    # rating_rows already has 'features' (own-offense+opp-defense) and 'points' per side; use
    # the SAME final model fit as production (intercept/weights below, copied from
    # nflhub/sources/team_ratings.py POINTS_INTERCEPT/POINTS_WEIGHTS).
    INTERCEPT = 7.21333
    WEIGHTS = np.array([2.472559, 6.259427, 0.371737, -0.0, 2.470530, 0.666745, 0.317523, -0.0])
    pred_by_game_side = {}
    for r in rating_rows:
        pred = INTERCEPT + float(np.dot(WEIGHTS, r["features"]))
        pred_by_game_side[(r["game_id"], r["side"])] = (pred, r["points"])

    rows = []
    for g in outdoor:
        if "_wx" not in g:
            continue
        hp = pred_by_game_side.get((g["game_id"], "home"))
        ap = pred_by_game_side.get((g["game_id"], "away"))
        if not hp or not ap:
            continue
        wx = g["_wx"]
        rows.append({
            "season": int(g["season"]), "temp_f": wx["temp_max_c"] * 9 / 5 + 32,
            "precip_mm": wx["precip_mm"], "snowfall_cm": wx["snowfall_cm"], "wind_mph": wx["wind_kmh"] * 0.621371,
            "home_residual": hp[1] - hp[0], "away_residual": ap[1] - ap[0],
            "total_residual": (hp[1] + ap[1]) - (hp[0] + ap[0]),
        })
    print(f"{len(rows)} games with weather + a model prediction to compare against\n")
    df = pd.DataFrame(rows)

    def bucket_report(col, bins, labels):
        df["_b"] = pd.cut(df[col], bins=bins, labels=labels)
        g = df.groupby("_b", observed=True).agg(
            n=("total_residual", "size"),
            avg_total_residual=("total_residual", "mean"),
            avg_home_residual=("home_residual", "mean"),
        )
        print(g.to_string())
        print()

    print("=== Total-points residual (actual - predicted, both teams combined) by TEMPERATURE ===")
    bucket_report("temp_f", [-30, 20, 32, 45, 60, 75, 120], ["<20F", "20-32F", "32-45F", "45-60F", "60-75F", "75F+"])

    print("=== by WIND (mph) ===")
    bucket_report("wind_mph", [0, 5, 10, 15, 20, 25, 100], ["0-5", "5-10", "10-15", "15-20", "20-25", "25+"])

    print("=== by PRECIPITATION (mm, whole day) ===")
    bucket_report("precip_mm", [-0.01, 0.1, 2, 10, 500], ["none", "trace-2mm", "2-10mm", "10mm+"])

    print("=== SNOW days only (snowfall_cm > 0) vs no-snow ===")
    df["had_snow"] = df["snowfall_cm"] > 0
    g = df.groupby("had_snow").agg(n=("total_residual", "size"), avg_total_residual=("total_residual", "mean"))
    print(g.to_string())

    print("\n=== correlation of total_residual with each raw weather variable ===")
    for col in ("temp_f", "wind_mph", "precip_mm", "snowfall_cm"):
        print(f"  {col:12s} r = {df['total_residual'].corr(df[col]):+.4f}")


if __name__ == "__main__":
    main()
