"""Per-team rolling ratings across 8 stats: rush/pass offense EPA, rush/pass defense EPA
allowed, points scored, points allowed, turnovers committed, turnovers forced. EPA and
turnovers exclude garbage time (win probability outside [0.05, 0.95] dropped); points are
final-score based (not garbage-time filtered -- a final score IS the whole-game outcome, so
filtering it would fight the stat's own meaning).

Ported from research/*.py in this repo, where the methodology was validated: walk-forward
backtesting (2007-2025) showed none of these, alone or combined, beats the closing market
spread at predicting winners (see research/output/ and the AUC comparisons in
research/edge_signal_test*.py). They're stored as descriptive context for pool decisions and
as inputs to the power ranking below, not a betting model. NOTE: that backtest validated the
OLD season-boundary mechanism (every research/edge_signal_test*.py script still carries its
own CARRYOVER=0.65 regression-to-league-mean, unchanged) -- compute_ratings() below now uses
a different, more aggressive forgetting rule (see SEASON_PRIOR_GAMES) that hasn't itself been
re-backtested. The "doesn't beat the market" finding is unlikely to flip (if anything a
faster-forgetting rating is more responsive, not less), but it's not literally re-validated.

Strength of schedule (`sos` in compute_ratings' output): an EWMA-weighted average of each
opponent's OWN composite power score at the time that specific game was played, fading out
the SAME way every other rating here does at a season boundary -- last season's SOS is this
season's starting prior, gone entirely (not just diluted) by SEASON_PRIOR_GAMES games into
the new season. Purely descriptive, not folded into the composite score itself.

Power ranking: a single composite score per team, `POWER_WEIGHTS[m] * ORIENTATION[m] *
z-score(team, m)` summed across all 8 stats. Weights come from ONE joint model across all 8
stats at once (not 8 independent single-stat fits, which is what an earlier version did) --
specifically an L2-regularized logistic regression against historical home_win, with
coefficients constrained >= 0 after each stat is oriented so higher-is-always-better
(ORIENTATION flips the sign of allowed/committed stats first). The non-negativity constraint
is the fix for a real failure mode: an unconstrained joint fit sign-flipped weak/collinear
stats (turnovers committed came out positively weighted) because points, EPA, and turnovers
overlap heavily -- non-negative coefficients can only shrink a redundant stat toward zero,
never flip its sign. Regularization strength (L2=0.3) was chosen by walk-forward validation
(research/edge_signal_test_v8_power_weights.py, 2007-2025, out-of-sample AUC), not in-sample
fit quality. Notably, turnovers_def_forced optimizes to exactly 0.0 -- once the other 7 stats
are known, takeaways forced adds no information, it's not just weak on its own. Computed once
offline, not refit daily -- see POWER_WEIGHTS/ORIENTATION below.

Display: every UI surface shows `display_ratings()`/the `*_display` fields rather than raw
signed EPA. The 4 EPA metrics (and the composite power score) are rescaled to a fixed 0-100
scale anchored to the best/worst value EVER recorded across the full 2007-present dataset --
not this season's 32 teams -- specifically so a mediocre season's "best" team doesn't get
displayed as a misleadingly inflated 100 (see compute_ratings' historical_bounds, computed
once from a full point-in-time replay of the whole dataset). Points/turnovers are left as
their raw per-game average, not rescaled at all -- they're already a directly countable,
intuitive unit, so normalizing them would only obscure them. All of this is cosmetic: the
model itself still fits on the raw/z-scored values above, this never feeds back into any
calculation.

Completed seasons' play-by-play is pre-aggregated once and committed as small parquet files
in nflhub/data/team_ratings/ (~390KB total for 2007-2025) so this doesn't re-download and
re-crunch ~350MB of historical play-by-play every run. Only the CURRENT season is fetched
fresh from nflverse each time this rebuilds, since it's still accumulating games. Rebuilt at
most once/day (see `refresh`), same cadence as sources/history.py's hist_distribution.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
from collections import defaultdict
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import requests

from .edge_teams import UnknownTeamError, normalize_team

log = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "team_ratings")
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
TIMEOUT = 30
FIRST_SEASON = 2007
EWMA_ALPHA = 0.2  # ~4-game half-life; reused convention, not independently tuned
# A new season's rating starts as last season's own ending rating (NOT a multi-generation
# blend that still carries a sliver of every season back to 2007 forever, which is what the
# old league-mean-regression carryover did -- a decade-old season should have ZERO path to
# this year's rating once last season is itself fully behind us, not just an ever-shrinking
# nonzero weight). That prior fades out linearly as this season's own games accumulate,
# reaching exactly zero weight once SEASON_PRIOR_GAMES have been played -- "the sample size
# is too small to trust yet" is a real, finite problem (solved by ~5 games), not a reason to
# keep leaning on history forever.
SEASON_PRIOR_GAMES = 5
WP_LO, WP_HI = 0.05, 0.95  # garbage-time filter: drop plays outside this win-probability band
RATING_METRICS = (
    "rush_off_epa", "pass_off_epa", "rush_def_epa_allowed", "pass_def_epa_allowed",
    "points_off", "points_def_allowed", "turnovers_off", "turnovers_def_forced",
)
# EPA metrics get a 0-100 display scale anchored to the best/worst ever recorded across the
# whole dataset (see compute_ratings' historical_bounds). Points/turnovers are "definitively
# countable" -- displayed as their own raw per-game average, no rescaling at all.
EPA_DISPLAY_METRICS = ("rush_off_epa", "pass_off_epa", "rush_def_epa_allowed", "pass_def_epa_allowed")
COUNTABLE_METRICS = ("points_off", "points_def_allowed", "turnovers_off", "turnovers_def_forced")
# How many teams need a complete rating before a point-in-time snapshot counts toward the
# historical composite-score bounds -- high enough to skip the first week or two of a new
# season (when only a handful of teams have played their first game and z-scores against a
# tiny pool are noisy), without requiring literally every team (bye weeks, mid-week ties).
MIN_TEAMS_FOR_HISTORICAL_SNAPSHOT = 28
PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "play_type",
            "epa", "wp", "interception", "fumble_lost"]
MISMATCH_Z_THRESHOLD = 1.5  # combined (offense z + opposing defense-allowed z) needed to flag a callout

# +1 if higher is already better, -1 if the raw stat needs flipping to be "higher=better"
# before weighting (allowed/committed stats).
ORIENTATION = {
    "rush_off_epa": 1, "pass_off_epa": 1, "rush_def_epa_allowed": -1, "pass_def_epa_allowed": -1,
    "points_off": 1, "points_def_allowed": -1, "turnovers_off": -1, "turnovers_def_forced": 1,
}

# Non-negative joint logistic-regression weights vs. historical home_win, L2=0.3, walk-forward
# validated (research/edge_signal_test_v13_new_window_weights.py, 4927 games, 2007-2025).
# Re-fit here against the CURRENT season_prior-fade rating windowing (see SEASON_PRIOR_GAMES)
# -- the original v8 fit (4436 games) used the old forever-decaying carryover mechanism,
# a different predictor, so its weights weren't technically valid for this one. The refit
# landed very close to the old values (same relative ranking, turnovers_def_forced again
# optimizing to exactly 0.0) and the same best L2=0.3, so the windowing change didn't
# meaningfully change which stats matter -- reassuring, not a coincidence worth skipping.
# Walk-forward AUC: composite alone 0.6746, market spread alone 0.7224, both together 0.7226
# -- same conclusion as before, just confirmed under the new windowing: doesn't beat the
# market, and adding it to the market doesn't move the needle. Apply as
# POWER_WEIGHTS[m] * ORIENTATION[m] * z-score -- these are all >= 0 by construction (see
# module docstring). Recompute that script and update these if you want to refresh the fit;
# not refit automatically.
POWER_WEIGHTS = {
    "rush_off_epa": 0.0850,
    "pass_off_epa": 0.1536,
    "rush_def_epa_allowed": 0.0572,
    "pass_def_epa_allowed": 0.0578,
    "points_off": 0.1587,
    "points_def_allowed": 0.1006,
    "turnovers_off": 0.0217,
    "turnovers_def_forced": 0.0000,
}

# Points prediction: predicted_points(team) = POINTS_INTERCEPT + sum(POINTS_WEIGHTS[k] * value),
# where `value` is the TEAM's own rating for an "own_*" key and the OPPONENT's rating for an
# "opp_*" key (own offense vs opponent defense). Non-negative L2-regularized (L2=0.01) linear
# regression on actual points scored, walk-forward validated (research/edge_signal_test_v9_*.py,
# 8872 team-game rows, 2007-2025): MAE 7.62 points/team-game (vs. 8.02 for a naive
# league-average guess), but the derived spread (home pred - away pred) is LESS accurate than
# the market's own spread at picking winners (0.686 AUC vs 0.726) -- same pattern as everything
# else in this project, shown as a second data point alongside the market/ELWAY lines, not a
# replacement. turnovers zeroed out here too (same redundancy finding as the power ranking, in
# a completely different model). Computed once offline, not refit daily.
POINTS_INTERCEPT = 7.21333
POINTS_WEIGHTS = {
    "own_rush_off_epa": 2.472559,
    "own_pass_off_epa": 6.259427,
    "own_points_off": 0.371737,
    "own_turnovers_off": -0.000000,
    "opp_rush_def_epa_allowed": 2.470530,
    "opp_pass_def_epa_allowed": 0.666745,
    "opp_points_def_allowed": 0.317523,
    "opp_turnovers_def_forced": -0.000000,
}


def predict_points(own: dict[str, float], opp: dict[str, float], weather: dict[str, float] | None = None) -> float | None:
    """Predicted points for a team with rating dict `own`, facing a team with rating dict
    `opp`. Returns None if either side is missing a needed rating. `weather`, if given, adds
    the WEATHER_ADJUSTMENT correction (see below) on top -- omit it (or pass None) for a dome
    game or when no forecast is available; the base prediction is unbiased either way since it
    was fit across all games, indoor and outdoor."""
    total = POINTS_INTERCEPT
    for key, w in POINTS_WEIGHTS.items():
        side, metric = key.split("_", 1)
        val = (own if side == "own" else opp).get(metric)
        if val is None:
            return None
        total += w * val
    if weather is not None:
        total += WEATHER_ADJUSTMENT_INTERCEPT
        for key, w in WEATHER_ADJUSTMENT_WEIGHTS.items():
            val = weather.get(key)
            if val is not None:
                total += w * val
    return round(total, 2)


# Stadium coordinates for teams that are UNAMBIGUOUSLY, permanently outdoor as of the current
# roof (verified against 2022+ games.csv roof values -- each of these showed ONLY 'outdoors',
# never 'dome'/'closed'/'open'). Deliberately excludes: fixed domes (DET/LV/LAR/LAC/MIN/NO --
# current venue), and retractable-roof teams (ARI/ATL/DAL/HOU/IND) whose roof is closed most
# of the time and can't be known in advance for a future game -- auto-applying an outdoor
# weather adjustment to those would be wrong more often than right. BUF shows one historical
# 'dome' entry (the 2014 Bills-Jets game relocated to Ford Field for a blizzard) -- a one-off
# anomaly, not current reality, so it's kept. research/weather_scoring_analysis.py has the
# broader historical coordinate set used for backtesting (includes eras like LA's 2016-2019
# outdoor Coliseum stint), which is intentionally wider than this live-application set.
STADIUM_COORDS = {
    "BAL": (39.2780, -76.6227), "BUF": (42.7738, -78.7870), "CAR": (35.2258, -80.8528),
    "CHI": (41.8623, -87.6167), "CIN": (39.0954, -84.5160), "CLE": (41.5061, -81.6995),
    "DEN": (39.7439, -105.0201), "GB": (44.5013, -88.0622), "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839), "MIA": (25.9580, -80.2389), "NE": (42.0909, -71.2643),
    "NYG": (40.8135, -74.0745), "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675),
    "PIT": (40.4468, -80.0158), "SEA": (47.5952, -122.3316), "SF": (37.4032, -121.9698),
    "TB": (27.9759, -82.5033), "TEN": (36.1665, -86.7713), "WSH": (38.9076, -76.8645),
}

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Weather ADJUSTMENT on top of predict_points()'s base prediction -- fit as a separate
# residual regression (actual_points - base_prediction) ~ wind/precip/cold, NOT refit jointly
# with the 8 team-rating weights above, specifically so "no weather data" cleanly means
# "adjustment = 0" with zero risk of implicitly assuming calm/dry conditions for a game we
# simply don't have a forecast for yet. Non-negative L2-regularized (L2=0.01), walk-forward
# validated (research/edge_signal_test_v10_weather.py, 5926 team-game rows, 2007-2025):
# reduces points MAE 7.647 -> 7.603 out-of-sample -- a real but modest improvement (~0.6%).
# cold_flag = 1.0 if forecast high temp < 20F else 0.0. Computed once offline, not refit daily.
WEATHER_ADJUSTMENT_INTERCEPT = 1.6502
WEATHER_ADJUSTMENT_WEIGHTS = {
    "wind_mph": -0.165478,
    "precip_mm": -0.051583,
    "cold_flag": -3.021398,
}


# WMO weather codes (Open-Meteo's `weathercode` daily value) -> a short display string.
# Not exhaustive -- just enough to label the common cases; an unmapped code is simply omitted.
_WEATHER_CODE_DESC = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "light freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "light freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "heavy showers",
    85: "light snow showers", 86: "snow showers",
    95: "thunderstorm", 96: "thunderstorm w/ hail", 99: "severe thunderstorm w/ hail",
}


def fetch_forecast_weather(team: str, date_str: str) -> dict[str, float] | None:
    """Live forecast for `team`'s stadium on `date_str` (YYYY-MM-DD), or None if the team has
    no outdoor stadium, the date is outside the forecast's reliable range (~16 days), or the
    request fails for any reason (soft-fail, matching every other best-effort external source
    in this project). `wind_mph`/`precip_mm`/`cold_flag` are the exact inputs predict_points'
    weather adjustment is fit on -- don't change their units/meaning. Everything else
    (temp_hi_f/temp_lo_f/snow_in/precip_chance/conditions) is display-only, for the full
    forecast shown in the matchup card's details -- predict_points never reads them."""
    coords = STADIUM_COORDS.get(team)
    if not coords:
        return None
    try:
        resp = requests.get(FORECAST_URL, params={
            "latitude": coords[0], "longitude": coords[1],
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum,"
                     "precipitation_probability_max,windspeed_10m_max,weathercode",
            "timezone": "America/New_York", "forecast_days": 16,
        }, timeout=TIMEOUT)
        resp.raise_for_status()
        daily = resp.json()["daily"]
        if date_str not in daily["time"]:
            return None
        i = daily["time"].index(date_str)
        temp_hi_f = daily["temperature_2m_max"][i] * 9 / 5 + 32
        temp_lo_f = daily["temperature_2m_min"][i] * 9 / 5 + 32
        code = (daily.get("weathercode") or [None] * (i + 1))[i]
        return {
            "wind_mph": daily["windspeed_10m_max"][i] * 0.621371,
            "precip_mm": daily["precipitation_sum"][i],
            "cold_flag": 1.0 if temp_hi_f < 20 else 0.0,
            "temp_hi_f": round(temp_hi_f, 1),
            "temp_lo_f": round(temp_lo_f, 1),
            "snow_in": round((daily.get("snowfall_sum") or [0] * (i + 1))[i] * 0.393701, 2),
            "precip_chance": (daily.get("precipitation_probability_max") or [None] * (i + 1))[i],
            "conditions": _WEATHER_CODE_DESC.get(code),
        }
    except Exception as exc:  # noqa: BLE001 - best-effort; never block the rest of the refresh
        log.warning("forecast fetch failed for %s on %s: %s", team, date_str, exc)
        return None


def _phase_stats_for_season(season: int, current_season: int) -> pd.DataFrame:
    """One row per (game_id, team): garbage-time-filtered rush/pass offensive EPA/play plus
    turnovers committed (same filtered play population, for consistency)."""
    if season < current_season:
        path = os.path.join(DATA_DIR, f"team_game_phase_{season}.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)

    df = pd.read_csv(PBP_URL.format(season=season), compression="gzip", usecols=PBP_COLS, low_memory=False)
    df = df[(df["season_type"] == "REG") & (df["play_type"].isin(["run", "pass"]))]
    df = df.dropna(subset=["wp", "epa", "posteam"])
    df = df[(df["wp"] >= WP_LO) & (df["wp"] <= WP_HI)]
    df["turnover"] = df["interception"].fillna(0) + df["fumble_lost"].fillna(0)

    off = df.groupby(["game_id", "posteam", "play_type"])["epa"].agg(["size", "sum"]).reset_index()
    off = off.pivot_table(index=["game_id", "posteam"], columns="play_type", values=["size", "sum"])
    off.columns = [f"{a}_{b}" for a, b in off.columns]
    off = off.reset_index().rename(columns={"posteam": "team"})
    for col in ("size_run", "size_pass", "sum_run", "sum_pass"):
        if col not in off.columns:
            off[col] = 0.0
    off["rush_off_epa"] = off["sum_run"] / off["size_run"].replace(0, np.nan)
    off["pass_off_epa"] = off["sum_pass"] / off["size_pass"].replace(0, np.nan)

    tov = df.groupby(["game_id", "posteam"])["turnover"].sum().reset_index() \
        .rename(columns={"posteam": "team", "turnover": "turnovers_committed"})
    out = off[["game_id", "team", "rush_off_epa", "pass_off_epa"]].merge(tov, on=["game_id", "team"], how="left")
    out["turnovers_committed"] = out["turnovers_committed"].fillna(0.0)
    return out.dropna(subset=["rush_off_epa", "pass_off_epa"])


def compute_ratings(
    played_games: list[dict], current_season: int, game_log: list[dict] | None = None
) -> tuple[dict[str, dict[str, float]], dict[str, tuple[float, float]]]:
    """`played_games`: nflverse games.csv REG rows with a result, any seasons needed.
    Returns ({team: {metric: rating}} as of the most recent game passed in, historical_bounds)
    where historical_bounds is {metric: (min, max)} (oriented, higher=better) for the 4 EPA
    metrics plus "composite" for the power-ranking score -- each the most extreme value ever
    observed at any point in the replayed history, used to anchor the fixed 0-100 display
    scale (see module docstring) instead of a per-season min/max that would make a mediocre
    season's best team look inflated.

    If `game_log` is given, every game with a complete pre-game rating for both teams gets
    appended to it as {game_id, season, week, gameday, home, away, home_score, away_score,
    spread_line, diffs} -- `diffs[m]` is the SAME oriented home-minus-away rating diff
    build_dataset() in research/edge_signal_test_v13_new_window_weights.py computes, using
    the rating as it stood immediately BEFORE this game (no lookahead). This is the single
    source of truth for that per-game rating walk -- the Historical Power tab's backtest
    scatter (compute_backtest_scatter() below) reuses this instead of its own copy, so it
    can never drift from what the live rating actually did."""
    # Both sides of this join need normalize_team(): the play-by-play files (the source of
    # `teams` below) retroactively relabel relocated franchises as LA/LAC/LV for every
    # season back to 2007, while games.csv's home/away columns correctly use the
    # contemporary code (STL/SD/OAK) for those years -- joining on the raw codes silently
    # dropped every St. Louis/San Diego/Oakland-era game entirely, for that team AND
    # whoever they played that week (same bug found and fixed in
    # compute_historical_season_averages(); see that function's comment for the full
    # verification against the committed parquet caches).
    by_game: dict[str, dict[str, dict]] = defaultdict(dict)
    seasons_needed = sorted({int(g["season"]) for g in played_games})
    for season in seasons_needed:
        for row in _phase_stats_for_season(season, current_season).to_dict("records"):
            try:
                team_key = normalize_team(row["team"])
            except UnknownTeamError:
                continue
            by_game[row["game_id"]][team_key] = row

    games_by_id = {g["game_id"]: g for g in played_games}

    # `rating`/`sos` are the BLENDED values everything else reads (bounds tracking, SOS
    # opponent lookups, the final return value) -- a mix of last season's own ending value
    # (season_prior/sos_prior, frozen at the moment a new season starts) and this season's
    # own EWMA so far (season_ewma/sos_ewma, reset to None at every season boundary, so it
    # NEVER carries anything from two or more seasons back). See SEASON_PRIOR_GAMES above
    # for the fade-out schedule.
    rating: dict[str, dict] = defaultdict(lambda: {m: None for m in RATING_METRICS})
    season_ewma: dict[str, dict] = defaultdict(lambda: {m: None for m in RATING_METRICS})
    season_prior: dict[str, dict] = defaultdict(lambda: {m: None for m in RATING_METRICS})
    sos: dict[str, float | None] = defaultdict(lambda: None)
    sos_ewma: dict[str, float | None] = defaultdict(lambda: None)
    sos_prior: dict[str, float | None] = defaultdict(lambda: None)
    games_played: dict[str, int] = defaultdict(int)
    last_season: dict[str, int] = {}
    epa_bounds = {m: [float("inf"), float("-inf")] for m in EPA_DISPLAY_METRICS}
    composite_bounds = [float("inf"), float("-inf")]
    sos_bounds = [float("inf"), float("-inf")]

    for gid, teams in sorted(by_game.items()):
        g = games_by_id.get(gid)
        if not g or len(teams) != 2:
            continue
        season = int(g["season"])
        try:
            home, away = normalize_team(g["home_team"]), normalize_team(g["away_team"])
        except UnknownTeamError:
            continue
        if home not in teams or away not in teams:
            continue
        try:
            home_pts, away_pts = float(g["home_score"]), float(g["away_score"])
        except (ValueError, TypeError):
            continue

        home_raw, away_raw = dict(teams[home]), dict(teams[away])
        home_raw["rush_def_epa_allowed"] = away_raw["rush_off_epa"]
        away_raw["rush_def_epa_allowed"] = home_raw["rush_off_epa"]
        home_raw["pass_def_epa_allowed"] = away_raw["pass_off_epa"]
        away_raw["pass_def_epa_allowed"] = home_raw["pass_off_epa"]
        home_raw["points_off"], away_raw["points_off"] = home_pts, away_pts
        home_raw["points_def_allowed"], away_raw["points_def_allowed"] = away_pts, home_pts
        home_raw["turnovers_off"] = home_raw.pop("turnovers_committed")
        away_raw["turnovers_off"] = away_raw.pop("turnovers_committed")
        home_raw["turnovers_def_forced"] = away_raw["turnovers_off"]
        away_raw["turnovers_def_forced"] = home_raw["turnovers_off"]

        # Season boundary: freeze whatever season_ewma/sos_ewma ended the PREVIOUS season at
        # as this new season's starting prior, then reset them to None -- a team's rating
        # two or more seasons back has no path into season_prior at all, by construction.
        # prior_w is computed here (not per-metric below) so the SOS blend and the rating
        # blend for this exact game use the identical fade schedule.
        prior_w: dict[str, float] = {}
        for team in (home, away):
            if last_season.get(team) is not None and last_season[team] != season:
                season_prior[team] = dict(season_ewma[team])
                season_ewma[team] = {m: None for m in RATING_METRICS}
                sos_prior[team] = sos_ewma[team]
                sos_ewma[team] = None
                games_played[team] = 0
            last_season[team] = season
            games_played[team] += 1
            prior_w[team] = max(0.0, (SEASON_PRIOR_GAMES - games_played[team]) / SEASON_PRIOR_GAMES)

        # Cross-sectional composite-score snapshot of the league as it stood right BEFORE this
        # game (i.e. not yet touched by either team's result today) -- used to (a) feed each
        # team's strength-of-schedule average with its opponent's strength AT THE TIME they
        # played, with no lookahead, and (b) track the most extreme composite score ever
        # observed, for the fixed historical display scale (see module docstring).
        complete = {t: r for t, r in rating.items() if all(r[m] is not None for m in RATING_METRICS)}
        pre_home_score = pre_away_score = None
        if len(complete) >= MIN_TEAMS_FOR_HISTORICAL_SNAPSHOT:
            z_snap: dict[str, dict[str, float]] = defaultdict(dict)
            for m in RATING_METRICS:
                zm, _ = _rank_and_z({t: r[m] for t, r in complete.items()})
                for t, v in zm.items():
                    z_snap[t][m] = v
            composite_snap = {t: sum(POWER_WEIGHTS[m] * ORIENTATION[m] * zt.get(m, 0.0) for m in RATING_METRICS)
                               for t, zt in z_snap.items()}
            for score in composite_snap.values():
                if score < composite_bounds[0]: composite_bounds[0] = score
                if score > composite_bounds[1]: composite_bounds[1] = score
            pre_home_score, pre_away_score = composite_snap.get(home), composite_snap.get(away)

        if pre_away_score is not None:
            prev = sos_ewma[home]
            sos_ewma[home] = pre_away_score if prev is None else (1 - EWMA_ALPHA) * prev + EWMA_ALPHA * pre_away_score
        if pre_home_score is not None:
            prev = sos_ewma[away]
            sos_ewma[away] = pre_home_score if prev is None else (1 - EWMA_ALPHA) * prev + EWMA_ALPHA * pre_home_score
        for team in (home, away):
            pv, ev = sos_prior[team], sos_ewma[team]
            sos[team] = (prior_w[team] * pv + (1 - prior_w[team]) * ev) if (pv is not None and ev is not None) else ev
            if sos[team] is not None:
                if sos[team] < sos_bounds[0]: sos_bounds[0] = sos[team]
                if sos[team] > sos_bounds[1]: sos_bounds[1] = sos[team]

        if game_log is not None and all(rating[home][m] is not None and rating[away][m] is not None for m in RATING_METRICS):
            try:
                spread_line = float(g["spread_line"])
            except (ValueError, TypeError):
                spread_line = None
            if spread_line is not None:
                game_log.append({
                    "game_id": gid, "season": season, "week": int(g.get("week") or 0),
                    "gameday": g.get("gameday", ""), "home": home, "away": away,
                    "home_score": int(home_pts), "away_score": int(away_pts),
                    "spread_line": spread_line,
                    "diffs": {m: ORIENTATION[m] * (rating[home][m] - rating[away][m]) for m in RATING_METRICS},
                })

        for team, raw in ((home, home_raw), (away, away_raw)):
            for m in RATING_METRICS:
                prev = season_ewma[team][m]
                season_ewma[team][m] = raw[m] if prev is None else (1 - EWMA_ALPHA) * prev + EWMA_ALPHA * raw[m]
                prior_val = season_prior[team].get(m)
                rating[team][m] = (
                    prior_w[team] * prior_val + (1 - prior_w[team]) * season_ewma[team][m]
                    if prior_val is not None else season_ewma[team][m]
                )
                if m in epa_bounds:
                    oriented = ORIENTATION[m] * rating[team][m]
                    b = epa_bounds[m]
                    if oriented < b[0]: b[0] = oriented
                    if oriented > b[1]: b[1] = oriented

    ratings_out = {
        team: {
            **{m: (round(v, 4) if v is not None else None) for m, v in r.items()},
            "sos": round(sos[team], 4) if sos[team] is not None else None,
        }
        for team, r in rating.items()
    }
    historical_bounds = {m: tuple(b) for m, b in epa_bounds.items()}
    historical_bounds["composite"] = tuple(composite_bounds)
    historical_bounds["sos"] = tuple(sos_bounds)
    return ratings_out, historical_bounds


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _rank_and_z(values: dict[str, float]) -> tuple[dict[str, float], dict[str, int]]:
    """rank 1 = highest raw value (for an offense metric, best; for a defense-allowed
    metric, worst -- callers phrase the sentence accordingly, this just ranks numerically)."""
    arr = np.array(list(values.values()))
    mu, sd = arr.mean(), (arr.std() or 1.0)
    z = {t: (v - mu) / sd for t, v in values.items()}
    order = sorted(values, key=lambda t: -values[t])
    rank = {t: i + 1 for i, t in enumerate(order)}
    return z, rank


def _normalize_fixed(value: float, lo: float, hi: float) -> float:
    """Scale an already-oriented (higher=better) value to [0, 100] against FIXED bounds --
    not the current team pool's own min/max, so a mediocre season's best team doesn't show up
    as a misleadingly inflated 100. Clipped defensively, though in practice `value` is always
    one of the snapshots `lo`/`hi` were themselves computed from (see compute_ratings), so it
    should never actually fall outside [lo, hi]."""
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi == lo:
        return 50.0  # no (or degenerate) historical range yet -- e.g. too little data to ever hit MIN_TEAMS_FOR_HISTORICAL_SNAPSHOT
    return round(max(0.0, min(100.0, 100 * (value - lo) / (hi - lo))), 1)


def display_ratings(ratings: dict[str, dict[str, float]], historical_bounds: dict[str, tuple[float, float]]) -> dict[str, dict[str, float]]:
    """Display-ready version of `ratings`: the 4 EPA metrics rescaled to a fixed 0-100 scale
    anchored to the best/worst ever recorded across the whole dataset (`historical_bounds`,
    from compute_ratings); points/turnovers left as their own raw per-game average, since
    they're already a directly countable, intuitive unit that normalizing would only obscure."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for t, r in ratings.items():
        for m in RATING_METRICS:
            v = r.get(m)
            if v is None:
                continue
            if m in EPA_DISPLAY_METRICS:
                lo, hi = historical_bounds[m]
                out[t][m] = _normalize_fixed(ORIENTATION[m] * v, lo, hi)
            else:
                out[t][m] = round(v, 2)
    return out


def matchup_callouts(ratings: dict[str, dict[str, float]], upcoming: list[dict], historical_bounds: dict[str, tuple[float, float]]) -> dict[str, dict]:
    """For each upcoming game (already normalized home/away team codes), rule-based sentences
    for any offense-vs-opposing-defense pairing whose combined z-score clears
    MISMATCH_Z_THRESHOLD -- a real strength meeting a real weakness, not just noise. Keyed
    "AWAY@HOME" so the frontend can look it up directly from its own game row. (The Parlays
    tab's own matchup identifier no longer reads anything from here -- it classifies teams
    into tiers off a frozen ratings_display snapshot instead; see refresh()'s snapshot step.)"""
    n_teams = len(ratings)
    if n_teams < 4:
        return {}
    z: dict[str, dict[str, float]] = {}
    rank: dict[str, dict[str, int]] = {}
    for m in RATING_METRICS:
        vals = {t: r[m] for t, r in ratings.items() if r.get(m) is not None}
        zm, rankm = _rank_and_z(vals)
        for t in vals:
            z.setdefault(t, {})[m] = zm[t]
            rank.setdefault(t, {})[m] = rankm[t]
    ratings_display = display_ratings(ratings, historical_bounds)

    # Rank by the already-ORIENTED display value (1 = best), not the raw _rank_and_z rank
    # above (which ranks offense and defense-allowed in opposite directions) -- this way a
    # callout's "(Nth)" always means the same thing, no "worst"/"fewest allowed" branching.
    display_rank: dict[str, dict[str, int]] = {}
    for m in RATING_METRICS:
        vals = {t: r[m] for t, r in ratings_display.items() if m in r}
        for i, t in enumerate(sorted(vals, key=lambda t: -vals[t])):
            display_rank.setdefault(t, {})[m] = i + 1

    out: dict[str, dict] = {}
    for g in upcoming:
        home, away = g.get("home_team"), g.get("away_team")
        if home not in ratings or away not in ratings:
            continue
        callouts = []
        for phase in ("rush", "pass"):
            off_m, def_m = f"{phase}_off_epa", f"{phase}_def_epa_allowed"
            for off_team, def_team in ((home, away), (away, home)):
                if off_m not in z.get(off_team, {}) or def_m not in z.get(def_team, {}):
                    continue
                combined = z[off_team][off_m] + z[def_team][def_m]
                if abs(combined) < MISMATCH_Z_THRESHOLD:
                    continue
                off_rank, def_rank = display_rank[off_team][off_m], display_rank[def_team][def_m]
                favors_offense = combined > 0
                verdict = "a lopsided matchup on paper" if favors_offense else "a tough matchup on paper"
                callouts.append(
                    f"{off_team}'s {phase} offense ({_ordinal(off_rank)}) "
                    f"{'faces' if favors_offense else 'runs into'} {def_team}'s {phase} defense "
                    f"({_ordinal(def_rank)}) -- {verdict}."
                )
        weather = None
        gameday = g.get("gameday")
        # skip the forecast fetch entirely for games outside its ~16-day reliable window --
        # `upcoming` covers the rest of the season, and most of those calls would just come
        # back empty; no point making 100+ live HTTP requests every refresh for that.
        if gameday and home in STADIUM_COORDS:
            try:
                days_out = (date.fromisoformat(gameday) - date.today()).days
            except ValueError:
                days_out = None
            if days_out is not None and 0 <= days_out <= 15:
                weather = fetch_forecast_weather(home, gameday)
        predicted_home_points = predict_points(ratings[home], ratings[away], weather)
        predicted_away_points = predict_points(ratings[away], ratings[home], weather)
        # How many of those predicted points are the weather adjustment itself, not the base
        # model -- the difference against the same prediction with weather=None. Lets a caller
        # flag "weather is worth N points here" without re-implementing predict_points' formula.
        weather_points_delta = None
        if weather is not None:
            base_home = predict_points(ratings[home], ratings[away], None)
            base_away = predict_points(ratings[away], ratings[home], None)
            if None not in (base_home, base_away, predicted_home_points, predicted_away_points):
                weather_points_delta = round(
                    (predicted_home_points - base_home) + (predicted_away_points - base_away), 2
                )
        out[f"{away}@{home}"] = {
            "home": home, "away": away,
            "home_ratings": ratings[home], "away_ratings": ratings[away],
            "home_ratings_display": ratings_display.get(home), "away_ratings_display": ratings_display.get(away),
            "callouts": callouts,
            "predicted_home_points": predicted_home_points,
            "predicted_away_points": predicted_away_points,
            "weather": weather,
            "weather_points_delta": weather_points_delta,
        }
    return out


def power_rankings(ratings: dict[str, dict[str, float]], historical_bounds: dict[str, tuple[float, float]]) -> dict[str, dict]:
    """Composite score per team: sum of POWER_WEIGHTS[m] * ORIENTATION[m] * z-score(team, m)
    across all 8 stats, ranked 1 = best. Each z-score is cross-sectional (against the other 31
    current teams), so the underlying fit still reflects current relative standing, not an
    absolute scale -- only the DISPLAY representation below is anchored to history instead.

    `score`/`ratings` keep the raw signed values the model is actually fit on (z-score-weighted
    sum, raw EPA/points/turnovers); `score_display`/`ratings_display` are the display-friendly
    version: the composite score and the 4 EPA stats rescaled to [0, 100] against the most
    extreme value ever seen across the full dataset (`historical_bounds`), so a mediocre
    season's best team doesn't read as an inflated 100; points/turnovers are left as their own
    raw per-game average (see display_ratings).

    `sos`/`sos_display`: strength of schedule -- an EWMA-weighted average of opponents' own
    composite scores AT THE TIME each game was played (so a team that's since gotten better or
    worse doesn't retroactively change how tough it was to play them back then), fading the
    same way every other rating here does at a season boundary (see SEASON_PRIOR_GAMES),
    computed once in compute_ratings' single historical replay and just passed through on
    `ratings[t]["sos"]`. Descriptive only -- not folded into the composite score weighting
    above."""
    if len(ratings) < 4:
        return {}
    z: dict[str, dict[str, float]] = defaultdict(dict)
    for m in RATING_METRICS:
        vals = {t: r[m] for t, r in ratings.items() if r.get(m) is not None}
        zm, _ = _rank_and_z(vals)
        for t, v in zm.items():
            z[t][m] = v

    scores = {t: sum(POWER_WEIGHTS[m] * ORIENTATION[m] * zt.get(m, 0.0) for m in RATING_METRICS) for t, zt in z.items()}
    composite_lo, composite_hi = historical_bounds["composite"]
    scores_display = {t: _normalize_fixed(s, composite_lo, composite_hi) for t, s in scores.items()}
    ratings_display = display_ratings(ratings, historical_bounds)
    sos_lo, sos_hi = historical_bounds["sos"]
    order = sorted(scores, key=lambda t: -scores[t])
    return {
        t: {
            "rank": i + 1, "score": round(scores[t], 4), "score_display": scores_display[t],
            "ratings": ratings[t], "ratings_display": ratings_display.get(t, {}),
            "sos": ratings[t].get("sos"),
            "sos_display": _normalize_fixed(ratings[t]["sos"], sos_lo, sos_hi) if ratings[t].get("sos") is not None else None,
        }
        for i, t in enumerate(order)
    }


# nflverse's games.csv uses the REAL contemporary team code for every season it covers --
# STL 2007-2015, SD 2007-2016, OAK 2007-2019, then LA/LAC/LV starting the season each
# franchise actually moved (verified directly against the live file: "STL" never appears
# after 2015, "LA" never appears before 2016, etc.). The play-by-play files are NOT
# consistent with that -- they retroactively relabel every St. Louis/San Diego/Oakland game
# as LA/LAC/LV even back to 2007 (verified directly against the committed 2007 and 2010
# parquet caches: no "STL"/"SD"/"OAK" in either, "LA"/"LAC"/"LV" in both). _phase_stats_for_
# season() comes from the play-by-play files, so games.csv's home/away code has to be
# normalize_team()'d (same canonical mapping, -> LAR/LAC/LV) to even FIND the matching row
# at all -- using the raw games.csv code directly against that dict, like compute_ratings()
# itself still does, silently drops every STL/SD/OAK-era game for both that team AND
# whoever they played that week. Unlike that join key, the team code actually STORED below
# is the games.csv one -- the team fans actually watched that year, not today's rebrand --
# with only two pure-spelling fixes that apply regardless of season: nflverse spells
# Washington "WAS" (this app uses "WSH") and the Rams "LA" from 2016 on (this app uses
# "LAR"); neither is a relocation-era distinction, just a different abbreviation for the
# same current team.
_HISTORICAL_SPELLING_FIX = {"WAS": "WSH", "LA": "LAR"}


def _historical_team_code(raw: str) -> str:
    return _HISTORICAL_SPELLING_FIX.get(raw, raw)


def compute_historical_season_averages(
    played_games: list[dict], current_season: int
) -> tuple[dict[tuple[str, int], dict[str, float]], dict[tuple[str, int], list[str]]]:
    """One row per (team, season) for every FULLY COMPLETED season in `played_games`
    (season < current_season) -- the simple, equally-weighted average of that team's own
    games THAT SEASON for each of the 8 RATING_METRICS, deliberately NOT the EWMA-decayed
    rolling rating compute_ratings() produces for the live dashboard. A live rating is a
    point-in-time snapshot answering "how good is this team right now, weighted toward
    recent form" -- correct for a live pick'em tool, but wrong for "how good was the 2007
    Patriots": that question wants the whole season weighted equally, not decayed toward
    whatever they looked like at the specific week the rating happened to be measured.
    Reuses _phase_stats_for_season()'s own per-season parquet cache (nflhub/data/
    team_ratings/, already committed for every completed season through 2025), so this is
    cheap -- no new downloads for any season that's already in that cache.

    Also returns `opponents`: {(team, season): [opponent_team, ...]} for every game played
    that season -- used by historical_power_rankings() to compute a season's strength of
    schedule as a plain second pass over already-known composite scores (see that
    function's docstring for why that's NOT circular, unlike it first looked)."""
    by_game: dict[str, dict[str, dict]] = defaultdict(dict)
    seasons_needed = sorted({int(g["season"]) for g in played_games if int(g["season"]) < current_season})
    for season in seasons_needed:
        for row in _phase_stats_for_season(season, current_season).to_dict("records"):
            # The play-by-play files' own team code needs the same normalize_team() pass as
            # games.csv's home/away below -- they use "LA" for the Rams even in 2007 (unlike
            # games.csv's "STL"), so without this the join key mismatches for this one
            # franchise even though Chargers/Raiders already match (PBP already spells those
            # "LAC"/"LV" outright, no gap to close there).
            try:
                team_key = normalize_team(row["team"])
            except UnknownTeamError:
                continue
            by_game[row["game_id"]][team_key] = row

    games_by_id = {g["game_id"]: g for g in played_games}
    sums: dict[tuple[str, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[tuple[str, int], int] = defaultdict(int)
    opponents: dict[tuple[str, int], list[str]] = defaultdict(list)

    for gid, teams in by_game.items():
        g = games_by_id.get(gid)
        if not g or len(teams) != 2:
            continue
        season = int(g["season"])
        if season >= current_season:
            continue
        home_raw_code, away_raw_code = g["home_team"], g["away_team"]
        try:
            home, away = normalize_team(home_raw_code), normalize_team(away_raw_code)
        except UnknownTeamError:
            continue
        if home not in teams or away not in teams:
            continue
        try:
            home_pts, away_pts = float(g["home_score"]), float(g["away_score"])
        except (ValueError, TypeError):
            continue

        home_raw, away_raw = dict(teams[home]), dict(teams[away])
        home_raw["rush_def_epa_allowed"] = away_raw["rush_off_epa"]
        away_raw["rush_def_epa_allowed"] = home_raw["rush_off_epa"]
        home_raw["pass_def_epa_allowed"] = away_raw["pass_off_epa"]
        away_raw["pass_def_epa_allowed"] = home_raw["pass_off_epa"]
        home_raw["points_off"], away_raw["points_off"] = home_pts, away_pts
        home_raw["points_def_allowed"], away_raw["points_def_allowed"] = away_pts, home_pts
        home_raw["turnovers_off"] = home_raw.pop("turnovers_committed")
        away_raw["turnovers_off"] = away_raw.pop("turnovers_committed")
        home_raw["turnovers_def_forced"] = away_raw["turnovers_off"]
        away_raw["turnovers_def_forced"] = home_raw["turnovers_off"]

        home_team = _historical_team_code(home_raw_code)
        away_team = _historical_team_code(away_raw_code)
        for team, opp, raw in ((home_team, away_team, home_raw), (away_team, home_team, away_raw)):
            key = (team, season)
            counts[key] += 1
            opponents[key].append(opp)
            for m in RATING_METRICS:
                sums[key][m] += raw[m]

    averages = {
        key: {m: round(sums[key][m] / n, 4) for m in RATING_METRICS}
        for key, n in counts.items()
    }
    return averages, opponents


def historical_power_rankings(
    team_season_avgs: dict[tuple[str, int], dict[str, float]],
    opponents: dict[tuple[str, int], list[str]] | None = None,
) -> list[dict]:
    """Composite Power Score for every (team, season) in `team_season_avgs`, z-scored and
    ranked against the WHOLE pooled historical dataset at once -- every team-season from
    every completed year, not cross-sectionally within just that one season's 32 teams. That
    pooling is the entire point: comparing "2023 KC's offense" against "2007 NE's offense"
    directly needs one shared scale across eras, not 19 separate single-season scales that
    can't be compared to each other. Same POWER_WEIGHTS/ORIENTATION as the live power
    ranking (power_rankings() above); the 4 EPA columns get a fresh 0-100 scale anchored to
    THIS pooled dataset's own extremes (the best/worst full-season average ever recorded),
    analogous to team_ratings.py's module-level historical_bounds but computed from season
    averages, not point-in-time EWMA snapshots.

    SOS (if `opponents` is given): a plain second pass, NOT an iterative/simultaneous solve
    -- composite scores are computed first from the 8 base stats alone (no SOS input), so by
    the time SOS is computed every opponent's score is already a known, fixed number; there
    is no circularity to resolve. That's genuinely different from the live dashboard's own
    SOS (EWMA-weighted, "AT THE TIME each game was played"), which has to special-case
    causality because a team's ROLLING rating keeps changing week to week -- using an
    opponent's end-of-season rating for a game played in week 3 would be a look-ahead. A
    full-season average has no such evolving state to protect against: the whole point of
    this dataset is the retrospective, fully-known season, so using opponents' own (equally
    retrospective, equally fully-known) season scores is exactly the right comparison, not a
    shortcut around a harder problem."""
    if len(team_season_avgs) < 4:
        return []
    z: dict[tuple, dict[str, float]] = defaultdict(dict)
    for m in RATING_METRICS:
        vals = {k: r[m] for k, r in team_season_avgs.items() if r.get(m) is not None}
        zm, _ = _rank_and_z(vals)
        for k, v in zm.items():
            z[k][m] = v

    scores = {k: sum(POWER_WEIGHTS[m] * ORIENTATION[m] * z[k].get(m, 0.0) for m in RATING_METRICS) for k in team_season_avgs}
    composite_lo, composite_hi = min(scores.values()), max(scores.values())

    epa_bounds: dict[str, tuple[float, float]] = {}
    for m in EPA_DISPLAY_METRICS:
        oriented_vals = [ORIENTATION[m] * r[m] for r in team_season_avgs.values() if r.get(m) is not None]
        if oriented_vals:
            epa_bounds[m] = (min(oriented_vals), max(oriented_vals))

    sos: dict[tuple, float] = {}
    if opponents:
        for key in team_season_avgs:
            team, season = key
            opp_scores = [scores[(opp, season)] for opp in opponents.get(key, []) if (opp, season) in scores]
            if opp_scores:
                sos[key] = sum(opp_scores) / len(opp_scores)
    sos_lo, sos_hi = (min(sos.values()), max(sos.values())) if sos else (0.0, 0.0)

    order = sorted(scores, key=lambda k: -scores[k])
    out = []
    for i, key in enumerate(order):
        team, season = key
        r = team_season_avgs[key]
        ratings_display: dict[str, float] = {}
        for m in EPA_DISPLAY_METRICS:
            if r.get(m) is not None and m in epa_bounds:
                lo, hi = epa_bounds[m]
                ratings_display[m] = _normalize_fixed(ORIENTATION[m] * r[m], lo, hi)
        for m in COUNTABLE_METRICS:
            if r.get(m) is not None:
                ratings_display[m] = round(r[m], 2)
        out.append({
            "team": team, "season": season, "rank": i + 1,
            "score": round(scores[key], 4),
            "score_display": _normalize_fixed(scores[key], composite_lo, composite_hi),
            "sos": round(sos[key], 4) if key in sos else None,
            "sos_display": _normalize_fixed(sos[key], sos_lo, sos_hi) if key in sos else None,
            "ratings": r, "ratings_display": ratings_display,
        })
    return out


def compute_backtest_scatter(game_log: list[dict]) -> list[dict]:
    """Turns compute_ratings()'s optional `game_log` into ready-to-plot rows for the
    Historical Power tab's backtest scatter: market spread vs. the CURRENT production
    POWER_WEIGHTS applied to each game's pre-game rating diffs, standardized ONCE across the
    whole dataset (not walk-forward -- this isn't re-deriving weights, just showing what
    today's weights would have said about each past game), colored by whether the model's
    favorite actually won. Same idea as research/edge_signal_test_v13_new_window_weights.py's
    final "fit on all data" step, applied for display instead of re-fitting."""
    if not game_log:
        return []
    mu = {m: float(np.mean([r["diffs"][m] for r in game_log])) for m in RATING_METRICS}
    sd = {m: float(np.std([r["diffs"][m] for r in game_log])) or 1.0 for m in RATING_METRICS}

    out = []
    for r in game_log:
        delta = sum(POWER_WEIGHTS[m] * ((r["diffs"][m] - mu[m]) / sd[m]) for m in RATING_METRICS)
        if r["home_score"] == r["away_score"]:
            outcome = "tie"
        else:
            actual_fav = "home" if r["home_score"] > r["away_score"] else "away"
            predicted_fav = "home" if delta > 0 else "away" if delta < 0 else None
            outcome = "push" if predicted_fav is None else ("hit" if predicted_fav == actual_fav else "miss")
        out.append({
            "season": r["season"], "week": r["week"], "date": r["gameday"],
            "home": r["home"], "away": r["away"],
            "home_score": r["home_score"], "away_score": r["away_score"],
            "spread": round(r["spread_line"], 1), "delta": round(delta, 4),
            "outcome": outcome,
        })
    return out


def refresh_historical(store, force: bool = False) -> str:
    """Rebuild the full historical (every completed season, pooled) power rankings dataset.
    Cheap after the first run -- every completed season's play-by-play is already cached as
    parquet (see _phase_stats_for_season), so this just re-aggregates small local files, not
    re-downloading anything. Rebuilt at most once/day anyway (same cadence as refresh()
    above), since completed seasons' data never changes except once a year at season end."""
    today = date.today().isoformat()
    if not force and store.kv_get("historical_power_date") == today:
        return "cached"

    from . import nfl_schedule  # local import: avoid a hard dependency for callers that don't need it

    current_season, _current_week = nfl_schedule.current_week()

    resp = requests.get(GAMES_URL, timeout=TIMEOUT)
    resp.raise_for_status()
    all_rows = [r for r in csv.DictReader(io.StringIO(resp.text)) if r["game_type"] == "REG"]
    played = [r for r in all_rows if r.get("result") not in ("", "NA", None) and int(r["season"]) >= FIRST_SEASON]

    season_avgs, opponents = compute_historical_season_averages(played, current_season)
    rankings = historical_power_rankings(season_avgs, opponents)
    seasons_covered = sorted({s for _, s in season_avgs})

    # Per-game backtest log for the scatter plot below the table -- same compute_ratings()
    # the live tab uses, just asked to also record its own pre-game snapshot of every game
    # along the way (see that function's `game_log` param) instead of only returning the
    # final state.
    game_log: list[dict] = []
    compute_ratings(played, current_season, game_log=game_log)
    scatter = compute_backtest_scatter(game_log)

    store.kv_set("historical_power_rankings", json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(),
        "seasons": seasons_covered,
        "rankings": rankings,
        "backtest_scatter": scatter,
    }))
    store.kv_set("historical_power_date", today)
    return (
        f"built ({len(rankings)} team-seasons, "
        f"{seasons_covered[0] if seasons_covered else '?'}-{seasons_covered[-1] if seasons_covered else '?'}, "
        f"{len(scatter)} backtest games)"
    )


def _refresh_parlay_snapshot(store, season: int, week: int, power: dict[str, dict], all_rows: list[dict]) -> None:
    """Freeze this week's team EPA display-ratings for the Parlays tab's matchup identifier,
    independent of team_ratings' own always-moving numbers -- those shift as each of this
    week's games gets folded into the EWMA the moment it finishes, which would otherwise make
    a matchup flagged before kickoff silently stop qualifying once the very game it described
    is in the books. Recomputed every refresh until this week's first kickoff (by nflverse's
    own `gameday`, date granularity -- good enough for a freeze point), then left untouched;
    resets fresh the moment the week number changes. Same freeze pattern as refresh_all's own
    odds/elway_odds handling and history.py's budget_snapshot. The very first time this runs,
    `prev` is None so `same_week` is False regardless of how far into the week we already are
    -- meaning a brand-new deploy mid-week backfills immediately off current data instead of
    waiting for a week boundary that already passed."""
    existing = store.kv_get("parlay_ratings_snapshot")
    try:
        prev = json.loads(existing) if existing else None
    except (TypeError, ValueError):
        prev = None
    same_week = bool(prev) and prev.get("season") == season and prev.get("week") == week

    this_week_days = [
        r.get("gameday") for r in all_rows
        if int(r["season"]) == season and str(r.get("week")) == str(week) and r.get("gameday")
    ]
    first_day = min(this_week_days) if this_week_days else None
    still_pregame = first_day is None or date.today().isoformat() < first_day

    if same_week and not still_pregame:
        return  # this week's snapshot is locked in -- don't touch it

    teams = {
        t: {m: info["ratings_display"].get(m) for m in EPA_DISPLAY_METRICS}
        for t, info in power.items() if info.get("ratings_display")
    }
    store.kv_set("parlay_ratings_snapshot", json.dumps({"season": season, "week": week, "teams": teams}))


def refresh(store, force: bool = False) -> str:
    """Rebuild team EPA ratings at most once per day."""
    today = date.today().isoformat()
    if not force and store.kv_get("team_ratings_date") == today:
        return "cached"

    from . import nfl_schedule  # local import: avoid a hard dependency for callers that don't need it

    current_season, current_week = nfl_schedule.current_week()

    resp = requests.get(GAMES_URL, timeout=TIMEOUT)
    resp.raise_for_status()
    all_rows = [r for r in csv.DictReader(io.StringIO(resp.text)) if r["game_type"] == "REG"]
    played = [r for r in all_rows if r.get("result") not in ("", "NA", None) and int(r["season"]) >= FIRST_SEASON]

    raw_ratings, historical_bounds = compute_ratings(played, current_season)

    # nflverse spells some current teams differently than nfl-hub's own canonical codes
    # (e.g. "LA" for the Rams, "WAS" for Washington) -- normalize so the frontend can look
    # these up directly against its own game rows without a second translation step.
    ratings: dict[str, dict[str, float]] = {}
    for raw_team, r in raw_ratings.items():
        try:
            ratings[normalize_team(raw_team)] = r
        except UnknownTeamError:
            continue  # a relocated franchise's old code with no current games -- not needed live

    upcoming_raw = [
        r for r in all_rows
        if int(r["season"]) == current_season and r.get("result") in ("", "NA", None)
    ]
    upcoming = []
    for r in upcoming_raw:
        try:
            upcoming.append({"home_team": normalize_team(r["home_team"]), "away_team": normalize_team(r["away_team"]),
                              "gameday": r.get("gameday")})
        except UnknownTeamError:
            continue
    matchups = matchup_callouts(ratings, upcoming, historical_bounds)
    power = power_rankings(ratings, historical_bounds)
    _refresh_parlay_snapshot(store, current_season, current_week, power, all_rows)

    store.kv_set("team_ratings", json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(),
        "season": current_season,
        "week": current_week,
        "teams": ratings,
        "matchups": matchups,
        "power_rankings": power,
        "power_ranking_meta": {
            "method": "L2-regularized (L2=0.3) logistic regression, coefficients constrained "
                      ">= 0, jointly fit across all 8 stats at once against real game outcomes "
                      "(2007-2025, 4927 games), chosen by walk-forward out-of-sample validation "
                      "-- not an in-sample fit, and re-fit against this rating's own season-"
                      "prior-fade windowing (not the old forever-decaying one). Walk-forward "
                      "AUC 0.6746 vs. 0.7224 for the closing market spread alone -- does not "
                      "beat the market at predicting winners; shown as descriptive context only.",
            "weights": {m: round(POWER_WEIGHTS[m] * ORIENTATION[m], 4) for m in RATING_METRICS},
        },
    }))
    store.kv_set("team_ratings_date", today)
    return f"built ({len(ratings)} teams, {len(matchups)} upcoming matchups, through {current_season} wk {current_week})"
