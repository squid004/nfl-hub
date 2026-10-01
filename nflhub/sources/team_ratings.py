"""Per-team rolling ratings across 8 stats: rush/pass offense EPA, rush/pass defense EPA
allowed, points scored, points allowed, turnovers committed, turnovers forced. EPA and
turnovers exclude garbage time (win probability outside [0.05, 0.95] dropped); points are
final-score based (not garbage-time filtered -- a final score IS the whole-game outcome, so
filtering it would fight the stat's own meaning).

Ported from research/*.py in this repo, where the methodology was validated: walk-forward
backtesting (2007-2025) showed none of these, alone or combined, beats the closing market
spread at predicting winners (see research/output/ and the AUC comparisons in
research/edge_signal_test*.py). They're stored as descriptive context for pool decisions and
as inputs to the power ranking below, not a betting model.

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
CARRYOVER = 0.65  # season-boundary regression-to-mean weight (matches ELWAY_BLEND_WEIGHT elsewhere)
WP_LO, WP_HI = 0.05, 0.95  # garbage-time filter: drop plays outside this win-probability band
RATING_METRICS = (
    "rush_off_epa", "pass_off_epa", "rush_def_epa_allowed", "pass_def_epa_allowed",
    "points_off", "points_def_allowed", "turnovers_off", "turnovers_def_forced",
)
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
# validated (research/edge_signal_test_v8_power_weights.py, 4436 games, 2007-2025). Apply as
# POWER_WEIGHTS[m] * ORIENTATION[m] * z-score -- these are all >= 0 by construction (see
# module docstring). Recompute that script and update these if you want to refresh the fit;
# not refit automatically.
POWER_WEIGHTS = {
    "rush_off_epa": 0.0772,
    "pass_off_epa": 0.1543,
    "rush_def_epa_allowed": 0.0791,
    "pass_def_epa_allowed": 0.0600,
    "points_off": 0.1603,
    "points_def_allowed": 0.1021,
    "turnovers_off": 0.0195,
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


def fetch_forecast_weather(team: str, date_str: str) -> dict[str, float] | None:
    """Live forecast wind_mph/precip_mm/cold_flag for `team`'s stadium on `date_str`
    (YYYY-MM-DD), or None if the team has no outdoor stadium, the date is outside the
    forecast's reliable range (~16 days), or the request fails for any reason (soft-fail,
    matching every other best-effort external source in this project)."""
    coords = STADIUM_COORDS.get(team)
    if not coords:
        return None
    try:
        resp = requests.get(FORECAST_URL, params={
            "latitude": coords[0], "longitude": coords[1],
            "daily": "temperature_2m_max,precipitation_sum,windspeed_10m_max",
            "timezone": "America/New_York", "forecast_days": 16,
        }, timeout=TIMEOUT)
        resp.raise_for_status()
        daily = resp.json()["daily"]
        if date_str not in daily["time"]:
            return None
        i = daily["time"].index(date_str)
        temp_f = daily["temperature_2m_max"][i] * 9 / 5 + 32
        return {
            "wind_mph": daily["windspeed_10m_max"][i] * 0.621371,
            "precip_mm": daily["precipitation_sum"][i],
            "cold_flag": 1.0 if temp_f < 20 else 0.0,
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


class _RunningMean:
    """Online mean over data seen so far only -- safe as a season-boundary regression target
    with no look-ahead."""

    def __init__(self) -> None:
        self.n, self.mean = 0, 0.0

    def update(self, x: float) -> None:
        self.n += 1
        self.mean += (x - self.mean) / self.n


def compute_ratings(played_games: list[dict], current_season: int) -> dict[str, dict[str, float]]:
    """`played_games`: nflverse games.csv REG rows with a result, any seasons needed.
    Returns {team: {metric: rating}} as of the most recent game passed in."""
    by_game: dict[str, dict[str, dict]] = defaultdict(dict)
    seasons_needed = sorted({int(g["season"]) for g in played_games})
    for season in seasons_needed:
        for row in _phase_stats_for_season(season, current_season).to_dict("records"):
            by_game[row["game_id"]][row["team"]] = row

    games_by_id = {g["game_id"]: g for g in played_games}
    running_mean = {"rush_off_epa": _RunningMean(), "pass_off_epa": _RunningMean()}

    running_mean.update({"points_off": _RunningMean(), "turnovers_off": _RunningMean()})

    def league_mean(metric: str) -> float:
        base = metric.replace("_def_epa_allowed", "_off_epa") \
            .replace("_def_allowed", "_off").replace("_def_forced", "_off")
        return running_mean[base].mean

    rating: dict[str, dict] = defaultdict(lambda: {m: None for m in RATING_METRICS})
    last_season: dict[str, int] = {}

    for gid, teams in sorted(by_game.items()):
        g = games_by_id.get(gid)
        if not g or len(teams) != 2:
            continue
        season = int(g["season"])
        home, away = g["home_team"], g["away_team"]
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

        for team in (home, away):
            if last_season.get(team) is not None and last_season[team] != season:
                for m in RATING_METRICS:
                    r = rating[team][m]
                    if r is not None:
                        lm = league_mean(m)
                        rating[team][m] = lm + CARRYOVER * (r - lm)
            last_season[team] = season

        for m in ("rush_off_epa", "pass_off_epa", "points_off", "turnovers_off"):
            running_mean[m].update(home_raw[m])
            running_mean[m].update(away_raw[m])
        for team, raw in ((home, home_raw), (away, away_raw)):
            for m in RATING_METRICS:
                prev = rating[team][m]
                rating[team][m] = raw[m] if prev is None else (1 - EWMA_ALPHA) * prev + EWMA_ALPHA * raw[m]

    return {
        team: {m: (round(v, 4) if v is not None else None) for m, v in r.items()}
        for team, r in rating.items()
    }


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


def matchup_callouts(ratings: dict[str, dict[str, float]], upcoming: list[dict]) -> dict[str, dict]:
    """For each upcoming game (already normalized home/away team codes), rule-based sentences
    for any offense-vs-opposing-defense pairing whose combined z-score clears
    MISMATCH_Z_THRESHOLD -- a real strength meeting a real weakness, not just noise. Keyed
    "AWAY@HOME" so the frontend can look it up directly from its own game row."""
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
                off_val, def_val = ratings[off_team][off_m], ratings[def_team][def_m]
                off_rank, def_rank = rank[off_team][off_m], rank[def_team][def_m]
                favors_offense = combined > 0
                # each side's descriptor reflects that team's OWN rank (good/bad), independent
                # of which way the combined matchup leans -- otherwise a genuinely good offense
                # facing an elite defense could get mislabeled "worst" just because the pairing
                # nets out unfavorable for it.
                off_desc = f"{_ordinal(off_rank)} in the NFL" if z[off_team][off_m] >= 0 \
                    else f"{_ordinal(n_teams + 1 - off_rank)}-worst in the NFL"
                def_desc = f"{_ordinal(def_rank)}-most allowed in the NFL" if z[def_team][def_m] >= 0 \
                    else f"{_ordinal(n_teams + 1 - def_rank)}-fewest allowed in the NFL"
                verdict = "a lopsided matchup on paper" if favors_offense else "a tough matchup on paper"
                callouts.append(
                    f"{off_team}'s {phase} offense ({off_val:+.2f} EPA/play, {off_desc}) "
                    f"{'faces' if favors_offense else 'runs into'} {def_team}'s {phase} defense "
                    f"({def_val:+.2f} EPA/play allowed, {def_desc}) -- {verdict}."
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
        out[f"{away}@{home}"] = {
            "home": home, "away": away,
            "home_ratings": ratings[home], "away_ratings": ratings[away],
            "callouts": callouts,
            "predicted_home_points": predict_points(ratings[home], ratings[away], weather),
            "predicted_away_points": predict_points(ratings[away], ratings[home], weather),
            "weather": weather,
        }
    return out


def power_rankings(ratings: dict[str, dict[str, float]]) -> dict[str, dict]:
    """Composite score per team: sum of POWER_WEIGHTS[m] * ORIENTATION[m] * z-score(team, m)
    across all 8 stats, ranked 1 = best. Each z-score is cross-sectional (against the other 31
    current teams), so this reflects current relative standing, not an absolute scale."""
    if len(ratings) < 4:
        return {}
    z: dict[str, dict[str, float]] = defaultdict(dict)
    for m in RATING_METRICS:
        vals = {t: r[m] for t, r in ratings.items() if r.get(m) is not None}
        zm, _ = _rank_and_z(vals)
        for t, v in zm.items():
            z[t][m] = v

    scores = {t: sum(POWER_WEIGHTS[m] * ORIENTATION[m] * zt.get(m, 0.0) for m in RATING_METRICS) for t, zt in z.items()}
    order = sorted(scores, key=lambda t: -scores[t])
    return {
        t: {"rank": i + 1, "score": round(scores[t], 4), "ratings": ratings[t]}
        for i, t in enumerate(order)
    }


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

    raw_ratings = compute_ratings(played, current_season)

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
    matchups = matchup_callouts(ratings, upcoming)
    power = power_rankings(ratings)

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
                      "(2007-2025, 4436 games), chosen by walk-forward out-of-sample validation "
                      "-- not an in-sample fit. Does not beat the closing market spread at "
                      "predicting winners; shown as descriptive context only.",
            "weights": {m: round(POWER_WEIGHTS[m] * ORIENTATION[m], 4) for m in RATING_METRICS},
        },
    }))
    store.kv_set("team_ratings_date", today)
    return f"built ({len(ratings)} teams, {len(matchups)} upcoming matchups, through {current_season} wk {current_week})"
