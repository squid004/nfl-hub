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

QB/skill-position health adjustment: `delta` (the composite's whole point -- fed into both
the Historical Power tab's backtest scatter and the Moneyline Pick'em tab's Power Model pick)
is NOT pure POWER_WEIGHTS output anymore. It's that 8-stat composite PLUS two small,
data-driven corrections for information those 8 stats have zero visibility into (who's
actually playing THIS week): QB_QUALITY_GAP_WEIGHT * qb_quality_gap (see
qb_quality_gap_data()) and SKILL_EPA_OUT_WEIGHT * skill_epa_out (see
skill_epa_out_by_team_week()). Both weights were fit once offline (research/edge_signal_
test_v26_production_weights.py) and are real, statistically significant effects -- but
walk-forward backtesting of their effect on raw hit-rate came back flat (research/edge_
signal_test_v23/v24/v25_*.py all found net change indistinguishable from zero). The point of
including them isn't to flip more picks right; it's so `delta` -- and therefore delta_win_
prob()'s displayed confidence -- honestly reflects known health context instead of silently
ignoring it. `delta_raw` is kept alongside `delta` everywhere, for comparison.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import re
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
INJURIES_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv"
PLAYER_STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.csv"
TIMEOUT = 30
FIRST_SEASON = 2007
FIRST_INJURY_SEASON = 2009  # nflverse's injury reports start here; 2007/2008 return 404
SKILL_POSITIONS = {"RB", "WR", "TE", "FB"}
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

# Calibrates the composite power-ranking delta into "P(the model's favorite wins)" for the
# Moneyline Pick'em tab's Power Model pick, shown alongside ELWAY/History. Plain unconstrained
# 2-parameter logistic fit, P(correct) = sigmoid(a + b*|delta|) -- 1 feature, thousands of
# games, no regularization needed. Fit on the CLEAN (non-backup-QB) backtest games only (the
# QB-injury blind spot is shown as its own separate filter on the Historical Power tab, not
# baked into this curve), walk-forward validated (research/edge_signal_test_v15_delta_
# calibration.py, 4296 games, 2007-2025): beat a flat-baseline log-loss in 14 of 15 held-out
# seasons. DELTA_PROB_MAX clips the output -- the highest-confidence bin ever actually
# observed in that backtest was ~77%, so the raw curve's extrapolation past ~85% at rare
# extreme deltas isn't trusted blindly. Computed once offline, not refit daily.
DELTA_CALIBRATION_INTERCEPT = 0.0484
DELTA_CALIBRATION_SLOPE = 1.4497
DELTA_PROB_MAX = 0.85


def delta_win_prob(delta: float) -> float:
    """Calibrated P(the side `delta` favors wins) -- exactly 0.5 if delta is 0 (no favorite,
    a dead-even composite rating -- vanishingly rare but not impossible)."""
    if delta == 0:
        return 0.5
    p = 1.0 / (1.0 + np.exp(-(DELTA_CALIBRATION_INTERCEPT + DELTA_CALIBRATION_SLOPE * abs(delta))))
    return min(p, DELTA_PROB_MAX)


# Converts two current-week injury signals -- QB_QUALITY_GAP (see qb_quality_gap_data()
# below) and SKILL_EPA_OUT (see skill_epa_out_by_team_week() below) -- into the SAME units
# as the composite `delta`, so they can be added directly to it. Both are real EPA-unit
# measurements (not binary flags), computed from nflverse's own play-by-play/player-stats
# data: QB_QUALITY_GAP is the difference between a team's normal starter's trailing EPA/
# dropback and whoever's actually expected to play; SKILL_EPA_OUT is the summed trailing
# EPA/game of every RB/WR/TE/FB currently listed Out/Doubtful. Fit ONCE offline
# (research/edge_signal_test_v26_production_weights.py, 4431 games 2009-2025): one joint
# logistic regression of home_win ~ delta + qb_quality_gap_diff + skill_epa_out_diff, all
# three features in their RAW native units (not standardized) -- the ratio of each gap's own
# coefficient to delta's own IS directly "how many delta-equivalent units this gap is worth."
# Both terms came back significant with the expected sign (QB z=-3.63, skill z=-2.46 --
# more missing value lowers delta, as it should). This does NOT touch POWER_WEIGHTS or the
# underlying 8-stat composite at all -- it's a purely additive correction on top, informed
# by current-week injury context those 8 inputs have zero visibility into (they only know
# how a team has actually performed with whoever played, never who's playing THIS week).
# Walk-forward backtests of this adjustment's effect on raw hit-rate (research/edge_signal_
# test_v23/v24/v25_*.py) came back statistically flat (net change indistinguishable from
# zero, ~50/50 split of improved vs. regressed picks) -- the point of including it isn't to
# flip more picks, it's so `delta` (and therefore delta_win_prob's displayed confidence)
# honestly reflects known health context instead of silently ignoring it. Computed once
# offline, not refit daily.
QB_QUALITY_GAP_WEIGHT = -0.5903
SKILL_EPA_OUT_WEIGHT = -0.0362

# Home-field advantage: every OTHER signal in this model is venue-blind (a team's own rating
# reflects its average performance regardless of where it played), so `delta` had NO
# home-field term at all until now -- confirmed directly against real data, not assumed: even
# in games the model's own delta calls a dead-even matchup, home teams still win meaningfully
# more than 50% of the time (research/edge_signal_test_v28_hfa_window.py). Unlike QB/skill
# (rare, situational, variance-type corrections), this applies to EVERY game and corrects a
# persistent structural bias, not situational uncertainty -- so unlike those two, this one
# actually moves raw hit-rate, not just calibration (see refresh_historical()'s own backtest
# numbers). HFA_WEIGHT (0.9169, z=6.14, significant) is fit against home_field_logit_by_
# season()'s trailing estimate -- NOT a flat historical average, which would overstate
# today's effect (home win rate has genuinely declined, 2019-2021 especially, including
# 2020's no-fans anomaly), and NOT a short/tight trailing window either, which gets whipsawed
# by any single season's sampling noise (each season is only ~256 games). The winning
# estimator (of 11 tested by walk-forward log-loss) shrinks a trailing-5-season average
# toward the all-time-to-date average with a 100-pseudo-game weight -- same shrinkage
# convention nflhub/sources/history.py already uses for small-sample spread-bucket rates,
# just reused here instead of invented fresh. Recomputed every refresh from `played` (already
# fetched for everything else) -- never a stale, separately-maintained number.
HFA_SHRINKAGE_K = 100
HFA_WEIGHT = 0.9169

# Neutral-site games get none of the home-field term above -- confirmed as a real,
# recurring gap, not a one-off: nflverse's games.csv `location` column correctly flags 72
# REG-season neutral games since 2007 (growing fast with the international series -- 8
# already in 2026) and we use that as the primary signal. But `location` alone isn't fully
# reliable: the 2026 wk5 PHI@JAX game at Tottenham Hotspur Stadium is still tagged "Home" in
# nflverse's own data even though JAX obviously has no real home-field edge playing in
# London -- confirmed live, `stadium_id` for that row even still reads "JAX00" (their normal
# stadium's id), so this is a genuine upstream data quirk, not a misread on our end. Same
# pattern bit Buffalo's old "Toronto Series" (2008-2013, Rogers Centre) -- also tagged
# "Home." _NEUTRAL_SITE_STADIUMS is a small, manually-curated fallback for exactly these
# known mislabeled cases; extend it if the international series adds a new venue nflverse
# hasn't started flagging correctly yet.
_NEUTRAL_SITE_STADIUMS = {
    "Tottenham Hotspur Stadium", "Tottenham Stadium", "Wembley Stadium", "Twickenham Stadium",
    "Allianz Arena", "FC Bayern Munich Stadium", "Deutsche Bank Park",
    "Azteca Stadium", "Estadio Banorte", "Arena Corinthians", "Neo Química Arena",
    "Maracana Stadium", "Melbourne Cricket Ground", "Bernabeu", "Stade de France",
    "Rogers Centre",
}


def is_neutral_site(row: dict) -> bool:
    """True if this game gets no home-field edge for either side -- nflverse's own
    `location` field (anything other than "Home") first, then the manual fallback list
    above for known cases that field still mislabels (see HFA_WEIGHT's own comment)."""
    if row.get("location") and row["location"] != "Home":
        return True
    return (row.get("stadium") or "") in _NEUTRAL_SITE_STADIUMS


def _home_win_rate_by_season(played_games: list[dict]) -> dict[int, tuple[int, int]]:
    """{season: (home_wins, decided_games)} straight off the same `played` games.csv rows
    refresh()/refresh_historical() already fetch -- no new network call needed for this.
    Neutral-site games excluded entirely (see is_neutral_site()) -- a neutral game's result
    carries no home-field signal one way or the other, and leaving it in would dilute the
    real trailing home-win rate toward 50% for no good reason."""
    out: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for g in played_games:
        try:
            season = int(g["season"])
            home_pts, away_pts = float(g["home_score"]), float(g["away_score"])
        except (ValueError, TypeError):
            continue
        if home_pts == away_pts or is_neutral_site(g):
            continue
        out[season][1] += 1
        if home_pts > away_pts:
            out[season][0] += 1
    return {s: (v[0], v[1]) for s, v in out.items()}


def home_field_logit_by_season(played_games: list[dict]) -> dict[int, float]:
    """{season: logit(P(home win))} a model could honestly have used GOING INTO that season
    -- trailing 5 completed seasons' home win rate, shrunk toward the all-time-to-date
    average with HFA_SHRINKAGE_K pseudo-games (see HFA_WEIGHT's own docstring for why this
    estimator specifically). No lookahead: season S's value only ever uses games from
    seasons strictly before S -- the very first tracked season gets exactly 0.0 (no history
    yet to estimate from, not a guess)."""
    rates = _home_win_rate_by_season(played_games)
    seasons = sorted(rates)
    out: dict[int, float] = {}
    for i, season in enumerate(seasons):
        prior = seasons[:i]
        if not prior:
            out[season] = 0.0
            continue
        window = prior[-5:]
        w5 = sum(rates[s][0] for s in window)
        n5 = sum(rates[s][1] for s in window)
        rate5 = w5 / n5 if n5 else 0.5
        w_all = sum(rates[s][0] for s in prior)
        n_all = sum(rates[s][1] for s in prior)
        rate_all = w_all / n_all if n_all else 0.5
        p = (n5 * rate5 + HFA_SHRINKAGE_K * rate_all) / (n5 + HFA_SHRINKAGE_K)
        p = min(max(p, 1e-6), 1 - 1e-6)
        out[season] = math.log(p / (1 - p))
    return out


# Momentum: a team's own signed win/loss streak entering a game (+N = N straight wins, -N =
# N straight losses, reset to 0 at a season boundary and after any tie) -- fit ONCE offline
# (research/edge_signal_test_v35_momentum.py, 4977 games 2007-2025) as a joint logistic
# regression of home_win ~ delta + streak_diff, where delta is the full production composite
# (8-stat + QB/skill/HFA, already recency-weighted at the STAT level via compute_ratings'
# own EWMA). The question this answers isn't "do good teams win more" -- delta already
# reflects recent form -- it's whether streak length carries anything EXTRA once delta
# already knows the team's been playing well. It does: streak_diff survived at z=3.74
# (delta-equivalent weight 0.0262/win-loss-game of streak advantage), and -- unlike QB/skill
# -- it moves raw hit rate too (68.3% vs 60.9% baseline on big streak mismatches, z=4.43),
# plus a strong calibration signal (confidence reduced 25.8% of misses vs 20.4% of hits,
# z=4.35, p<0.0001). No new data source: games.csv's own chronological results are already
# fetched for everything else in this module.
MOMENTUM_WEIGHT = 0.0262


def team_streak_by_week(played_games: list[dict]) -> dict[tuple[str, int, int], int]:
    """{(team, season, week): signed streak ENTERING that week's game} -- +N for N
    consecutive wins, -N for N consecutive losses, reset to 0 at the first game of a season
    and after any tie. No lookahead: a team's streak for week W only ever uses games from
    weeks strictly before W (or earlier seasons for week 1, which is itself always 0 since a
    new season carries no streak over)."""
    by_team_season: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for g in played_games:
        try:
            season, week = int(g["season"]), int(g["week"])
            home, away = normalize_team(g["home_team"]), normalize_team(g["away_team"])
            home_pts, away_pts = float(g["home_score"]), float(g["away_score"])
        except (ValueError, TypeError, UnknownTeamError):
            continue
        if home_pts == away_pts:
            home_result = away_result = "tie"
        else:
            home_result = "win" if home_pts > away_pts else "loss"
            away_result = "loss" if home_result == "win" else "win"
        by_team_season[(home, season)].append({"week": week, "result": home_result})
        by_team_season[(away, season)].append({"week": week, "result": away_result})

    out: dict[tuple[str, int, int], int] = {}
    for (team, season), games in by_team_season.items():
        games.sort(key=lambda g: g["week"])
        streak = 0
        for g in games:
            out[(team, season, g["week"])] = streak
            if g["result"] == "tie":
                streak = 0
            elif g["result"] == "win":
                streak = streak + 1 if streak >= 0 else 1
            else:
                streak = streak - 1 if streak <= 0 else -1
    return out


def _norm_player_name(name: str) -> str:
    """Lowercase, strip punctuation/suffixes -- same normalization on both sides of every
    name-matched join below (injury report <-> player_stats.csv), since neither shares a
    common ID with the other (gsis_id vs. no stable id in player_stats' name columns)."""
    name = (name or "").lower().strip()
    name = re.sub(r"[.'`]", "", name)
    name = re.sub(r"\s+(jr|sr|ii|iii|iv|v)\.?$", "", name)
    return re.sub(r"\s+", " ", name)


def _passer_game_stats(season: int, current_season: int) -> pd.DataFrame:
    """One row per (game_id, team, passer_player_id): pass attempts and EPA sum that game
    (REG season, garbage time excluded -- same WP_LO/WP_HI filter as _phase_stats_for_
    season). NOT reusable from that function's own cache -- it doesn't carry passer
    identity. Completed seasons cached (nflhub/data/team_ratings/passer_game_{season}.
    parquet, same convention as _phase_stats_for_season); only the current season
    re-downloads play-by-play fresh."""
    if season < current_season:
        path = os.path.join(DATA_DIR, f"passer_game_{season}.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)
    cols = ["game_id", "season", "week", "season_type", "posteam", "play_type", "epa", "wp",
            "passer_player_id", "passer_player_name"]
    df = pd.read_csv(PBP_URL.format(season=season), compression="gzip", usecols=cols, low_memory=False)
    df = df[(df["season_type"] == "REG") & (df["play_type"] == "pass")]
    df = df.dropna(subset=["wp", "epa", "posteam", "passer_player_id"])
    df = df[(df["wp"] >= WP_LO) & (df["wp"] <= WP_HI)]
    out = df.groupby(["game_id", "posteam", "passer_player_id"]).agg(
        attempts=("epa", "size"), epa_sum=("epa", "sum"), name=("passer_player_name", "first")
    ).reset_index().rename(columns={"posteam": "team"})
    if season < current_season:
        out.to_parquet(os.path.join(DATA_DIR, f"passer_game_{season}.parquet"))
    return out


def qb_quality_gap_data(played_games: list[dict], current_season: int):
    """One chronological walk over every passer's pass attempts/EPA (FIRST_INJURY_SEASON
    through current_season -- matches the window QB_QUALITY_GAP_WEIGHT was fit against)
    that serves TWO callers from the same state, so they can never drift apart:
      - `gap_by_team_week` {(team, season, week): gap}: for every HISTORICAL game, the
        team's primary starter's own trailing EPA/dropback minus the ACTUAL passer's own
        trailing EPA/dropback -- 0 when the normal starter played, positive when whoever
        played was worse than usual. Feeds compute_backtest_scatter()'s adjusted delta.
      - `current_primary` {team: passer_id} / `current_trailing_epa` {passer_id: EPA/
        dropback} / `league_avg_epa`: the LATEST state as of the most recent game each team/
        passer has played -- "who's QB1 right now" and "how have they (and everyone else)
        been throwing." Feeds the live, UPCOMING-game adjustment in refresh() (no actual
        passer exists yet for a future game, so that path infers "league-average backup" if
        the live injury report has the primary starter Out/Doubtful, else assumes the
        healthy primary plays -- see refresh()'s own qb_gap_upcoming construction).
      - `passer_names` {passer_id: display name}: nflverse has no id shared with ESPN's live
        injury feed, so matching `current_primary`'s id against that feed (refresh()'s own
        live gate, see _primary_qb_out_live()) has to go through the player's name instead.
    `primary_so_far` is cumulative attempts THIS season so far, falling back to last
    season's leader before this season's first pass -- no lookahead, no full-season
    hindsight."""
    frames = []
    for season in range(FIRST_INJURY_SEASON, current_season + 1):
        try:
            df = _passer_game_stats(season, current_season)
        except Exception:  # noqa: BLE001 -- best-effort, same soft-fail convention as the rest of this module
            log.warning("passer_game_stats failed for %s", season, exc_info=True)
            continue
        df = df.copy()
        df["season"] = season
        frames.append(df)
    if not frames:
        return {}, {}, {}, 0.0
    passer_df = pd.concat(frames, ignore_index=True)

    games_by_id = {g["game_id"]: g for g in played_games}

    def week_of(gid):
        g = games_by_id.get(gid)
        if g is not None:
            try:
                return int(g["week"])
            except (TypeError, ValueError):
                pass
        try:
            return int(gid.split("_")[1])
        except (IndexError, ValueError):
            return None

    passer_df["week"] = [week_of(gid) for gid in passer_df["game_id"]]
    passer_df = passer_df.dropna(subset=["week"])
    passer_df["week"] = passer_df["week"].astype(int)

    actual_passer: dict[tuple[str, str], str] = {}
    game_team_epa: dict[tuple[str, str], tuple[float, int]] = {}
    for (gid, team), grp in passer_df.groupby(["game_id", "team"]):
        row = grp.loc[grp["attempts"].idxmax()]
        actual_passer[(gid, team)] = row["passer_player_id"]
        game_team_epa[(gid, team)] = (row["epa_sum"], row["attempts"])

    gt_meta = passer_df[["game_id", "team", "season", "week"]].drop_duplicates().sort_values(["season", "week", "game_id", "team"])
    game_order = list(gt_meta.itertuples(index=False, name=None))

    season_attempts: dict[tuple[str, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    last_season_leader: dict[str, str] = {}
    last_season_num: dict[str, int] = {}
    primary_so_far: dict[tuple[str, str], str] = {}
    passer_hist: dict[str, list[float]] = defaultdict(list)
    trailing_epa: dict[tuple[str, str], float | None] = {}
    league_avg: list[float] = []

    for gid, team, season, week in game_order:
        if last_season_num.get(team) is not None and last_season_num[team] != season and season_attempts[(team, season - 1)]:
            last_season_leader[team] = max(season_attempts[(team, season - 1)].items(), key=lambda kv: kv[1])[0]
        last_season_num[team] = season
        this_season = season_attempts[(team, season)]
        if this_season:
            primary_so_far[(team, gid)] = max(this_season.items(), key=lambda kv: kv[1])[0]
        elif team in last_season_leader:
            primary_so_far[(team, gid)] = last_season_leader[team]
        else:
            primary_so_far[(team, gid)] = actual_passer.get((gid, team))

        ap = actual_passer.get((gid, team))
        if ap is not None:
            hist = passer_hist[ap]
            trailing_epa[(ap, gid)] = (sum(hist) / len(hist)) if hist else None
            epa_sum, attempts = game_team_epa[(gid, team)]
            if attempts:
                per_play = epa_sum / attempts
                hist.append(per_play)
                league_avg.append(per_play)
            this_season[ap] += attempts

    league_avg_epa = (sum(league_avg) / len(league_avg)) if league_avg else 0.0

    gap_by_team_week: dict[tuple[str, int, int], float] = {}
    for gid, team, season, week in game_order:
        primary = primary_so_far.get((team, gid))
        actual = actual_passer.get((gid, team))
        if primary is None or actual is None:
            continue
        tp = trailing_epa.get((primary, gid))
        ta = trailing_epa.get((actual, gid))
        tp = tp if tp is not None else league_avg_epa
        ta = ta if ta is not None else league_avg_epa
        gap_by_team_week[(team, season, week)] = tp - ta

    current_primary: dict[str, str] = {}
    for team, season in season_attempts:
        if season != last_season_num.get(team):
            continue
        this_season = season_attempts[(team, season)]
        if this_season:
            current_primary[team] = max(this_season.items(), key=lambda kv: kv[1])[0]
        elif team in last_season_leader:
            current_primary[team] = last_season_leader[team]
    current_trailing_epa = {p: ((sum(h) / len(h)) if h else league_avg_epa) for p, h in passer_hist.items()}
    passer_names: dict[str, str] = dict(zip(passer_df["passer_player_id"], passer_df["name"]))

    return gap_by_team_week, current_primary, current_trailing_epa, league_avg_epa, passer_names


def skill_epa_history(current_season: int) -> tuple[dict[str, list], float]:
    """{name_norm: [(season, week, epa), ...]} sorted chronologically (rushing_epa +
    receiving_epa per game, nflverse's player_stats.csv, RB/WR/TE/FB only), plus the
    league-average fallback for a player with no tracked history yet. nflverse serves
    player_stats as ONE file covering every season (no per-season URL) -- every refresh
    still downloads the whole file, but completed seasons' rows are cached as their own
    small parquet (nflhub/data/team_ratings/skill_epa_{season}.parquet) and only the
    CURRENT season's rows get re-parsed from the fresh download each time."""
    frames = []
    missing = [s for s in range(FIRST_INJURY_SEASON, current_season + 1)
               if s == current_season or not os.path.exists(os.path.join(DATA_DIR, f"skill_epa_{s}.parquet"))]
    for s in range(FIRST_INJURY_SEASON, current_season + 1):
        if s in missing:
            continue
        frames.append(pd.read_parquet(os.path.join(DATA_DIR, f"skill_epa_{s}.parquet")))
    if missing:
        resp = requests.get(PLAYER_STATS_URL, timeout=90)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
        df = df[(df["position"].isin(SKILL_POSITIONS)) & (df["season_type"] == "REG")]
        df["epa"] = df["rushing_epa"].fillna(0) + df["receiving_epa"].fillna(0)
        df["name_norm"] = df["player_display_name"].fillna(df.get("player_name")).map(_norm_player_name)
        df = df[["name_norm", "season", "week", "epa"]].dropna(subset=["name_norm"])
        for s in missing:
            season_df = df[df["season"] == s]
            if s != current_season:
                season_df.to_parquet(os.path.join(DATA_DIR, f"skill_epa_{s}.parquet"))
            frames.append(season_df)

    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["name_norm", "season", "week", "epa"])
    player_games: dict[str, list] = defaultdict(list)
    for row in full.sort_values(["season", "week"]).itertuples(index=False):
        player_games[row.name_norm].append((row.season, row.week, row.epa))
    league_avg_epa = float(full["epa"].mean()) if len(full) else 0.0
    return dict(player_games), league_avg_epa


def _skill_injury_rows(current_season: int) -> list[dict]:
    """Every Out/Doubtful RB/WR/TE/FB report row -- same cached injuries_{season}.csv files
    as _qb_out_doubtful_by_week (shared cache, different position filter, no extra network
    cost beyond what that function already pays within the same refresh)."""
    out = []
    for season in range(FIRST_INJURY_SEASON, current_season + 1):
        cache_path = os.path.join(DATA_DIR, f"injuries_{season}.csv")
        if season < current_season and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                text = f.read()
        else:
            resp = requests.get(INJURIES_URL.format(season=season), timeout=TIMEOUT)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            text = resp.text
            if season < current_season:
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(text)
        for row in csv.DictReader(io.StringIO(text)):
            if row.get("game_type") != "REG" or row.get("position") not in SKILL_POSITIONS:
                continue
            if row.get("report_status") not in ("Out", "Doubtful"):
                continue
            try:
                team = normalize_team(row["team"])
                week = int(row["week"])
            except (UnknownTeamError, ValueError, TypeError):
                continue
            out.append({"team": team, "season": season, "week": week, "name": _norm_player_name(row.get("full_name", ""))})
    return out


def skill_epa_out_by_team_week(current_season: int) -> dict[tuple[str, int, int], float]:
    """{(team, season, week): sum of each flagged RB/WR/TE/FB's own trailing (career-to-
    date, no lookahead) rushing+receiving EPA per game} -- "how much known offensive value
    is unavailable this week," zero in a healthy week, scaling with both how many are out
    AND how productive they normally are. Each player's weight is a DISCRETE value (their
    own trailing EPA), just summed in aggregate across however many are flagged -- not a
    unit-wide average. Works unchanged for both historical weeks (full backtest) and the
    CURRENT week (live picks), since it's defined purely in terms of who's flagged + their
    own history, no "who's the normal starter" ambiguity the way QB's 1-for-1 swap has."""
    import bisect
    player_games, league_avg_epa = skill_epa_history(current_season)
    injury_rows = _skill_injury_rows(current_season)
    out: dict[tuple[str, int, int], float] = defaultdict(float)
    for r in injury_rows:
        games = player_games.get(r["name"])
        weight = league_avg_epa
        if games:
            keys = [(s, w) for s, w, _ in games]
            idx = bisect.bisect_left(keys, (r["season"], r["week"]))
            prior = games[:idx]
            if prior:
                weight = sum(e for _, _, e in prior) / len(prior)
        out[(r["team"], r["season"], r["week"])] += max(weight, 0.0)  # a below-average player's "loss" isn't negative value
    return dict(out)


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


# Per-team SCORE DISTRIBUTIONS (not just predict_points()'s single number) for the Against
# the Spread tab's Power Ranking spread/O&U and its score-distribution chart. Two-step, per
# explicit user direction (research/edge_signal_test_v32/v33_score_distribution*.py, 2429
# games validated): (1) a raw Normal(bias-corrected predict_points() mean, empirical
# residual sd) discretized to integers; (2) reweighted by the REAL historical frequency of
# each exact final score since the PAT distance increased (2015 season -- verified live,
# NFL owners approved the 15-yard-line PAT in May 2015), so the output favors realistic
# scores (14) over unrealistic ones (15) instead of spreading mass smoothly across every
# integer. Validated at scale: MAE improved on the uncorrected baseline (7.57 vs. 7.70), and
# Step 2 genuinely improves the distribution's accuracy, not just its look (+0.16 nats/
# game-side average log-likelihood of the actual score vs. Step 1 alone).
#
# Bias correction folds into Step 1's mean: home/away bias is real and DECLINING over time
# (same underlying shift as the win-probability HFA work) -- trailing-5-season-shrunk-
# toward-long-run, same structure as home_field_logit_by_season() but in raw points.
# Favorite/underdog bias is defined by predict_points()'s OWN predicted margin (not the
# market spread, so this module stays self-contained) -- flat/cumulative average, since no
# time trend was found there (unlike home/away), just noise.
#
# Reconciliation with the model's OWN win probability (delta_win_prob()): the two
# distributions are calibrated via bisection on a symmetric mean-shift so their IMPLIED
# P(home scores more) -- treating the two as INDEPENDENT, a real simplification, not a true
# joint model -- exactly matches delta_win_prob(). "Most likely outcome" (the Power Ranking
# spread/O&U shown in the UI) is each side's own distribution MODE -- under independence the
# single most probable (home, away) score PAIR is exactly (mode(home), mode(away)), so this
# is a well-defined joint answer, not just two separate marginal summaries pasted together.
SCORE_HIST_FIRST_SEASON = 2015
MAX_TEAM_SCORE = 60  # integer score bins 0..MAX_TEAM_SCORE; negligible real mass beyond this


def points_bias_by_season(game_log: list[dict]) -> dict[int, dict[str, float]]:
    """Trailing-5-season home/away points-prediction bias (actual - predicted), shrunk
    toward the all-time-to-date average with HFA_SHRINKAGE_K pseudo-games -- same structure
    as home_field_logit_by_season(), just in raw points instead of logit units. No
    lookahead: season S's value only uses games from seasons strictly before S. Neutral-site
    games excluded (see is_neutral_site()) -- same reasoning as _home_win_rate_by_season:
    there's no real home/away scoring asymmetry to measure in a game with no true home side."""
    by_season: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"home": [], "away": []})
    for r in game_log:
        if r.get("home_pred_points") is None or r.get("away_pred_points") is None or r.get("neutral"):
            continue
        by_season[r["season"]]["home"].append(r["home_score"] - r["home_pred_points"])
        by_season[r["season"]]["away"].append(r["away_score"] - r["away_pred_points"])
    seasons = sorted(by_season)
    out: dict[int, dict[str, float]] = {}
    for i, season in enumerate(seasons):
        prior = seasons[:i]
        if not prior:
            out[season] = {"home": 0.0, "away": 0.0}
            continue
        window = prior[-5:]
        result = {}
        for side in ("home", "away"):
            w_vals = [v for s in window for v in by_season[s][side]]
            a_vals = [v for s in prior for v in by_season[s][side]]
            rate5 = float(np.mean(w_vals)) if w_vals else 0.0
            n5 = len(w_vals)
            rate_all = float(np.mean(a_vals)) if a_vals else 0.0
            n_all = len(a_vals)
            result[side] = (n5 * rate5 + HFA_SHRINKAGE_K * rate_all) / (n5 + HFA_SHRINKAGE_K)
        out[season] = result
    return out


def points_fav_dog_bias_current(game_log: list[dict]) -> float:
    """Cumulative-to-date favorite bias (predict_points()'s OWN predicted margin decides
    "favorite", not the market) -- a single current value, not a per-game history, since
    this only feeds LIVE upcoming-game predictions. The underdog's bias is its negative
    (research/edge_signal_test_v32/v33 found these symmetric to 3 decimals)."""
    fav_sum, fav_n = 0.0, 0
    for r in sorted(game_log, key=lambda r: (r["season"], r["week"])):
        hp, ap = r.get("home_pred_points"), r.get("away_pred_points")
        if hp is None or ap is None:
            continue
        if hp > ap:
            fav_sum += r["home_score"] - hp
            fav_n += 1
        elif ap > hp:
            fav_sum += r["away_score"] - ap
            fav_n += 1
    return (fav_sum / fav_n) if fav_n else 0.0


def historical_score_frequency(played_games: list[dict], first_season: int = SCORE_HIST_FIRST_SEASON) -> dict[int, float]:
    """P(a team's final score == s) for s in 0..MAX_TEAM_SCORE, from every team-game's own
    final score since `first_season` -- the modern PAT-distance era. Both home and away
    scores counted (one entry per team per game)."""
    counts: dict[int, int] = defaultdict(int)
    n = 0
    for g in played_games:
        try:
            season = int(g["season"])
            home_pts, away_pts = float(g["home_score"]), float(g["away_score"])
        except (ValueError, TypeError):
            continue
        if season < first_season:
            continue
        for pts in (home_pts, away_pts):
            s = int(round(pts))
            if 0 <= s <= MAX_TEAM_SCORE:
                counts[s] += 1
                n += 1
    if not n:
        return {}
    return {s: counts.get(s, 0) / n for s in range(MAX_TEAM_SCORE + 1)}


def _raw_score_distribution(mean: float, sd: float) -> np.ndarray:
    """Step 1: Normal(mean, sd) discretized to integers 0..MAX_TEAM_SCORE via the
    continuity correction; mass outside that range (a team can't score negative, and a 61+
    point game is vanishingly rare) folds into the boundary bins."""
    from scipy.stats import norm
    edges = np.arange(-0.5, MAX_TEAM_SCORE + 1.5, 1.0)
    cdf = norm.cdf(edges, loc=mean, scale=sd)
    probs = np.diff(cdf)
    probs[0] += norm.cdf(-0.5, loc=mean, scale=sd)
    probs[-1] += 1 - norm.cdf(MAX_TEAM_SCORE + 0.5, loc=mean, scale=sd)
    return probs / probs.sum()


def _reweight_score_by_history(raw: np.ndarray, hist_freq: dict[int, float]) -> np.ndarray:
    """Step 2: multiply by the real historical per-score frequency, renormalize. A score
    with zero historical precedent gets a tiny floor instead of an outright zero."""
    hist = np.array([hist_freq.get(s, 0.0) for s in range(len(raw))])
    weighted = raw * (hist + 1e-6)
    return weighted / weighted.sum()


def _implied_home_score_win_prob(home_dist: np.ndarray, away_dist: np.ndarray) -> float:
    """P(home score > away score), treating the two distributions as INDEPENDENT."""
    away_cdf_below = np.cumsum(away_dist) - away_dist
    return float(np.sum(home_dist * away_cdf_below))


def build_score_distributions(
    mean_home: float, mean_away: float, sd_home: float, sd_away: float,
    hist_freq: dict[int, float], target_home_win_prob: float,
    tol: float = 0.0005, max_iter: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Full pipeline: Step 1 + Step 2 for both sides, then bisect on a symmetric mean-shift
    so the pair's implied win probability matches `target_home_win_prob` exactly (within
    `tol`) -- see this section's own module-level docstring for why. Returns (home_dist,
    away_dist), each a MAX_TEAM_SCORE+1-length probability array summing to 1."""
    lo, hi = -21.0, 21.0  # generous bracket; a 3-score swing in mean is already extreme
    h_dist = a_dist = None
    mid = 0.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        h_dist = _reweight_score_by_history(_raw_score_distribution(mean_home + mid, sd_home), hist_freq)
        a_dist = _reweight_score_by_history(_raw_score_distribution(mean_away - mid, sd_away), hist_freq)
        p = _implied_home_score_win_prob(h_dist, a_dist)
        if abs(p - target_home_win_prob) < tol:
            break
        if p < target_home_win_prob:
            lo = mid
        else:
            hi = mid
    return h_dist, a_dist


def score_dist_to_whole_percentages(dist: np.ndarray) -> list[dict[str, int]]:
    """Largest-remainder (Hamilton) apportionment -- integer percentages that sum to EXACTLY
    100, instead of naive rounding (which can land on 99 or 101). Only nonzero bins are kept
    (a 61-entry array of mostly-zero percentages isn't useful to ship to the frontend)."""
    raw_pct = dist * 100
    floors = np.floor(raw_pct).astype(int)
    remainder = 100 - int(floors.sum())
    order = np.argsort(-(raw_pct - floors))
    out = floors.copy()
    for i in range(max(remainder, 0)):
        out[order[i]] += 1
    return [{"points": int(s), "pct": int(out[s])} for s in range(len(out)) if out[s] > 0]


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
# Display-only "City, ST" / "City, Country" for every stadium that can show up as `stadium`
# on a live or backfilled game row -- the 32 current teams' own buildings (names taken
# verbatim from nflverse's own 2026 games.csv, which is also what every card displays, so
# this table only ever needs to AGREE with that text, never predict it) plus every known
# neutral/international-series venue from _NEUTRAL_SITE_STADIUMS above. Keyed by stadium
# NAME, not team -- a neutral-site game's `stadium` is the actual venue, not the home team's
# regular building, so this lookup works unchanged for both cases; an unmapped name (a new
# venue, or nflverse renaming one) just shows no location rather than guessing.
STADIUM_LOCATIONS = {
    "State Farm Stadium": "Glendale, AZ", "Mercedes-Benz Stadium": "Atlanta, GA",
    "M&T Bank Stadium": "Baltimore, MD", "Highmark Stadium": "Orchard Park, NY",
    "Bank of America Stadium": "Charlotte, NC", "Soldier Field": "Chicago, IL",
    "Paycor Stadium": "Cincinnati, OH", "Huntington Bank Field": "Cleveland, OH",
    "AT&T Stadium": "Arlington, TX", "Empower Field at Mile High": "Denver, CO",
    "Ford Field": "Detroit, MI", "Lambeau Field": "Green Bay, WI",
    "Reliant Stadium": "Houston, TX", "NRG Stadium": "Houston, TX",
    "Lucas Oil Stadium": "Indianapolis, IN", "EverBank Stadium": "Jacksonville, FL",
    "GEHA Field at Arrowhead Stadium": "Kansas City, MO", "SoFi Stadium": "Inglewood, CA",
    "Allegiant Stadium": "Las Vegas, NV", "Hard Rock Stadium": "Miami Gardens, FL",
    "U.S. Bank Stadium": "Minneapolis, MN", "Gillette Stadium": "Foxborough, MA",
    "Caesars Superdome": "New Orleans, LA", "MetLife Stadium": "East Rutherford, NJ",
    "Lincoln Financial Field": "Philadelphia, PA", "Acrisure Stadium": "Pittsburgh, PA",
    "Lumen Field": "Seattle, WA", "Levi's Stadium": "Santa Clara, CA",
    "Raymond James Stadium": "Tampa, FL", "Nissan Stadium": "Nashville, TN",
    "Northwest Stadium": "Landover, MD",
    # retired/former names for current teams' own buildings -- still show up verbatim on
    # older rows (e.g. a relocated-game host or a backfilled past week), confirmed live
    # against games.csv rather than guessed.
    "TIAA Bank Stadium": "Jacksonville, FL", "Alltel Stadium": "Jacksonville, FL",
    "FirstEnergy Stadium": "Cleveland, OH", "Qualcomm Stadium": "San Diego, CA",
    "Cowboys Stadium": "Arlington, TX", "Dolphin Stadium": "Miami Gardens, FL",
    "Mercedes-Benz Superdome": "New Orleans, LA", "Louisiana Superdome": "New Orleans, LA",
    "University of Phoenix Stadium": "Glendale, AZ",
    # neutral/international-series venues (see _NEUTRAL_SITE_STADIUMS) -- names verified
    # live against games.csv's own actual text, not guessed (nflverse uses more than one
    # spelling for some of these across seasons).
    "Tottenham Hotspur Stadium": "London, England", "Tottenham Stadium": "London, England",
    "Wembley Stadium": "London, England", "Twickenham Stadium": "London, England",
    "Allianz Arena": "Munich, Germany", "FC Bayern Munich Stadium": "Munich, Germany",
    "Deutsche Bank Park": "Frankfurt, Germany",
    "Azteca Stadium": "Mexico City, Mexico", "Estadio Banorte": "Monterrey, Mexico",
    "Arena Corinthians": "São Paulo, Brazil", "Neo Química Arena": "São Paulo, Brazil",
    "Maracana Stadium": "Rio de Janeiro, Brazil",
    "Melbourne Cricket Ground": "Melbourne, Australia",
    "Bernabeu": "Madrid, Spain", "Stade de France": "Paris, France",
    "Rogers Centre": "Toronto, ON, Canada",
}

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
                    "spread_line": spread_line, "neutral": is_neutral_site(g),
                    "diffs": {m: ORIENTATION[m] * (rating[home][m] - rating[away][m]) for m in RATING_METRICS},
                    # same pregame rating[home]/rating[away] snapshot the diffs above use --
                    # reuses predict_points() directly instead of a second, parallel rating
                    # walk (unlike the research scripts, which deliberately stay
                    # self-contained) so this can never drift from what matchup_callouts()
                    # shows live. Feeds the score-distribution pipeline's bias corrections
                    # (see points_bias_by_season()/points_fav_dog_bias() below).
                    "home_pred_points": predict_points(rating[home], rating[away]),
                    "away_pred_points": predict_points(rating[away], rating[home]),
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


def matchup_callouts(
    ratings: dict[str, dict[str, float]], upcoming: list[dict], historical_bounds: dict[str, tuple[float, float]],
    diff_mu: dict[str, float] | None = None, diff_sd: dict[str, float] | None = None,
    qb_gap_upcoming: dict[tuple[str, int], float] | None = None,
    skill_out_upcoming: dict[tuple[str, int], float] | None = None,
    hfa_logit: float = 0.0,
    points_bias: dict[str, float] | None = None, sd_home: float | None = None, sd_away: float | None = None,
    hist_score_freq: dict[int, float] | None = None,
    streak_upcoming: dict[tuple[str, int], int] | None = None,
) -> dict[str, dict]:
    """For each upcoming game (already normalized home/away team codes), rule-based sentences
    for any offense-vs-opposing-defense pairing whose combined z-score clears
    MISMATCH_Z_THRESHOLD -- a real strength meeting a real weakness, not just noise. Keyed
    "AWAY@HOME" so the frontend can look it up directly from its own game row. (The Parlays
    tab's own matchup identifier no longer reads anything from here -- it classifies teams
    into tiers off a frozen ratings_display snapshot instead; see refresh()'s snapshot step.)

    `diff_mu`/`diff_sd` (optional, from refresh()'s own game_log): per-metric mean/std of the
    oriented home-minus-away diff, same standardization compute_backtest_scatter() uses for
    the Historical Power tab's backtest -- applying it here to each upcoming game's CURRENT
    rating diff gives the exact same composite `delta` that chart plots, just prospectively.

    `qb_gap_upcoming`/`skill_out_upcoming` (optional, keyed (team, week) -- always this
    week's data, since that's as far as the live injury report goes): QB_QUALITY_GAP_WEIGHT/
    SKILL_EPA_OUT_WEIGHT applied to these adjust `delta` the same way compute_backtest_
    scatter() does. `hfa_logit` (from home_field_logit_by_season(), a single value for the
    CURRENT season -- home-field advantage isn't a per-team/per-week thing): HFA_WEIGHT
    applied to this adds the same constant to every game this season. When diff_mu/diff_sd
    are given, each entry gets a `power_model` block ({favorite, prob, delta, delta_raw}) for
    the Moneyline Pick'em tab's Power Model pick.

    `points_bias` ({"home", "away", "fav_dog"}, from points_bias_by_season()/points_fav_dog_
    bias_current())/`sd_home`/`sd_away`/`hist_score_freq` (from historical_score_frequency()):
    when all given (alongside power_model), each entry also gets a `score_distribution`
    block (home/away mode, spread, total, and each side's full percentage-by-score curve) for
    the Against the Spread tab -- see build_score_distributions()'s own module-level
    docstring for the full two-step methodology and how it's reconciled with power_model's
    own win probability. NOT frozen-at-kickoff here (this function is stateless, no store
    access) -- refresh() applies that after calling this."""
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
        week = g.get("week")
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
        # Also doubles as the NO-WEATHER base the score-distribution pipeline below anchors
        # on -- its bias corrections were fit against the no-weather residual, so mixing in a
        # weather-adjusted mean here would double-count/mismatch what they were fit to.
        base_home = predict_points(ratings[home], ratings[away], None)
        base_away = predict_points(ratings[away], ratings[home], None)
        weather_points_delta = None
        if weather is not None and None not in (base_home, base_away, predicted_home_points, predicted_away_points):
            weather_points_delta = round(
                (predicted_home_points - base_home) + (predicted_away_points - base_away), 2
            )

        power_model = None
        if diff_mu is not None and diff_sd is not None and all(
            ratings[home].get(m) is not None and ratings[away].get(m) is not None for m in RATING_METRICS
        ):
            diffs = {m: ORIENTATION[m] * (ratings[home][m] - ratings[away][m]) for m in RATING_METRICS}
            # Each of the 8 stats' own standardized, weighted contribution to delta_raw --
            # same sum as before, just captured per-term instead of only the total, so the
            # frontend can show "how much of delta came from pass offense EPA" etc.
            rating_terms = {m: POWER_WEIGHTS[m] * ((diffs[m] - diff_mu[m]) / diff_sd[m]) for m in RATING_METRICS}
            delta_raw = sum(rating_terms.values())
            home_qb_gap = (qb_gap_upcoming or {}).get((home, week), 0.0)
            away_qb_gap = (qb_gap_upcoming or {}).get((away, week), 0.0)
            home_skill_out = (skill_out_upcoming or {}).get((home, week), 0.0)
            away_skill_out = (skill_out_upcoming or {}).get((away, week), 0.0)
            home_streak = (streak_upcoming or {}).get((home, week), 0)
            away_streak = (streak_upcoming or {}).get((away, week), 0)
            qb_diff = home_qb_gap - away_qb_gap
            skill_diff = home_skill_out - away_skill_out
            streak_diff = home_streak - away_streak
            # Neutral-site games (international series, etc. -- see is_neutral_site()) get
            # NO home-field term at all -- there's no real "home" side to credit.
            effective_hfa_logit = 0.0 if g.get("neutral") else hfa_logit
            delta = (delta_raw + QB_QUALITY_GAP_WEIGHT * qb_diff + SKILL_EPA_OUT_WEIGHT * skill_diff
                     + HFA_WEIGHT * effective_hfa_logit + MOMENTUM_WEIGHT * streak_diff)
            fav = home if delta > 0 else away if delta < 0 else None
            # Every non-EPA term `delta` actually applied to THIS matchup -- surfaced so the
            # frontend can call each one out explicitly instead of a prose narrative (removed
            # 2026-10-07 per user direction). Weighted terms are in delta-equivalent units,
            # signed home-minus-away like delta itself; raw values let the UI phrase its own
            # sentence (e.g. "PHI on a 3-game win streak") without re-deriving anything.
            power_model = {"favorite": fav, "prob": round(delta_win_prob(delta), 4),
                            "delta": round(delta, 4), "delta_raw": round(delta_raw, 4),
                            "neutral_site": bool(g.get("neutral")),
                            "hfa_term": round(HFA_WEIGHT * effective_hfa_logit, 4),
                            "qb_term": round(QB_QUALITY_GAP_WEIGHT * qb_diff, 4),
                            "skill_term": round(SKILL_EPA_OUT_WEIGHT * skill_diff, 4),
                            "momentum_term": round(MOMENTUM_WEIGHT * streak_diff, 4),
                            "home_qb_gap": round(home_qb_gap, 4), "away_qb_gap": round(away_qb_gap, 4),
                            "home_skill_out": round(home_skill_out, 4), "away_skill_out": round(away_skill_out, 4),
                            "home_streak": home_streak, "away_streak": away_streak,
                            # All 8 of POWER_WEIGHTS' own stats, individually -- the frontend
                            # currently only charts the 4 EPA ones (rush/pass off+def), but
                            # points/turnovers are included too rather than arbitrarily left
                            # out of an otherwise-complete breakdown.
                            "rating_terms": {m: round(v, 4) for m, v in rating_terms.items()}}

        score_distribution = None
        if (power_model is not None and points_bias is not None and sd_home and sd_away and hist_score_freq
                and base_home is not None and base_away is not None):
            # Same neutral-site exception as the HFA term above -- no real home/away
            # scoring asymmetry to correct for when neither side is actually at home.
            hb = 0.0 if g.get("neutral") else points_bias.get("home", 0.0)
            ab = 0.0 if g.get("neutral") else points_bias.get("away", 0.0)
            fb = points_bias.get("fav_dog", 0.0)
            mean_home = base_home + hb + (fb if base_home > base_away else -fb)
            mean_away = base_away + ab + (fb if base_away > base_home else -fb)
            target_home_p = power_model["prob"] if power_model["favorite"] == home else (
                1 - power_model["prob"] if power_model["favorite"] == away else 0.5)
            h_dist, a_dist = build_score_distributions(mean_home, mean_away, sd_home, sd_away, hist_score_freq, target_home_p)
            h_mode, a_mode = int(np.argmax(h_dist)), int(np.argmax(a_dist))
            score_distribution = {
                "home_mode": h_mode, "away_mode": a_mode,
                "spread": a_mode - h_mode,  # home-spread convention: negative = home favored, matches el.spread_home
                "total": h_mode + a_mode,
                "home_pct": score_dist_to_whole_percentages(h_dist),
                "away_pct": score_dist_to_whole_percentages(a_dist),
            }

        out[f"{away}@{home}"] = {
            "home": home, "away": away, "game_id": g.get("game_id"), "gameday": gameday,
            "stadium": g.get("stadium"), "stadium_location": STADIUM_LOCATIONS.get(g.get("stadium"), ""),
            "neutral_site": bool(g.get("neutral")),
            "home_ratings": ratings[home], "away_ratings": ratings[away],
            "home_ratings_display": ratings_display.get(home), "away_ratings_display": ratings_display.get(away),
            "callouts": callouts,
            "predicted_home_points": predicted_home_points,
            "predicted_away_points": predicted_away_points,
            "weather": weather,
            "weather_points_delta": weather_points_delta,
            "power_model": power_model,
            "score_distribution": score_distribution,
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


def season_records(played_games: list[dict], season: int) -> dict[str, dict[str, int]]:
    """Each team's win-loss-tie record for ONE season, straight off final scores -- same
    normalize_team() mapping `ratings`'s own keys use, so the frontend can look a team up
    directly without a second translation step. Ties count for neither side (NFL regular-
    season ties are rare but real, e.g. overtime)."""
    records: dict[str, dict[str, int]] = {}
    for g in played_games:
        if int(g["season"]) != season:
            continue
        try:
            home, away = normalize_team(g["home_team"]), normalize_team(g["away_team"])
        except UnknownTeamError:
            continue
        home_pts, away_pts = float(g["home_score"]), float(g["away_score"])
        records.setdefault(home, {"wins": 0, "losses": 0, "ties": 0})
        records.setdefault(away, {"wins": 0, "losses": 0, "ties": 0})
        if home_pts > away_pts:
            records[home]["wins"] += 1
            records[away]["losses"] += 1
        elif away_pts > home_pts:
            records[away]["wins"] += 1
            records[home]["losses"] += 1
        else:
            records[home]["ties"] += 1
            records[away]["ties"] += 1
    return records


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


def _qb_out_doubtful_by_week(current_season: int) -> dict[tuple[str, int, int], bool]:
    """{(team, season, week): True} for every (team, season, week) where that team had a QB
    listed Out or Doubtful on nflverse's weekly injury report -- a proxy for "the backup
    probably played" (the dataset has no direct starter-confirmation field, so this is the
    closest available signal). Keys are only ever present when True; absence means either
    "healthy that week" or "no report available" (seasons before FIRST_INJURY_SEASON, which
    404 and are silently skipped) -- both read the same way downstream (no flag raised).
    Completed seasons are cached locally (nflhub/data/team_ratings/injuries_{season}.csv),
    same convention as _phase_stats_for_season's play-by-play cache; only the CURRENT season
    is fetched fresh every call. Same contemporary-team-code convention as games.csv (STL/SD/
    OAK pre-relocation, WAS not WSH) -- normalize_team() handles it, same as everywhere else
    in this module."""
    out: dict[tuple[str, int, int], bool] = {}
    for season in range(FIRST_INJURY_SEASON, current_season + 1):
        cache_path = os.path.join(DATA_DIR, f"injuries_{season}.csv")
        if season < current_season and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                text = f.read()
        else:
            resp = requests.get(INJURIES_URL.format(season=season), timeout=TIMEOUT)
            if resp.status_code == 404:
                continue  # no report published for this season
            resp.raise_for_status()
            text = resp.text
            if season < current_season:
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(text)
        for row in csv.DictReader(io.StringIO(text)):
            if row.get("game_type") != "REG" or row.get("position") != "QB":
                continue
            if row.get("report_status") not in ("Out", "Doubtful"):
                continue
            try:
                team = normalize_team(row["team"])
                week = int(row["week"])
            except (UnknownTeamError, ValueError, TypeError):
                continue
            out[(team, season, week)] = True
    return out


_ESPN_QB_OUT_STATUSES = {"Out", "Doubtful"}


def _primary_qb_out_live(team: str, primary_id: str | None, passer_names: dict[str, str],
                          injuries: dict[str, list[dict]] | None) -> bool:
    """True when `team`'s current_primary passer (identified by name -- nflverse's
    passer_player_id has no equivalent on ESPN's side) is listed at QB with status Out or
    Doubtful on ESPN's live injury feed (nfl_schedule.fetch_injuries(), the same feed the
    game-card's QB chip reads). Only used for refresh()'s own live/upcoming gate -- see its
    comment for why this stays a separate source from the nflverse-based historical path.

    Matches on LAST NAME only: nflverse spells passer names "F.Lastname" (e.g.
    "C.Williams", no space -- verified against a live pull), ESPN spells them
    "Firstname Lastname" ("Caleb Williams"), so there's no full-name string that's equal
    on both sides. Same last-name-only approach js/pickem.js's elwayStaleQb() already uses
    for a different source pairing with the same mismatch."""
    nflverse_name = passer_names.get(primary_id or "")
    if not nflverse_name or "." not in nflverse_name:
        return False
    target_last = nflverse_name.split(".", 1)[-1].strip().lower()
    for inj in (injuries or {}).get(team, []):
        if inj.get("position") != "QB":
            continue
        espn_name = _norm_player_name(inj.get("player", ""))
        espn_last = espn_name.split()[-1] if espn_name else ""
        if espn_last != target_last:
            continue
        if inj.get("status") in _ESPN_QB_OUT_STATUSES:
            return True
    return False


# Mid-game QB injury: a DIFFERENT signal from _qb_out_doubtful_by_week above (which only
# ever sees PRE-game report status) -- a starter healthy enough to start, then hurt partway
# through. Ported from research/edge_signal_test_v34_midgame_qb_injury.py once that
# investigation confirmed it as a real, statistically significant (z=3.36, p<0.001)
# contributor to misses: when the MODEL'S OWN FAVORITE's QB goes down mid-game, that game
# misses 2.34% of the time vs. 1.11% for an otherwise-identical hit -- small in absolute
# terms (43 of 1836 misses, 2007-2025) but real and mechanistically unknowable pregame,
# hence descriptive/filterable here rather than folded into `delta` itself.
MIDGAME_MIN_BACKUP_ATTEMPTS = 3  # filters out one-off gadget/trick-play passer changes


def _compute_midgame_qb_injury_season(season: int) -> pd.DataFrame:
    """One row per (team, week) where that team's starting passer changed mid-game outside
    garbage time (WP_LO/WP_HI) AND nflverse's own play text explicitly says the departing
    passer "was injured during the play" -- both signals required, see module docstring
    above. Returns an empty-but-correctly-columned frame on total failure (caller treats
    that as "nothing found," not a crash)."""
    cols = ["game_id", "season", "week", "season_type", "posteam", "play_type", "wp",
            "passer_player_id", "passer_player_name", "game_seconds_remaining", "desc"]
    df = pd.read_csv(PBP_URL.format(season=season), compression="gzip", usecols=cols, low_memory=False)
    df = df[df["season_type"] == "REG"]
    pass_df = df[(df["play_type"] == "pass") & df["passer_player_id"].notna()].copy()
    pass_df = pass_df.sort_values(["game_id", "posteam", "game_seconds_remaining"], ascending=[True, True, False])

    rows = []
    for (gid, team_raw), grp in pass_df.groupby(["game_id", "posteam"], sort=False):
        passers = grp["passer_player_id"].tolist()
        starter_id = passers[0]
        switch_idx = next((i for i, p in enumerate(passers) if p != starter_id), None)
        if switch_idx is None:
            continue
        new_id = passers[switch_idx]
        backup_attempts = sum(1 for p in passers[switch_idx:] if p == new_id)
        if backup_attempts < MIDGAME_MIN_BACKUP_ATTEMPTS:
            continue
        wp_at_switch = grp["wp"].iloc[switch_idx - 1] if switch_idx > 0 else grp["wp"].iloc[0]
        if pd.isna(wp_at_switch) or not (WP_LO <= wp_at_switch <= WP_HI):
            continue  # game already decided when the switch happened

        starter_name = grp["passer_player_name"].iloc[0]
        game_desc = df.loc[df["game_id"] == gid, "desc"].dropna()
        name_pat = re.escape(str(starter_name))
        injured = bool(game_desc.str.contains(
            rf"{re.escape(str(team_raw))}-\d+-{name_pat} was injured", regex=True, na=False
        ).any())
        if not injured:
            continue
        try:
            team = normalize_team(team_raw)
        except UnknownTeamError:
            continue
        rows.append({"team": team, "week": int(grp["week"].iloc[0])})
    return pd.DataFrame(rows, columns=["team", "week"])


def midgame_qb_injury_by_week(current_season: int) -> dict[tuple[str, int, int], bool]:
    """{(team, season, week): True} for every (team, season, week) with a confirmed mid-game
    QB injury (see _compute_midgame_qb_injury_season). Completed seasons cached (nflhub/data/
    team_ratings/midgame_qb_injury_{season}.parquet), same convention as _passer_game_stats;
    only the current season re-downloads. A per-season fetch failure is logged and skipped,
    never allowed to fail the whole historical refresh over one bad season."""
    out: dict[tuple[str, int, int], bool] = {}
    for season in range(FIRST_SEASON, current_season + 1):
        cache_path = os.path.join(DATA_DIR, f"midgame_qb_injury_{season}.parquet")
        if season < current_season and os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
        else:
            try:
                df = _compute_midgame_qb_injury_season(season)
            except Exception:  # noqa: BLE001 -- best-effort, same soft-fail convention as the rest of this module
                log.warning("midgame QB injury detection failed for %s", season, exc_info=True)
                continue
            if season < current_season:
                df.to_parquet(cache_path)
        for _, row in df.iterrows():
            out[(row["team"], season, int(row["week"]))] = True
    return out


def compute_backtest_scatter(
    game_log: list[dict], qb_out: dict[tuple[str, int, int], bool] | None = None,
    qb_gap: dict[tuple[str, int, int], float] | None = None,
    skill_out: dict[tuple[str, int, int], float] | None = None,
    hfa_by_season: dict[int, float] | None = None,
    midgame_qb_injury: dict[tuple[str, int, int], bool] | None = None,
    streak: dict[tuple[str, int, int], int] | None = None,
) -> list[dict]:
    """Turns compute_ratings()'s optional `game_log` into ready-to-plot rows for the
    Historical Power tab's backtest scatter: market spread vs. the CURRENT production
    POWER_WEIGHTS applied to each game's pre-game rating diffs, standardized ONCE across the
    whole dataset (not walk-forward -- this isn't re-deriving weights, just showing what
    today's weights would have said about each past game), colored by whether the model's
    favorite actually won. Same idea as research/edge_signal_test_v13_new_window_weights.py's
    final "fit on all data" step, applied for display instead of re-fitting.

    `qb_out` (from _qb_out_doubtful_by_week, optional): tags each row with whether the home/
    away team had a QB Out/Doubtful that week, so the frontend can filter on it.

    `qb_gap`/`skill_out`/`hfa_by_season` (from qb_quality_gap_data()/skill_epa_out_by_team_
    week()/home_field_logit_by_season(), optional): QB_QUALITY_GAP_WEIGHT/SKILL_EPA_OUT_
    WEIGHT/HFA_WEIGHT applied to these ADD to the pure 8-stat composite below, producing the
    `delta` this function actually returns -- `delta_raw` keeps the unadjusted composite for
    comparison. See each weight's own docstring for why it's a real, data-driven correction.

    `midgame_qb_injury` (from midgame_qb_injury_by_week, optional): tags each row with
    whether the home/away team's starter went down mid-game (confirmed, see that function's
    docstring) -- display/filter-only, NOT folded into `delta` (unlike qb_gap/skill_out),
    since it's unknowable before the game even starts.

    `streak` (from team_streak_by_week, optional): MOMENTUM_WEIGHT applied to the signed
    win/loss streak difference ADDS to delta like qb_gap/skill_out/hfa -- see that weight's
    own docstring for why this one, unlike the mid-game injury flag above, survived the bar
    to be folded in rather than staying display-only."""
    if not game_log:
        return []
    qb_out = qb_out or {}
    qb_gap = qb_gap or {}
    skill_out = skill_out or {}
    hfa_by_season = hfa_by_season or {}
    midgame_qb_injury = midgame_qb_injury or {}
    streak = streak or {}
    mu = {m: float(np.mean([r["diffs"][m] for r in game_log])) for m in RATING_METRICS}
    sd = {m: float(np.std([r["diffs"][m] for r in game_log])) or 1.0 for m in RATING_METRICS}

    out = []
    for r in game_log:
        delta_raw = sum(POWER_WEIGHTS[m] * ((r["diffs"][m] - mu[m]) / sd[m]) for m in RATING_METRICS)
        qb_gap_diff = qb_gap.get((r["home"], r["season"], r["week"]), 0.0) - qb_gap.get((r["away"], r["season"], r["week"]), 0.0)
        skill_diff = skill_out.get((r["home"], r["season"], r["week"]), 0.0) - skill_out.get((r["away"], r["season"], r["week"]), 0.0)
        hfa = 0.0 if r.get("neutral") else HFA_WEIGHT * hfa_by_season.get(r["season"], 0.0)
        streak_diff = streak.get((r["home"], r["season"], r["week"]), 0) - streak.get((r["away"], r["season"], r["week"]), 0)
        delta = (delta_raw + QB_QUALITY_GAP_WEIGHT * qb_gap_diff + SKILL_EPA_OUT_WEIGHT * skill_diff
                 + hfa + MOMENTUM_WEIGHT * streak_diff)
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
            "spread": round(r["spread_line"], 1), "delta": round(delta, 4), "delta_raw": round(delta_raw, 4),
            "outcome": outcome,
            "home_qb_out": qb_out.get((r["home"], r["season"], r["week"]), False),
            "away_qb_out": qb_out.get((r["away"], r["season"], r["week"]), False),
            "home_qb_injured_ingame": midgame_qb_injury.get((r["home"], r["season"], r["week"]), False),
            "away_qb_injured_ingame": midgame_qb_injury.get((r["away"], r["season"], r["week"]), False),
            "home_streak": streak.get((r["home"], r["season"], r["week"]), 0),
            "away_streak": streak.get((r["away"], r["season"], r["week"]), 0),
            "neutral": bool(r.get("neutral")),
        })
    return out


def compute_calibration_stats(scatter: list[dict]) -> dict | None:
    """Old (delta_raw) vs. new (delta) model comparison, computed fresh every refresh from
    the same backtest_scatter rows so it can never drift from what the chart itself shows.
    Standard ongoing monitoring metric as of 2026-10 (see the module docstring's QB/skill
    health adjustment section) -- NOT just hit-rate, which the adjustment was never expected
    to move much (walk-forward backtests all came back statistically flat, research/
    edge_signal_test_v23/v24/v25_*.py), but whether it's SELECTIVELY reducing confidence
    (|delta|) on games the old model actually got wrong, vs. games it got right. That's the
    real goal of folding current-week health context into delta -- honest uncertainty, not a
    better point pick. A working adjustment shows a HIGHER confidence-reduction rate on
    misses than on hits; a meaningless one shows the same rate on both, since adding ANY
    nonzero-variance correction mechanically inflates average |delta| a little either way
    (adding noise to a number tends to push its magnitude up, regardless of direction) --
    the RATE comparison is what's actually diagnostic, not the raw before/after magnitude.

    Returns None if there's nothing gradeable yet (e.g. a fresh/empty dataset). Every count
    here includes every graded game ever (2007+), not just seasons with QB/skill data --
    pre-2009 games have delta == delta_raw by construction (no injury data exists to adjust
    with), so they correctly land in neither the "reduced" nor "increased" bucket and dilute
    both rates toward whatever the no-adjustment baseline is, same as leaving them out of the
    denominator entirely would NOT do -- they're real games this model graded, counted
    honestly as "nothing changed" rather than quietly excluded."""
    def pick(delta: float) -> str | None:
        return "home" if delta > 0 else ("away" if delta < 0 else None)

    def winner(r: dict) -> str:
        return "home" if r["home_score"] > r["away_score"] else "away"

    graded = [r for r in scatter if r["home_score"] != r["away_score"] and r["delta_raw"] != 0 and r["delta"] != 0]
    if not graded:
        return None

    old_correct = [pick(r["delta_raw"]) == winner(r) for r in graded]
    new_correct = [pick(r["delta"]) == winner(r) for r in graded]
    n = len(graded)
    old_hits, new_hits = sum(old_correct), sum(new_correct)

    flips = [(r, oc, nc) for r, oc, nc in zip(graded, old_correct, new_correct) if pick(r["delta_raw"]) != pick(r["delta"])]
    improvements = sum(1 for _, oc, nc in flips if nc and not oc)
    regressions = sum(1 for _, oc, nc in flips if oc and not nc)

    misses = [r for r, oc in zip(graded, old_correct) if not oc]
    hits = [r for r, oc in zip(graded, old_correct) if oc]

    def reduced_rate(rows: list[dict]) -> tuple[int, int]:
        return sum(1 for r in rows if abs(r["delta"]) < abs(r["delta_raw"])), len(rows)

    miss_reduced, miss_n = reduced_rate(misses)
    hit_reduced, hit_n = reduced_rate(hits)

    z = p_value = None
    if miss_n >= 10 and hit_n >= 10:
        p1, p2 = miss_reduced / miss_n, hit_reduced / hit_n
        p_pool = (miss_reduced + hit_reduced) / (miss_n + hit_n)
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / miss_n + 1 / hit_n))
        if se > 0:
            z = (p1 - p2) / se
            p_value = math.erfc(abs(z) / math.sqrt(2))  # two-tailed normal p-value, no scipy dependency needed here

    return {
        "n": n,
        "old_hits": old_hits, "old_hit_rate": round(old_hits / n, 4),
        "new_hits": new_hits, "new_hit_rate": round(new_hits / n, 4),
        "net_games": new_hits - old_hits,
        "flips": len(flips), "improvements": improvements, "regressions": regressions,
        "miss_confidence_reduced": miss_reduced, "miss_confidence_n": miss_n,
        "miss_confidence_reduced_rate": round(miss_reduced / miss_n, 4) if miss_n else None,
        "hit_confidence_reduced": hit_reduced, "hit_confidence_n": hit_n,
        "hit_confidence_reduced_rate": round(hit_reduced / hit_n, 4) if hit_n else None,
        "calibration_z": round(z, 3) if z is not None else None,
        "calibration_p": round(p_value, 4) if p_value is not None else None,
    }


def refresh_historical(store, force: bool = False) -> str:
    """Rebuild the full historical (every completed season, pooled) power rankings dataset.
    Cheap after the first run -- every completed season's play-by-play (and passer-game
    data, see qb_quality_gap_data()) is already cached as parquet, so this just re-aggregates
    small local files for everything except skill_epa_history()'s player_stats.csv pull
    (nflverse serves that as one file covering every season, with no per-season URL to cache
    around -- see that function's own docstring). Rebuilt at most once/day anyway (same
    cadence as refresh() above), since completed seasons' data never changes except once a
    year at season end."""
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
    qb_out = _qb_out_doubtful_by_week(current_season)
    qb_gap, _primary, _trailing, _league_avg, _names = qb_quality_gap_data(played, current_season)
    skill_out = skill_epa_out_by_team_week(current_season)
    hfa_by_season = home_field_logit_by_season(played)
    midgame_qb_injury = midgame_qb_injury_by_week(current_season)
    streak = team_streak_by_week(played)
    scatter = compute_backtest_scatter(game_log, qb_out, qb_gap, skill_out, hfa_by_season, midgame_qb_injury, streak)
    calibration = compute_calibration_stats(scatter)

    store.kv_set("historical_power_rankings", json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(),
        "seasons": seasons_covered,
        "rankings": rankings,
        "backtest_scatter": scatter,
        "calibration": calibration,
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


def _week_matchups_asof(played_cutoff: list[dict], all_rows: list[dict], current_season: int, week: int) -> dict[str, dict]:
    """The Moneyline/ATS "matchup" block (power model pick, score distribution, etc.) for
    one week's own games, reconstructed from data as of right before that week's first
    kickoff -- the SAME pipeline refresh() runs live for upcoming games, just pointed at a
    week that's already been played instead of one that hasn't. Weather comes back None for
    any already-past gameday (fetch_forecast_weather's own ~16-day forecast window can't
    reach into the past), same graceful "no data" this project already shows for a game too
    far out to forecast -- not a bug, just this mechanism's natural domain.

    One known approximation, inherited from qb_quality_gap_data()/skill_epa_out_by_team_week()
    rather than introduced here: those two build their trailing-EPA state from the FULL
    current-season play-by-play fetch regardless of `played_cutoff`, so a backfilled week's
    QB/skill-health adjustment uses each player's trailing EPA as of TODAY, not strictly as
    of that week -- a small hindsight leak on an already-minor adjustment term, not a
    structural error in the ratings/bias/score-distribution pieces (which ARE properly
    cutoff-bound)."""
    game_log_cutoff: list[dict] = []
    raw_ratings, historical_bounds = compute_ratings(played_cutoff, current_season, game_log=game_log_cutoff)
    ratings: dict[str, dict[str, float]] = {}
    for raw_team, r in raw_ratings.items():
        try:
            ratings[normalize_team(raw_team)] = r
        except UnknownTeamError:
            continue

    diff_mu = diff_sd = None
    if game_log_cutoff:
        diff_mu = {m: float(np.mean([r["diffs"][m] for r in game_log_cutoff])) for m in RATING_METRICS}
        diff_sd = {m: float(np.std([r["diffs"][m] for r in game_log_cutoff])) or 1.0 for m in RATING_METRICS}

    week_games_raw = [
        r for r in all_rows
        if int(r["season"]) == current_season and str(r.get("week")) == str(week)
    ]
    week_games = []
    for r in week_games_raw:
        try:
            week_games.append({"home_team": normalize_team(r["home_team"]), "away_team": normalize_team(r["away_team"]),
                                "gameday": r.get("gameday"), "week": week, "game_id": r.get("game_id"),
                                "neutral": is_neutral_site(r), "stadium": r.get("stadium")})
        except (UnknownTeamError, ValueError, TypeError):
            continue

    qb_out_asof = _qb_out_doubtful_by_week(current_season)
    _qb_gap_hist, current_primary, current_trailing_epa, league_avg_epa, _names = qb_quality_gap_data(played_cutoff, current_season)
    qb_gap_week: dict[tuple[str, int], float] = {}
    for (team, season, wk), is_out in qb_out_asof.items():
        if season != current_season or wk != week or not is_out:
            continue
        primary = current_primary.get(team)
        if primary is None:
            continue
        tp = current_trailing_epa.get(primary, league_avg_epa)
        qb_gap_week[(team, wk)] = tp - league_avg_epa
    skill_out_by_tw = skill_epa_out_by_team_week(current_season)
    skill_out_week = {(team, wk): v for (team, season, wk), v in skill_out_by_tw.items() if season == current_season and wk == week}
    hfa_logit = home_field_logit_by_season(played_cutoff).get(current_season, 0.0)

    points_bias_by_season_map = points_bias_by_season(game_log_cutoff)
    bias_asof = points_bias_by_season_map.get(current_season, {"home": 0.0, "away": 0.0})
    points_bias_asof = {
        "home": bias_asof["home"], "away": bias_asof["away"],
        "fav_dog": points_fav_dog_bias_current(game_log_cutoff),
    }
    sd_home = sd_away = None
    if game_log_cutoff:
        home_resids = [r["home_score"] - r["home_pred_points"] for r in game_log_cutoff if r.get("home_pred_points") is not None]
        away_resids = [r["away_score"] - r["away_pred_points"] for r in game_log_cutoff if r.get("away_pred_points") is not None]
        sd_home = (float(np.std(home_resids)) or 1.0) if home_resids else None
        sd_away = (float(np.std(away_resids)) or 1.0) if away_resids else None
    hist_score_freq = historical_score_frequency(played_cutoff)

    return matchup_callouts(ratings, week_games, historical_bounds, diff_mu, diff_sd,
                             qb_gap_week, skill_out_week, hfa_logit,
                             points_bias_asof, sd_home, sd_away, hist_score_freq)


# Bump whenever matchup_callouts()'s/power_model's own output schema meaningfully changes
# (a new field, a corrected calculation) -- refresh_weekly_power_rankings' self-heal below
# recomputes a frozen week's `matchups` whenever its stored schema version is behind this
# one, so a fix like this one doesn't silently stay broken in weeks already backfilled
# before it shipped. Ratings/power_rankings/team_records are NOT versioned this way -- they
# don't change retroactively, only `matchups`' shape (and the live model logic feeding it)
# evolves. Current bump (2026-10-07): momentum terms, the non-EPA breakdown fields, and the
# neutral-site HFA fix + stadium/location fields were all missing from weeks backfilled
# before those shipped. v3: corrected STADIUM_LOCATIONS spellings for a few international
# venues (Azteca/Corinthians/Maracana/Melbourne) that were wrong or missing in v2. v4: added
# `rating_terms`, each of the 8 POWER_WEIGHTS stats' own individual weighted contribution to
# delta_raw (previously only the combined delta_raw was exposed).
MATCHUPS_SCHEMA_VERSION = 4


def refresh_weekly_power_rankings(store, all_rows: list[dict], current_season: int, current_week: int) -> None:
    """Backfills AND maintains a permanent per-week snapshot of the power rankings (kv
    "power_rankings_by_week", {season: {week: {...}}}) for the week dropdown's Power
    Rankings/Parlays history -- explicit user direction: reconstruct missing weeks once and
    store them, don't recompute on every view; each week's snapshot is "ratings as of the
    last refresh before that week's first kickoff," frozen permanently once computed, never
    touched again (the underlying games that defined it don't change in hindsight).

    "As of right before week W's first kickoff" has an exact, unambiguous definition given
    final historical results: every fully-completed PRIOR season, plus this season's weeks
    strictly before W -- no week-W (or later) game has a result yet at that point by
    definition, so this cutoff is identical whether reconstructed today or computed live
    back when week W actually started. That's what makes one-time reconstruction exact, not
    an approximation of what the live snapshot would have shown.

    Idempotent and incremental by construction: a week already present in the stored blob is
    never recomputed (this IS the "store so we don't recompute every time" behavior, not a
    separate backfill-vs-maintain code path) -- the very first run after this shipped simply
    found every past week missing and filled them all in one pass; every run after that only
    ever computes whichever single NEW week most recently had its first kickoff pass, if any.
    The CURRENT week stays unsnapshotted until its own first kickoff -- team_ratings' own
    live, reactive "teams"/"power_rankings" already serves as its pre-kickoff preview, so
    there's nothing to freeze yet."""
    existing = store.kv_get("power_rankings_by_week")
    try:
        by_week: dict[str, dict[str, dict]] = json.loads(existing) if existing else {}
    except (TypeError, ValueError):
        by_week = {}
    season_key = str(current_season)
    season_weeks = by_week.setdefault(season_key, {})

    changed = False
    for week in range(1, current_week + 1):
        week_key = str(week)
        existing_week = season_weeks.get(week_key)
        if existing_week is not None:
            # Ratings/power_rankings/parlay_teams are frozen forever once computed -- but a
            # field added to this snapshot AFTER a week was already frozen (e.g. team_records,
            # matchups) would otherwise never backfill into it. Self-heals those fields in
            # place without recomputing (or re-freezing) anything else about an already-frozen
            # week.
            records_cutoff = [
                r for r in all_rows
                if r.get("result") not in ("", "NA", None) and int(r["season"]) == current_season
                and int(r["week"]) < week
            ]
            if "team_records" not in existing_week:
                existing_week["team_records"] = season_records(records_cutoff, current_season)
                changed = True
            if existing_week.get("matchups_schema") != MATCHUPS_SCHEMA_VERSION:
                played_cutoff = [
                    r for r in all_rows
                    if r.get("result") not in ("", "NA", None) and int(r["season"]) >= FIRST_SEASON
                    and (int(r["season"]) < current_season or int(r["week"]) < week)
                ]
                existing_week["matchups"] = _week_matchups_asof(played_cutoff, all_rows, current_season, week)
                existing_week["matchups_schema"] = MATCHUPS_SCHEMA_VERSION
                changed = True
            continue

        this_week_days = [
            r.get("gameday") for r in all_rows
            if int(r["season"]) == current_season and str(r.get("week")) == str(week) and r.get("gameday")
        ]
        first_day = min(this_week_days) if this_week_days else None
        still_pregame = first_day is None or date.today().isoformat() < first_day
        if week == current_week and still_pregame:
            continue  # nothing to freeze yet -- live team_ratings IS this week's preview

        played_cutoff = [
            r for r in all_rows
            if r.get("result") not in ("", "NA", None) and int(r["season"]) >= FIRST_SEASON
            and (int(r["season"]) < current_season or int(r["week"]) < week)
        ]
        raw_ratings, historical_bounds = compute_ratings(played_cutoff, current_season)
        ratings: dict[str, dict[str, float]] = {}
        for raw_team, r in raw_ratings.items():
            try:
                ratings[normalize_team(raw_team)] = r
            except UnknownTeamError:
                continue
        power = power_rankings(ratings, historical_bounds)
        season_weeks[week_key] = {
            "generated": datetime.now(timezone.utc).isoformat(),
            "teams": ratings,
            "power_rankings": power,
            # same shape as parlay_ratings_snapshot's own "teams" field, for direct reuse by
            # the Parlays tab's history view without a second extraction step client-side.
            "parlay_teams": {
                t: {m: info["ratings_display"].get(m) for m in EPA_DISPLAY_METRICS}
                for t, info in power.items() if info.get("ratings_display")
            },
            # each team's record AS OF right before week W's first kickoff -- same cutoff as
            # everything else in this snapshot, so a browsed past week shows the record that
            # was actually true then, not today's.
            "team_records": season_records(played_cutoff, current_season),
            # Moneyline/ATS card data (power model, score distribution) for THIS week's own
            # games -- see _week_matchups_asof()'s own docstring for the cutoff/caveats.
            "matchups": _week_matchups_asof(played_cutoff, all_rows, current_season, week),
            "matchups_schema": MATCHUPS_SCHEMA_VERSION,
        }
        changed = True

    if changed:
        store.kv_set("power_rankings_by_week", json.dumps(by_week))


def _freeze_score_distributions(store, matchups: dict[str, dict], today: str) -> dict[str, dict]:
    """Freezes each game's `score_distribution` the moment its kickoff (gameday, date
    granularity -- same freeze-point convention as _refresh_parlay_snapshot above and
    refresh.py's own odds freezing, "good enough" per that precedent) has passed, so a live
    or just-finished game keeps showing its last PRE-kickoff score distribution instead of
    drifting if refresh() reruns later the same day -- explicit user direction, matching how
    the market line is already frozen at kickoff elsewhere in this project. Keyed by each
    matchup's own `game_id` (stable across weeks/seasons, unlike the "AWAY@HOME" dict key
    matchup_callouts() itself uses, which recurs every season). Snapshot is pruned to only
    this week's games each call, so it can't grow unbounded over a season."""
    existing = store.kv_get("score_distribution_snapshot")
    try:
        frozen: dict[str, dict] = json.loads(existing) if existing else {}
    except (TypeError, ValueError):
        frozen = {}

    changed = False
    for key, info in matchups.items():
        gid = info.get("game_id")
        gameday = info.get("gameday")
        if not gid or info.get("score_distribution") is None:
            continue
        past_kickoff = bool(gameday) and today >= gameday
        if past_kickoff:
            if gid in frozen:
                info["score_distribution"] = frozen[gid]
            else:
                frozen[gid] = info["score_distribution"]
                changed = True

    stale = [gid for gid in frozen if gid not in {m.get("game_id") for m in matchups.values()}]
    for gid in stale:
        del frozen[gid]
        changed = True

    if changed:
        store.kv_set("score_distribution_snapshot", json.dumps(frozen))
    return matchups


def refresh(store, force: bool = False, injuries: dict[str, list[dict]] | None = None) -> str:
    """Rebuild team EPA ratings at most once per day. `injuries` (ESPN's live feed, see
    _primary_qb_out_live()) is normally passed in by refresh.py, which already fetches it
    for the game-card QB chip earlier in the same run -- re-fetched here only for a
    standalone/test call that doesn't have it."""
    today = date.today().isoformat()
    if not force and store.kv_get("team_ratings_date") == today:
        return "cached"

    from . import nfl_schedule  # local import: avoid a hard dependency for callers that don't need it

    if injuries is None:
        injuries = nfl_schedule.fetch_injuries()

    current_season, current_week = nfl_schedule.current_week()

    resp = requests.get(GAMES_URL, timeout=TIMEOUT)
    resp.raise_for_status()
    all_rows = [r for r in csv.DictReader(io.StringIO(resp.text)) if r["game_type"] == "REG"]
    played = [r for r in all_rows if r.get("result") not in ("", "NA", None) and int(r["season"]) >= FIRST_SEASON]

    # game_log (same param compute_backtest_scatter() consumes on the Historical Power tab)
    # doubles here as the standardization basis for the Power Model pick below -- same mu/sd
    # per metric, just computed live instead of from refresh_historical()'s own full sweep.
    game_log: list[dict] = []
    raw_ratings, historical_bounds = compute_ratings(played, current_season, game_log=game_log)
    diff_mu = diff_sd = None
    if game_log:
        diff_mu = {m: float(np.mean([r["diffs"][m] for r in game_log])) for m in RATING_METRICS}
        diff_sd = {m: float(np.std([r["diffs"][m] for r in game_log])) or 1.0 for m in RATING_METRICS}

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
                              "gameday": r.get("gameday"), "week": int(r.get("week") or 0), "game_id": r.get("game_id"),
                              "neutral": is_neutral_site(r), "stadium": r.get("stadium")})
        except (UnknownTeamError, ValueError, TypeError):
            continue

    # QB/skill health adjustment for the Power Model pick (see QB_QUALITY_GAP_WEIGHT's own
    # docstring) -- qb_gap_upcoming only has entries for THIS week's teams with a QB
    # currently Out/Doubtful (no actual passer exists yet for a future game, so "league-
    # average backup" is the best available estimate of who plays; a healthy primary
    # starter gets gap=0, same as compute_backtest_scatter()'s historical rows).
    #
    # The LIVE gate here uses ESPN's injury feed (`injuries`, same one the game-card's QB
    # chip reads), not nflverse's weekly report that every historical/backtested path below
    # still uses: nflverse's `report_status` field routinely sits blank until the Friday
    # pregame report, days after ESPN already has a real answer (confirmed against Caleb
    # Williams' 2026 wk5 report -- ESPN had him Out Monday, nflverse's report_status was
    # still blank Wednesday). This is a timeliness swap for THIS week's live prediction
    # only, not an accuracy claim -- QB_QUALITY_GAP_WEIGHT was fit against, and every
    # backtested number still comes from, nflverse exclusively (ESPN's feed has no
    # historical archive to validate against, same reason weather comes from a separate
    # live source instead of nflverse).
    _qb_gap_hist, current_primary, current_trailing_epa, league_avg_epa, passer_names = qb_quality_gap_data(played, current_season)
    qb_gap_upcoming: dict[tuple[str, int], float] = {}
    for team, primary in current_primary.items():
        if not _primary_qb_out_live(team, primary, passer_names, injuries):
            continue
        tp = current_trailing_epa.get(primary, league_avg_epa)
        qb_gap_upcoming[(team, current_week)] = tp - league_avg_epa
    skill_out_by_tw = skill_epa_out_by_team_week(current_season)
    skill_out_upcoming = {(team, week): v for (team, season, week), v in skill_out_by_tw.items() if season == current_season}
    hfa_logit_current = home_field_logit_by_season(played).get(current_season, 0.0)
    streak_by_tsw = team_streak_by_week(played)
    streak_upcoming = {(team, week): v for (team, season, week), v in streak_by_tsw.items() if season == current_season}

    # Score-distribution pipeline (Against the Spread tab) -- reuses the SAME game_log
    # compute_ratings() already built above (now carries home_pred_points/away_pred_points,
    # see compute_ratings()'s own game_log.append), so this costs nothing extra beyond a
    # handful of cheap aggregations over data already in memory.
    points_bias_by_season_map = points_bias_by_season(game_log)
    current_points_bias = points_bias_by_season_map.get(current_season, {"home": 0.0, "away": 0.0})
    points_bias_live = {
        "home": current_points_bias["home"], "away": current_points_bias["away"],
        "fav_dog": points_fav_dog_bias_current(game_log),
    }
    sd_home = sd_away = None
    if game_log:
        home_resids = [r["home_score"] - r["home_pred_points"] for r in game_log if r.get("home_pred_points") is not None]
        away_resids = [r["away_score"] - r["away_pred_points"] for r in game_log if r.get("away_pred_points") is not None]
        sd_home = (float(np.std(home_resids)) or 1.0) if home_resids else None
        sd_away = (float(np.std(away_resids)) or 1.0) if away_resids else None
    hist_score_freq = historical_score_frequency(played)

    matchups = matchup_callouts(ratings, upcoming, historical_bounds, diff_mu, diff_sd,
                                 qb_gap_upcoming, skill_out_upcoming, hfa_logit_current,
                                 points_bias_live, sd_home, sd_away, hist_score_freq,
                                 streak_upcoming)
    matchups = _freeze_score_distributions(store, matchups, date.today().isoformat())
    power = power_rankings(ratings, historical_bounds)
    team_records = season_records(played, current_season)
    _refresh_parlay_snapshot(store, current_season, current_week, power, all_rows)
    refresh_weekly_power_rankings(store, all_rows, current_season, current_week)

    store.kv_set("team_ratings", json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(),
        "season": current_season,
        "week": current_week,
        "teams": ratings,
        "matchups": matchups,
        "power_rankings": power,
        "team_records": team_records,
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
