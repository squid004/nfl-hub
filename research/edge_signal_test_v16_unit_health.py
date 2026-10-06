"""v16: does RELATIVE health of the 6 natural position groups (not just QB) explain where
the rating's own EWMA is currently wrong?

The rating (nflhub/sources/team_ratings.py compute_ratings()) is an alpha=0.2 EWMA of actual
performance -- it has no live injury awareness, but it DOES self-correct over ~4-5 games as
a unit's new (injured, or newly-healthy) performance accumulates. So an injury isn't a
constant blind spot: it's a TRANSIENT one, worst right after it happens (or right after a
return), and fading out as the EWMA catches up. This script builds that "relative health"
signal directly instead of a flat Out/Doubtful count:

Step 1: for each (team, position-group, week), count Out/Doubtful players in that group,
then compute a health EWMA of that same count (same alpha=0.2 half-life as the rating,
reset every season -- unlike the rating's own season_prior, injury status has no reason to
carry over an offseason). `health_gap = this_week's count - the EWMA going into this week`.
Negative = currently WORSE than what's been feeding the rating lately (a fresh injury);
positive = currently BETTER (a fresh return) -- both are exactly the windows where the
rating's own memory hasn't caught up yet.

Step 2: for each of the 4 EPA metrics, regress (actual this-game stat - the rating's own
PRE-GAME value for that team/metric, i.e. the rating's prediction error) against the health
gaps of every position group that plausibly touches it (OL/skill/QB -> offense, DL/LB/
secondary -> defense). A significant negative coefficient on a group's gap = "when this unit
is unusually battered relative to its own recent average, the rating overshoots actual
performance by about that many EPA/play" -- real, currently-unused signal.

Step 3 (only if Step 2 finds something): walk-forward backtest -- fit the health-gap
regression on train seasons only, apply the resulting adjustment to each test-season game's
pregame rating before computing the composite delta, compare hit rate against the
unadjusted model on the same held-out games.

Run: python research/edge_signal_test_v16_unit_health.py
"""
from __future__ import annotations

import csv
import io
import os
import sys
from collections import defaultdict

import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nflhub.sources.edge_teams import UnknownTeamError, normalize_team  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
INJURIES_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv"
SEASONS = range(2009, 2026)  # injury reports start 2009; stay in lockstep with that, not 2007
FIRST_INJURY_SEASON = 2009
EWMA_ALPHA = 0.2
SEASON_PRIOR_GAMES = 5
WP_LO, WP_HI = 0.05, 0.95
TEST_START_SEASON = 2013  # a few more train seasons needed here than v13 (6 extra regression inputs)
RATING_METRICS = (
    "rush_off_epa", "pass_off_epa", "rush_def_epa_allowed", "pass_def_epa_allowed",
    "points_off", "points_def_allowed", "turnovers_off", "turnovers_def_forced",
)
ORIENTATION = {
    "rush_off_epa": 1, "pass_off_epa": 1, "rush_def_epa_allowed": -1, "pass_def_epa_allowed": -1,
    "points_off": 1, "points_def_allowed": -1, "turnovers_off": -1, "turnovers_def_forced": 1,
}
PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "epa", "wp", "interception", "fumble_lost"]
PRODUCTION_WEIGHTS = {
    "rush_off_epa": 0.0850, "pass_off_epa": 0.1536, "rush_def_epa_allowed": 0.0572,
    "pass_def_epa_allowed": 0.0578, "points_off": 0.1587, "points_def_allowed": 0.1006,
    "turnovers_off": 0.0217, "turnovers_def_forced": 0.0000,
}

POSITION_GROUPS = {
    "qb": {"QB"},
    "ol": {"T", "G", "C"},
    "skill": {"RB", "WR", "TE", "FB"},
    "dl": {"DE", "DT"},
    "lb": {"LB"},
    "secondary": {"CB", "S"},
}
# Which groups plausibly touch which EPA metric -- OL/skill/QB on offense, DL/LB/secondary
# on defense. QB -> pass primarily (not rush: a backup QB's running ability isn't the issue
# here, and run-game EPA is driven much more by OL/RB).
METRIC_UNITS = {
    "rush_off_epa": ["ol", "skill"],
    "pass_off_epa": ["ol", "skill", "qb"],
    "rush_def_epa_allowed": ["dl", "lb"],
    "pass_def_epa_allowed": ["dl", "lb", "secondary"],
}


def _cached_fetch_text(url: str, cache_name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, cache_name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(path, "w", encoding="utf-8") as f:
        f.write(resp.text)
    return resp.text


def load_games() -> dict[str, dict]:
    text = _cached_fetch_text(GAMES_URL, "games.csv")
    games = {}
    for r in csv.DictReader(io.StringIO(text)):
        if r["game_type"] != "REG":
            continue
        try:
            season = int(r["season"])
        except ValueError:
            continue
        if season in SEASONS:
            games[r["game_id"]] = r
    return games


def team_game_phase_stats(season: int):
    import pandas as pd
    cache_path = os.path.join(CACHE_DIR, f"team_game_phase_turnovers_{season}.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    print(f"  downloading/parsing pbp {season}...", file=sys.stderr)
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
    tov = df.groupby(["game_id", "posteam"])["turnover"].sum().reset_index().rename(columns={"posteam": "team", "turnover": "turnovers_committed"})
    out = off[["game_id", "team", "rush_off_epa", "pass_off_epa"]].merge(tov, on=["game_id", "team"], how="left")
    out["turnovers_committed"] = out["turnovers_committed"].fillna(0.0)
    out = out.dropna(subset=["rush_off_epa", "pass_off_epa"])
    out.to_parquet(cache_path)
    return out


def load_position_group_counts() -> dict[tuple[str, int, int, str], int]:
    """{(team, season, week, group): count of DISTINCT players Out/Doubtful in that group}."""
    counts: dict[tuple[str, int, int, str], set] = defaultdict(set)
    pos_to_group = {}
    for grp, positions in POSITION_GROUPS.items():
        for p in positions:
            pos_to_group[p] = grp

    for season in range(FIRST_INJURY_SEASON, SEASONS.stop):
        text = _cached_fetch_text(INJURIES_URL.format(season=season), f"injuries_{season}.csv")
        for row in csv.DictReader(io.StringIO(text)):
            if row.get("game_type") != "REG":
                continue
            grp = pos_to_group.get(row.get("position"))
            if grp is None or row.get("report_status") not in ("Out", "Doubtful"):
                continue
            try:
                team = normalize_team(row["team"])
                week = int(row["week"])
            except (UnknownTeamError, ValueError, TypeError):
                continue
            counts[(team, season, week, grp)].add(row.get("gsis_id") or row.get("full_name"))
    return {k: len(v) for k, v in counts.items()}


def build_health_gaps(team_weeks: list[tuple[str, int, int]], raw_counts: dict) -> dict[tuple[str, int, int, str], float]:
    """`team_weeks`: sorted [(team, season, week), ...] for every team-week actually played
    (so bye weeks don't appear, but a week with ZERO Out/Doubtful in a group still does --
    needed for the EWMA to decay back down after a player returns, not just skip silently).
    Returns {(team, season, week, group): gap}, gap = this week's count - the EWMA of
    counts strictly BEFORE this week (reset at every season boundary -- see module
    docstring). First week of data for a team each season gets gap=0 (no prior memory to be
    stale relative to yet)."""
    ewma: dict[tuple[str, str], float | None] = {}
    last_season: dict[str, int] = {}
    gaps: dict[tuple[str, int, int, str], float] = {}
    for team, season, week in team_weeks:
        if last_season.get(team) is not None and last_season[team] != season:
            for grp in POSITION_GROUPS:
                ewma[(team, grp)] = None
        last_season[team] = season
        for grp in POSITION_GROUPS:
            count = raw_counts.get((team, season, week, grp), 0)
            prev = ewma.get((team, grp))
            gaps[(team, season, week, grp)] = 0.0 if prev is None else count - prev
            ewma[(team, grp)] = count if prev is None else (1 - EWMA_ALPHA) * prev + EWMA_ALPHA * count
    return gaps


def build_team_game_rows() -> list[dict]:
    """One row per (game, team) with that team's ACTUAL stat this game AND the rating's own
    PRE-GAME value for the same metric (no lookahead -- same rating walk as compute_ratings),
    for every RATING_METRIC. Also carries season/week/team for the health-gap join."""
    by_game: dict[str, dict[str, dict]] = defaultdict(dict)
    for season in SEASONS:
        for row in team_game_phase_stats(season).to_dict("records"):
            try:
                team_key = normalize_team(row["team"])
            except UnknownTeamError:
                continue
            by_game[row["game_id"]][team_key] = row

    games = load_games()
    rating: dict[str, dict] = defaultdict(lambda: {m: None for m in RATING_METRICS})
    season_ewma: dict[str, dict] = defaultdict(lambda: {m: None for m in RATING_METRICS})
    season_prior: dict[str, dict] = defaultdict(lambda: {m: None for m in RATING_METRICS})
    games_played: dict[str, int] = defaultdict(int)
    last_season: dict[str, int] = {}

    rows_out = []
    for gid, teams in sorted(by_game.items()):
        g = games.get(gid)
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

        prior_w: dict[str, float] = {}
        for team in (home, away):
            if last_season.get(team) is not None and last_season[team] != season:
                season_prior[team] = dict(season_ewma[team])
                season_ewma[team] = {m: None for m in RATING_METRICS}
                games_played[team] = 0
            last_season[team] = season
            games_played[team] += 1
            prior_w[team] = max(0.0, (SEASON_PRIOR_GAMES - games_played[team]) / SEASON_PRIOR_GAMES)

        week = int(g["week"])
        for team, opp, raw in ((home, away, home_raw), (away, home, away_raw)):
            if all(rating[team][m] is not None for m in RATING_METRICS):
                rows_out.append({
                    "game_id": gid, "season": season, "week": week, "team": team, "opp": opp,
                    **{f"{m}_actual": raw[m] for m in RATING_METRICS},
                    **{f"{m}_pregame": rating[team][m] for m in RATING_METRICS},
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

    return rows_out


def ols_with_stats(X: np.ndarray, y: np.ndarray, names: list[str]):
    """Plain OLS with intercept -- coefficients, standard errors, t-stats, two-tailed
    p-values (normal approx, n is in the thousands here so that's fine), R^2."""
    n = len(y)
    Xb = np.hstack([np.ones((n, 1)), X])
    beta, _, _, _ = np.linalg.lstsq(Xb, y, rcond=None)
    resid = y - Xb @ beta
    k = Xb.shape[1]
    sigma2 = (resid @ resid) / (n - k)
    cov = sigma2 * np.linalg.inv(Xb.T @ Xb)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(np.abs(t)))
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - (resid @ resid) / ss_tot
    print(f"  {'term':14s} {'coef':>9s} {'se':>8s} {'t':>7s} {'p':>8s}")
    print(f"  {'intercept':14s} {beta[0]:9.4f} {se[0]:8.4f} {t[0]:7.2f} {p[0]:8.4f}")
    for i, nm in enumerate(names):
        print(f"  {nm:14s} {beta[i+1]:9.4f} {se[i+1]:8.4f} {t[i+1]:7.2f} {p[i+1]:8.4f}")
    print(f"  n={n}  R^2={r2:.4f}")
    return beta, se, t, p


def main():
    print("Loading position-group injury counts...", file=sys.stderr)
    raw_counts = load_position_group_counts()

    print("Building per-team-game rating/actual rows...", file=sys.stderr)
    rows = build_team_game_rows()
    print(f"{len(rows)} team-game rows\n")

    team_weeks = sorted({(r["team"], r["season"], r["week"]) for r in rows})
    gaps = build_health_gaps(team_weeks, raw_counts)
    for r in rows:
        for grp in POSITION_GROUPS:
            r[f"gap_{grp}"] = gaps.get((r["team"], r["season"], r["week"], grp), 0.0)

    print("--- health-gap distribution sanity check ---")
    for grp in POSITION_GROUPS:
        vals = np.array([r[f"gap_{grp}"] for r in rows])
        print(f"  {grp:10s} mean={vals.mean():+.3f} std={vals.std():.3f} "
              f"min={vals.min():+.2f} max={vals.max():+.2f} nonzero={np.mean(vals != 0)*100:.1f}%")

    print("\n=== Step 2: does a unit's health gap explain the rating's pregame prediction error? ===")
    for metric, units in METRIC_UNITS.items():
        print(f"\n[{metric}]  residual = actual - pregame_rating  ~  " + " + ".join(f"gap_{u}" for u in units))
        y = np.array([r[f"{metric}_actual"] - r[f"{metric}_pregame"] for r in rows])
        X = np.array([[r[f"gap_{u}"] for u in units] for r in rows])
        ols_with_stats(X, y, [f"gap_{u}" for u in units])


if __name__ == "__main__":
    main()
