"""v21: refine the one tail finding that independently replicated (v19/v20: a short-handed
skill-position corps hurts win probability) by weighting each Out/Doubtful RB/WR/TE/FB by
USAGE instead of snap share -- touches (carries+receptions) or scrimmage yards (rushing+
receiving), trailing average, no lookahead. A player who plays 70% of snaps in two-TE sets
blocking is a very different loss than one who plays 70% of snaps as the every-down back --
touches/yards should separate "on the field a lot" from "the ball actually goes through
them a lot," which is closer to what "biggest impact" means.

Verified before building: nflverse publishes a single combined weekly player-stats file
(github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.csv,
confirmed live, 1999-2024/25, `carries`/`receptions`/`rushing_yards`/`receiving_yards` per
player per game) -- same retroactively-relabeled team-code convention as the play-by-play
files (LA/LAC/LV even pre-relocation, confirmed directly: 2015 rows show "LA" not "STL"),
handled the same way as everywhere else in this project via normalize_team(). Crucially,
this dataset ISN'T bounded by snap_counts' 2012 start (v17/v19/v20's limit) -- it covers the
injury report's full 2009+ window, so this version uses 2009-2025, not 2012-2025, a real
power increase on top of the weighting change itself.

Two weight schemes, tested side by side against the ORIGINAL snap-share version (v19/v20)
on the identical skill-group test (game outcome ~ delta + short/decimated dummy), each run
through the SAME early/late split-half replication check before trusting anything --
established as standard practice for this whole investigation, not optional.

Run: python research/edge_signal_test_v21_skill_touches_weighted.py
"""
from __future__ import annotations

import csv
import io
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
from nflhub.sources.edge_teams import UnknownTeamError, normalize_team  # noqa: E402

import importlib.util
def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

v14 = _load("v14", "edge_signal_test_v14_qb_injury_dropout.py")
v16 = _load("v16", "edge_signal_test_v16_unit_health.py")

CACHE_DIR = os.path.join(HERE, "cache")
FIRST_SEASON = 2009  # matches the injury report's own start -- not snap_counts' 2012 limit
LAST_SEASON = 2025
SPLIT_SEASON = 2018  # roughly even halves over 2009-2025 (9 vs 8 seasons)
SKILL_POSITIONS = {"RB", "WR", "TE", "FB"}


def _norm_name(name: str) -> str:
    name = (name or "").lower().strip()
    name = re.sub(r"[.'`]", "", name)
    name = re.sub(r"\s+(jr|sr|ii|iii|iv|v)\.?$", "", name)
    return re.sub(r"\s+", " ", name)


def load_skill_usage() -> tuple[dict, dict]:
    """{name_norm: [(season, week, touches, yards), ...]} sorted chronologically, plus
    {"touches"|"yards": league-average fallback for a player's first tracked game}."""
    path = os.path.join(CACHE_DIR, "player_stats.csv")
    if not os.path.exists(path):
        import requests
        resp = requests.get("https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.csv", timeout=90)
        resp.raise_for_status()
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(resp.text)

    player_games: dict[str, list] = defaultdict(list)
    touch_sum = yard_sum = n = 0.0
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("position") not in SKILL_POSITIONS or row.get("season_type") != "REG":
                continue
            try:
                season = int(row["season"])
            except (ValueError, TypeError):
                continue
            if not (FIRST_SEASON <= season <= LAST_SEASON):
                continue
            try:
                week = int(row["week"])
                touches = float(row["carries"] or 0) + float(row["receptions"] or 0)
                yards = float(row["rushing_yards"] or 0) + float(row["receiving_yards"] or 0)
            except (ValueError, TypeError):
                continue
            name = _norm_name(row.get("player_display_name") or row.get("player_name"))
            if not name:
                continue
            player_games[name].append((season, week, touches, yards))
            touch_sum += touches
            yard_sum += yards
            n += 1

    for name in player_games:
        player_games[name].sort()
    fallback = {"touches": touch_sum / n if n else 0.0, "yards": yard_sum / n if n else 0.0}
    return player_games, fallback


def load_skill_injury_rows() -> list[dict]:
    out = []
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        text = v16._cached_fetch_text(v16.INJURIES_URL.format(season=season), f"injuries_{season}.csv")
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
            out.append({"team": team, "season": season, "week": week, "name": _norm_name(row.get("full_name", ""))})
    return out


def weighted_severity(injury_rows, player_games, fallback, scheme: str) -> dict:
    """scheme: 'touches' or 'yards'. {(team, season, week): sum of each flagged player's own
    trailing average `scheme` value, no lookahead}."""
    import bisect
    out: dict[tuple, float] = defaultdict(float)
    idx_touch, idx_yard = 2, 3
    use_idx = idx_touch if scheme == "touches" else idx_yard
    for r in injury_rows:
        games = player_games.get(r["name"])
        weight = fallback[scheme]
        if games:
            keys = [(s, w) for s, w, _, _ in games]
            idx = bisect.bisect_left(keys, (r["season"], r["week"]))
            prior = games[:idx]
            if prior:
                weight = sum(g[use_idx] for g in prior) / len(prior)
        out[(r["team"], r["season"], r["week"])] += weight
    return out


def fit_logistic(X: np.ndarray, y: np.ndarray):
    n, p = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])

    def nll_grad(w):
        z = Xb @ w
        p_hat = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-12
        nll = -np.sum(y * np.log(p_hat + eps) + (1 - y) * np.log(1 - p_hat + eps))
        grad = Xb.T @ (p_hat - y)
        return nll, grad

    res = minimize(nll_grad, x0=np.zeros(p + 1), jac=True, method="L-BFGS-B")
    w = res.x
    z = Xb @ w
    p_hat = 1.0 / (1.0 + np.exp(-z))
    W = p_hat * (1 - p_hat)
    cov = np.linalg.inv((Xb * W[:, None]).T @ Xb)
    se = np.sqrt(np.diag(cov))
    return w, se, -res.fun


def run_outcome_test(games_subset, sev_by_tw, tag_key, delta_key="delta"):
    y = np.array([g["home_win"] for g in games_subset], dtype=float)
    delta = np.array([g[delta_key] for g in games_subset]).reshape(-1, 1)
    diffs = []
    for g in games_subset:
        hv = sev_by_tw.get((g["home"], g["season"], g["week"]), {}).get(tag_key, 0)
        av = sev_by_tw.get((g["away"], g["season"], g["week"]), {}).get(tag_key, 0)
        diffs.append(hv - av)
    dcol = np.array(diffs, dtype=float).reshape(-1, 1)
    if np.std(dcol) == 0:
        return None
    w_r, se_r, ll_r = fit_logistic(delta, y)
    w_f, se_f, ll_f = fit_logistic(np.hstack([delta, dcol]), y)
    lr = 2 * (ll_f - ll_r)
    p = 1 - chi2.cdf(lr, df=1)
    return w_f[2], se_f[2], p


def main():
    print("Loading skill-position usage (touches/yards, 2009-2025)...", file=sys.stderr)
    player_games, fallback = load_skill_usage()
    print(f"{len(player_games)} tracked skill players; league avg touches/game={fallback['touches']:.2f}, yards/game={fallback['yards']:.2f}")

    print("Loading skill-position injury rows...", file=sys.stderr)
    injury_rows = load_skill_injury_rows()

    print("Building game-level dataset...", file=sys.stderr)
    rows_all = v14.build_dataset()
    cols = [f"{m}_diff" for m in v14.RATING_METRICS]
    X_all = np.array([[r[c] for c in cols] for r in rows_all])
    mu_all, sd_all = X_all.mean(axis=0), X_all.std(axis=0)
    sd_all[sd_all == 0] = 1.0
    PW = v14.PRODUCTION_WEIGHTS
    for i, r in enumerate(rows_all):
        x = X_all[i]
        r["delta"] = sum(PW[m] * ((x[j] - mu_all[j]) / sd_all[j]) for j, m in enumerate(v14.RATING_METRICS))
    games = [r for r in rows_all if FIRST_SEASON <= r["season"] <= LAST_SEASON]
    print(f"{len(games)} games, {FIRST_SEASON}-{LAST_SEASON}\n")

    for scheme in ("touches", "yards"):
        print(f"=== scheme: {scheme} ===")
        sev = weighted_severity(injury_rows, player_games, fallback, scheme)
        vals = np.array(list(sev.values())) if sev else np.array([0.0])
        # team-weeks with NO flagged skill player never enter `sev` at all (defaultdict only
        # materializes on write) -- those are legitimately zero, so pad before taking
        # percentiles or p85 would be computed only over the already-nonzero tail.
        all_team_weeks = {(r[s], r["season"], r["week"]) for r in games for s in ("home", "away")}
        full_vals = np.array([sev.get(tw, 0.0) for tw in all_team_weeks])
        p85, p95 = np.percentile(full_vals, 85), np.percentile(full_vals, 95)
        print(f"  severity distribution: p85={p85:.2f} p95={p95:.2f} max={full_vals.max():.2f}")

        sev_by_tw = {tw: {"short": 1 if sev.get(tw, 0.0) >= p85 else 0,
                           "decimated": 1 if sev.get(tw, 0.0) >= p95 else 0}
                     for tw in all_team_weeks}

        for tag in ("short", "decimated"):
            full = run_outcome_test(games, sev_by_tw, tag)
            early = run_outcome_test([g for g in games if g["season"] < SPLIT_SEASON], sev_by_tw, tag)
            late = run_outcome_test([g for g in games if g["season"] >= SPLIT_SEASON], sev_by_tw, tag)
            if not (full and early and late):
                print(f"  {tag:9s}: degenerate, skipped")
                continue
            fc, _, fp = full
            ec, _, ep = early
            lc, _, lp = late
            same_sign = (ec < 0) == (lc < 0)
            verdict = "YES" if (same_sign and ep < 0.15 and lp < 0.15) else ("partial" if same_sign else "NO")
            print(f"  {tag:9s}: FULL coef={fc:+.3f} p={fp:.4f}   |  EARLY coef={ec:+.3f} p={ep:.4f}  "
                  f"LATE coef={lc:+.3f} p={lp:.4f}   replicates? {verdict}")
        print()


if __name__ == "__main__":
    main()
