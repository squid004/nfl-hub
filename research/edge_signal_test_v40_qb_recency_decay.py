"""v40: tests two independent hypotheses about the QB-health correction, both raised during a
live bug report (the Oct 2026 BAL@ATL Details card mislabeled BAL's Lamar Jackson injury as
Atlanta's, traced upstream to `current_primary` still pointing at Cooper Rush over Michael
Penix Jr. even though Penix has started the last two games straight -- Rush just has a
cumulative-attempts head start from starting weeks 1-2 while Penix was hurt).

(1) RECENCY: production's `primary_so_far`/`current_primary` (team_ratings.qb_quality_gap_
data, same definition as v25 below) picks whoever has the most CUMULATIVE season attempts.
That's robust to a single-game injury blip (needs sustained volume to flip), but lags a
genuine in-season QB change whenever the eventual real starter opened the season hurt and the
fill-in got an early cumulative lead -- exactly the Penix/Rush case. Tests a TRAILING-WINDOW
definition instead (most starts in the last K games, ties broken toward the more recent
start), sweeping K against the cumulative baseline.

(2) DECAY + REINSERTION: qb_quality_gap compares the primary's flat, never-decaying CAREER-
average EPA against whoever's actually playing, and applies that AS A CONSTANT regardless of
how long the current passer has already been starting. But the team-level rolling rating
`delta` is already built from EWMA_ALPHA=0.2 (~4-game half-life, team_ratings.py's own
comment) -- by ~4-5 games into a backup's run, the rolling rating has ALREADY absorbed most of
his true quality, so the flat gap doubles up on a correction the rolling average is making on
its own. Tests replacing the static primary-EPA reference with a synthetic EWMA
("implied_quality") that updates toward whoever actually played each week using the SAME
alpha the team ratings use -- gap_to_apply = implied_quality_before_this_week - trailing_epa
(actual_passer_this_week). This has a clean property: in a team's very FIRST week of a QB
switch, implied_quality hasn't moved yet, so this reduces to exactly the same "full gap" as
production's static formula (meaning it's a strict generalization of the already-validated
case, not a replacement of it) -- it only differs in extended-duration/return situations,
precisely the dimension raised and never tested before. The same mechanism naturally produces
a REINSERTION boost for free: when the real primary returns after a long absence,
implied_quality is still sitting near the backup's level, so the gap flips sign (negative =
boost delta) instead of snapping to 0 the moment he's back (production's current behavior).

Both axes are varied independently (primary-definition x gap-construction) against a single
shared `delta` from frozen PRODUCTION POWER_WEIGHTS -- only the qb_gap term built on top of it
is being tested here, same scope v25/v26 originally validated. Walk-forward,
TEST_START_SEASON=2014, matching v25's own methodology (the script that validated
qb_quality_gap_diff for production in the first place).

Run: python research/edge_signal_test_v40_qb_recency_decay.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict, deque

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

import importlib.util
def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

v14 = _load("v14", "edge_signal_test_v14_qb_injury_dropout.py")

CACHE_DIR = os.path.join(HERE, "cache")
FIRST_SEASON = 2009
TEST_START_SEASON = 2014
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
WP_LO, WP_HI = 0.05, 0.95
PASS_COLS = ["game_id", "season", "week", "season_type", "posteam", "play_type", "epa", "wp", "passer_player_id"]
EWMA_ALPHA = 0.2  # same constant team_ratings.py uses for the team-level rolling ratings


def passer_game_stats(season: int) -> pd.DataFrame:
    """Same shape/cache as v25 -- reuses its already-downloaded per-season parquet files."""
    cache_path = os.path.join(CACHE_DIR, f"passer_game_{season}.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    print(f"  downloading/parsing pbp {season} for passer identity...", file=sys.stderr)
    df = pd.read_csv(PBP_URL.format(season=season), compression="gzip", usecols=PASS_COLS, low_memory=False)
    df = df[(df["season_type"] == "REG") & (df["play_type"] == "pass")]
    df = df.dropna(subset=["wp", "epa", "posteam", "passer_player_id"])
    df = df[(df["wp"] >= WP_LO) & (df["wp"] <= WP_HI)]
    out = df.groupby(["game_id", "posteam", "passer_player_id"]).agg(
        attempts=("epa", "size"), epa_sum=("epa", "sum")
    ).reset_index().rename(columns={"posteam": "team"})
    os.makedirs(CACHE_DIR, exist_ok=True)
    out.to_parquet(cache_path)
    return out


def load_passer_df() -> pd.DataFrame:
    print("Loading passer-level game stats (2009-2025)...", file=sys.stderr)
    frames = []
    for season in range(FIRST_SEASON, 2026):
        try:
            df = passer_game_stats(season)
        except Exception as exc:  # noqa: BLE001
            print(f"  {season}: unavailable ({exc}), skipping", file=sys.stderr)
            continue
        df = df.copy()
        df["season"] = season
        frames.append(df)
    passer_df = pd.concat(frames, ignore_index=True)

    games_meta = {r["game_id"]: (r["season"], r["week"]) for r in v14.build_dataset()}
    def week_of(gid, season):
        _, w = games_meta.get(gid, (None, None))
        if w is not None:
            return w
        try:
            return int(gid.split("_")[1])
        except (IndexError, ValueError):
            return None
    passer_df["week"] = [week_of(gid, s) for gid, s in zip(passer_df["game_id"], passer_df["season"])]
    passer_df = passer_df.dropna(subset=["week"])
    passer_df["week"] = passer_df["week"].astype(int)
    return passer_df


def build_game_order(passer_df: pd.DataFrame):
    """actual_passer(team,gid), game_team_epa(team,gid)->(epa_sum,attempts), and the single
    chronological (gid, team, season, week) walk order every no-lookahead computation below
    shares -- identical to v25's own construction."""
    actual_passer: dict[tuple[str, str], str] = {}
    game_team_epa: dict[tuple[str, str], tuple[float, int]] = {}
    for (gid, team), grp in passer_df.groupby(["game_id", "team"]):
        row = grp.loc[grp["attempts"].idxmax()]
        actual_passer[(gid, team)] = row["passer_player_id"]
        game_team_epa[(gid, team)] = (row["epa_sum"], row["attempts"])

    gt_meta = passer_df[["game_id", "team", "season", "week"]].drop_duplicates()
    gt_meta = gt_meta.sort_values(["season", "week", "game_id", "team"])
    game_order = list(gt_meta.itertuples(index=False, name=None))
    return actual_passer, game_team_epa, game_order


def build_primary_so_far(game_order, actual_passer, game_team_epa, mode: str):
    """mode='cumulative' reproduces production's qb_quality_gap_data() exactly (most
    cumulative attempts this season, falling back to last season's leader). mode='windowK'
    uses the last K actual-passer starts instead (any season boundary), ties broken toward
    the more recent start -- reacts to a real in-season change as soon as the new guy has
    started a plurality of the last K games, instead of waiting for a season-cumulative
    crossover that a early-season-opener absence can delay for a long time."""
    primary_so_far: dict[tuple[str, str], str] = {}

    if mode == "cumulative":
        season_attempts: dict[tuple[str, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
        last_season_leader: dict[str, str] = {}
        last_season_num: dict[str, int] = {}
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
                _, attempts = game_team_epa[(gid, team)]
                this_season[ap] += attempts  # cumulative ATTEMPTS, matching production exactly
        return primary_so_far

    k = int(mode.replace("window", ""))
    history: dict[str, deque] = defaultdict(lambda: deque(maxlen=k))
    for gid, team, season, week in game_order:
        hist = history[team]
        if hist:
            counts = Counter(hist)
            best = max(counts.values())
            candidates = {p for p, c in counts.items() if c == best}
            for p in reversed(hist):
                if p in candidates:
                    primary_so_far[(team, gid)] = p
                    break
        else:
            primary_so_far[(team, gid)] = actual_passer.get((gid, team))
        ap = actual_passer.get((gid, team))
        if ap is not None:
            hist.append(ap)
    return primary_so_far


def build_trailing_epa(game_order, actual_passer, game_team_epa):
    passer_hist: dict[str, list[float]] = defaultdict(list)
    trailing_epa: dict[tuple[str, str], float] = {}
    league_avg: list[float] = []
    for gid, team, season, week in game_order:
        ap = actual_passer.get((gid, team))
        if ap is None:
            continue
        hist = passer_hist[ap]
        trailing_epa[(ap, gid)] = (sum(hist) / len(hist)) if hist else None
        epa_sum, attempts = game_team_epa[(gid, team)]
        if attempts:
            per_play = epa_sum / attempts
            hist.append(per_play)
            league_avg.append(per_play)
    league_avg_epa = float(np.mean(league_avg)) if league_avg else 0.0
    return trailing_epa, league_avg_epa


def build_qb_gap(game_order, actual_passer, primary_so_far, trailing_epa, league_avg_epa, dynamic: bool):
    """gap(team, gid): positive = today's actual passer worse than the reference, applied the
    same way production does (subtracted via a negative QB_QUALITY_GAP_WEIGHT-equivalent, so
    positive gap penalizes, negative gap boosts).
    dynamic=False: production's existing static formula, primary_so_far's own trailing EPA
    minus actual's -- identical to v25/production, used here as the control group.
    dynamic=True: the FIRST game of a backup's run gets the full static gap (that's known
    pre-game from the injury report, same as production's live path -- nothing to decay yet).
    Each ADDITIONAL consecutive start by that same non-primary passer decays the gap by
    (1-EWMA_ALPHA)**(n-1), n = how many straight games he's now started, mirroring exactly how
    fast the team's own rolling rating (same alpha) is independently catching up to his real
    level -- by n~5 the flat correction would mostly double-count what delta already reflects.
    When the real primary returns, the rolling rating is still sitting near the backup's
    level (it hasn't forgotten him yet either), so this applies a REINSERTION boost sized by
    how much of the absence's gap had NOT yet decayed away (i.e. how much the rolling rating
    still has left to recover) at the moment of return, itself decaying the same way over the
    games right after the return -- instead of snapping straight to 0 like the static/
    production formula does the instant the primary is back under center."""
    gap: dict[tuple[str, str], float] = {}
    if not dynamic:
        for gid, team, season, week in game_order:
            primary = primary_so_far.get((team, gid))
            actual = actual_passer.get((gid, team))
            if primary is None or actual is None:
                gap[(team, gid)] = 0.0
                continue
            tp = trailing_epa.get((primary, gid))
            ta = trailing_epa.get((actual, gid))
            tp = tp if tp is not None else league_avg_epa
            ta = ta if ta is not None else league_avg_epa
            gap[(team, gid)] = tp - ta
        return gap

    # pending_boost[team] IS "how much reinsertion credit is left to pay back," decaying by
    # EWMA_ALPHA every HEALTHY week after an absence ends until it's negligible -- no separate
    # "weeks since return" counter needed (an EARLIER version of this used one, return_k, but
    # it reset streak_n to 0 on the FIRST healthy week and gated the boost on streak_n>0, so it
    # only ever credited ONE week of boost instead of decaying over several -- caught when
    # porting this same mechanism into production team_ratings.py and fixed there and here).
    streak_n: dict[str, int] = defaultdict(int)          # consecutive starts by the CURRENT non-primary passer
    pending_boost: dict[str, float] = defaultdict(float)  # undecayed remainder owed back on return
    for gid, team, season, week in game_order:
        primary = primary_so_far.get((team, gid))
        actual = actual_passer.get((gid, team))
        if primary is None or actual is None:
            gap[(team, gid)] = 0.0
            continue
        tp = trailing_epa.get((primary, gid))
        ta = trailing_epa.get((actual, gid))
        tp = tp if tp is not None else league_avg_epa
        ta = ta if ta is not None else league_avg_epa
        raw_gap = tp - ta

        if actual != primary:
            streak_n[team] += 1
            decay_factor = (1 - EWMA_ALPHA) ** (streak_n[team] - 1)
            gap[(team, gid)] = raw_gap * decay_factor
            pending_boost[team] = raw_gap * decay_factor
        else:
            streak_n[team] = 0
            if pending_boost[team]:
                gap[(team, gid)] = -pending_boost[team]
                pending_boost[team] *= (1 - EWMA_ALPHA)
            else:
                gap[(team, gid)] = 0.0
    return gap


def fit_logistic(X, y):
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
    return res.x


def evaluate(games, gap_diff_key):
    seasons = sorted({g["season"] for g in games})
    all_y, all_p, all_old, all_new = [], [], [], []
    for test_season in seasons:
        if test_season < TEST_START_SEASON:
            continue
        train = [g for g in games if g["season"] < test_season]
        test = [g for g in games if g["season"] == test_season]
        if len(train) < 300 or not test:
            continue
        gap_train = np.array([g[gap_diff_key] for g in train])
        gmu, gsd = gap_train.mean(), (gap_train.std() or 1.0)
        def feat(g):
            return [g["delta"], (g[gap_diff_key] - gmu) / gsd]
        X_train = np.array([feat(g) for g in train])
        y_train = np.array([g["home_win"] for g in train], dtype=float)
        w = fit_logistic(X_train, y_train)
        for g in test:
            f = feat(g)
            z = w[0] + w[1] * f[0] + w[2] * f[1]
            p_hat = 1.0 / (1.0 + np.exp(-z))
            all_y.append(g["home_win"])
            all_p.append(p_hat)
            all_old.append(1 if g["delta"] > 0 else 0)
            all_new.append(1 if z > 0 else 0)
    all_y = np.array(all_y)
    acc_old = float(np.mean(np.array(all_old) == all_y))
    acc_new = float(np.mean(np.array(all_new) == all_y))
    auc = roc_auc_score(all_y, all_p)
    ll = log_loss(all_y, all_p)
    return len(all_y), acc_old, acc_new, auc, ll


def main():
    passer_df = load_passer_df()
    print(f"{len(passer_df)} (game, team, passer) rows\n")
    actual_passer, game_team_epa, game_order = build_game_order(passer_df)

    print("Building game dataset + frozen production delta...", file=sys.stderr)
    rows_all = v14.build_dataset()
    cols = [f"{m}_diff" for m in v14.RATING_METRICS]
    X_all = np.array([[r[c] for c in cols] for r in rows_all])
    mu_all, sd_all = X_all.mean(axis=0), X_all.std(axis=0)
    sd_all[sd_all == 0] = 1.0
    PW = v14.PRODUCTION_WEIGHTS
    for i, r in enumerate(rows_all):
        x = X_all[i]
        r["delta"] = sum(PW[m] * ((x[j] - mu_all[j]) / sd_all[j]) for j, m in enumerate(v14.RATING_METRICS))
    games = [r for r in rows_all if r["season"] >= FIRST_SEASON and r["delta"] != 0]

    primary_modes = ["cumulative", "window3", "window4", "window5", "window6"]
    gap_modes = [("static", False), ("decay", True)]

    print("\n=== how often each primary definition disagrees with production (cumulative) ===")
    cum_primary = build_primary_so_far(game_order, actual_passer, game_team_epa, "cumulative")
    for mode in primary_modes[1:]:
        alt_primary = build_primary_so_far(game_order, actual_passer, game_team_epa, mode)
        keys = set(cum_primary) & set(alt_primary)
        disagree = sum(1 for k in keys if cum_primary[k] != alt_primary[k])
        print(f"  {mode:10s}: disagrees on {disagree}/{len(keys)} team-games ({100*disagree/len(keys):.2f}%)")

    print(f"\n=== walk-forward comparison, {TEST_START_SEASON}-2025, delta + qb_gap_diff (home-away) ===")
    header = f"{'primary':10s} {'gap':7s} {'n':>6s} {'acc(delta)':>11s} {'acc(+gap)':>10s} {'AUC':>7s} {'logloss':>8s}"
    print(header)
    results = []
    for p_mode in primary_modes:
        primary_so_far = build_primary_so_far(game_order, actual_passer, game_team_epa, p_mode)
        trailing_epa, league_avg_epa = build_trailing_epa(game_order, actual_passer, game_team_epa)
        for g_label, decay in gap_modes:
            gap = build_qb_gap(game_order, actual_passer, primary_so_far, trailing_epa, league_avg_epa, decay)
            key = f"gap_{p_mode}_{g_label}"
            for g in games:
                g[key] = gap.get((g["home"], g["game_id"]), 0.0) - gap.get((g["away"], g["game_id"]), 0.0)
            n, acc_old, acc_new, auc, ll = evaluate(games, key)
            print(f"{p_mode:10s} {g_label:7s} {n:6d} {100*acc_old:10.2f}% {100*acc_new:9.2f}% {auc:7.4f} {ll:8.4f}")
            results.append((p_mode, g_label, n, acc_old, acc_new, auc, ll))

    print("\n=== best variant by AUC ===")
    best = max(results, key=lambda r: r[5])
    print(f"primary={best[0]} gap={best[1]}  AUC={best[5]:.4f}  acc={100*best[4]:.2f}%  logloss={best[6]:.4f}")
    prod = next(r for r in results if r[0] == "cumulative" and r[1] == "static")
    print(f"production (cumulative/static) AUC={prod[5]:.4f}  acc={100*prod[4]:.2f}%  logloss={prod[6]:.4f}")
    print(f"delta AUC vs production: {best[5]-prod[5]:+.4f}")


if __name__ == "__main__":
    main()
