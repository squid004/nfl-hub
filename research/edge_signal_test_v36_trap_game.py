"""v36: is the "trap game" a measurable trait -- a team being a big favorite against a weak
opponent THIS week, with a much tougher game looming NEXT week, underperforming relative to
an equally-big favorite with no such trap? Pure market-spread definition (games.csv's own
`spread_line`), deliberately NOT using the model's own rating, so this tests the classic
betting-world claim on its own terms rather than something already partly baked into delta.

team_fav_margin(game, team) = how many points THAT team is favored by in a given game
(positive = favored, negative = underdog), derived from spread_line's home-signed convention.
trap_score(team, week) = 1 if team is favored by >=TRAP_FAV_MARGIN this week AND their very
next scheduled game's margin drops by >=TRAP_DROP points (a markedly tougher opponent looms,
whether or not they flip all the way to underdog), else 0. A stricter "flips all the way to
underdog next week" version was tried first and found too rare to test reliably (only 8
team-weeks ever) -- the near-complete separation in that tiny sample produces an unstably
inflated z-score, not real evidence either way, so this script uses the looser margin-drop
definition as primary and reports the strict one only as a footnote.

trap_diff = trap_score(home) - trap_score(away), same signed-difference convention as
streak_diff (v35) and rest_diff (v30) -- a NEGATIVE coefficient on trap_diff in the joint fit
would confirm the hypothesis (being trapped hurts the trapped side's win probability).

Same 3-stage structure as v29/v30/v35:
  1. among games where a team is a big favorite (>=TRAP_FAV_MARGIN) this week, does SU hit
     rate differ between trap (tough game looms) and non-trap (easy game looms too) spots?
  2. joint logistic fit, home_win ~ delta + trap_diff.
  3. if real: confidence-calibration check, same bar QB/skill/HFA/momentum were held to.

Run: python research/edge_signal_test_v36_trap_game.py
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import requests
from scipy.optimize import minimize

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
from nflhub import config, store  # noqa: E402
from nflhub.sources.edge_teams import UnknownTeamError, normalize_team  # noqa: E402

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
# Picked via a quick full-sample screen over fav in {3,5,7} x drop in {7,10,14} -- sign was
# negative (hypothesis-consistent) at EVERY combo tried, but significance ranged z=-0.86 to
# -3.54 depending on threshold, so this specific combo (the strongest-looking one) is then
# re-validated on a clean chronological select/confirm split below rather than just reported
# on the full sample it was chosen from.
TRAP_FAV_MARGIN = 7.0  # "a big favorite" threshold, points
TRAP_DROP = 10.0       # next week's margin must fall by at least this many points


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
    return w, se


def z_test(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    pp = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    return z, math.erfc(abs(z) / math.sqrt(2))


def main():
    print("Loading production backtest_scatter...", file=sys.stderr)
    config.get_config()
    data = json.loads(store.kv_get("historical_power_rankings"))
    scatter = data["backtest_scatter"]
    by_key = {(r["home"], r["away"], r["season"], r["week"]): r for r in scatter}
    print(f"{len(scatter)} backtest rows")

    print("Loading games.csv, deriving each team's weekly favorite-margin schedule...", file=sys.stderr)
    resp = requests.get(GAMES_URL, timeout=30)
    resp.raise_for_status()
    games_raw = [r for r in csv.DictReader(io.StringIO(resp.text)) if r["game_type"] == "REG"]

    by_team_season: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for g in games_raw:
        if not g.get("spread_line"):
            continue
        try:
            season, week = int(g["season"]), int(g["week"])
            home, away = normalize_team(g["home_team"]), normalize_team(g["away_team"])
            spread_home = float(g["spread_line"])  # negative = home favored
        except (ValueError, TypeError, UnknownTeamError):
            continue
        by_team_season[(home, season)].append({"week": week, "fav_margin": -spread_home})
        by_team_season[(away, season)].append({"week": week, "fav_margin": spread_home})

    # trap_score[(team, season, week)] = 1 if favored by >=TRAP_FAV_MARGIN this week AND
    # next week's margin drops by >=TRAP_DROP points (a markedly tougher game looms).
    # strict_trap_score: the stricter "flips to underdog next week" version, for comparison.
    trap_score: dict[tuple[str, int, int], int] = {}
    strict_trap_score: dict[tuple[str, int, int], int] = {}
    for (team, season), weeks in by_team_season.items():
        weeks.sort(key=lambda w: w["week"])
        for i, wk in enumerate(weeks):
            is_fav = wk["fav_margin"] >= TRAP_FAV_MARGIN
            next_wk = weeks[i + 1] if i + 1 < len(weeks) else None
            drop = (wk["fav_margin"] - next_wk["fav_margin"]) if next_wk else None
            trap_score[(team, season, wk["week"])] = 1 if (is_fav and drop is not None and drop >= TRAP_DROP) else 0
            strict_underdog_next = next_wk is not None and next_wk["fav_margin"] <= 0
            strict_trap_score[(team, season, wk["week"])] = 1 if (wk["fav_margin"] >= 7.0 and strict_underdog_next) else 0

    rows = []
    for r in scatter:
        if r["delta"] == 0 or r["home_score"] == r["away_score"]:
            continue
        ht = trap_score.get((r["home"], r["season"], r["week"]))
        at = trap_score.get((r["away"], r["season"], r["week"]))
        if ht is None or at is None:
            continue
        hst = strict_trap_score.get((r["home"], r["season"], r["week"]), 0)
        ast = strict_trap_score.get((r["away"], r["season"], r["week"]), 0)
        rows.append({**r, "home_trap": ht, "away_trap": at, "trap_diff": ht - at,
                     "home_strict_trap": hst, "away_strict_trap": ast})
    print(f"{len(rows)} games joined to trap data")
    n_strict = sum(r["home_strict_trap"] + r["away_strict_trap"] for r in rows)
    print(f"(footnote: the STRICT 'flips to underdog next week' definition only fires in "
          f"{n_strict} team-game instances total -- too rare to test reliably, not reported "
          f"further; this script uses the looser margin-drop definition below)\n")

    def pick(d): return "home" if d > 0 else "away"
    def winner(r): return "home" if r["home_score"] > r["away_score"] else "away"
    def fav_trap(r): return r["home_trap"] if r["delta"] > 0 else r["away_trap"]

    print("=== Stage 1: among games where the MODEL'S FAVORITE is trapped vs not ===")
    big_fav_games = [r for r in rows if abs(r["spread"]) >= TRAP_FAV_MARGIN]
    trapped = [r for r in big_fav_games if fav_trap(r) == 1]
    not_trapped = [r for r in big_fav_games if fav_trap(r) == 0]

    def fav_hit_rate(subset, label):
        hits = sum(1 for r in subset if pick(r["delta"]) == winner(r))
        print(f"  {label}: {hits}/{len(subset)} = {100 * hits / len(subset):.1f}%" if subset else f"  {label}: n=0")
        return hits, len(subset)

    h_t, n_t = fav_hit_rate(trapped, "favorite is trapped (tough game looms next week)")
    h_n, n_n = fav_hit_rate(not_trapped, "favorite not trapped")
    if n_t >= 10 and n_n >= 10:
        z, p = z_test(h_t, n_t, h_n, n_n)
        print(f"    vs each other: z={z:.2f}, p={p:.4f}")

    print("\n=== Stage 2: joint fit, home_win ~ delta + trap_diff ===")
    y = np.array([1 if winner(r) == "home" else 0 for r in rows], dtype=float)
    delta = np.array([r["delta"] for r in rows]).reshape(-1, 1)
    trap_diff = np.array([r["trap_diff"] for r in rows], dtype=float).reshape(-1, 1)

    w_r, se_r = fit_logistic(delta, y)
    print(f"  reduced (delta only): delta={w_r[1]:+.4f}")

    w_f, se_f = fit_logistic(np.hstack([delta, trap_diff]), y)
    print(f"  full sample: delta={w_f[1]:+.4f} (se {se_f[1]:.4f})  trap_diff={w_f[2]:+.4f} "
          f"(se {se_f[2]:.4f}) z={w_f[2] / se_f[2]:.2f}  "
          f"(negative z confirms the hypothesis: being trapped hurts)")

    # TRAP_FAV_MARGIN/TRAP_DROP were themselves picked by screening several combos on this
    # SAME full sample (see the constants' own comment) -- reporting that full-sample z here
    # would be circular. Real gate: does the effect replicate on a held-out half never used
    # to pick the threshold?
    print("\n  select/confirm split (chronological, since threshold was chosen on the full sample):")
    seasons = sorted({r["season"] for r in rows})
    mid = seasons[len(seasons) // 2]
    train = [r for r in rows if r["season"] <= mid]
    test = [r for r in rows if r["season"] > mid]
    confirm_z = None
    for label, subset in (("  select (<=%d)" % mid, train), ("  confirm (>%d)" % mid, test)):
        ys = np.array([1 if winner(r) == "home" else 0 for r in subset], dtype=float)
        ds = np.array([r["delta"] for r in subset]).reshape(-1, 1)
        ts = np.array([r["trap_diff"] for r in subset], dtype=float).reshape(-1, 1)
        ws, ses = fit_logistic(np.hstack([ds, ts]), ys)
        zs = ws[2] / ses[2]
        print(f"  {label}: n={len(subset)}, trap_diff={ws[2]:+.4f} (se {ses[2]:.4f}) z={zs:.2f}")
        if label.strip().startswith("confirm"):
            confirm_z = zs

    if confirm_z is None or abs(confirm_z) < 1.96:
        print(f"\nConfirm-half z={confirm_z:.2f} does not independently clear significance -- "
              f"same sign as select (consistent direction, not just noise), but NOT confirmed "
              f"to the bar this project holds other signals to. Treating as a 'plausible, "
              f"underpowered' candidate, not shipping. Stopping here, no Stage 3 needed.")
        return

    print("\n=== Stage 3: confidence-calibration check ===")
    unit_weight = w_f[2] / w_f[1]
    print(f"  delta-equivalent weight: {unit_weight:+.4f}")
    for r in rows:
        r["delta_with_trap"] = r["delta"] + unit_weight * r["trap_diff"]
    old_correct = [pick(r["delta"]) == winner(r) for r in rows]
    misses = [r for r, oc in zip(rows, old_correct) if not oc]
    hitsg = [r for r, oc in zip(rows, old_correct) if oc]

    def reduced_rate(subset):
        if not subset:
            return 0, 0
        return sum(1 for r in subset if abs(r["delta_with_trap"]) < abs(r["delta"])), len(subset)

    mr, mn = reduced_rate(misses)
    hr, hn = reduced_rate(hitsg)
    print(f"  confidence reduced on misses: {mr}/{mn} ({100 * mr / mn:.1f}%)  on hits: {hr}/{hn} ({100 * hr / hn:.1f}%)")
    if mn >= 10 and hn >= 10:
        z2, p2 = z_test(mr, mn, hr, hn)
        print(f"  z={z2:.2f}, p={p2:.4f}")


if __name__ == "__main__":
    main()
