"""v73: does the betting market (Vegas) have a persistent bias for particular teams?

Question (user request 2026-10-26): are any teams consistently over- or under-valued by the closing line, independent of the
power rankings? "Over-valued" = the team covers less often than the line implies (negative mean ATS margin); "under-valued" =
covers more.

Data: nflverse games.csv, REG season 1999-present (every season it carries a closing spread), closing spread_line (home-favored positive), final scores. For each game the
market residual for the home team is e = (home_score - away_score) - spread_line; the away team's is -e. A team's market bias is
its mean residual in points per game (positive = the market under-rates it). Franchise moves (STL/SD/OAK) count as one team.

A. Is there anything to find? Is the spread of team-average residuals wider than luck allows? Observed Q = sum_t n_t * mean_t^2 /
   sigma^2 against a parametric bootstrap (game residuals ~ iid Normal(0, sigma^2), same schedule; handles the fact that every
   game appears in two teams' rows). The excess implies a between-team sd (tau, points per game). Also: split-half persistence,
   the correlation between a team's mean residual in odd and even seasons.
B. Does it predict? Walk-forward from 2012: for each test season S, shrunk bias per team from seasons < S only
   (bias_t = mean_t * n_t / (n_t + K), K in {200, 400, 800, 1600} games, a fixed grid, never tuned on the outcome),
   predicted residual for a game = bias_home - bias_away. Scored on (i) the slope of the actual residual on the prediction with a
   season-clustered bootstrap CI (1.0 would mean the shrunk bias is exactly right, 0 means no information), and (ii) the ATS
   hit rate of betting the side with the positive predicted residual (pushes dropped), overall, by half (<=2018 / >2018), and on
   the quarter of games with the largest |prediction|.

Pre-registered rule (fixed before running): the bias counts as a USABLE signal only if, for a given K, the slope is > 0 in both
halves AND the ATS hit rate exceeds the 52.38% break-even in both halves AND the pooled one-sided binomial p <= 0.01 (four K
values are tried, so the p bar is deliberately strict). Anything less is "descriptive only": the table below can be shown, but
nothing is claimed about future games.

Run: PYTHONPATH=. python research/edge_signal_test_v73_team_market_bias.py
"""
from __future__ import annotations

import csv
import io
import os
import sys

import numpy as np
import requests
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nflhub.sources.team_ratings as tr  # noqa: E402
from nflhub.sources.edge_teams import TEAMS, normalize_team  # noqa: E402

TEST_START = 2012
SELECT_MAX = 2018
KS = (200, 400, 800, 1600)
BREAKEVEN = 0.5238
N_BOOT = 2000
rng = np.random.default_rng(73)


def load():
    resp = requests.get(tr.GAMES_URL, timeout=tr.TIMEOUT)
    resp.raise_for_status()
    rows = []
    for r in csv.DictReader(io.StringIO(resp.text)):
        if r["game_type"] != "REG" or r.get("result") in ("", "NA", None) or r.get("spread_line") in ("", "NA", None):
            continue
        try:
            rows.append((int(r["season"]), int(r["week"]), normalize_team(r["home_team"]), normalize_team(r["away_team"]),
                         float(r["home_score"]), float(r["away_score"]), float(r["spread_line"])))
        except (ValueError, KeyError):
            continue
    return rows


def main():
    rows = load()
    teams = sorted({r[2] for r in rows} | {r[3] for r in rows})
    ti = {t: i for i, t in enumerate(teams)}
    season = np.array([r[0] for r in rows])
    home = np.array([ti[r[2]] for r in rows])
    away = np.array([ti[r[3]] for r in rows])
    e = np.array([(r[4] - r[5]) - r[6] for r in rows])       # home-team residual, points
    sigma = float(e.std())
    print(f"{len(rows)} games, seasons {season.min()}-{season.max()}, {len(teams)} teams; residual sd {sigma:.2f} pts, mean {e.mean():+.3f}\n")

    def team_stats(mask):
        s = np.zeros(len(teams)); n = np.zeros(len(teams))
        np.add.at(s, home[mask], e[mask]); np.add.at(n, home[mask], 1)
        np.add.at(s, away[mask], -e[mask]); np.add.at(n, away[mask], 1)
        return s, n

    # ---------------- A. is there anything to find
    s, n = team_stats(np.ones(len(rows), bool))
    mean_t = s / n
    Q = float((n * mean_t ** 2).sum() / sigma ** 2)
    sims = []
    for _ in range(N_BOOT):
        es = rng.normal(0, sigma, len(rows))
        ss = np.zeros(len(teams)); np.add.at(ss, home, es); np.add.at(ss, away, -es)
        sims.append(float((ss ** 2 / n).sum() / sigma ** 2))
    sims = np.array(sims)
    p_q = float(np.mean(sims >= Q))
    # E[Q] rises by sum_t n_t * tau^2 / sigma^2 when each team has a true bias with sd tau (points per game)
    tau = float(np.sqrt(max(Q - sims.mean(), 0.0) * sigma ** 2 / n.sum()))
    print("=== A. spread of team-average residuals vs. luck ===")
    print(f"  Q = {Q:.1f} vs. bootstrap mean {sims.mean():.1f} (95th pct {np.percentile(sims, 95):.1f}); P(sim >= observed) = {p_q:.3f}")
    print(f"  implied true between-team sd (tau) about {tau:.2f} pts/game (a team's own sampling sd over {n.mean():.0f} games is {sigma / np.sqrt(n.mean()):.2f})")
    odd, even = season % 2 == 1, season % 2 == 0
    so, no = team_stats(odd); se_, ne = team_stats(even)
    r_split = float(np.corrcoef(so / no, se_ / ne)[0, 1])
    boots = []
    for _ in range(N_BOOT):
        ix = rng.integers(0, len(teams), len(teams))
        boots.append(np.corrcoef((so / no)[ix], (se_ / ne)[ix])[0, 1])
    print(f"  odd-season vs even-season team means: r = {r_split:+.2f} (team-bootstrap 95% CI {np.nanpercentile(boots, 2.5):+.2f} to {np.nanpercentile(boots, 97.5):+.2f})\n")

    # ---------------- B. walk-forward prediction
    test = season >= TEST_START
    print("=== B. walk-forward (bias from earlier seasons only), test seasons 2012-2026 ===")
    results = {}
    for K in KS:
        pred = np.full(len(rows), np.nan)
        for S in sorted(set(season[test])):
            s_, n_ = team_stats(season < S)
            bias = (s_ / np.maximum(n_, 1)) * n_ / (n_ + K)
            m = season == S
            pred[m] = bias[home[m]] - bias[away[m]]
        results[K] = pred
        # slope of actual residual on prediction, season-clustered bootstrap
        seas = sorted(set(season[test]))
        def slope(mask):
            x, y = pred[mask], e[mask]
            return float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
        sl_all = slope(test)
        bs = []
        for _ in range(N_BOOT):
            pick = rng.choice(seas, len(seas))
            idx = np.concatenate([np.where(season == p)[0] for p in pick])
            x, y = pred[idx], e[idx]
            bs.append(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
        sel, con = test & (season <= SELECT_MAX), test & (season > SELECT_MAX)
        # ATS: bet the side with the positive predicted residual
        side_home = pred > 0
        covered_home = e > 0
        nopush = (e != 0) & (pred != 0)
        hit = (side_home == covered_home)

        def rate(mask):
            m = mask & nopush
            return float(hit[m].mean()), int(m.sum()), int(hit[m].sum())
        r_all, r_sel, r_con = rate(test), rate(sel), rate(con)
        thr = np.nanpercentile(np.abs(pred[test & nopush]), 75)
        r_top = rate(test & (np.abs(np.nan_to_num(pred)) >= thr))
        pbin = binomtest(r_all[2], r_all[1], BREAKEVEN, alternative="greater").pvalue
        ok = slope(sel) > 0 and slope(con) > 0 and r_sel[0] > BREAKEVEN and r_con[0] > BREAKEVEN and pbin <= 0.01
        print(f"  K={K:5d}: slope {sl_all:+.2f} (95% CI {np.percentile(bs, 2.5):+.2f} to {np.percentile(bs, 97.5):+.2f}; halves {slope(sel):+.2f}/{slope(con):+.2f}) | "
              f"ATS hit {r_all[0]:.4f} ({r_all[2]}/{r_all[1]}), halves {r_sel[0]:.4f}/{r_con[0]:.4f}, top-quartile |pred| {r_top[0]:.4f} (n={r_top[1]}); "
              f"p(vs 52.38%) {pbin:.3f} -> {'USABLE SIGNAL' if ok else 'descriptive only'}")

    # ---------------- exploratory (NOT part of the pre-registered rule): recency-weighted bias, half-life in seasons
    print(chr(10) + "  exploratory, recency-weighted (weights halve every H seasons; K = 800 pseudo-games; not part of the rule):")
    for H in (3, 6):
        pred = np.full(len(rows), np.nan)
        for S in sorted(set(season[test])):
            w = np.where(season < S, 0.5 ** ((S - 1 - season) / H), 0.0)
            sw = np.zeros(len(teams)); nw = np.zeros(len(teams))
            np.add.at(sw, home, w * e); np.add.at(nw, home, w)
            np.add.at(sw, away, -w * e); np.add.at(nw, away, w)
            bias = sw / np.maximum(nw, 1e-9) * nw / (nw + 800.0 * (nw.sum() / max(np.sum(season < S) * 2, 1)))
            m = season == S
            pred[m] = bias[home[m]] - bias[away[m]]
        x, y = pred[test], e[test]
        slope = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
        nopush = (e != 0) & (pred != 0)
        hit = ((pred > 0) == (e > 0))
        m = test & nopush
        print(f"    half-life {H} seasons: slope {slope:+.2f}; ATS hit {hit[m].mean():.4f} ({int(hit[m].sum())}/{int(m.sum())}), "
              f"p(vs 52.38%) {binomtest(int(hit[m].sum()), int(m.sum()), BREAKEVEN, alternative='greater').pvalue:.3f}")

    # ---------------- the table (descriptive): all seasons, and the last 5 seasons
    print("\n=== team market bias, points per game (positive = under-rated by the line, covers more than it should) ===")
    k_show = 800
    last5 = season >= season.max() - 4
    s5, n5 = team_stats(last5)
    print(f"  {'team':5s} {'all: mean':>9} {'se':>5} {'shrunk':>7} {'games':>6} | {'last5: mean':>11} {'se':>5} {'games':>6}")
    order = np.argsort(-(s / n * n / (n + k_show)))
    for i in order:
        print(f"  {teams[i]:5s} {s[i] / n[i]:+9.2f} {sigma / np.sqrt(n[i]):5.2f} {s[i] / (n[i] + k_show):+7.2f} {int(n[i]):6d} | {s5[i] / n5[i]:+11.2f} {sigma / np.sqrt(n5[i]):5.2f} {int(n5[i]):6d}")


if __name__ == "__main__":
    main()
