"""v63: does a FiveThirtyEight-style Elo rating add anything to the production composite?

Context: a cowork check (docs/cowork-tests/elo_check.py) found Elo alone scores AUC 0.692 on
2011-2025 and carries most of the information a slower rating update recovers (corr 0.91 with
points ratings). v61/v62 then shipped the slower update. Question now: on top of THAT composite,
is Elo worth a 10th stat?

Elo: K = 20, home-field 48 Elo points inside the UPDATE only, margin-of-victory multiplier
ln(|margin|+1) * 2.2 / (0.001 * (winner Elo diff incl. HFA) + 2.2), one-third regression to 1505 at
each team's first game of a new season, neutral sites get no HFA. Warmed up on every REG game back
to 1999 (all market-free). The FEATURE is the pre-game home-minus-away Elo difference WITHOUT the
home-field term (delta adds its own HFA correction, so including it here would double count).
Constants are the published 538 values, not tuned here.

Models, all walk-forward 2011-2026 (weights refit per fold, non-negative L2 = 0.03, same recipe as
production), on the production compute_ratings game_log (v61 shipped settings):
  (a) 9-stat composite (production), (b) composite + Elo as a 10th non-negative stat, (c) Elo alone.

Pre-registered rule (v57/v61/v62, fixed before running): (b) is adopted over (a) only if dAUC > 0
AND dLogLoss > 0 in BOTH halves (<=2018 / >2018), paired-bootstrap P(dLogLoss <= 0) <= 0.01
(N_BOOT = 2000), and log loss better in >= 60% of test seasons. If (b) fails, Elo stays out of delta.

Implementation note: the first run used a standalone Elo warmed up from 1999 (result: ADOPT, see
output/v63_run_standalone.txt). Production (team_ratings.elo_ratings) can only see games from
FIRST_SEASON = 2007, so the decisive run below uses PRODUCTION's own Elo (the "elo" key of the
compute_ratings game_log diffs, cold-started in 2007); the 1999-warmed standalone Elo is kept as a
sensitivity check. Same rule, same constants.

Run: PYTHONPATH=. python research/edge_signal_test_v63_elo_tenth_stat.py
"""
from __future__ import annotations

import csv
import io
import os
import sys

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(__file__))
import nflhub.sources.team_ratings as tr  # noqa: E402
from edge_signal_test_v54_pen_presnap_production_refit import M9, fit_nonneg, metrics, walk_forward  # noqa: E402
from edge_signal_test_v61_ewma_speed_season_prior import compare  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402

L2 = 0.03
K, HFA_ELO, REGRESS, MEAN_ELO = 20.0, 48.0, 1 / 3, 1505.0


def elo_pre_games(all_rows):
    """{game_id: home-minus-away pre-game Elo difference WITHOUT home field}."""
    rows = [r for r in all_rows if r.get("home_score") not in ("", "NA", None) and r.get("home_score") is not None]
    rows.sort(key=lambda r: (r["gameday"], r["game_id"]))
    rating: dict[str, float] = {}
    last: dict[str, int] = {}
    out: dict[str, float] = {}
    for r in rows:
        season = int(r["season"])
        h, a = r["home_team"], r["away_team"]
        for t in (h, a):
            rating.setdefault(t, 1500.0)
            if last.get(t, season) != season:
                rating[t] = rating[t] * (1 - REGRESS) + MEAN_ELO * REGRESS
            last[t] = season
        margin = float(r["home_score"]) - float(r["away_score"])
        diff = rating[h] - rating[a]
        out[r["game_id"]] = diff
        d = diff + (0.0 if tr.is_neutral_site(r) else HFA_ELO)
        e = 1.0 / (1.0 + 10 ** (-d / 400.0))
        s = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
        mov = np.log(abs(margin) + 1.0) * 2.2 / ((d if margin > 0 else -d) * 0.001 + 2.2)
        rating[h] += K * mov * (s - e)
        rating[a] -= K * mov * (s - e)
    return out


def verdict(c):
    rep = all(c[f"dAUC_{h}"] > 0 and c[f"dLL_{h}"] > 0 for h in ("select", "confirm"))
    return rep and c["p_ll_le0"] <= 0.01 and c["ll_season_win"] >= 0.6, rep


def fmt(c):
    return (f"dAUC {c['dAUC_all']:+.4f} (sel {c['dAUC_select']:+.4f}, con {c['dAUC_confirm']:+.4f}); "
            f"dLL {c['dLL_all']:+.4f} (sel {c['dLL_select']:+.4f}, con {c['dLL_confirm']:+.4f}); "
            f"P(dLL<=0)={c['p_ll_le0']:.3f}; season-win={c['ll_season_win']:.2f}")


def main():
    resp = requests.get(tr.GAMES_URL, timeout=tr.TIMEOUT)
    resp.raise_for_status()
    all_rows = [r for r in csv.DictReader(io.StringIO(resp.text)) if r["game_type"] == "REG"]
    played = [r for r in all_rows if r.get("result") not in ("", "NA", None) and int(r["season"]) >= tr.FIRST_SEASON]
    current = max(int(r["season"]) for r in played)
    elo = elo_pre_games([r for r in all_rows if r.get("result") not in ("", "NA", None)])

    log: list[dict] = []
    tr.compute_ratings(played, current, game_log=log)
    log = [r for r in log if r["home_score"] != r["away_score"] and r["game_id"] in elo]
    seasons = np.array([r["season"] for r in log])
    y = np.array([1.0 if r["home_score"] > r["away_score"] else 0.0 for r in log])
    X9 = np.column_stack([[r["diffs"][m] for r in log] for m in M9])
    E99 = np.array([elo[r["game_id"]] for r in log])[:, None]
    E = np.array([r["diffs"]["elo"] for r in log])[:, None]
    X10 = np.hstack([X9, E])
    comp = ((X9 - X9.mean(0)) / X9.std(0)) @ np.array([tr.POWER_WEIGHTS[m] for m in M9])
    print(f"{len(log)} decided games {seasons.min()}-{seasons.max()}; corr(Elo diff, 9-stat composite) = "
          f"{np.corrcoef(E[:, 0], comp)[0, 1]:.3f}\n")

    pa, yy, ss, wa = walk_forward(seasons, X9, y, L2)
    pb, _, _, wb = walk_forward(seasons, X10, y, L2)
    pc, _, _, _ = walk_forward(seasons, E, y, L2)
    rng = np.random.default_rng(63)
    print(f"=== walk-forward 2011-2026 ({len(yy)} games) ===")
    for lab, p in (("(a) 9-stat composite", pa), ("(b) composite + Elo", pb), ("(c) Elo alone", pc)):
        a, l = metrics(p, yy)
        sel = ss <= 2018
        print(f"  {lab:22s} AUC {a:.4f} LL {l:.4f} acc {np.mean((p > 0.5) == (yy == 1)):.4f} | "
              f"<=2018 AUC {roc_auc_score(yy[sel], p[sel]):.4f} | >2018 AUC {roc_auc_score(yy[~sel], p[~sel]):.4f}")

    c = compare(yy, ss, pa, pb, rng)
    ok, rep = verdict(c)
    print(f"\n(b) vs (a): {fmt(c)}; replicated both halves={rep} -> {'ADOPT' if ok else 'DO NOT ADOPT'}")
    p99, _, _, _ = walk_forward(seasons, np.hstack([X9, E99]), y, L2)
    c99 = compare(yy, ss, pa, p99, rng)
    print(f"sensitivity, Elo warmed up from 1999: AUC {metrics(p99, yy)[0]:.4f} LL {metrics(p99, yy)[1]:.4f}; {fmt(c99)}")
    c2 = compare(yy, ss, pa, pc, rng)
    print(f"(c) vs (a) (information only): {fmt(c2)}")

    print("\n=== per-fold weight of the Elo stat in (b) (standardized units; mean / min / folds at zero) ===")
    we = wb[:, -1]
    print(f"  mean {we.mean():.4f} min {we.min():.4f} zero in {(we == 0).sum()}/{len(we)} folds")

    print("\n=== all-data fit of (b) (standardized-diff weights) ===")
    mu, sd = X10.mean(0), X10.std(0)
    w = fit_nonneg((X10 - mu) / sd, y, L2)
    for m, v in zip(list(M9) + ["elo"], w[1:]):
        print(f"  {m:22s} {v:.4f}")


if __name__ == "__main__":
    main()
