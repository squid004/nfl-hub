"""v55: re-run every v52 candidate under the market-free framing.

v52 screened candidates partly on whether the unexplained part correlated with the closing spread /
market residual, and v53 only walk-forwarded four hand-picked sets chosen from that screen. The
production model never uses the market spread, so that screen was the wrong gate: a candidate that
adds out-of-sample signal to the market-free composite is a real improvement even if it adds nothing
once the spread is in. This sweep drops the market from the decision entirely.

Baseline = the 9 production stats (the original 8 + pen_presnap_yds, shipped in v54).
Each persistence-passing v52 candidate is added ONE AT A TIME, walk-forward by season from 2011
(same framework as v53: pre-game EWMA diffs, one joint non-negative L2=0.3 logistic).

Pre-registered SHIP tier (all must hold, fixed before running):
  1. dAUC > 0 and dLogLoss > 0 in BOTH chronological halves (select 2011-2018, confirm 2019-2025)
  2. paired-bootstrap P(dLogLoss <= 0) <= 0.01
  3. fitted weight > 0 in >= 90% of walk-forward folds
  4. better log loss in >= 60% of individual test seasons
Anything that clears 1 and 3 but misses 2 or 4 is WATCH. With N looks the chance of a false SHIP
is reported (N * 0.01 upper bound before the replication requirement, which cuts it further).
Pace (sec_per_play) has no obvious sign, so it is tested in both orientations.

Winners are then re-run greedily (best SHIP first, re-judged against the grown baseline) so a
stack of correlated survivors is not double counted.

Run: PYTHONPATH=. python research/edge_signal_test_v55_market_free_candidate_sweep.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from persistence import pregame_ewma  # noqa: E402
from edge_signal_test_v52_extra_terms_screen import BENCH, build_team_games  # noqa: E402
from edge_signal_test_v53_walkforward_extra_terms import (  # noqa: E402
    SELECT_END, TEST_START, compare, half, summarize, walk_forward,
)

HERE = os.path.dirname(__file__)
BASE9 = BENCH + ["pen_presnap_c"]
CANDS = [
    "expl_rush_off_c", "expl_pass_off_c", "expl_rush_def_c", "expl_pass_def_c",
    "pen_yds_c", "pen_drawn_c",
    "top_share_c", "sec_per_play_c", "sec_per_play_fast_c", "drive_score_off_c", "drive_score_def_c",
    "st_epa_c", "st_kp_c", "st_kicking_c", "st_returning_c", "fp_net_c", "fg_oe_c",
]


def build_diffs() -> pd.DataFrame:
    tg = build_team_games()
    tg["sec_per_play_fast_c"] = -tg["sec_per_play_c"]
    cols = sorted(set(BASE9 + CANDS))
    pre = pregame_ewma(tg, cols)
    pre.columns = [c.removeprefix("pre_") for c in pre.columns]
    side = pd.concat([tg[["game_id", "season", "week", "is_home"]], pre], axis=1)
    h = side[side.is_home == 1].set_index("game_id")
    a = side[side.is_home == 0].set_index("game_id").reindex(h.index)
    d = h[cols] - a[cols]
    d["season"], d["week"] = h["season"], h["week"]
    gm = tg[tg.is_home == 1].set_index("game_id")[["home_score", "away_score"]].reindex(h.index)
    d = d.join(gm)
    d["margin"] = d["home_score"] - d["away_score"]
    d["home_win"] = (d["margin"] > 0).astype(int)
    return d.dropna(subset=cols).reset_index()


def judge(base: pd.DataFrame, alt: pd.DataFrame, ws: list, n_base: int, rng) -> dict:
    r = compare("x", base, alt, rng)
    w = np.array(ws)[:, n_base:]
    r["w_mean"] = float(w.mean())
    r["w_pos_frac"] = float(np.mean(w > 0))
    sb = base.groupby("season").apply(lambda g: summarize(g)["ll"], include_groups=False)
    sa = alt.groupby("season").apply(lambda g: summarize(g)["ll"], include_groups=False)
    r["ll_season_win"] = float(np.mean(sa < sb))
    ab = base.groupby("season").apply(lambda g: summarize(g)["auc"], include_groups=False)
    aa = alt.groupby("season").apply(lambda g: summarize(g)["auc"], include_groups=False)
    r["auc_season_win"] = float(np.mean(aa > ab))
    rep = all(r[f"dAUC_{h}"] > 0 and r[f"dLL_{h}"] > 0 for h in ("select", "confirm"))
    r["tier"] = ("SHIP" if rep and r["p_ll_le0"] <= 0.01 and r["w_pos_frac"] >= 0.9 and r["ll_season_win"] >= 0.6
                 else "WATCH" if rep and r["w_pos_frac"] >= 0.9 else "no")
    return r


def sweep(d: pd.DataFrame, base_cols: list[str], cands: list[str], rng) -> pd.DataFrame:
    base, _ = walk_forward(d, base_cols)
    rows = []
    for c in cands:
        alt, ws = walk_forward(d, base_cols + [c])
        r = judge(base, alt, ws, len(base_cols), rng)
        r["cand"] = c
        rows.append(r)
    return pd.DataFrame(rows).set_index("cand").sort_values("dLL_all", ascending=False)


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    d = build_diffs()
    print(f"{len(d)} games with all candidate diffs ({d.season.min()}-{d.season.max()}); "
          f"test seasons {TEST_START}-{d.season.max()}, select <= {SELECT_END} < confirm\n")
    rng = np.random.default_rng(55)

    base, _ = walk_forward(d, BASE9)
    sb = summarize(base)
    print(f"BASELINE (9 production stats): n={len(base)} AUC={sb['auc']:.4f} logloss={sb['ll']:.4f} acc={sb['acc']:.4f}\n")

    show = ["dAUC_all", "dAUC_select", "dAUC_confirm", "dLL_all", "dLL_select", "dLL_confirm",
            "p_ll_le0", "w_mean", "w_pos_frac", "ll_season_win", "auc_season_win", "tier"]
    chosen, cols = [], list(BASE9)
    remaining = list(CANDS)
    rnd = 1
    while remaining:
        res = sweep(d, cols, remaining, rng)
        print(f"=== Round {rnd}: one-at-a-time vs baseline of {len(cols)} stats ({'+'.join(chosen) or 'no additions'}) ===")
        print(res[show].to_string(float_format=lambda v: f"{v:+.4f}" if isinstance(v, float) else v))
        if rnd == 1:
            n = len(remaining)
            print(f"\n  {n} looks; P(at least one false SHIP) <= {1 - 0.99 ** n:.2f} before the replication requirement\n")
            res.to_csv(os.path.join(HERE, "output", "v55_round1.csv"))
        ship = res[res.tier == "SHIP"]
        if ship.empty:
            print("\n  no SHIP-tier candidate this round; stopping.")
            break
        best = ship.index[0]
        chosen.append(best)
        cols.append(best)
        remaining.remove(best)
        print(f"\n  -> add {best}\n")
        rnd += 1

    if chosen:
        final, ws = walk_forward(d, cols)
        r = compare("final", base, final, rng)
        print(f"FINAL STACK {chosen}: dAUC={r['dAUC_all']:+.4f} dLL={r['dLL_all']:+.4f} "
              f"(select {r['dAUC_select']:+.4f}/{r['dLL_select']:+.4f}, confirm {r['dAUC_confirm']:+.4f}/{r['dLL_confirm']:+.4f}) "
              f"P(dLL<=0)={r['p_ll_le0']:.3f}")
    else:
        print("\nNo candidate reached SHIP tier.")


if __name__ == "__main__":
    main()
