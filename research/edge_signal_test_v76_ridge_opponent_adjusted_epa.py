"""v76: simultaneous opponent adjustment of the four EPA stats (a time-decayed ridge fit of offense and defense effects together).

User concern (2026-10-31): a bad rush defense against a very strong rush offense should count for less; offense and defense EPA should be
normalized for who is across the line. v74 tried the naive version (subtract K x the opponent's pre-game rating from each game's value)
and found nothing, plausibly because it subtracts a NOISY rating (defensive EPA ratings repeat at only ~0.38). The principled version
fits both sides at once and shrinks them:

    y(team, game) = mu + off(team) + def(opponent)        (separately for rush EPA/play and pass EPA/play)

fit by weighted ridge on every earlier game (observation weight halves every H weeks, last season counted with its age, offseason gap of
4 weeks; ridge penalty lambda on off and def, none on mu). Each team's pre-game offense rating is its `off` and its defense rating is
its `def` (the EPA/play it allows relative to an average opponent), so an offense that only beat bad defenses is credited less, a defense
that only faced weak offenses is credited less, and shrinkage is automatic. Pre-game ratings only (games strictly before the target
week), no lookahead. Equal weight per game (play counts are not in the cached extracts); no home-field term (HFA is added later).

Model R1 replaces R0's four EWMA EPA stats with these four ridge ratings; turnovers, pre-snap penalties and Elo are unchanged, weights are
refit walk-forward as in v57-v75 (non-negative, L2 = 0.03, seasons < S, test 2011-2025). Grid, fixed before running: half-life H in
{10, 20} weeks x lambda in {5, 15, 40} (effective observations per team are about H / ln 2, so lambda 5-40 spans light to heavy shrinkage).
Six candidates share one baseline (R0), reported with that multiplicity.

Control added after the first run (every pre-registered candidate lost to R0, and the loss grew with the half-life, which says time
structure and not adjustment may be the cause): the SAME time weighting and shrinkage with no opponent term, so the adjustment can be judged
on its own (adjusted vs control) as well as against R0.

Pre-registered rule (v57+): vs R0, AUC and log loss better in BOTH halves (<=2018 / >2018), paired-bootstrap P(dLL <= 0) <= 0.01
(N_BOOT = 2000), better log loss in >= 60% of seasons.

Run: PYTHONPATH=. python research/edge_signal_test_v76_ridge_opponent_adjusted_epa.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import edge_signal_test_v53_walkforward_extra_terms as v53  # noqa: E402
import edge_signal_test_v75_research_model_no_points_sweep as v75  # noqa: E402
from edge_signal_test_v52_extra_terms_screen import build_team_games  # noqa: E402

L2 = 0.03
GRID_H = (10.0, 20.0)
EXPLORATORY_H = (5.0,)     # added after the first run (see "Second pass" in main); NOT part of the pre-registered grid
GRID_LAMBDA = (5.0, 15.0, 40.0)
SEASON_SPAN = 22          # weeks per season on the time axis: 18 regular-season weeks + a 4-week offseason gap
WINDOW_WEEKS = 3 * SEASON_SPAN
STATS = {"rush": "rush_off_epa", "pass": "pass_off_epa"}
KEEP = ["turnovers_off_b", "turnovers_def_b", "pen_presnap_c", "elo_b"]


def ridge_pregame(tg: pd.DataFrame, half_life: float, lam: float, adjust: bool = True) -> pd.DataFrame:
    """Pre-game ridge ratings per (game_id, team): off_rush, def_rush, off_pass, def_pass for that team entering the game."""
    teams = sorted(set(tg["team"]) | set(tg["opp"]))
    ti = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    g = tg[["game_id", "season", "week", "team", "opp"] + list(STATS.values())].copy()
    g["t"] = (g["season"] - g["season"].min()) * SEASON_SPAN + g["week"]
    g["o"] = g["team"].map(ti)
    g["d"] = g["opp"].map(ti)
    g = g.sort_values("t").reset_index(drop=True)
    t_arr, o_arr, d_arr = g["t"].to_numpy(), g["o"].to_numpy(), g["d"].to_numpy()
    decay = np.log(2.0) / half_life

    out_rows = []
    for t_now in sorted(g["t"].unique()):
        lo = np.searchsorted(t_arr, t_now - WINDOW_WEEKS, side="left")
        hi = np.searchsorted(t_arr, t_now, side="left")       # games strictly before this week
        ratings = {s: (np.zeros(T), np.zeros(T)) for s in STATS}
        if hi - lo >= 40:
            w = np.exp(-decay * (t_now - t_arr[lo:hi]))
            o, d = o_arr[lo:hi], d_arr[lo:hi]
            for s, col in STATS.items():
                y = g[col].to_numpy()[lo:hi]
                ok = ~np.isnan(y)
                ww, oo, dd, yy = w[ok], o[ok], d[ok], y[ok]
                if not adjust:
                    # control with the SAME time weighting and shrinkage but no opponent term: each side's rating is just the
                    # shrunk weighted mean of what that team's offense produced / what its defense allowed
                    mu = (ww * yy).sum() / ww.sum()
                    so, no_ = np.zeros(T), np.zeros(T)
                    np.add.at(so, oo, ww * (yy - mu)); np.add.at(no_, oo, ww)
                    sd_, nd_ = np.zeros(T), np.zeros(T)
                    np.add.at(sd_, dd, ww * (yy - mu)); np.add.at(nd_, dd, ww)
                    ratings[s] = (so / (no_ + lam), sd_ / (nd_ + lam))
                    continue
                n = 1 + 2 * T
                M = np.zeros((n, n))
                b = np.zeros(n)
                M[0, 0] = ww.sum()
                np.add.at(M[0], 1 + oo, ww); np.add.at(M[0], 1 + T + dd, ww)
                M[:, 0] = M[0, :]
                np.add.at(M, (1 + oo, 1 + oo), ww)
                np.add.at(M, (1 + T + dd, 1 + T + dd), ww)
                np.add.at(M, (1 + oo, 1 + T + dd), ww)
                np.add.at(M, (1 + T + dd, 1 + oo), ww)
                b[0] = (ww * yy).sum()
                np.add.at(b, 1 + oo, ww * yy); np.add.at(b, 1 + T + dd, ww * yy)
                M[np.arange(1, n), np.arange(1, n)] += lam
                beta = np.linalg.solve(M, b)
                ratings[s] = (beta[1:1 + T], beta[1 + T:])
        idx = np.where(t_arr == t_now)[0]
        for i in idx:
            row = {"game_id": g.at[i, "game_id"], "team": teams[o_arr[i]]}
            for s in STATS:
                row[f"off_{s}"] = ratings[s][0][o_arr[i]]
                row[f"def_{s}"] = ratings[s][1][o_arr[i]]      # this team's own defense rating (EPA/play it allows vs average)
            out_rows.append(row)
    return pd.DataFrame(out_rows)


def adj_diffs(tg: pd.DataFrame, half_life: float, lam: float, adjust: bool = True) -> pd.DataFrame:
    pre = ridge_pregame(tg, half_life, lam, adjust)
    side = tg[["game_id", "team", "is_home"]].merge(pre, on=["game_id", "team"], how="inner")
    h = side[side.is_home == 1].set_index("game_id")
    a = side[side.is_home == 0].set_index("game_id").reindex(h.index)
    out = pd.DataFrame(index=h.index)
    for s in STATS:
        out[f"adj_off_{s}"] = h[f"off_{s}"] - a[f"off_{s}"]
        out[f"adj_def_{s}"] = a[f"def_{s}"] - h[f"def_{s}"]       # higher = home defense allows less
    return out


def main() -> None:
    pd.set_option("display.width", 250)
    v53.L2 = L2
    d = v75.build_diffs()
    d = d[d.margin != 0].reset_index(drop=True)
    tg = build_team_games()
    base_cols = v75.EPA_ETC[:4] + KEEP
    base, _ = v53.walk_forward(d, base_cols)
    sb = v53.summarize(base)
    print(f"R0 (EWMA EPA stats): AUC={sb['auc']:.4f} logloss={sb['ll']:.4f} acc={sb['acc']:.4f}  ({len(base)} test games)\n")

    rng = np.random.default_rng(76)
    adj_names = [f"adj_{k}_{s}" for s in STATS for k in ("off", "def")]
    cols = adj_names + KEEP

    def run(H, lam, adjust):
        adj = adj_diffs(tg, H, lam, adjust)
        dd = d.merge(adj.reset_index(), on="game_id", how="inner")
        res, ws = v53.walk_forward(dd, cols)
        return dd, res, ws

    def judge(b, a):
        r = v53.compare("x", b, a, rng)
        sb_ = b.groupby("season").apply(lambda g: v53.summarize(g)["ll"], include_groups=False)
        sa_ = a.groupby("season").apply(lambda g: v53.summarize(g)["ll"], include_groups=False)
        r["ll_season_win"] = float(np.mean(sa_ < sb_))
        rep = all(r[f"dAUC_{h}"] > 0 and r[f"dLL_{h}"] > 0 for h in ("select", "confirm"))
        r["verdict"] = "ADOPT" if rep and r["p_ll_le0"] <= 0.01 and r["ll_season_win"] >= 0.6 else "DO NOT ADOPT"
        return r

    show = ["dAUC_all", "dAUC_select", "dAUC_confirm", "dLL_all", "dLL_select", "dLL_confirm", "p_ll_le0", "ll_season_win", "verdict"]
    for label, grid in (("Pre-registered grid", GRID_H), ("Second pass (exploratory, H added after seeing the first run)", EXPLORATORY_H)):
        rows = []
        for H in grid:
            for lam in GRID_LAMBDA:
                dd, adj_res, ws = run(H, lam, True)
                _, ctl_res, _ = run(H, lam, False)
                b2, _ = v53.walk_forward(dd, base_cols)
                sa, sc = v53.summarize(adj_res), v53.summarize(ctl_res)
                r1 = judge(ctl_res, adj_res)        # opponent adjustment, same time structure
                r2 = judge(b2, ctl_res)             # time structure alone vs production-style EWMA (R0)
                r3 = judge(b2, adj_res)             # adjusted vs R0 (the pre-registered comparison)
                rows.append({"cfg": f"H={H:.0f} lam={lam:.0f}", "AUC adj": sa["auc"], "AUC ctl": sc["auc"], "LL adj": sa["ll"], "LL ctl": sc["ll"],
                             **{f"adjVSctl {k}": r1[k] for k in show}, **{f"adjVSR0 {k}": r3[k] for k in ("dAUC_all", "dLL_all", "verdict")},
                             **{f"ctlVSR0 {k}": r2[k] for k in ("dAUC_all", "dLL_all")}})
        out = pd.DataFrame(rows).set_index("cfg")
        print(f"=== {label} ===")
        print(out.T.to_string(float_format=lambda v: f"{v:+.4f}" if isinstance(v, float) else v))
        print()


if __name__ == "__main__":
    main()
