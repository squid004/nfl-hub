"""v79: is the injury gain (v78 F4, mostly QB gaps) real once the QB gap uses only what was knowable before kickoff?

The historical QB gap in production's backtest (and in v78 F4) is the primary starter's trailing EPA minus the trailing EPA of the passer who
ACTUALLY played, so it also credits surprise changes: an unannounced benching, a starter hurt in warm-ups, an in-game replacement. None of
that is known at kickoff. The live production gate is stricter: the gap applies only when the primary QB is listed Out/Doubtful on the
injury report. This test builds that version for every historical game:

  primary    = plurality of the team's last 5 actual passers before this game (ties: most recent); no history -> no gap
  tp         = primary's mean per-game EPA/dropback over his earlier games (league mean if none)
  backup avg = attempt-weighted EPA/dropback of all earlier non-primary passer appearances (pooled, as in production)
  flag       = the primary's gsis id is Out/Doubtful for that team-week on the nflverse injury report
  gap        = flag x (tp - backup avg) x 0.6^(straight games before this one started by a non-primary)    [QB_GAP_DECAY_RATE 0.4]

All state is updated week by week after the games, so a gap never uses its own game or later ones. FantasyPros variants are not rebuilt
here (only the EPA gap).

Variants in the v77/v78 points-target model (baseline B = no injury terms, lambda 300), own and opponent roles, judged on win metrics:
  L  v78's leaky QB gaps (EPA gap, FantasyPros gap where available)      reference
  K  the leak-free EPA gap                                                 the test
  K+S  K plus skill-position EPA out
Pre-registered rule (v57+): dAUC and dLL > 0 in both halves, P(dLL <= 0) <= 0.01, better log loss in >= 60% of seasons, K vs B. Also reported:
how many team-games each version flags, and how many of the leaky version's flags the injury report would never have shown.

Run: PYTHONPATH=. python research/edge_signal_test_v79_leak_free_qb_gap.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict, deque

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import nflhub.sources.team_ratings as tr  # noqa: E402
import edge_signal_test_v53_walkforward_extra_terms as v53  # noqa: E402
import edge_signal_test_v75_research_model_no_points_sweep as v75  # noqa: E402
import edge_signal_test_v77_team_points_target as v77  # noqa: E402
import edge_signal_test_v78_points_model_new_families as v78  # noqa: E402
from nflhub.sources.edge_teams import UnknownTeamError, normalize_team  # noqa: E402

LAM = 300.0
FIRST = 2009
DECAY = 0.4


def _norm(code):
    try:
        return normalize_team(code)
    except UnknownTeamError:
        return None


def injured_primaries(current: int) -> set[tuple[str, int, int, str]]:
    out = set()
    for season in range(FIRST, current + 1):
        path = os.path.join(tr.DATA_DIR, f"injuries_{season}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, usecols=["season", "game_type", "team", "week", "gsis_id", "position", "report_status"])
        df = df[(df.game_type == "REG") & (df.position == "QB") & df.report_status.isin(["Out", "Doubtful"])]
        for t, w, g in zip(df.team, df.week, df.gsis_id):
            nt = _norm(t)
            if nt is not None and isinstance(g, str):
                out.add((nt, int(season), int(w), g))
    return out


def leak_free_gaps(current: int) -> dict[tuple[str, int, int], float]:
    frames = []
    for season in range(FIRST, current):
        f = tr._passer_game_stats(season, current).copy()
        f["season"] = season
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df["team"] = df["team"].map(_norm)
    df = df.dropna(subset=["team"])
    df["week"] = df["game_id"].str.split("_").str[1].astype(int)
    df["epa_game"] = df["epa_sum"] / df["attempts"]
    starters = df.loc[df.groupby(["game_id", "team"])["attempts"].idxmax()].sort_values(["season", "week", "game_id", "team"])
    inj = injured_primaries(current)

    hist5: dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
    passer_hist: dict[str, list[float]] = defaultdict(list)
    streak: dict[str, int] = defaultdict(int)
    league: list[float] = []
    bk_sum = bk_att = 0.0
    gaps: dict[tuple[str, int, int], float] = {}

    for (season, week), wk in starters.groupby(["season", "week"], sort=True):
        league_avg = float(np.mean(league)) if league else 0.0
        backup_avg = (bk_sum / bk_att) if bk_att else league_avg
        updates = []
        for r in wk.itertuples(index=False):
            h = hist5[r.team]
            primary = None
            if h:
                counts: dict[str, int] = defaultdict(int)
                for p in h:
                    counts[p] += 1
                best = max(counts.values())
                cands = {p for p, c in counts.items() if c == best}
                primary = next(p for p in reversed(h) if p in cands)
            gap = 0.0
            if primary is not None and (r.team, int(season), int(week), primary) in inj:
                ph = passer_hist.get(primary)
                tp = float(np.mean(ph)) if ph else league_avg
                gap = (tp - backup_avg) * (1 - DECAY) ** streak[r.team]
            gaps[(r.team, int(season), int(week))] = gap
            updates.append((r, primary))
        for r, primary in updates:           # state updates only after the whole week's gaps are fixed
            ap = r.passer_player_id
            if primary is not None and ap != primary:
                bk_sum += r.epa_sum
                bk_att += r.attempts
                streak[r.team] += 1
            else:
                streak[r.team] = 0
            passer_hist[ap].append(float(r.epa_game))
            league.append(float(r.epa_game))
            hist5[r.team].append(ap)
    return gaps


def main() -> None:
    pd.set_option("display.width", 250)
    v53.L2 = v77.L2_WIN
    d = v75.build_diffs()
    d = d[d.margin != 0].reset_index(drop=True)
    base_win, _ = v53.walk_forward(d, v75.R0)
    elo_diff = d.set_index("game_id")["elo_b"]
    current = max(int(g["season"]) for g in [{"season": s} for s in d.season.unique()]) + 1

    tg = v78.build_frame()                       # carries the leaky injury columns (qb_epa_gap, qb_fp_gap, skill_out)
    gaps = leak_free_gaps(current)
    key = list(zip(tg["team"], tg["season"], tg["week"]))
    tg["qb_free_gap"] = np.array([gaps.get(k, 0.0) for k in key])

    leaky = (tg["qb_epa_gap"].abs() + tg["qb_fp_gap"].abs()) > 0
    free = tg["qb_free_gap"].abs() > 0
    ok = tg["season"] >= 2011
    print("=== how many team-games each QB gap flags (2011-2025) ===")
    print(f"  leaky gap nonzero: {int((leaky & ok).sum())}   leak-free gap nonzero: {int((free & ok).sum())}   both: {int((leaky & free & ok).sum())}   "
          f"leaky only (not on the injury report, i.e. surprise changes): {int((leaky & ~free & ok).sum())}   leak-free only: {int((free & ~leaky & ok).sum())}")
    print(f"  share of the leaky gap's total |size| that the report would have shown: "
          f"{float(tg.loc[ok & leaky & free, ['qb_epa_gap', 'qb_fp_gap']].abs().sum().sum() / tg.loc[ok & leaky, ['qb_epa_gap', 'qb_fp_gap']].abs().sum().sum()):.0%}\n")

    smooth = sorted(set(v77.OWN_OFF + v77.OPP_DEF + v77.DIFFS))
    raw = ["qb_epa_gap", "qb_fp_gap", "skill_out", "qb_free_gap"]
    m = v78.team_rows(tg, smooth, raw)
    m = m[m.game_id.isin(elo_diff.index)].reset_index(drop=True)

    def run(inj_cols):
        X, cols = v77.features(m, [], elo_diff)
        for c in inj_cols:
            X["own_" + c] = m[c]
            X["opp_" + c] = m[c + "_o"]
        okm = X.notna().all(axis=1)
        mm, XX = m[okm].reset_index(drop=True), X[okm].reset_index(drop=True)
        res, rp = v77.walk_points(mm, XX, list(XX.columns), LAM)
        res = res[res.game_id.isin(base_win.game_id)]
        b = base_win[base_win.game_id.isin(res.game_id)].reset_index(drop=True)
        return res.set_index("game_id").loc[b.game_id].reset_index(), rp

    rng = np.random.default_rng(79)
    r0, rp0 = run([])
    s0 = v53.summarize(r0)
    print(f"B (no injury terms): AUC={s0['auc']:.4f} LL={s0['ll']:.4f} acc={s0['acc']:.4f} MAE={(rp0.pts - rp0.pred).abs().mean():.3f}\n")
    rows = []
    for name, cols in (("L  leaky QB gaps (v78 F4a)", ["qb_epa_gap", "qb_fp_gap"]), ("K  leak-free EPA gap", ["qb_free_gap"]),
                       ("K+S leak-free + skill-out", ["qb_free_gap", "skill_out"])):
        r, rp = run(cols)
        c = v53.compare("x", r0, r, rng)
        sb = r0.groupby("season").apply(lambda g: v53.summarize(g)["ll"], include_groups=False)
        sa = r.groupby("season").apply(lambda g: v53.summarize(g)["ll"], include_groups=False)
        c["seasons"] = float(np.mean(sa < sb))
        rep = all(c[f"dAUC_{h}"] > 0 and c[f"dLL_{h}"] > 0 for h in ("select", "confirm"))
        s = v53.summarize(r)
        rows.append({"variant": name, "AUC": s["auc"], "LL": s["ll"], "acc": s["acc"], "dAUC": c["dAUC_all"], "dLL": c["dLL_all"],
                     "dAUC_sel": c["dAUC_select"], "dAUC_con": c["dAUC_confirm"], "dLL_sel": c["dLL_select"], "dLL_con": c["dLL_confirm"],
                     "p": c["p_ll_le0"], "seasons": c["seasons"], "MAE": (rp.pts - rp.pred).abs().mean(),
                     "verdict": "ADOPT" if rep and c["p_ll_le0"] <= 0.01 and c["seasons"] >= 0.6 else "DO NOT ADOPT"})
    print(pd.DataFrame(rows).set_index("variant").to_string(float_format=lambda v: f"{v:+.4f}" if isinstance(v, float) else v))


if __name__ == "__main__":
    main()
