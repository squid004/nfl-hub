"""v80: does production's QB term still work when it uses only the pre-game (injury-report) QB gap?

v79 found that production's historical QB gap (the passer who ACTUALLY played, v64's `qb`/`fa` terms) flags 1,278 team-games in 2011-2025 against 309
for a gap built from the pre-game injury report alone, and that the live gate only ever applies the gap to report-flagged starters. So production's
QB weight and calibration were fit on a quantity the live model never sees. This re-runs v64's fully walk-forward pipeline (z-score stats from
training seasons, composite weights non-negative L2 = 0.03, corrections by win/loss logit on seasons >= 2009, calibration refit; momentum removed as
in production) with three QB treatments:

  P0  production today: EPA gap (FantasyPros gap where available), hindsight construction
  P1  the leak-free EPA gap (v79) in place of both QB terms
  P3  no QB term at all
Skill-out and home-field terms are as in production in all three. Test seasons 2011-2026.

Pre-registered rule (v57+) applied to P1 vs P3 (does the pre-game QB term earn its place?) and P1 vs P0 (is it at least as good as today's?): AUC and
log loss better in BOTH halves (<=2018 / >2018), bootstrap P(dLL <= 0) <= 0.01, better log loss in >= 60% of seasons. Also reported: the fitted QB
weight in each version (last fold), and calibration on the games the live gate would actually adjust (primary QB flagged Out/Doubtful): mean
predicted P(win) of the QB-affected side against its actual win rate.

Run: PYTHONPATH=. python research/edge_signal_test_v80_production_qb_term_leak_free.py
"""
from __future__ import annotations

import csv
import io
import os
import sys

import numpy as np
import requests
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import nflhub.sources.team_ratings as tr  # noqa: E402
from edge_signal_test_v54_pen_presnap_production_refit import fit_logit, fit_nonneg  # noqa: E402
from edge_signal_test_v61_ewma_speed_season_prior import compare  # noqa: E402
from edge_signal_test_v64_honest_walkforward_delta import p_home  # noqa: E402
from edge_signal_test_v68_qb_weight_ats_residual import fmt, verdict  # noqa: E402
import edge_signal_test_v79_leak_free_qb_gap as v79  # noqa: E402

M = tr.POWER_METRICS
L2 = 0.03
TEST_START, CORR_FROM = 2011, 2009


def main():
    resp = requests.get(tr.GAMES_URL, timeout=tr.TIMEOUT)
    resp.raise_for_status()
    rows = [r for r in csv.DictReader(io.StringIO(resp.text)) if r["game_type"] == "REG"]
    played = [r for r in rows if r.get("result") not in ("", "NA", None) and int(r["season"]) >= tr.FIRST_SEASON]
    current = max(int(r["season"]) for r in played)
    log: list[dict] = []
    tr.compute_ratings(played, current, game_log=log)
    log = [r for r in log if r["home_score"] != r["away_score"]]
    seasons = np.array([r["season"] for r in log])
    y = np.array([1.0 if r["home_score"] > r["away_score"] else 0.0 for r in log])
    X = np.column_stack([[r["diffs"][m] for r in log] for m in M])

    qb_gap, *_ = tr.qb_quality_gap_data(played, current)
    fantasy_gap, _t, _l, _b, fantasy_available = tr.qb_fantasy_gap_data(played, current, 1)
    skill_out = tr.skill_epa_out_by_team_week(current)
    hfa_by_season = tr.home_field_logit_by_season(played)
    qb_out = tr._qb_out_doubtful_by_week(current)
    free_gaps = v79.leak_free_gaps(current)

    def g(d, r, side):
        return d.get((r[side], r["season"], r["week"]), 0.0)

    qb = np.array([g(qb_gap, r, "home") - g(qb_gap, r, "away") for r in log])
    fa = np.array([g(fantasy_gap, r, "home") - g(fantasy_gap, r, "away") for r in log])
    free = np.array([g(free_gaps, r, "home") - g(free_gaps, r, "away") for r in log])
    sk = np.array([g(skill_out, r, "home") - g(skill_out, r, "away") for r in log])
    hfa = np.array([0.0 if r.get("neutral") else hfa_by_season.get(r["season"], 0.0) for r in log])
    use_fp = np.array([fantasy_available.get((r["season"], r["week"]), False) for r in log])
    clean = np.array([not (qb_out.get((r["home"], r["season"], r["week"]), False) or qb_out.get((r["away"], r["season"], r["week"]), False)) for r in log])
    flagged_home = np.array([free_gaps.get((r["home"], r["season"], r["week"]), 0.0) != 0 for r in log])
    flagged_away = np.array([free_gaps.get((r["away"], r["season"], r["week"]), 0.0) != 0 for r in log])

    names = ("P0 production (hindsight gap)", "P1 leak-free EPA gap", "P3 no QB term")
    keys = ("P0", "P1", "P3")
    test_seasons = [s for s in sorted(set(seasons)) if s >= TEST_START]
    probs = {k: np.zeros(len(y)) for k in keys}
    qbw = {}
    for S in test_seasons:
        tri, tei = seasons < S, seasons == S
        mu_t, sd_t = X[tri].mean(0), X[tri].std(0)
        sd_t[sd_t == 0] = 1.0
        w = fit_nonneg((X[tri] - mu_t) / sd_t, y[tri], L2)[1:]
        d_raw = ((X - mu_t) / sd_t) @ w
        kp = tri & (seasons >= CORR_FROM)
        for k in keys:
            if k == "P0":
                feats = np.column_stack([d_raw, qb, fa, sk, hfa])[kp]
            elif k == "P1":
                feats = np.column_stack([d_raw, free, sk, hfa])[kp]
            else:
                feats = np.column_stack([d_raw, sk, hfa])[kp]
            live = feats.std(0) > 0
            live[0] = True
            wj, _se = fit_logit(feats[:, live], y[kp])
            ratio = np.zeros(feats.shape[1])
            ratio[live] = np.concatenate([[1.0], wj[2:] / wj[1]])
            if k == "P0":
                c = (ratio[1], ratio[2])
                d = d_raw + np.where(use_fp, ratio[2] * fa, ratio[1] * qb) + ratio[3] * sk + ratio[4] * hfa
            elif k == "P1":
                c = (ratio[1],)
                d = d_raw + ratio[1] * free + ratio[2] * sk + ratio[3] * hfa
            else:
                c = (0.0,)
                d = d_raw + ratio[1] * sk + ratio[2] * hfa
            km = kp & clean & (d != 0)
            fav = np.where(d > 0, y, 1 - y)
            wc, _ = fit_logit(np.abs(d[km])[:, None], fav[km])
            for i in np.where(tei)[0]:
                probs[k][i] = p_home(d[i], wc[0], wc[1], True)
            if S == test_seasons[-1]:
                qbw[k] = c
    test = seasons >= TEST_START
    yt, ss = y[test], seasons[test]
    sel, con = ss <= 2018, ss > 2018
    print(f"{test.sum()} decided test games, seasons {TEST_START}-{max(test_seasons)}\n")
    print(f"{'variant':32s} {'AUC':>7} {'LL':>7} {'acc':>7} | AUC<=18 AUC>18 | QB weight(s), last fold")
    for nm, k in zip(names, keys):
        p = probs[k][test]
        print(f"{nm:32s} {roc_auc_score(yt, p):7.4f} {log_loss(yt, p):7.4f} {np.mean((p > .5) == (yt == 1)):7.4f} | "
              f"{roc_auc_score(yt[sel], p[sel]):.4f} {roc_auc_score(yt[con], p[con]):.4f} | {tuple(round(float(v), 3) for v in qbw[k])}")
    rng = np.random.default_rng(80)
    print()
    for lab, base, alt in (("P1 vs P3 (does the pre-game QB term earn its place?)", "P3", "P1"),
                           ("P1 vs P0 (as good as today's hindsight term?)", "P0", "P1"),
                           ("P0 vs P3 (what the hindsight term claims)", "P3", "P0")):
        c = compare(yt, ss, probs[base][test], probs[alt][test], rng)
        ok, rep = verdict(c)
        print(f"{lab}: {fmt(c)} -> {'ADOPT' if ok else 'DO NOT ADOPT'}")

    # the games the live gate would adjust: the QB-affected side is the team whose primary is flagged
    print("\n=== calibration on games where one team's primary QB is flagged Out/Doubtful (the live gate's cases) ===")
    side_home = flagged_home & ~flagged_away
    side_away = flagged_away & ~flagged_home
    one = test & (side_home | side_away)
    for nm, k in zip(names, keys):
        p_h = probs[k][one]
        p_aff = np.where(side_home[one], p_h, 1 - p_h)           # P(the QB-affected team wins)
        y_aff = np.where(side_home[one], y[one], 1 - y[one])
        print(f"  {nm:32s} n={one.sum()}: mean predicted P(affected team wins) {p_aff.mean():.3f} vs actual {y_aff.mean():.3f}  "
              f"(log loss {log_loss(y_aff, np.clip(p_aff, 1e-4, 1 - 1e-4)):.4f})")


if __name__ == "__main__":
    main()
