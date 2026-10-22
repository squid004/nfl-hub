"""v68: does refitting the QB / fantasy-QB weight against the ATS residual beat the win/loss fit? (cowork plan B6.1)

Context: model-side ATS in QB-flag games is 46.8% (v60). The QB-gap weight in `delta` is fit to maximise
win/loss likelihood. If the market prices a QB difference more (or less) heavily than the model, the
market residual (actual margin - closing spread) should correlate with the QB gap even after the model
has used it. This test updates the QB weights from that residual, walk-forward, and asks whether the
resulting `delta` predicts game WINNERS better. The spread is a TRAINING TARGET only; it is never an input
to `delta` at prediction time, so the model stays market-free in the sense of v52-v58.

Baseline (B): the fully walk-forward pipeline of v64's V4 (z-score from training seasons, composite weights,
corrections incl. QB / fantasy refit by win/loss logit, calibration refit), test seasons 2011+.
Candidates (same pipeline, QB weights changed in the correction step only):
  C1  QB weight (non-FantasyPros games) shifted by beta_qb / k, beta_qb = OLS slope of the ATS residual
      (home margin - closing home spread) on the QB gap over all training games from 2009
  C2  C1 + the FantasyPros-QB weight shifted the same way, from training games where it was available
  C3  C2, but the residual regressions use only QB-flag training games (a QB listed out/doubtful)
k = margin points per delta unit (OLS through the origin of home margin on the baseline delta, training).

Pre-registered ADOPT rule (v57/v61-v63, fixed before running): a candidate replaces the baseline only if
dAUC > 0 AND dLogLoss > 0 in BOTH halves (<=2018 / >2018), paired-bootstrap P(dLL <= 0) <= 0.01 (N_BOOT=2000),
and log loss better in >= 60% of test seasons. Diagnostics (not part of the rule): model-side ATS hit rate
overall and on QB-flag games, by half; the fitted shifts by season.

Run: PYTHONPATH=. python research/edge_signal_test_v68_qb_weight_ats_residual.py
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
from edge_signal_test_v54_pen_presnap_production_refit import fit_logit, fit_nonneg  # noqa: E402
from edge_signal_test_v61_ewma_speed_season_prior import compare  # noqa: E402

M = tr.POWER_METRICS
L2 = 0.03
TEST_START = 2011
CORR_FROM = 2009
SELECT_MAX = 2018
BREAKEVEN = 0.5238


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def p_home(delta, a, b):
    pf = np.minimum(np.maximum(sig(a + b * np.abs(delta)), 0.5), tr.DELTA_PROB_MAX)
    return np.where(delta > 0, pf, 1 - pf)


def ols_slope(x, r):
    """Slope of r on x with an intercept; 0 when x has no variance."""
    if len(x) < 30 or x.std() == 0:
        return 0.0
    xc = x - x.mean()
    return float((xc * (r - r.mean())).sum() / (xc ** 2).sum())


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
    rows = [r for r in csv.DictReader(io.StringIO(resp.text)) if r["game_type"] == "REG"]
    played = [r for r in rows if r.get("result") not in ("", "NA", None) and int(r["season"]) >= tr.FIRST_SEASON]
    current = max(int(r["season"]) for r in played)
    log: list[dict] = []
    tr.compute_ratings(played, current, game_log=log)
    log = [r for r in log if r["home_score"] != r["away_score"]]
    seasons = np.array([r["season"] for r in log])
    y = np.array([1.0 if r["home_score"] > r["away_score"] else 0.0 for r in log])
    margin = np.array([r["home_score"] - r["away_score"] for r in log], float)
    spread = np.array([r["spread_line"] for r in log], float)      # home-favored positive
    ats_resid = margin - spread
    X = np.column_stack([[r["diffs"][m] for r in log] for m in M])

    qb_gap, *_ = tr.qb_quality_gap_data(played, current)
    fantasy_gap, _t, _l, _b, fantasy_available = tr.qb_fantasy_gap_data(played, current, 1)
    skill_out = tr.skill_epa_out_by_team_week(current)
    hfa_by_season = tr.home_field_logit_by_season(played)
    streak = tr.team_streak_by_week(played)
    qb_out = tr._qb_out_doubtful_by_week(current)

    def g(d, r, side):
        return d.get((r[side], r["season"], r["week"]), 0.0)

    qb = np.array([g(qb_gap, r, "home") - g(qb_gap, r, "away") for r in log])
    fa = np.array([g(fantasy_gap, r, "home") - g(fantasy_gap, r, "away") for r in log])
    sk = np.array([g(skill_out, r, "home") - g(skill_out, r, "away") for r in log])
    hfa = np.array([0.0 if r.get("neutral") else hfa_by_season.get(r["season"], 0.0) for r in log])
    stk = np.array([g(streak, r, "home") - g(streak, r, "away") for r in log])
    use_fp = np.array([fantasy_available.get((r["season"], r["week"]), False) for r in log])
    flag = np.array([qb_out.get((r["home"], r["season"], r["week"]), False) or qb_out.get((r["away"], r["season"], r["week"]), False) for r in log])
    clean = ~flag

    def delta_from(dr, c):
        qbh = np.where(use_fp, c["fantasy"] * fa, c["qb"] * qb)
        return dr + qbh + c["skill"] * sk + c["hfa"] * hfa + c["streak"] * stk

    test_seasons = [s for s in sorted(set(seasons)) if s >= TEST_START]
    names = ("B", "C1", "C2", "C3")
    delta_v = {n: np.full(len(y), np.nan) for n in names}
    cal_v = {n: np.zeros((len(y), 2)) for n in names}
    shifts = {}

    for S in test_seasons:
        tri, tei = seasons < S, seasons == S
        mu_t, sd_t = X[tri].mean(0), X[tri].std(0)
        sd_t[sd_t == 0] = 1.0
        w = fit_nonneg((X[tri] - mu_t) / sd_t, y[tri], L2)[1:]
        d_raw = ((X - mu_t) / sd_t) @ w
        kp = tri & (seasons >= CORR_FROM)
        feats = np.column_stack([d_raw, qb, fa, sk, hfa, stk])[kp]
        live = feats.std(0) > 0
        live[0] = True
        wj, _se = fit_logit(feats[:, live], y[kp])
        ratio = np.zeros(6)
        ratio[live] = np.concatenate([[1.0], wj[2:] / wj[1]])
        base_c = dict(qb=ratio[1], fantasy=ratio[2], skill=ratio[3], hfa=ratio[4], streak=ratio[5])
        d_base = delta_from(d_raw, base_c)
        # k: margin points per delta unit on training games (through the origin)
        k = float((d_base[kp] * margin[kp]).sum() / (d_base[kp] ** 2).sum())
        cands = {"B": base_c}
        for name, use_flag_only, shift_fp in (("C1", False, False), ("C2", False, True), ("C3", True, True)):
            c = dict(base_c)
            sel = kp & ~use_fp & (flag if use_flag_only else True)
            b_qb = ols_slope(qb[sel], ats_resid[sel])
            c["qb"] = base_c["qb"] + b_qb / k
            b_fa = 0.0
            if shift_fp:
                self_fp = kp & use_fp & (flag if use_flag_only else True)
                b_fa = ols_slope(fa[self_fp], ats_resid[self_fp])
                c["fantasy"] = base_c["fantasy"] + b_fa / k
            cands[name] = c
            shifts[(S, name)] = (b_qb / k, b_fa / k)
        for name, c in cands.items():
            d = delta_from(d_raw, c)
            km = kp & clean & (d != 0)
            fav = np.where(d > 0, y, 1 - y)
            wc, _ = fit_logit(np.abs(d[km])[:, None], fav[km])
            delta_v[name][tei] = d[tei]
            cal_v[name][tei] = (wc[0], wc[1])
        shifts[(S, "k")] = k

    test = seasons >= TEST_START
    yt, ss = y[test], seasons[test]
    sel_m, con_m = ss <= SELECT_MAX, ss > SELECT_MAX
    probs = {}
    for n in names:
        d, cal = delta_v[n][test], cal_v[n][test]
        probs[n] = np.array([p_home(d[i], cal[i, 0], cal[i, 1]) for i in range(len(d))]).reshape(-1)
    print(f"{len(yt)} decided test games, seasons {TEST_START}-{max(test_seasons)}\n")
    rng = np.random.default_rng(68)
    print("=== ADOPT test vs baseline B (walk-forward, win/loss-fit corrections) ===")
    for n in names[1:]:
        c = compare(yt, ss, probs["B"], probs[n], rng)
        ok, rep = verdict(c)
        print(f"{n}: {fmt(c)}; replicated both halves={rep} -> {'ADOPT' if ok else 'DO NOT ADOPT'}")

    print("\n=== diagnostic: model-side ATS hit rate (pick home if k*delta > closing home spread), pushes dropped ===")
    margin_t, spread_t, flag_t = margin[test], spread[test], flag[test]
    for n in names:
        kk = np.array([shifts[(s, "k")] for s in ss])
        pick_home = kk * delta_v[n][test] > spread_t
        cover_home = margin_t > spread_t
        nopush = margin_t != spread_t
        hit = (pick_home == cover_home)
        row = []
        for lab, m in (("all", np.ones(len(yt), bool)), ("QB-flag", flag_t)):
            for hl, hm in (("<=18", sel_m), (">18", con_m)):
                mm = m & hm & nopush
                row.append(f"{lab} {hl}: {hit[mm].mean():.3f} (n={mm.sum()})")
        print(f"  {n}: " + " | ".join(row) + f"   [breakeven {BREAKEVEN}]")

    print("\n=== fitted QB-weight shifts (delta units) by test season: qb / fantasy; k = margin pts per delta ===")
    for S in test_seasons:
        print(f"  {S}: C1 qb {shifts[(S,'C1')][0]:+.3f} | C2 fa {shifts[(S,'C2')][1]:+.4f} | C3 qb {shifts[(S,'C3')][0]:+.3f} fa {shifts[(S,'C3')][1]:+.4f} | k {shifts[(S,'k')]:.2f}")


if __name__ == "__main__":
    main()
