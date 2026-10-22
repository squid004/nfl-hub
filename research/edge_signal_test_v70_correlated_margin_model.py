"""v70: independent-scores vs. direct margin model for the ATS tab (cowork plan B7.2).

Production's score pipeline (team_ratings.build_score_distributions) builds each side's score as an independent
Normal(mean, sd) discretized to integers and re-weighted by the historical per-score frequency (key scores 3, 6, 7, 10...), then
the margin follows by treating the two scores as independent. Real scores are correlated (pace, game script), and the margin has
its own key-number mass at 3 and 7 that independent scores smear. Alternative: model the MARGIN directly.

Both models use the same pregame mean margin mu = home_pred_points - away_pred_points from the production game_log, so only the
SHAPE of the margin distribution differs:
  A  independent scores: each side Normal(pred, training residual sd), discretized, historically re-weighted (Steps 1-2 of
     production, no bisection onto a target win probability -- the means already carry the model's view), margin by convolution
  A2 = A with production's bisection (build_score_distributions): the symmetric mean shift that makes A's P(home win) equal B's, i.e.
     A given B's win probability (a deliberately strong baseline -- it isolates the SHAPE of the margin distribution from the mean/scale)
  B  direct margin: empirical margin distribution of the K = 400 training games whose mu is nearest this game's mu, each margin
     shifted by round(mu_target - mu_neighbor); +0.02 pseudo-mass per integer margin in -45..45 so unseen margins are not zero

Walk-forward: for test season S (2017+, so the 2015+ score-frequency table has training data) everything is fit on seasons < S.
Metrics on the same test games: (1) mean log-likelihood of the ACTUAL integer margin under each pmf (shape accuracy);
(2) calibration of P(home covers the closing spread) (pushes dropped): log loss / Brier, plus reliability by predicted-probability bin.

DECISIVE comparison is B vs A2 (A alone is a straw man: it never gets the mean calibration production's bisection provides).
Pre-registered rule (fixed before running; same discipline as v57+): B replaces A2 only if its margin log-likelihood AND cover log
loss are both better in BOTH halves (<=2021 / >2021), paired-bootstrap P(no improvement) <= 0.01 on each of the two metrics
(N_BOOT = 2000), and the margin log-likelihood is better in >= 60% of test seasons.

Run: PYTHONPATH=. python research/edge_signal_test_v70_correlated_margin_model.py
"""
from __future__ import annotations

import csv
import io
import sys

import numpy as np
import requests

import nflhub.sources.team_ratings as tr

TEST_START = 2017
HALF = 2021
K_NEIGHBORS = 400
PSEUDO = 0.02
MARGIN_RANGE = 45
N_BOOT = 2000
MAXS = tr.MAX_TEAM_SCORE


def margin_pmf_independent(mean_h, mean_a, sd_h, sd_a, hist):
    h = tr._reweight_score_by_history(tr._raw_score_distribution(mean_h, sd_h), hist)
    a = tr._reweight_score_by_history(tr._raw_score_distribution(mean_a, sd_a), hist)
    conv = np.convolve(h, a[::-1])            # index i <-> margin i - MAXS
    margins = np.arange(len(conv)) - MAXS
    return margins, conv


def pmf_on_grid(margins, p):
    grid = np.arange(-MARGIN_RANGE, MARGIN_RANGE + 1)
    out = np.zeros(len(grid))
    for m, pr in zip(margins, p):
        j = int(np.clip(m, -MARGIN_RANGE, MARGIN_RANGE)) + MARGIN_RANGE
        out[j] += pr
    return out / out.sum()


def cover_prob(pmf, line):
    """P(home margin > line) given no push; pmf on the integer grid -MARGIN_RANGE..MARGIN_RANGE."""
    grid = np.arange(-MARGIN_RANGE, MARGIN_RANGE + 1)
    above = pmf[grid > line].sum()
    push = pmf[grid == line].sum() if float(line).is_integer() else 0.0
    return above / max(1e-12, 1 - push)


def boot_p(d, rng):
    n = len(d)
    means = [d[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)]
    return float(np.mean(np.array(means) <= 0))


def main():
    resp = requests.get(tr.GAMES_URL, timeout=tr.TIMEOUT)
    resp.raise_for_status()
    rows = [r for r in csv.DictReader(io.StringIO(resp.text)) if r["game_type"] == "REG"]
    played = [r for r in rows if r.get("result") not in ("", "NA", None) and int(r["season"]) >= tr.FIRST_SEASON]
    current = max(int(r["season"]) for r in played)
    log: list[dict] = []
    tr.compute_ratings(played, current, game_log=log)
    log = [r for r in log if r.get("home_pred_points") is not None and r.get("away_pred_points") is not None]
    seasons = np.array([r["season"] for r in log])
    mu = np.array([r["home_pred_points"] - r["away_pred_points"] for r in log])
    hp = np.array([r["home_pred_points"] for r in log])
    ap = np.array([r["away_pred_points"] for r in log])
    hs = np.array([r["home_score"] for r in log], float)
    as_ = np.array([r["away_score"] for r in log], float)
    margin = (hs - as_).astype(int)
    line = np.array([r["spread_line"] for r in log], float)    # home-favored positive: home covers if margin > line
    grid = np.arange(-MARGIN_RANGE, MARGIN_RANGE + 1)

    ll_a, ll_b, ll_a2, cov_a, cov_b, cov_a2, cov_y, cov_s = [], [], [], [], [], [], [], []
    for S in [s for s in sorted(set(seasons)) if s >= TEST_START]:
        tri, tei = np.where(seasons < S)[0], np.where(seasons == S)[0]
        hist = tr.historical_score_frequency([g for g in played if int(g["season"]) < S])
        sd_h = float(np.std(hs[tri] - hp[tri]))
        sd_a = float(np.std(as_[tri] - ap[tri]))
        mu_tr, mar_tr = mu[tri], margin[tri]
        for i in tei:
            m_t = int(np.clip(margin[i], -MARGIN_RANGE, MARGIN_RANGE))
            # A: independent scores
            mg, p = margin_pmf_independent(hp[i], ap[i], sd_h, sd_a, hist)
            pa = pmf_on_grid(mg, p)
            # B: direct margin from the K nearest training games by mu
            nn = np.argsort(np.abs(mu_tr - mu[i]))[:K_NEIGHBORS]
            shifted = np.rint(mar_tr[nn] + np.rint(mu[i] - mu_tr[nn])).astype(int)
            pb = np.full(len(grid), PSEUDO)
            for m in shifted:
                pb[int(np.clip(m, -MARGIN_RANGE, MARGIN_RANGE)) + MARGIN_RANGE] += 1.0
            pb /= pb.sum()
            target = pb[grid > 0].sum()
            h2, a2 = tr.build_score_distributions(hp[i], ap[i], sd_h, sd_a, hist, float(target))
            mg2 = np.arange(2 * MAXS + 1) - MAXS
            pa2 = pmf_on_grid(mg2, np.convolve(h2, a2[::-1]))
            ll_a2.append((S, np.log(max(pa2[m_t + MARGIN_RANGE], 1e-9))))
            ll_a.append((S, np.log(max(pa[m_t + MARGIN_RANGE], 1e-9))))
            ll_b.append((S, np.log(pb[m_t + MARGIN_RANGE])))
            if margin[i] != line[i]:                          # drop pushes
                cov_a.append(cover_prob(pa, line[i]))
                cov_b.append(cover_prob(pb, line[i]))
                cov_a2.append(cover_prob(pa2, line[i]))
                cov_y.append(1.0 if margin[i] > line[i] else 0.0)
                cov_s.append(S)
    s_ll = np.array([s for s, _ in ll_a])
    la, lb = np.array([v for _, v in ll_a]), np.array([v for _, v in ll_b])
    la2 = np.array([v for _, v in ll_a2])
    ca2 = np.clip(np.array(cov_a2), 1e-6, 1 - 1e-6)
    cs, cy = np.array(cov_s), np.array(cov_y)
    ca, cb = np.clip(np.array(cov_a), 1e-6, 1 - 1e-6), np.clip(np.array(cov_b), 1e-6, 1 - 1e-6)

    def cll(p):
        return -(cy * np.log(p) + (1 - cy) * np.log(1 - p))

    rng = np.random.default_rng(70)
    print(f"{len(la)} test games ({len(cy)} after dropping pushes), seasons {TEST_START}-{s_ll.max()}\n")
    print("=== (1) mean log-likelihood of the actual margin (higher = better) ===")
    for lab, m in (("all", s_ll > 0), (f"<={HALF}", s_ll <= HALF), (f">{HALF}", s_ll > HALF)):
        print(f"  {lab:7s}: A {la[m].mean():.4f}   B {lb[m].mean():.4f}   B-A {np.mean(lb[m] - la[m]):+.4f}")
    d1 = lb - la
    p1 = boot_p(d1, rng)
    wins1 = np.mean([d1[s_ll == s].mean() > 0 for s in sorted(set(s_ll))])
    print(f"  paired bootstrap P(B-A <= 0) = {p1:.3f}; B better in {wins1:.0%} of seasons")

    print("\n=== (2) P(home covers) calibration (lower log loss / Brier = better) ===")
    for lab, m in (("all", cs > 0), (f"<={HALF}", cs <= HALF), (f">{HALF}", cs > HALF)):
        print(f"  {lab:7s}: logloss A {cll(ca)[m].mean():.4f} B {cll(cb)[m].mean():.4f} (A-B {np.mean(cll(ca)[m] - cll(cb)[m]):+.4f}) | "
              f"Brier A {np.mean((ca - cy)[m] ** 2):.4f} B {np.mean((cb - cy)[m] ** 2):.4f}")
    d2 = cll(ca) - cll(cb)
    p2 = boot_p(d2, rng)
    print(f"  paired bootstrap P(A-B <= 0) = {p2:.3f}")
    print("  reliability by predicted cover probability (home):")
    for lo, hi in ((0, .35), (.35, .45), (.45, .5), (.5, .55), (.55, .65), (.65, 1.01)):
        for lab, p in (("A", ca), ("B", cb)):
            m = (p >= lo) & (p < hi)
            if m.sum() >= 30:
                print(f"    {lab} {lo:.2f}-{min(hi, 1):.2f}: n={m.sum():4d} predicted {p[m].mean():.3f} actual {cy[m].mean():.3f}")

    print(chr(10) + "=== vs the strong baseline A2 (A with B's win probability) ===")
    for lab, m, m2 in (("all", s_ll > 0, cs > 0), (f"<={HALF}", s_ll <= HALF, cs <= HALF), (f">{HALF}", s_ll > HALF, cs > HALF)):
        print(f"  {lab:7s}: margin LL A2 {la2[m].mean():.4f} B {lb[m].mean():.4f} (B-A2 {np.mean(lb[m] - la2[m]):+.4f}) | "
              f"cover logloss A2 {cll(ca2)[m2].mean():.4f} B {cll(cb)[m2].mean():.4f} (A2-B {np.mean(cll(ca2)[m2] - cll(cb)[m2]):+.4f})")
    q1 = boot_p(lb - la2, rng)
    q2 = boot_p(cll(ca2) - cll(cb), rng)
    wins_a2 = np.mean([(lb - la2)[s_ll == s].mean() > 0 for s in sorted(set(s_ll))])
    ok2 = all(np.mean((lb - la2)[mk]) > 0 and np.mean((cll(ca2) - cll(cb))[mk2]) > 0
              for mk, mk2 in (((s_ll <= HALF), (cs <= HALF)), ((s_ll > HALF), (cs > HALF))))
    print(f"  paired bootstrap P(B-A2 <= 0): margin LL {q1:.3f}, cover logloss {q2:.3f}; margin-LL season-win {wins_a2:.2f}; both halves={ok2}")
    print(f"  -> {'ADOPT B over production-style A2' if ok2 and q1 <= 0.01 and q2 <= 0.01 and wins_a2 >= 0.6 else 'DO NOT ADOPT over A2'}")

    ok_halves = all(np.mean((lb - la)[mk]) > 0 and np.mean((cll(ca) - cll(cb))[mk2]) > 0
                    for mk, mk2 in (((s_ll <= HALF), (cs <= HALF)), ((s_ll > HALF), (cs > HALF))))
    adopt = ok_halves and p1 <= 0.01 and p2 <= 0.01 and wins1 >= 0.6
    print(f"\n(informational only -- weak baseline A without mean calibration: rule would pass = {adopt}; "
          f"the DECISIVE comparison is the A2 verdict above)")


if __name__ == "__main__":
    main()
