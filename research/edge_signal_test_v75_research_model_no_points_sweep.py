"""v75: research model R -- a composite that never uses points scored / points allowed -- and which candidate stats win its weight back.

Context (user direction 2026-10-29): leave production as it is, and build a parallel research model whose stats are all play-level
measures (never PPG / PAPG), then add back the information those two stats carry through better-defined inputs. v74 showed that
forcing both to 0 costs ~0.2 AUC points (and hands defense real weight); v52-v58 already screened 17 candidate stats (explosive
plays, penalties, tempo/drive quality, special teams, field position) but always ON TOP of a composite that contained PPG/PAPG, where
most of what they offered was redundant. This re-runs that sweep with PPG/PAPG removed, so a candidate can earn the weight they used to hold.

Model R0 (baseline): rush/pass offense EPA, rush/pass defense EPA, both turnover stats, pre-snap penalty yards, Elo (the production
composite minus points_off and points_def_allowed). Reference P10 = R0 + the two points stats (production's stat set).
Ratings use production's settings (EWMA alpha 0.08, bias-corrected, season prior fading over 8 games); weights are the same
non-negative L2 = 0.03 logistic refit walk-forward by season from 2011.

Candidates (v52's team-game extracts, 17 stats): explosive plays (4), penalty yards (2), time of possession / pace / drive score (5),
special teams (6: ST EPA, kick+punt, kicking, returning, field-position net, field goals over expected). Each is added one at a time to
R0, then greedily stacked.

Pre-registered rule (v55's SHIP / WATCH tiers, fixed before running): SHIP = dAUC > 0 and dLL > 0 in both halves (<=2018 / >2018),
paired-bootstrap P(dLL <= 0) <= 0.01, fitted weight > 0 in >= 90% of folds, better log loss in >= 60% of seasons. 17 looks (a false
SHIP somewhere is ~16% likely before the replication requirement, which cuts it further). The stacked model is then compared with R0
and with P10: how much of the gap to production did it recover?

Run: PYTHONPATH=. python research/edge_signal_test_v75_research_model_no_points_sweep.py
"""
from __future__ import annotations

import csv
import io
import os
import sys

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
import nflhub.sources.team_ratings as tr  # noqa: E402
import edge_signal_test_v53_walkforward_extra_terms as v53  # noqa: E402
from edge_signal_test_v52_extra_terms_screen import BENCH, build_team_games  # noqa: E402
from edge_signal_test_v55_market_free_candidate_sweep import CANDS, judge  # noqa: E402

# nflverse's release downloads (the uncached current-season play-by-play file in particular) intermittently time out with a 504; research runs
# retry instead of dying 20 minutes in. Research-only: production's own function is untouched.
_orig_phase_stats = tr._phase_stats_for_season


def _phase_stats_retry(season: int, current_season: int):
    import time
    for attempt in range(6):
        try:
            return _orig_phase_stats(season, current_season)
        except Exception as exc:  # noqa: BLE001
            if attempt == 5:
                raise
            print(f"    play-by-play {season} download failed ({exc}); retry {attempt + 1}/5", file=sys.stderr)
            time.sleep(20 * (attempt + 1))


tr._phase_stats_for_season = _phase_stats_retry

L2 = 0.03
ALPHA, PRIOR_GAMES = 0.08, 8
POINTS = ["points_off_b", "points_def_b"]
EPA_ETC = [c for c in BENCH if c not in POINTS]            # 4 EPA stats + 2 turnover stats
R0 = EPA_ETC + ["pen_presnap_c", "elo_b"]
P10 = R0 + POINTS


def pregame_prod(tg: pd.DataFrame, cols: list[str], alpha: float = ALPHA, prior_games: int = PRIOR_GAMES) -> pd.DataFrame:
    """Pre-game rating per team-game with production's scheme: in-season bias-corrected EWMA blended with last season's ending
    rating, whose weight fades linearly to 0 over `prior_games` games. NaN observations do not update the rating."""
    arr = np.full((len(tg), len(cols)), np.nan)
    pos_of = {ix: p for p, ix in enumerate(tg.index)}
    for _, grp in tg.sort_values(["season", "week"]).groupby("team"):
        vals = grp[cols].to_numpy(float)
        seasons = grp["season"].to_numpy()
        acc = np.zeros(len(cols)); n_upd = np.zeros(len(cols)); ewma = np.full(len(cols), np.nan)
        prior = np.full(len(cols), np.nan)
        last_season, k = None, 0
        for i, ix in enumerate(grp.index):
            if seasons[i] != last_season:
                # last season's ending blended rating becomes the prior; the in-season state resets
                if last_season is not None:
                    w_end = max(0.0, (prior_games - k) / prior_games)
                    end = np.where(np.isnan(prior), ewma, np.where(np.isnan(ewma), prior, w_end * prior + (1 - w_end) * ewma))
                    prior = end
                acc[:] = 0.0; n_upd[:] = 0.0; ewma = np.full(len(cols), np.nan)
                last_season, k = seasons[i], 0
            w = max(0.0, (prior_games - k) / prior_games)
            blend = np.where(np.isnan(prior), ewma, np.where(np.isnan(ewma), prior, w * prior + (1 - w) * ewma))
            arr[pos_of[ix]] = blend
            v = vals[i]
            ok = ~np.isnan(v)
            k += 1
            n_upd[ok] += 1
            acc[ok] = (1 - alpha) * acc[ok] + alpha * v[ok]
            ewma[ok] = acc[ok] / (1 - (1 - alpha) ** n_upd[ok])
    return pd.DataFrame(arr, index=tg.index, columns=cols)


def build_diffs() -> pd.DataFrame:
    tg = build_team_games()
    tg["sec_per_play_fast_c"] = -tg["sec_per_play_c"]
    # Elo (production's, from compute_ratings' own game log) as a per-team-game value: the home/away pre-game Elo implied by diffs is a
    # DIFFERENCE, so carry it as a home-minus-away column after the team-game EWMA instead (Elo needs no smoothing).
    resp = requests.get(tr.GAMES_URL, timeout=tr.TIMEOUT)
    resp.raise_for_status()
    rows = [r for r in csv.DictReader(io.StringIO(resp.text)) if r["game_type"] == "REG"]
    played = [r for r in rows if r.get("result") not in ("", "NA", None) and int(r["season"]) >= tr.FIRST_SEASON]
    current = max(int(r["season"]) for r in played)
    log: list[dict] = []
    tr.compute_ratings(played, current, game_log=log)
    elo_diff = pd.Series({r["game_id"]: r["diffs"]["elo"] for r in log})

    cols = sorted(set(EPA_ETC + ["pen_presnap_c"] + POINTS + CANDS))
    pre = pregame_prod(tg, cols)
    side = pd.concat([tg[["game_id", "season", "week", "is_home"]], pre], axis=1)
    h = side[side.is_home == 1].set_index("game_id")
    a = side[side.is_home == 0].set_index("game_id").reindex(h.index)
    d = h[cols] - a[cols]
    d["season"], d["week"] = h["season"], h["week"]
    gm = tg[tg.is_home == 1].set_index("game_id")[["home_score", "away_score"]].reindex(h.index)
    d = d.join(gm)
    d["margin"] = d["home_score"] - d["away_score"]
    d["home_win"] = (d["margin"] > 0).astype(int)
    d["elo_b"] = elo_diff.reindex(d.index)
    return d.dropna(subset=cols + ["elo_b"]).reset_index()


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    v53.L2 = L2
    d = build_diffs()
    d = d[d.margin != 0].reset_index(drop=True)
    print(f"{len(d)} decided games ({d.season.min()}-{d.season.max()}); test seasons {v53.TEST_START}-{d.season.max()}\n")
    rng = np.random.default_rng(75)

    p10, _ = v53.walk_forward(d, P10)
    r0, _ = v53.walk_forward(d, R0)
    s10, s0 = v53.summarize(p10), v53.summarize(r0)
    print(f"P10 (production stat set, points in):  AUC={s10['auc']:.4f} logloss={s10['ll']:.4f} acc={s10['acc']:.4f}")
    print(f"R0  (points out):                      AUC={s0['auc']:.4f} logloss={s0['ll']:.4f} acc={s0['acc']:.4f}   gap to P10: "
          f"AUC {s0['auc'] - s10['auc']:+.4f}, LL {s0['ll'] - s10['ll']:+.4f}\n")

    show = ["dAUC_all", "dAUC_select", "dAUC_confirm", "dLL_all", "dLL_select", "dLL_confirm", "p_ll_le0", "w_mean", "w_pos_frac",
            "ll_season_win", "tier"]
    chosen, cols, remaining, rnd = [], list(R0), list(CANDS), 1
    while remaining:
        base, _ = v53.walk_forward(d, cols)
        rows = []
        for c in remaining:
            alt, ws = v53.walk_forward(d, cols + [c])
            r = judge(base, alt, ws, len(cols), rng)
            r["cand"] = c
            rows.append(r)
        res = pd.DataFrame(rows).set_index("cand").sort_values("dLL_all", ascending=False)
        print(f"=== Round {rnd}: one-at-a-time vs R0{'+' + '+'.join(chosen) if chosen else ''} ({len(cols)} stats) ===")
        print(res[show].to_string(float_format=lambda v: f"{v:+.4f}" if isinstance(v, float) else v))
        if rnd == 1:
            res.to_csv(os.path.join(os.path.dirname(__file__), "output", "v75_round1.csv"))
        ship = res[res.tier == "SHIP"]
        if ship.empty:
            print("\n  no SHIP-tier candidate this round; stopping.\n")
            break
        best = ship.index[0]
        chosen.append(best); cols.append(best); remaining.remove(best)
        print(f"\n  -> add {best}\n")
        rnd += 1

    final, ws = v53.walk_forward(d, cols)
    sf = v53.summarize(final)
    print(f"R-stack {chosen or '(none)'}: AUC={sf['auc']:.4f} logloss={sf['ll']:.4f} acc={sf['acc']:.4f}")
    print(f"  vs R0:  dAUC {sf['auc'] - s0['auc']:+.4f}, dLL {s0['ll'] - sf['ll']:+.4f}   (positive = better)")
    print(f"  vs P10: dAUC {sf['auc'] - s10['auc']:+.4f}, dLL {s10['ll'] - sf['ll']:+.4f}   share of the points-stats cost recovered: "
          f"AUC {(sf['auc'] - s0['auc']) / (s10['auc'] - s0['auc']) if s10['auc'] != s0['auc'] else float('nan'):.0%}, "
          f"LL {(s0['ll'] - sf['ll']) / (s0['ll'] - s10['ll']) if s0['ll'] != s10['ll'] else float('nan'):.0%}")
    w = np.array(ws).mean(0)
    print("  mean walk-forward weights:", {c: round(float(x), 3) for c, x in zip(cols, w)})


if __name__ == "__main__":
    main()
