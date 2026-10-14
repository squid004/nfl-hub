"""Reusable persistence screen for any candidate team-level stat (research helper, not production).

A stat that does not repeat from game to game (or season to season) cannot help a forward-looking
rating, no matter how it correlates with results after the fact. Run this BEFORE spending any
walk-forward effort on a new feature.

Input: a team-game DataFrame with columns `team`, `season`, `week` plus one column per stat, each
already oriented however you like (the screen is sign-agnostic). Returns one row per stat:

  n             team-games with a value
  splithalf_r   corr of odd-game vs even-game means across team-seasons
  reliability   Spearman-Brown of splithalf_r: share of between-team-season variance that is repeatable
  yoy_r         corr of a team's season mean with its next season's mean
  ewma_r        corr of the PRE-game EWMA rating (same alpha / prior-fade scheme as production
                team_ratings.py) with the stat's actual value in that game
  ewma_r_k6     same, restricted to the team's 6th+ game of a season (prior fully faded out)
  z             ewma_r * sqrt(n), a rough significance read for ewma_r > 0
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EWMA_ALPHA = 0.2
SEASON_PRIOR_GAMES = 5


def pregame_ewma(tg: pd.DataFrame, cols: list[str], alpha: float = EWMA_ALPHA,
                 prior_games: int = SEASON_PRIOR_GAMES) -> pd.DataFrame:
    """Per team-game rating as it stood BEFORE that game, mirroring team_ratings.compute_ratings:
    in-season EWMA, seeded by last season's ending EWMA which fades linearly to zero weight over
    `prior_games` games. NaN values are skipped (the rating just doesn't update that game)."""
    arr = np.full((len(tg), len(cols)), np.nan)
    pos_of = {ix: p for p, ix in enumerate(tg.index)}
    for _, grp in tg.sort_values(["season", "week"]).groupby("team"):
        vals = grp[cols].to_numpy(float)
        seasons = grp["season"].to_numpy()
        ewma = np.full(len(cols), np.nan)
        prior = np.full(len(cols), np.nan)
        last_season, k = None, 0
        for i, ix in enumerate(grp.index):
            if seasons[i] != last_season:
                prior, ewma, last_season, k = ewma.copy(), np.full(len(cols), np.nan), seasons[i], 0
            k += 1
            w = max(0.0, (prior_games - k) / prior_games)
            blend = np.where(np.isnan(prior), ewma, np.where(np.isnan(ewma), prior, w * prior + (1 - w) * ewma))
            arr[pos_of[ix]] = blend
            v = vals[i]
            ewma = np.where(np.isnan(v), ewma, np.where(np.isnan(ewma), v, (1 - alpha) * ewma + alpha * v))
    return pd.DataFrame(arr, index=tg.index, columns=[f"pre_{c}" for c in cols])


def _corr(a: pd.Series, b: pd.Series) -> float:
    m = pd.concat([a, b], axis=1).dropna()
    return float(m.iloc[:, 0].corr(m.iloc[:, 1])) if len(m) > 2 else float("nan")


def persistence_report(tg: pd.DataFrame, cols: list[str], alpha: float = EWMA_ALPHA,
                       prior_games: int = SEASON_PRIOR_GAMES) -> pd.DataFrame:
    pre = pregame_ewma(tg, cols, alpha, prior_games)
    k = tg.groupby(["team", "season"]).cumcount() + 1
    rows = []
    for c in cols:
        d = pd.DataFrame({"y": tg[c], "x": pre[f"pre_{c}"], "k": k, "team": tg["team"], "season": tg["season"]})
        d = d.dropna(subset=["y"])
        m = d.dropna(subset=["x"])
        n = len(m)
        ewma_r = _corr(m["x"], m["y"])
        ewma_r6 = _corr(m.loc[m.k >= 6, "x"], m.loc[m.k >= 6, "y"])
        odd = d[d.k % 2 == 1].groupby(["team", "season"])["y"].mean()
        even = d[d.k % 2 == 0].groupby(["team", "season"])["y"].mean()
        sh = _corr(odd, even)
        mean_ts = d.groupby(["team", "season"])["y"].mean().reset_index()
        nxt = mean_ts.assign(season=mean_ts.season - 1).rename(columns={"y": "y_next"})
        yoy = mean_ts.merge(nxt, on=["team", "season"])
        rows.append({
            "stat": c, "n": n, "splithalf_r": sh,
            "reliability": 2 * sh / (1 + sh) if sh == sh and sh > -1 else float("nan"),
            "yoy_r": _corr(yoy["y"], yoy["y_next"]),
            "ewma_r": ewma_r, "ewma_r_k6": ewma_r6, "z": ewma_r * np.sqrt(n) if n else float("nan"),
        })
    return pd.DataFrame(rows).set_index("stat")
