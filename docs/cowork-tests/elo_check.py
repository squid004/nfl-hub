"""Elo vs. the points ratings and vs. the closing spread, walk-forward 2011-2025.
Result on 2026-10-15: Elo alone 0.692 AUC; adds a lot to alpha=0.2 ratings (p<0.001) but ~nothing once alpha=0.1 (p=0.21); adds nothing to the spread."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from _common import load_games, points_ewma, elo, walk_forward, report, boot_dll

g = load_games()
P2, P1 = points_ewma(g, 0.2, 0.65), points_ewma(g, 0.1, 0.8)
E = elo(g)[:, None]
M = g.spread_line.values[:, None]
ok = (g.result != 0).values & ~np.isnan(P1).any(1) & ~np.isnan(P2).any(1)
for K in (16, 20, 24):
    for rev in (0.25, 1 / 3, 0.5):
        p, y, s = walk_forward(g, elo(g, K=K, REV=rev)[:, None], ok); report(f"Elo K={K} rev={rev:.2f}", p, y, s)
runs = {}
for name, X in [("points a=.2", P2), ("points a=.2 + Elo", np.c_[P2, E]), ("points a=.1", P1),
                ("points a=.1 + Elo", np.c_[P1, E]), ("closing spread", M), ("spread + Elo", np.c_[M, E])]:
    p, y, s = walk_forward(g, X, ok); runs[name] = p; report(name, p, y, s)
for a, b in [("points a=.2", "points a=.2 + Elo"), ("points a=.1", "points a=.1 + Elo")]:
    d, pv = boot_dll(y, runs[a], runs[b]); print(f"{b} vs {a}: dLL {d:.4f}, P(<=0) {pv:.3f}")
print("corr(Elo, points a=.1 sum):", np.corrcoef(E[ok, 0], P1[ok].sum(1))[0, 1].round(3))
