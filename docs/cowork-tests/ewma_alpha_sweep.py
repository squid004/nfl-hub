"""Avenue 1 lead: EWMA update speed. Points-for/against ratings only, walk-forward 2011-2025.
Result on 2026-10-15: alpha 0.08-0.12 beat 0.20 by ~+0.013 AUC in both halves, dLL +0.0065 (p=0.0005), 14/15 seasons."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from _common import load_games, points_ewma, walk_forward, report, boot_dll

g = load_games()
base = points_ewma(g, 0.2, 0.65)
for a in (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30):
    for c in (0.5, 0.65, 0.8, 1.0):
        X = points_ewma(g, a, c)
        ok = (g.result != 0).values & ~np.isnan(X).any(1) & ~np.isnan(base).any(1)
        p, y, s = walk_forward(g, X, ok)
        report(f"alpha={a:.2f} carry={c:.2f}", p, y, s)
ok = (g.result != 0).values & ~np.isnan(base).any(1)
p0, y, s = walk_forward(g, base, ok)
p1, _, _ = walk_forward(g, points_ewma(g, 0.10, 0.8), ok)
d, pv = boot_dll(y, p0, p1)
print(f"alpha 0.10/carry 0.8 vs 0.20/0.65: dLL {d:.4f}, bootstrap P(<=0) {pv:.4f}")
