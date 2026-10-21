"""v67: do entrants have personal Monday-night tiebreak habits, and does modelling them beat the pooled guess model?

The pool optimizer's first version drew every opponent's tiebreak guess from one pooled distribution
(Normal(posted total - 1.5, 4.5), v65). The user's observation: at least one entrant always guesses 42, which matters
whenever a tie is possible. Data: research/cache/pool_weeks.json (weeks 1-5, every entrant's final-tab tiebreak guess
and the Monday game's posted total).

Per-entrant model (what pool_mc uses): with probability p_fixed the entrant repeats their favorite number m (their modal
past guess); otherwise the guess is Normal(posted total + offset_j, SD) rounded. p_fixed = (count_of_m + K*P0) / (n + K),
offset_j = (sum of past offsets + K*MU0) / (n + K): shrunk toward the pooled values, so a one- or two-week history moves
little. Pooled model: Normal(posted total - 1.5, 4.5) for everyone.

Score: mean log probability (integer guesses; normals discretized) of each HELD-OUT guess. Evaluation 1: walk-forward over
weeks 3-5 (habits from earlier weeks only). Evaluation 2: leave-one-week-out over all 5 weeks. No pass/fail rule: this is a
model-selection check on ~130 guesses, reported with the per-entrant habit table (entrants anonymized by index).

Run: PYTHONPATH=. python research/edge_signal_test_v67_tiebreak_habits.py
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter

import numpy as np

CACHE = os.path.join(os.path.dirname(__file__), "cache", "pool_weeks.json")
MU0, SD = -1.5, 4.5     # pooled offset / sd (v65)
K = 2.0                 # shrinkage strength (pseudo-weeks)
P0 = 0.15               # prior probability that a guess repeats the entrant's favorite number


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def disc_normal(g, mu, sd):
    return max(1e-9, phi((g + 0.5 - mu) / sd) - phi((g - 0.5 - mu) / sd))


def habit(history):
    """history = [(guess, line), ...] -> (mode value, p_fixed, offset)."""
    n = len(history)
    if n == 0:
        return None, 0.0, MU0
    c = Counter(int(g) for g, _ in history)
    m, cnt = c.most_common(1)[0]
    p_fixed = (cnt + K * P0) / (n + K) if cnt >= 2 else K * P0 / (n + K)
    offset = (sum(g - line for g, line in history) + K * MU0) / (n + K)
    return m, p_fixed, offset


def prob_entrant(g, line, hist):
    m, p_fixed, offset = habit(hist)
    cont = disc_normal(g, line + offset, SD)
    return (p_fixed * (1.0 if m is not None and int(g) == m else 0.0) + (1 - p_fixed) * cont) if m is not None else disc_normal(g, line + MU0, SD)


def main():
    D = json.load(open(CACHE))
    by_week = {}
    for w, d in D.items():
        mnf = [g for g in d["games"] if g["weekday"] == "Monday"][-1]
        by_week[int(w)] = (mnf["total_line"], {e["name"]: e["tiebreak"] for e in d["entrants"] if e["tiebreak"] is not None
                                                 and d["my_name"].lower() not in e["name"].lower()})
    names = sorted({n for _, g in by_week.values() for n in g})

    def history(name, weeks):
        return [(by_week[w][1][name], by_week[w][0]) for w in sorted(weeks) if name in by_week[w][1]]

    for label, plans in (("walk-forward (weeks 3-5, earlier weeks only)", [(w, [x for x in by_week if x < w]) for w in (3, 4, 5)]),
                         ("leave-one-week-out (weeks 1-5)", [(w, [x for x in by_week if x != w]) for w in by_week])):
        pooled, per = [], []
        for w, train in plans:
            line, guesses = by_week[w]
            for n, g in guesses.items():
                pooled.append(math.log(disc_normal(g, line + MU0, SD)))
                per.append(math.log(prob_entrant(g, line, history(n, train))))
        print(f"{label}: {len(pooled)} held-out guesses | mean log-prob pooled {np.mean(pooled):.3f}  per-entrant {np.mean(per):.3f}  "
              f"(+{np.mean(per) - np.mean(pooled):.3f} nats/guess; per-entrant better on {np.mean(np.array(per) > np.array(pooled)):.0%} of guesses)")

    print("\n=== habits from all 5 weeks (entrants anonymized by index; sorted by repeat share) ===")
    rows = []
    for i, n in enumerate(names):
        h = history(n, by_week)
        if not h:
            continue
        m, p_fixed, offset = habit(h)
        rows.append((Counter(int(g) for g, _ in h).most_common(1)[0][1] / len(h), i, [int(g) for g, _ in h], m, p_fixed, offset))
    for share, i, g, m, pf, off in sorted(rows, reverse=True)[:12]:
        print(f"  #{i:2d} guesses {g}: favorite {m}, P(repeat) {pf:.2f}, offset vs posted {off:+.1f}")
    allg = [int(g) for _, gs in by_week.values() for g in gs.values()]
    print("\n  pooled share of guesses that are exactly 42:", f"{allg.count(42) / len(allg):.0%}", "| 45:", f"{allg.count(45) / len(allg):.0%}")


if __name__ == "__main__":
    main()
