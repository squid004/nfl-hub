"""One-off: rebuild the committed completed-season team_game_phase_{season}.parquet caches so
they carry the new pen_presnap_yds column (production code recomputes any cache lacking it).
Run: PYTHONPATH=. python research/backfill_pen_presnap.py"""
import os
import sys

import pandas as pd

from nflhub.sources import team_ratings as tr

CURRENT = 2026
for season in range(2007, CURRENT):
    path = os.path.join(tr.DATA_DIR, f"team_game_phase_{season}.parquet")
    old = pd.read_parquet(path)
    if "pen_presnap_yds" in old.columns:
        continue
    new = tr._phase_stats_for_season(season, CURRENT)
    same = old.merge(new, on=["game_id", "team"], suffixes=("_o", "_n"))
    assert len(same) == len(old) == len(new), (season, len(old), len(new), len(same))
    for c in ("rush_off_epa", "pass_off_epa", "turnovers_committed"):
        assert (same[c + "_o"] - same[c + "_n"]).abs().max() < 1e-9, (season, c)
    new.to_parquet(path)
    print(season, len(new), float(new["pen_presnap_yds"].mean().round(2)), file=sys.stderr)
