"""v12: walk-forward backtest of the production "upset budget" strategy (take the smooth
curve's most-live dogs, up to the week's expected-upset count -- nflhub/sources/history.py
week_budget()) against always picking the favorite ("chalk"), on raw season-long pick accuracy.

Also decomposes *why* the strategy wins or loses relative to chalk, to test a specific
hypothesis: that a wrong upset pick costs roughly TWO games' worth of ground, not one --
the game itself (favorite won, strategy picked the dog) PLUS a true upset elsewhere that
week going uncaptured because the budget was already spent on the wrong game. That's measured
directly here as false positives (flagged dog that lost) vs false negatives (an actual upset
on a game the strategy chalked), not assumed.

Walk-forward, same discipline as every other backtest in this project: the distribution used
to generate week N's picks is fit ONLY on seasons strictly before that week's season -- no
reuse of the single-pooled-everything production distribution (which is fit on the full
2007-present range and would leak future seasons into picks for, e.g., 2012).

No ELWAY data available historically in a clean per-game form, so this omits it from the
ranking blend (falls back to ranking live-ness by market spread alone via predict()) -- ELWAY
only affects WHICH dog gets flagged within a bucket when the budget is < the bucket's games,
never the total budget count itself, so this is a reasonable simplification, not a different
strategy.

Run: python research/edge_signal_test_v12_upset_budget_backtest.py
"""
from __future__ import annotations

import csv
import io
import os
import sys
from collections import defaultdict

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nflhub.sources import history

TEST_START_SEASON = 2011  # same convention as the rest of this project's backtests
MIN_SEASON = history.MIN_SEASON  # 2007, matches production build_distributions()


def load_games() -> list[dict]:
    """Returns the RAW nflverse CSV row dicts (filtered to usable REG games), unmodified --
    build_distributions() expects exactly this shape (game_type/season/week/spread_line/
    result/home_score/away_score/location as nflverse spells them), so these rows are fed to
    it directly for the walk-forward refit, with a few derived fields (season/week as int,
    scores as float, spread_line as float) added for this script's own use without disturbing
    the fields build_distributions reads."""
    resp = requests.get(history.GAMES_CSV, timeout=30)
    resp.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(resp.text)))
    out = []
    for r in rows:
        if r.get("game_type") != "REG":
            continue
        try:
            season, week = int(r["season"]), int(r["week"])
        except (KeyError, ValueError):
            continue
        if season < MIN_SEASON:
            continue
        sl_raw = r.get("spread_line", "")
        if sl_raw in ("", "NA", None):
            continue
        try:
            sl = float(sl_raw)
        except ValueError:
            continue
        try:
            home_score, away_score = float(r["home_score"]), float(r["away_score"])
        except (KeyError, ValueError, TypeError):
            continue
        if home_score == away_score:
            continue  # ties: no SU winner, excluded from both strategies same as a push
        r = dict(r)  # don't mutate the DictReader row in place
        r["_season"] = season
        r["_week"] = week
        r["_spread_line"] = sl
        r["_home_score"] = home_score
        r["_away_score"] = away_score
        out.append(r)
    return out


def main() -> None:
    print("Loading games...", file=sys.stderr)
    games = load_games()
    seasons = sorted({g["_season"] for g in games})
    print(f"{len(games)} games, seasons {seasons[0]}-{seasons[-1]}\n")

    tot_games = 0
    chalk_correct = 0
    strat_correct = 0
    flagged_total = 0
    flagged_hit = 0          # strategy took the dog and the dog won
    true_upsets = 0          # dog actually won, regardless of what either strategy picked
    true_upsets_on_flagged = 0    # true upset AND strategy had it flagged (correctly caught)
    true_upsets_on_unflagged = 0  # true upset but strategy chalked it (missed)
    season_rows = []

    for test_season in seasons:
        if test_season < TEST_START_SEASON:
            continue
        train_rows = [g for g in games if g["_season"] < test_season]
        test_season_games = [g for g in games if g["_season"] == test_season]
        if len(train_rows) < 500 or not test_season_games:
            continue

        # Refit the distribution on strictly-prior seasons only (walk-forward) -- reuses the
        # exact production function, just fed a season-restricted row set. build_distributions
        # expects nflverse-games.csv-shaped dict rows, which `train_rows` already are (plus a
        # few harmless extra _-prefixed fields it doesn't look at), so pass them straight through.
        dist = history.build_distributions(train_rows, min_season=MIN_SEASON)

        by_week: dict[int, list[dict]] = defaultdict(list)
        for g in test_season_games:
            by_week[g["_week"]].append(g)

        season_tot = season_chalk = season_strat = 0
        for week, wk_games in sorted(by_week.items()):
            games_in = [{"game_id": g["game_id"], "home": g["home_team"], "away": g["away_team"]} for g in wk_games]
            # dashboard spread convention: negative/zero = home favored (nflverse spread_line
            # is the opposite sign -- see history.py module docstring).
            odds_map = {g["game_id"]: {"spread": -g["_spread_line"]} for g in wk_games}
            budget = history.week_budget(dist, games_in, odds_map, elway_map=None)
            rows = [row for bin_row in budget["ml"] if bin_row["bin"] != "TOTAL" for row in bin_row["games"]]
            by_gid = {r["game_id"]: r for r in rows}

            for g in wk_games:
                r = by_gid.get(g["game_id"])
                if r is None:
                    continue  # no valid spread bucket (shouldn't happen given load_games filtering)
                fav_won = (r["fav"] == g["home_team"] and g["_home_score"] > g["_away_score"]) or \
                          (r["fav"] == g["away_team"] and g["_away_score"] > g["_home_score"])
                dog_won = not fav_won

                tot_games += 1
                season_tot += 1
                if fav_won:
                    chalk_correct += 1
                    season_chalk += 1
                if dog_won:
                    true_upsets += 1

                if r["flagged"]:
                    flagged_total += 1
                    if dog_won:
                        strat_correct += 1
                        season_strat += 1
                        flagged_hit += 1
                        true_upsets_on_flagged += 1
                    else:
                        true_upsets_on_flagged += 0  # fav won; nothing to count here
                else:
                    if fav_won:
                        strat_correct += 1
                        season_strat += 1
                    else:
                        true_upsets_on_unflagged += 1

        season_rows.append((test_season, season_tot, season_chalk, season_strat))

    print("=== Season-by-season: chalk vs. upset-budget strategy (# correct of n games) ===")
    print(f"{'season':>6} {'n':>5} {'chalk':>7} {'strategy':>9} {'delta':>7}")
    for s, n, c, st in season_rows:
        print(f"{s:>6} {n:>5} {c:>7} {st:>9} {st - c:>+7}")

    print()
    print("=== Totals ===")
    print(f"games tested:        {tot_games}")
    print(f"chalk correct:       {chalk_correct}  ({chalk_correct / tot_games:.4f})")
    print(f"strategy correct:    {strat_correct}  ({strat_correct / tot_games:.4f})")
    print(f"delta vs. chalk:     {strat_correct - chalk_correct:+d} games "
          f"({(strat_correct - chalk_correct) / tot_games:+.4f} rate)")

    print()
    print("=== Decomposition: false positives (wrong dog picks) vs false negatives (missed true upsets) ===")
    flagged_miss = flagged_total - flagged_hit
    print(f"flagged (budget took the dog):     {flagged_total}")
    print(f"  hit (dog won):                   {flagged_hit}  ({flagged_hit / flagged_total:.3f} hit rate)" if flagged_total else "  (none)")
    print(f"  miss (favorite won anyway):       {flagged_miss}   <- false positives, cost vs. chalk")
    print(f"true upsets total:                  {true_upsets}")
    print(f"  caught (flagged correctly):       {true_upsets_on_flagged}")
    print(f"  missed (strategy had chalked it): {true_upsets_on_unflagged}   <- false negatives")
    print()
    print(f"ratio of false negatives to false positives: "
          f"{true_upsets_on_unflagged / flagged_miss:.2f}" if flagged_miss else "n/a")
    print("(the 'does a wrong upset pick cost ~2 games' hypothesis would show this ratio near 1:1 --")
    print(" i.e., for every wrong dog pick, there's roughly one true upset elsewhere that week the")
    print(" budget didn't have room for or didn't flag -- not a causal link, just how often both")
    print(" happen together at the rates the model and real upset variance actually produce.)")


if __name__ == "__main__":
    main()
