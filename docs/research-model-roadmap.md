# Research model R: a composite that never uses points scored / points allowed

Started 2026-10-29 at the owner's direction. **Production is unchanged** (`team_ratings.py` keeps its 10-stat composite, PPG/PAPG
included). Model R is a parallel, background track whose only job is to find a stronger model; nothing here ships without the same
walk-forward, pre-registered ADOPT rule every production change has passed (docs/model-methodology.md, section 2).

## Why

Points scored / allowed carry about a third of today's weight, but they depend on special teams, field position, turnover returns,
garbage-time and opponent circumstances that the other stats are blind to, and they leave pass defense at weight 0 (it overlaps
points allowed at r = 0.73). The goal is a composite whose every input is a play-level measure with a clear football meaning, and
which gets back whatever PPG/PAPG were really contributing through better-defined inputs.

## Where R stands (measured)

| Model | Stats | Walk-forward AUC / log loss / accuracy (2011-2025, 4,913 games, research/output/v75_run.txt) |
|---|---|---|
| P10 | production's 10 stats (points in) | 0.6989 / 0.6265 / 64.5% |
| **R0** | the same minus `points_off`, `points_def_allowed` | 0.6967 / 0.6278 / 64.2% |

R0 gives up about 0.002 AUC (and 0.0013 log loss) and hands real weight to defense: rush defense 0.08, pass defense 0.05 (production:
0.06 and 0), Elo 0.40, pass offense 0.27. v74 and v76: opponent-adjusting the four EPA stats did not help, naively (v74) or with a proper simultaneous ridge fit of offense and
defense effects (v76, adjusted vs. an otherwise identical unadjusted control: dAUC -0.0007 to +0.0003, p >= 0.17); Elo and the rolling
averages already carry schedule strength. v76 also showed the recency of the rating matters: a long-memory ridge loses 0.003-0.006 AUC to
the production EWMA, a 5-week half-life nearly matches it. v75: re-running the v52 candidate sweep on top of R0, no candidate reaches SHIP; the closest are kick/punt
special-teams EPA (`st_kp`, +0.0009 AUC, p = 0.07, better in 73% of seasons, WATCH), pace (WATCH) and drive score (select half only).
v77: switching R to a team-points target (two rows per game, offense-vs-defense structure, win probability from the predicted margin) ties the win/loss logistic on the same stats (dAUC -0.0002) and gains +0.0010 AUC with the 17 extra stats added (p = 0.22). It does not buy accuracy by itself, but it gives points-valued components, totals and a spread, so it becomes R's working structure. Read: the missing information is small and spread thin, so one-at-a-time tests cannot resolve it. R0 is already within noise of
production, which makes it a sound foundation. The larger upside is probably NEW information (below), not recovering PPG.

## Update 2026-11-05 (v78)

Added to the points-target model one family at a time: sacks and QB hits (F1, +0.0013 AUC, p = 0.18, WATCH), pass-game shape (F2, nothing), turnover quality (F3, nothing), injuries (F4, +0.0043 AUC, p < 0.001, driven by the QB gaps). Stack F1-F4 + 17 candidates: AUC 0.7020 vs. 0.6966 for the baseline. Pass protection is the most repeatable stat screened (QB hits allowed reliability 0.79); pass rush and turnovers are not (0.2-0.5). The F4 gain needs a leak-free re-test first (the historical QB gap uses who actually played). Next: v79 leak-free QB injury features, then a points-valued QB term.

## Update 2026-11-05 (v79)

The F4 injury gain holds up in reduced form once the QB gap uses only pre-game information (primary QB flagged Out/Doubtful on the injury report): +0.0027 AUC / +0.0014 LL (p = 0.05) against +0.0040 / +0.0026 for the hindsight version; 79% of the hindsight version's flags were surprise changes the report never showed. So R's injury term should be the leak-free QB gap plus skill-out (+0.0030 AUC, p = 0.03). Side finding for production: its QB weight was fit on the hindsight gap, so the live injury-report-only use may be mis-scaled.

## Update 2026-11-05 (v81, v82): penalties are part of EPA in model R

Owner direction: a spot foul such as pass interference is the same outcome as a completed pass, so penalties belong in EPA, on both sides of the ball, pre-snap penalties included. R's rush and pass EPA therefore count every penalty play (live-ball by the play's own run/pass flag, pre-snap by the type of the replayed down); a defense's version is the opponent offense's value. Measured: persistence unchanged or slightly better (pass offense 0.696 vs 0.691), prediction neutral (AUC -0.0004, p = 0.89). Code: `research/r_penalty_features.py` (columns `rush_epa_allpen`, `pass_epa_allpen`), test `edge_signal_test_v82_*`. Also from v81: drawing live-ball flags is a weak repeatable offensive trait (0.44); committing them and pre-snap discipline are nearly noise.

## Update 2026-11-05 (v83): expected points per drive (EPD), experimental

New whole-drive metric for offense and defense (summed play EPA per drive, penalties included, garbage-time DRIVES discarded if any play is snapped under 5% or over 95% win probability, drives starting with 3:00 or less in the half tracked as a separate 3-minute set). Code `research/r_drive_features.py`, test `edge_signal_test_v83_*`; the EPA stats are kept alongside. Persistence beats every single EPA stat (offense 0.72, defense 0.42; the 3-minute set is noise). In the points model, EPD swapped in for the four rush/pass EPA stats gives +0.0007 AUC with half the inputs (p = 0.16); adding it on top +0.0006 to +0.0014. Not significant, consistently positive. Per-team offensive and defensive EPD tables for any season: `PYTHONPATH=. python research/r_epd_team_table.py 2026` (writes `research/output/r_epd_team_table_{season}.csv`; offense = drives the team ran, defense = drives it faced, ranks, regular and 3-minute sets). Across teams offense EPD and defense EPD allowed are nearly uncorrelated (r = +0.19 in 2026, +0.05 in 2025), so the two roles carry separate information. Candidate directions: EPD split by what ended the drive, EPD by starting field position, and a 3-minute set once more seasons accrue.

## Rules of the track

1. R never uses points scored or allowed, in any form.
2. Every addition is one pre-registered test (script `research/edge_signal_test_vNN_*.py`, rule written in the docstring before running):
   walk-forward 2011-onward, AUC and log loss better in BOTH halves, bootstrap p <= 0.01, better in >= 60% of seasons, fitted weight
   positive in >= 90% of folds; WATCH when it misses only the p or season bar. Stats that earn WATCH twice (new data, same sign) are
   promoted.
3. Persistence screen first (`research/persistence.py`): a stat that does not repeat cannot help a forward-looking rating.
4. Market-free: the spread is never an input.
5. Out-of-sample evidence accrues in 2026+: once R's definition is frozen, its pre-game probabilities are logged weekly next to
   production's and the market's (the prospective log) so the model is judged on games nobody tuned on.
6. Data that cannot be redistributed (PFF) stays out of the public repo and the publicly readable Supabase tables; only derived
   team-level numbers may be stored, and only if the licence allows it.

## Backlog (in rough order of expected value per effort)

**A. Play-by-play only, full 2007+ history, no new data source**
- Sack rate / QB-hit rate, offense (allowed) and defense (forced), per dropback: a pass-rush and protection proxy, never tested
  (v52 tested explosive plays, penalties, tempo, special teams, field position; not sacks or hits). Direct route to real pass-defense
  information.
- Pass-defense decomposition: completion rate, air yards and yards after catch allowed, EPA on pressured vs. clean dropbacks.
- Turnover luck split: fumbles lost vs. forced (recovery is near random) against interceptions; expected turnovers from pressure and
  air yards. Turnovers carry 0 weight today partly because they are mostly luck.
- Special teams: promote `st_kp` if it reproduces (WATCH); returning and field-position net behave as noise.
- Success rate, early-down EPA, red-zone and third-down splits (low prior: noisy).

**B. Free structured data, shorter history**
- `pfr_advstats` (2018+): pressures, hurries, blitz rate, drops, coverage stats allowed per defender. The closest free substitute for
  PFF's pass-rush and coverage grades, but 8 seasons.
- `nextgen_stats` (2016+): time to throw, CPOE, separation. `snap_counts` (2012+): depth and who actually plays. `ftn_charting`
  (2022+): play-action, motion, pressure; 4 seasons, tracked but not built on.
- Constraint from docs/nflverse-data-catalog.md: `pbp_participation` is post-season only, so it can never feed a live weekly pick.

**C. PFF player and unit grades (decision pending)**
- What it would answer: the one live survivor of the earlier unit-health sweep, DB/S health (p = 0.08-0.11), was parked
  "pending a decision on PFF", and OL/DL/LB/secondary quality is exactly what play-by-play cannot see.
- Key assumption to verify before any pipeline: that PFF's grades can be obtained as structured weekly (or per-game) team-unit or
  per-player data, with enough history (ideally 2006+) and a join key to nflverse, under a licence that allows model research. PFF
  has no public API and prohibits scraping, so the realistic route is a manual export from a paid subscription.
- Verification, cheapest first: (1) confirm which product and exports the owner has; (2) pull ONE season of weekly team-unit grades
  as a sample and check columns, granularity and ids; (3) run the persistence and redundancy screens against R0 on that sample before
  buying more history; (4) only then build an ingest. Compare against path B (free) first: if `pfr_advstats` recovers most of the
  same signal the subscription may not be needed.

**D. Model structure** (after the inputs settle)
- Unit-level composites (offense / defense / special teams) with a shared shrinkage, instead of ten free weights.
- Matchup terms (this team's pass rush against that team's protection) once pressure data exist.
- A QB-adjusted rating (team-without-QB + QB value) and the state-space (Kalman) ratings from the earlier plan, as a research branch.

## Next concrete steps

1. v76: sack rate and QB-hit rate (offense and defense) added to R0, plus a pass-defense decomposition.
2. v77: turnover-luck split.
3. Decide path B vs. C for player-level information after the owner answers what PFF access exists.
4. Freeze R's definition and start the weekly prospective log (production, R and the market side by side on Model Trends).
