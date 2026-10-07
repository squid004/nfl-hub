# nflverse data catalog — what's usable for backtestable, in-season models

Written 2026-10-06. Every season/cadence claim below was verified directly against the
live `nflverse-data` GitHub releases (API asset listings + `timestamp.txt` + a handful of
CSV header/content checks) and the live `nfldata/games.csv`, not just nflreadr's docs —
docs quoted "2006 onward" for `games.csv` where the raw file actually starts 1999. Re-verify
anything load-bearing before trusting it blindly a second time; this is a point-in-time scan.

## The filter this project actually needs

A dataset is only useful for a predictive model here if it clears **both** bars:

1. **Enough history to backtest.** Walk-forward validation (this project's standing bar —
   see `research/edge_signal_test*.py`) needs years of completed seasons, not one.
2. **Available before next week's games, every week, not just eventually.** A dataset that
   only shows up after the season ends (or days after the fact) can describe history but can
   never feed a live weekly pick — no amount of backtest history fixes that.

Most of nflverse's catalog clears #1 easily (that's the point of the project). The real
filter is #2, and it varies a lot by dataset — some update multiple times a day during the
season, some once a day, some only when the source changes, and at least one explicitly
**does not update during the season at all**.

## Master table

"Lag" = how soon after a game the data for that game shows up, per nflreadr's own schedule
doc plus this scan's live timestamp checks. "First season" was verified against the actual
files (not just docs), per-file below.

| Dataset (release tag) | First season | Update cadence (in-season) | Usable for a live weekly model? |
|---|---|---|---|
| `schedules` (games.csv) | **1999** | every 5 min | **Yes** — already the backbone of this project |
| `pbp` (play-by-play, EPA/WP) | **1999** | nightly (+ intra-day on game days) | **Yes** — already used (`team_ratings.py`) |
| `stats_team` (team-week box score + EPA) | **1999** | nightly | **Yes, unused** — see Opportunity 1 below |
| `stats_player` (player-week stats, successor to the now-frozen `player_stats` tag) | **1999** | nightly | **Yes, unused** |
| `injuries` | **2009** | daily (7am UTC) | **Yes** — already used (`team_ratings.py` QB-health filter) |
| `snap_counts` (PFR) | **2012** | 4x/day (0/6/12/18 UTC) | **Yes, unused** — explored in `research/edge_signal_test_v17*`, shelved for crosswalk cost, not unavailability |
| `nextgen_stats` (NGS speed/separation/CPOE/time-to-throw) | **2016** | nightly (3-5am ET) | **Yes, unused** |
| `pfr_advstats` (pressure/blitz/drop rates) | **2018** | 4x/day | **Yes, unused** |
| `ftn_charting` (play-level: motion, RPO, play-action, pressure, coverage shell) | **2022** | 4x/day | **Yes, unused, but only 4 backtest seasons** |
| `weekly_rosters` (active/IR/PS status per week) | **2002** | daily (7am UTC) | **Yes, unused** |
| `depth_charts` | **2001** | daily (7am UTC, year-round) | **Yes, unused** |
| `espn_data` (ESPN QBR, week + season level) | **2006** | irregular (source-change triggered, confirmed fresh as of yesterday) | **Probably yes** — verify same-week lag before relying on it |
| `officials` (referee assignments/history) | **2015** | source-change triggered, **observed stale 34 days** at scan time | **No for next-game prediction** — not on the guaranteed schedule; the current game's ref isn't reliably in here before kickoff |
| `pbp_participation` (on-field personnel per play) | **2016** | **post-season only** — 2023+ explicitly "provided after all post-season games are completed... does not update during the season" | **No, categorically** — great for historical research, structurally unusable for a weekly model |
| `combine` | 2000 | source-change triggered (last touched March) | No — pre-season only by nature, not a weekly signal |
| `draft_picks` | 1980 | source-change triggered | No — same, not a weekly signal |
| `contracts` (OTC) | varies | source-change triggered | No — cap/contract data, not game-predictive at weekly grain |
| `trades` | long history | source-change triggered | No — same category |

## Already powering nfl-hub

- `games.csv` (`schedules`): schedule, scores, closing `spread_line`/`total_line`,
  moneylines (from **2006** on — confirmed by direct scan; earlier seasons have the spread
  but not the moneyline), `roof`/`surface`/`temp`/`wind` (from 1999), `referee`,
  `away_qb_id`/`home_qb_id`. Drives `nflhub/sources/history.py` (favorite-vs-spread rates)
  and the backbone of `team_ratings.py`.
- `pbp` (play-by-play): source of the 4 EPA metrics in `team_ratings.py`, filtered to
  `FIRST_SEASON = 2007` by this project's own choice (not nflverse's — the raw files go back
  to 1999; 2007 is where this project decided to start trusting it, likely for play-type/EPA
  model consistency rather than a hard availability floor — worth re-examining, see
  Opportunity 1).
- `injuries`: QB Out/Doubtful flag, `FIRST_INJURY_SEASON = 2009` — matches the dataset's
  real floor exactly.

## Unused but immediately viable (clears both bars today)

**Opportunity 1 — `stats_team` could extend the backtest window from 2007 back to 1999.**
`team_ratings.py`'s EPA metrics are hand-aggregated from raw play-by-play starting 2007. The
`stats_team` release already ships pre-aggregated `passing_epa`/`rushing_epa` per team-week
back to **1999** — confirmed directly (`stats_team_week_1999.csv` exists and has both
columns populated). That's 8 more backtest seasons for free. The catch: this project's own
EPA pipeline excludes garbage-time plays (win probability outside [0.05, 0.95]) before
averaging, and `stats_team`'s pre-aggregated EPA does **not** apply that filter — so it's not
a drop-in replacement, it's a different (unfiltered) version of the same stat. Worth
backtesting both ways before trusting the extra 8 seasons: does the garbage-time filter
actually matter enough to justify the shorter window, or does the longer window net out
ahead even unfiltered?

**Opportunity 2 — `weekly_rosters`' `status` field is a cheap, broad injury-adjacent signal.**
Separate from the `injuries` report (QB-specific, already used), `weekly_rosters` carries an
Active/IR/PUP/Practice-Squad `status` per player per week, back to 2002, updated daily. Could
extend the existing QB-health work (`[[project_nfl_hub]]` memory, 2026-10-05 note on the
blind spot) toward a roster-depth signal beyond just the QB position, without waiting on a
new data source.

**Opportunity 3 — `nextgen_stats` / `pfr_advstats` as genuinely new signal types, not more EPA.**
Both are already-structured, nightly/4x-daily-updated, multi-season (2016+ / 2018+) datasets
that measure things EPA doesn't: NGS has CPOE, time-to-throw, separation; PFR adv has
pressure rate, blitz rate, drop rate. The [[project_nfl_hub]] 2026-10-05 QB-injury finding
explicitly concluded the existing 8 EPA/points/turnover stats carry "ANY current-week injury
signal regardless of how they're combined" and a fix needs "a genuinely NEW model input" —
these two datasets are closer to that than another reweighting of the same 8 stats would be.
Shorter history (10 / 8 seasons) than the EPA pipeline's 19 (or 27, see Opportunity 1), so
walk-forward validation on these would have less out-of-sample room — a real cost, not just a
footnote.

**Opportunity 4 — `ftn_charting` for scheme-level matchup context.** Play-level motion/RPO/
play-action/blitz/pressure flags, 4x-daily during season — but only since 2022, so only ~4
completed backtest seasons. Worth tracking as it accumulates more than building on now.

## Disqualified, and why (don't re-attempt without new reasoning)

- **`pbp_participation`**: nflreadr's own docs are explicit — 2023+ participation data "is
  provided after all post-season games are completed. It does not update during the
  season." This has rich personnel/formation history (back to 2016) but is structurally
  incapable of feeding a live weekly pick; it's a post-mortem research dataset only. Don't
  build toward this expecting an in-season fix later — the constraint is permanent by design
  (FTN's licensing/release choice), not a temporary gap.
- **`officials`**: not on nflreadr's guaranteed-cadence list (that list is `pbp`,
  `player_stats`/`team_stats`, `snap_counts`, `pfr_advstats`, `ftn_charting`,
  `nextgen_stats`, `rosters`, `depth_charts`, `injury_data`, `schedules`). It updates "when
  the source changes," and this scan caught it 34 days stale mid-season — the current week's
  referee assignment is not reliably present before kickoff. Fine for historical ref-tendency
  research; don't rely on it as a live pregame input without re-checking freshness close to
  game time every time.
- **`combine` / `draft_picks` / `contracts` / `trades`**: all real, all historically deep,
  none weekly-cadence by nature (draft/contract/trade events don't happen on a weekly
  schedule) — not disqualified by a data-freshness problem, just the wrong grain for a
  week-to-week predictive signal.

## Gotchas worth remembering

- **`player_stats` (the old release tag) is frozen as of 2025-05-07** — its last timestamp
  predates this scan by 5 months, while its replacement `stats_player` updated today.
  `research/edge_signal_test_v21_skill_touches_weighted.py` still points at the old
  `player_stats.csv` URL; any future work from that script should repoint to `stats_player`
  first, or it'll silently train on stale data.
- **Moneylines start later than spreads in `games.csv`**: `spread_line`/`total_line` are
  populated from 1999, but `home_moneyline`/`away_moneyline` only from **2006** (confirmed by
  direct scan, not docs). A backtest mixing spread-based and moneyline-based signals over the
  same window needs to pick the later floor, or explicitly handle the gap.
- **nflreadr's own docs undersell `games.csv`'s history** — the published dictionary/README
  describes betting and weather columns as "NEW Feb 2020," which reads like a 2020 floor; the
  live file actually has those columns backfilled to 1999. Don't take a doc's stated
  introduction date as the data's actual availability floor without checking the raw file,
  same lesson as the "2006 onward" schedules claim that turned out to be wrong too.
- **Team-code normalization is still required everywhere**: `games.csv` uses the
  contemporary code per season (STL/SD/OAK pre-relocation), play-by-play-derived files
  retroactively relabel those as LA/LAC/LV for every season — this project's existing
  `normalize_team()` already handles it (see `team_ratings.py` module comments); any new
  dataset joined against either source needs the same treatment, not a fresh one.
