# Live pool picks: sources, format, and how to add the CBS fetch

The pool optimizer needs everyone's locked picks right after each game day's first kickoff (picks lock by game
day and become visible then), not on Tuesday when the final sheet tab is written. `nflhub/sources/pool_live.py`
merges every working source into one store (`pool_live_picks`); the optimizer, the app-vs-CBS pick flags and the
ntfy reminders read from that store.

Deduplication: a pick that several sources agree on is one pick. If two snapshots disagree about the same
entrant's pick in the same game, the later capture wins and the disagreement is listed in `conflicts` (and counted
on the Pipeline summary). A source that errors is recorded and never blocks the others. Once the final "Week N" tab
exists the live data for that week is no longer needed.

## Source 1: the "Week N live" sheet tab (works now)

A Cowork run (PC on, Claude app open) writes a tab named exactly `Week N live` (e.g. `Week 6 live`) in the pool
sheet right after each game day's first kickoff: about Thursday 8:25 PM, Sunday 1:10 PM (9:30 AM on London
weeks) and Monday 8:25 PM ET, plus a run per extra game day on holiday weeks. Layout, same as the final tab:

| name | weekly | season | pick1 | pick2 | ... |
| --- | --- | --- | --- | --- | --- |
| `1st Jaime Flores` | | | `GB` | `MIN` | |
| `Captured` | `2026-10-18T17:12:00-04:00` | | | | |

- One row per entrant. Column 1 = the name as on CBS (a rank ordinal prefix is fine). Weekly/season can be blank.
- Pick cells hold team codes for the picks visible at capture time (locked games for everyone; the user's own
  picks may include open games). Leave unlocked games blank.
- One row whose first cell is `Captured` and whose second cell is an ISO timestamp (with time zone). The optimizer
  only treats a missing pick as "no pick submitted" if the capture is later than that game day's first kickoff.
- Overwrite the same tab each run (latest capture wins); never put the final results here.

Suggested Cowork task prompt: "After the first kickoff of each NFL game day, open the CBS pool weekly picks page,
and write every entrant's visible picks into the tab `Week N live` of the 2026 Pickem sheet using the layout in
docs/pool-live-picks.md, with the `Captured` row set to the current time."

## Source 2: a direct CBS fetch from the refresh job (built, needs your session cookie)

`nflhub/sources/cbs_pool.py` runs the same GraphQL operation the pool's weekly-standings page runs in the browser
(`FootballPickemManagerWeeklyStandingsContent`, document in `cbs_pool_query.graphql`, copied from CBS's public page
JavaScript) against `https://picks.cbssports.com/graphql`. For the current week it returns every entry's visible picks
(other entrants' picks appear once their game locks; yours always), tiebreaker answers, and each game's `isLocked` flag.
It is strictly read-only: it only ever sends that query, never CBS's save-picks mutation.

Setup:

1. Log in to the pool on CBS in a desktop browser, open the weekly standings page
   (`https://picks.cbssports.com/football/pickem/pools/kbxw63b2ge3dknryg44ts===/standings/weekly`).
2. Open dev tools (F12), Network tab, tick Preserve log, filter on `graphql`, and reload the page. Every request to that address
   is listed as just `graphql` (the operation name is in the request payload, not the Name column), and they all carry the same
   cookie, so click any one of them, open Headers, Request Headers, and copy the whole `cookie:` value.
3. Save it as the GitHub Actions secret `CBS_COOKIE` (repo settings, Secrets and variables, Actions). Optional variable
   `CBS_POOL_ID` (defaults to the pool id above). The workflow already passes both to the refresh job.
4. Optional local check before relying on it: `CBS_COOKIE="<paste>" PYTHONPATH=. python -m nflhub.sources.cbs_pool 6`
   prints how many picks it mapped and each entrant's picks.

Behavior: at most one fetch per 20 minutes. A failure (expired cookie, CBS blocking GitHub's IP range, a response shape it
can't map) is recorded as that source's error in `pool_live_picks.sources` and the previous capture is kept, so the sheet
source and any earlier data keep working. The response mapping follows the query's field names and has not yet been run
against a live response; if it can't map any pick it fails loudly with a diagnostic rather than returning nothing. Cookies
expire (typically weeks): when the source starts showing `HTTP 401/403` or "not logged in", paste a fresh one.

Either way: the project never writes picks to CBS.
