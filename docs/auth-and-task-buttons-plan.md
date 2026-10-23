# NFL Hub: owner sign-in and task buttons plan

Oct 17, 2026 · Jeremy · Exported from the Claude doc: https://claude.ai/code/artifact/5e847e12-cd35-4f7d-a763-9aa2ad0b7e27 (the doc is the live version; this file is a snapshot for Claude Code).

## Goal and today's exposure

Goal: anyone can still view NFL Hub, but only Jeremy, signed in, can change picks, request a refresh, or start a Claude task. Today anyone can do all three, because every table's policy is `anon all` and the anon key ships in `js/db.js` of a public repo.

| Who can do it today | What | Where in code |
| --- | --- | --- |
| Anyone with the page | Set or change Moneyline picks | `DB.setPickemPick` → `pickem_pick` |
| Anyone with the page | Set ATS picks, record gut picks | `setAtsPick` → `ats_pick`; `recordGutPick` → `kv.gut_picks` |
| Anyone with the page | Request a refresh | `requestRefresh` → `refresh_request` |
| Anyone with the page | Write national pick % | `edge_national_pct` upsert in `js/db.js` |
| Anyone with the anon key | Overwrite or delete any row in any table | `supabase/schema.sql` RLS loop (`for all … using (true)`) |

The last row matters most: once the CBS sync task reads `pickem_pick`, a stranger who edits that table could change Jeremy's real pool picks.

## Part 1: owner sign-in and locked-down writes

Use Supabase Auth, which the hub already loads through `supabase-js`, so no new service is needed. Reads stay public; every write requires Jeremy's signed-in session, enforced by the database, not the page.

**Auth setup (Supabase dashboard)**

1. Turn on the Google provider: create an OAuth client in Google Cloud and paste its client ID and secret into Supabase. Add `https://squid004.github.io/nfl-hub/` as the Site URL and an allowed redirect URL.
2. Create Jeremy's user once, then turn off "Allow new users to sign up", so no one else can make an account.

**Database (new migration `supabase/auth.sql`)**

1. Add an `app_owner (user_id uuid primary key)` table with Jeremy's user id, and an `is_owner()` SQL function that checks `auth.uid()` against it.
2. Replace the `anon all` loop with two policies per table: `public read` (select, to anon and authenticated, `using (true)`) and `owner write` (insert, update, delete, to authenticated, `using (is_owner()) with check (is_owner())`).
3. Lock `app_owner` itself to owner-only, so no one can add themselves.

**Backend (GitHub Actions refresh job)**

1. The refresh job writes with `SUPABASE_ANON_KEY` today, which would fail once anon can't write. Add a `SUPABASE_SECRET_KEY` GitHub secret (Supabase's secret or service-role key, which bypasses RLS).
2. In `nflhub/config.py` and `nflhub/store.py`, use the secret key for all server calls. It never goes in `js/`.

**Frontend (`index.html`, `js/app.js`, `js/db.js`)**

1. Add a Sign in / Sign out control to the header. Sign in opens Google sign-in; `supabase-js` keeps the session in the browser.
2. Signed out = read-only mode: pick buttons, gut picks, ATS picks, Refresh now and the new task buttons are hidden or disabled, with a short "Sign in to edit" note.
3. Make every write check for failure. Under RLS a blocked update can quietly touch 0 rows, so writes add `.select()` and show an error toast when nothing came back.

**Unchanged:** the CBS sync task and anyone viewing the site still read with the public key, which becomes read-only.

## Part 2: buttons that start Claude tasks

Two owner-only buttons on the Moneyline Pick'em tab call a small Supabase Edge Function, which checks Jeremy's session and then fires the matching Claude task through its API trigger. The trigger tokens live only in Supabase secrets.

Flow: **NFL Hub button** (Jeremy signed in; sends `read` or `write`) → **`run-task` Edge Function** (checks `is_owner()`, 5-minute cooldown, holds the API tokens; refuses if not Jeremy or too soon) → **Claude API trigger** (starts a new run, returns the session link) → **Claude task** (runs on Jeremy's PC; reads or sets CBS picks).

The page sends only Jeremy's session; the function decides, so copying the page's code gives no one a way in.

| Button | Claude task it fires | What the run does | Needs Jeremy's computer |
| --- | --- | --- | --- |
| Read pool picks | CBS pool live picks (Sun/Mon) | Copies whatever CBS picks are visible into the "Week N live" sheet tab; the hub updates the week from it, or closes the week out once every game is final | Yes |
| Send my picks to CBS | NFL Hub picks to CBS pool | Makes Jeremy's CBS picks match his hub picks (optimizer fills gaps), never touching the tiebreaker; then re-reads the CBS standings, rewrites the "Week N live" tab and requests a hub refresh | Yes |

**Edge Function `run-task` (`supabase/functions/run-task/index.ts`)**

1. Accepts `POST {task: "read" | "write"}` from the hub's origin only (CORS limited to `squid004.github.io`).
2. Reads the caller's session with `supabase.auth.getUser()` and refuses unless `is_owner()` is true.
3. Enforces a cooldown: a new `task_run` table (task, requested_at, session_url, error) refuses a second fire of the same task within 5 minutes. Claude also caps each task at 30 fires an hour.
4. Posts to `https://api.anthropic.com/v1/claude_code/routines/<trig id>/fire` with the task's token and a fixed note ("Started from the NFL Hub button at <time>"). It never forwards text from the page.
5. Saves the returned session link in `task_run` and returns it to the page.

**Hub UI**

1. Each button shows its last start time and a "View run" link to the Claude session.
2. While a cooldown is active the button is disabled with "Started 2 min ago".
3. A one-line note says the Claude desktop app must be open on Jeremy's computer.

**Run results in Pipeline Health**

1. Each browser task already opens the hub in the Claude app's browser, where Jeremy is signed in. As its last step, it writes one row to `task_run` (task, status OK / PARTIAL / SKIPPED / FAILED, one-line summary, finished_at) through `supabase-js` on that page, using his session. No new secret is needed.
2. Pipeline Health gets a "Claude tasks" card: the latest row per task, its status colour, summary and time, and the session link when the run came from a button.
3. If the hub session has expired, the task notes that in its Drive outcome file (which stays as the backup record) and the card shows the run as unreported.

**Refresh now:** stays as it is (it updates `refresh_request`, which the refresh job handles), but becomes owner-only under Part 1.

## Rollout, testing and rollback

Ship in this order so nothing breaks mid-week: the backend gets its secret key before anon loses write access.

1. **Backend key first.** Add `SUPABASE_SECRET_KEY`, switch `store.py` to it, and confirm one refresh run succeeds. Policies are unchanged, so this is safe alone.
2. **Sign-in UI.** Ship the header control and read-only mode; writes still work for everyone at this point.
3. **Lock the database.** Run `supabase/auth.sql` (owner table, `is_owner()`, new policies). Best on a Tuesday or Wednesday, away from pick deadlines.
4. **Edge Function and buttons.** Deploy `run-task`, add the two API triggers and their secrets, then ship the buttons.

**Testing checklist**

- [ ] Signed out in a private window: the page loads, picks are visible, and a pick click does nothing and says "Sign in to edit"
- [ ] With only the public key, a direct REST `PATCH` on `pickem_pick` changes 0 rows
- [ ] Signed in as Jeremy: set a pick, gut pick, ATS pick and Refresh now; each saves
- [ ] GitHub Actions refresh run succeeds after step 3
- [ ] The CBS sync task's read query still returns this week's picks
- [ ] `run-task` with no session → 401; with Jeremy's session → Claude run starts and the link opens
- [ ] A second press within 5 minutes is refused

**Rollback:** if writes break after step 3, re-run the old `anon all` loop from `schema.sql` (one SQL statement) and fix forward. The backend key from step 1 keeps refreshes working either way.

## What only Jeremy can do, and decisions

Claude Code can write all the code; these steps need Jeremy's own logins.

- [ ] Supabase: enable the Google provider (OAuth client from Google Cloud), set the Site URL, create his user, turn off new sign-ups, and copy his user id into `app_owner`
- [ ] Supabase: copy the secret key into the GitHub secret `SUPABASE_SECRET_KEY`
- [ ] Supabase: install the CLI (or use the dashboard) to deploy `run-task`, and set its secrets
- [ ] claude.ai/code/routines: on "CBS pool live picks (Sun/Mon)" and "NFL Hub picks to CBS pool", choose Add another trigger → API → Generate token; store each URL and token as Supabase secrets (tokens show once)

**Decisions (from Jeremy's answers)**

- Sign-in: Google.
- Reading picks: one task. It copies whatever CBS shows into the live tab; the hub decides whether that updates the week or closes it out. The Read button fires "CBS pool live picks (Sun/Mon)".
- Run results: shown in Pipeline Health (Part 2).
- Sending picks: one task does the write, the re-read and the sheet update, then requests a refresh. One task is enough, as long as the sheet is written from a fresh read of the CBS standings page after saving, not from the hub's target picks. That re-read is the independent check that the write stuck, and any mismatch marks the run PARTIAL.

The API trigger is experimental, so its request shape or limits could change ([Anthropic docs](https://platform.claude.com/docs/en/api/claude-code/routines-fire)).
