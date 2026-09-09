# NFL Hub

One page for your NFL season: both fantasy leagues (Yahoo + ESPN), CBS pick'em, Yahoo
survivor, betting lines, and the lineup FantasyPros would start — plus ntfy phone reminders
before every lineup lock and pick deadline.

**Read + reminder only.** It never submits a lineup or a pick for you.

## How it's built

- **Frontend:** static `index.html` + `js/*.js` on **GitHub Pages**. Reads Supabase live,
  renders every panel client-side. Pick'em / survivor buttons write straight to Supabase.
- **Data store:** **Supabase** (Postgres). Anon key is committed (client-side use); every
  table's RLS policy is "anon all" — personal single-tenant tool.
- **Updater:** a **GitHub Actions** cron job (`.github/workflows/refresh.yml`, every ~10 min
  + manual "Run workflow") runs `python -m nflhub refresh` to pull all sources into Supabase
  and `python -m nflhub tick` to send any due reminders to ntfy. Your PC does not need to be on.
- **"Refresh now" button:** sets a flag in Supabase; the next cron run refreshes immediately.
  No token in the page.

## API notes

| Source | What it gives us |
|---|---|
| Yahoo fantasy | Official OAuth2 API. Roster, matchup, opponent. (Lineup *write* is possible but out of scope.) |
| ESPN fantasy | Unofficial `espn-api`. Private league needs `ESPN_S2` + `ESPN_SWID` cookies. |
| CBS pick'em | No API. Slate derived from the NFL schedule; you record picks on the page. |
| Yahoo survivor | No API. "Teams used" tracked from your recorded picks. |
| Odds | Free ESPN scoreboard lines by default; SportsGameOdds / The Odds API optional. |
| FantasyPros | Paid key. Weekly projections + ECR. **No league sync / lineup access** — the "optimal lineup" is computed locally in `nflhub/optimizer.py`. |

## Setup

### 1. Supabase

1. Create a new project at supabase.com.
2. SQL Editor → paste [supabase/schema.sql](supabase/schema.sql) → Run.
3. Settings → API → copy the **Project URL** and the **anon public** key.
4. Paste both into [js/db.js](js/db.js) (`SUPABASE_URL`, `SUPABASE_ANON`).

### 2. GitHub

Push this folder to a new repo, then in **Settings**:

- **Pages** → Source: *Deploy from a branch* → `main` / `/ (root)`.
- **Secrets and variables → Actions → Secrets:**
  `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `FANTASYPROS_API_KEY`,
  `ESPN_S2`, `ESPN_SWID`, `NTFY_TOPIC`, `NTFY_TOKEN` (opt), `ODDS_API_KEY` (opt),
  `YAHOO_OAUTH_JSON` (contents of `oauth2.json`, see step 3).
- **Variables:** `YAHOO_LEAGUE_ID` (`nfl.l.<num>`), `ESPN_LEAGUE_ID`, `ESPN_TEAM_ID`,
  `FP_SCORING_YAHOO`/`FP_SCORING_ESPN` (`PPR`|`HALF`|`STD`), `ODDS_PROVIDER` (`espn`),
  `NTFY_SERVER` (`https://ntfy.sh`), `TZ_NAME` (`America/New_York`),
  `DASHBOARD_URL` (your Pages URL), `REMINDER_OFFSETS` (`24,3,0.75`),
  `QUIET_START` (`23`), `QUIET_END` (`7`).

The FantasyPros key is the same one used by the `ff-draft-edge` project.

### 3. Yahoo one-time consent (local)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env      # fill in YAHOO_CLIENT_ID / YAHOO_CLIENT_SECRET
.\.venv\Scripts\python.exe -m nflhub yahoo-auth
```

Paste the contents of the generated `oauth2.json` into the `YAHOO_OAUTH_JSON` secret.
If Yahoo ever invalidates the token, re-run this and update the secret.

### 4. ntfy

Pick an unguessable `NTFY_TOPIC`, subscribe to it in the ntfy phone app, then:

```powershell
.\.venv\Scripts\python.exe -m nflhub notify --test
```

## Local development

`.env` with at least `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `FANTASYPROS_API_KEY`:

```powershell
.\.venv\Scripts\python.exe -m nflhub refresh      # pull everything into Supabase
.\.venv\Scripts\python.exe -m nflhub tick         # send due reminders
.\.venv\Scripts\python.exe -m nflhub notify --dry-run
```

Open `index.html` — it talks to the same Supabase project as the deployed page.

## Layout

```
index.html  style.css  js/*.js       static dashboard (GitHub Pages)
supabase/schema.sql                   Postgres schema + RLS
nflhub/                               the Actions job
  __main__.py   refresh | tick | notify | yahoo-auth
  store.py      Supabase REST read/write
  config.py     env-only config
  refresh.py  deadlines.py  notify.py  optimizer.py  util.py
  sources/  nfl_schedule  odds  fantasypros  espn_fantasy  yahoo_fantasy  pickem  survivor
.github/workflows/refresh.yml         cron + manual dispatch
```

## Known trade-offs

- Public repo + RLS "anon all": anyone with the Supabase URL can read/write your picks
  (same posture as the meal-planner project). The FantasyPros paid key is **not** exposed —
  it lives only in Actions secrets.
- "Refresh now" latency = one cron interval; GitHub's scheduled runs can lag 5–20 min.
- Reminder precision is bounded by the cron cadence.
