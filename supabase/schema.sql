-- Run this whole file once in the Supabase SQL editor
-- (supabase.com -> your project -> SQL Editor -> New query -> paste -> Run).
-- Personal tool, no auth: RLS is enabled but every policy is "anon all".

create table if not exists kv (
  key        text primary key,
  value      text,
  updated_at timestamptz default now()
);

create table if not exists game (
  game_id    text primary key,
  season     int,
  week       int,
  kickoff    timestamptz,
  home       text,
  away       text,
  home_full  text,
  away_full  text,
  state      text,               -- pre | in | post
  home_score int,
  away_score int,
  updated_at timestamptz default now()
);
create index if not exists game_week_idx on game (week);

create table if not exists odds (
  game_id      text primary key references game (game_id) on delete cascade,
  week         int,
  spread       real,             -- home spread, negative = home favored
  total        real,
  ml_home      int,
  ml_away      int,
  implied_home real,             -- de-vigged win probability
  implied_away real,
  book         text,
  updated_at   timestamptz default now()
);
create index if not exists odds_week_idx on odds (week);

create table if not exists roster_snapshot (
  league      text,              -- yahoo | espn
  week        int,
  captured_at timestamptz default now(),
  payload     jsonb,             -- {team_name, opponent_name, score, starters[], bench[], fp{...}, ...}
  primary key (league, captured_at)
);
create index if not exists roster_latest_idx on roster_snapshot (league, captured_at desc);

create table if not exists news (
  id          text primary key,
  published   timestamptz,
  headline    text,
  description text,
  players     jsonb,             -- rostered player names mentioned
  link        text,
  updated_at  timestamptz default now()
);

create table if not exists pickem_pick (
  week       int,
  game_id    text,
  pick       text,               -- team abbreviation
  confidence int,                -- nullable (confidence pools)
  created_at timestamptz default now(),
  primary key (week, game_id)
);
alter table pickem_pick add column if not exists spread_at_pick real;  -- home spread when picked

-- Weekly "take N" suggestion frozen for end-of-season analysis. One row per
-- (week, mode, spread bin), refreshed each cron run until the week's first kickoff.
create table if not exists budget_snapshot (
  week        int,
  mode        text,              -- 'ml' | 'ats'
  bin         text,              -- spread bucket label, or 'TOTAL'
  n_games     int,
  rate        real,              -- bin's historical dog upset / cover rate
  suggested   int,               -- rounded "take N dogs" for the bin
  games       jsonb,             -- [{game_id, fav, dog, dog_home, hist_su, hist_ats, flagged}]
  captured_at timestamptz default now(),
  primary key (week, mode, bin)
);

create table if not exists survivor_pick (
  week       int primary key,
  team       text,
  created_at timestamptz default now()
);

create table if not exists ats_pick (
  week           int,
  game_id        text,
  pick           text,             -- team abbreviation picked to COVER
  spread_at_pick real,             -- home spread when the pick was made (line can move)
  created_at     timestamptz default now(),
  primary key (week, game_id)
);

-- ---------------------------------------------------------------------------
-- pickem-edge integration: leverage/fade recommendations for the straight
-- moneyline pick'em pool, ported from github.com/squid004/pickem-edge.
-- 'edge_' prefix keeps this distinct from the unrelated budget_snapshot
-- (historical upset-rate) feature above.
-- ---------------------------------------------------------------------------

create table if not exists edge_national_pct (
  season     int,
  week       int,
  team       text,
  pct        real,               -- 0..1, fraction of the national pool picking this team
  source     text,               -- 'nflpickwatch' | 'manual'
  fetched_at timestamptz default now(),
  is_stale   boolean not null default false,
  primary key (season, week, team, source)
);

create table if not exists edge_opponent_pick (
  season      int,
  week        int,
  opponent    text,               -- pool member's name
  team_picked text,
  imported_at timestamptz default now(),
  source      text not null default 'paste',
  primary key (season, week, opponent, team_picked)
);

create table if not exists edge_bias (
  team           text primary key,
  bias_value     real not null default 0,
  n_observations int not null default 0,
  last_updated   timestamptz default now(),
  overridden     boolean not null default false  -- true = skip in the auto-recompute step
);

create table if not exists edge_season_standing (
  season          int,
  week            int,
  standing_bucket text not null,   -- LEADING | EARLY | MIDDLE | BEHIND (always manual)
  pool_size       int not null,
  correct_picks   int,
  total_picks     int,
  rank            int,
  updated_at      timestamptz default now(),
  primary key (season, week)
);

-- Frozen at the week's first kickoff (same lock pattern as budget_snapshot) so
-- season-log stats reflect what was actually recommended, not hindsight.
create table if not exists edge_recommendation_log (
  season         int,
  week           int,
  game_id        text,
  favorite_team  text,
  underdog_team  text,
  p_favorite     real not null,
  vig            real,
  f_estimate     real,             -- null if leverage was gated out (p > MAX_P)
  leverage       real,
  eligible       boolean not null,
  recommendation text not null,    -- FADE | CHALK | NO_PLAY
  budget_at_time int,
  generated_at   timestamptz default now(),
  primary key (season, week, game_id)
);

-- Best price across real sportsbooks, scraped from actionnetwork.com/nfl/odds (no
-- official API; see nflhub/sources/actionnetwork.py). One row per game, overwritten
-- each refresh — no history kept, this is "what can I get right now."
create table if not exists best_price_odds (
  game_id           text primary key references game (game_id) on delete cascade,
  week              int,
  avg_spread_home   real,             -- mean home spread across all quoted real books
  ml_home_book      text,
  ml_home_price     int,
  ml_away_book      text,
  ml_away_price     int,
  spread_home_book  text,
  spread_home_line  real,
  spread_home_price int,
  spread_away_book  text,
  spread_away_line  real,
  spread_away_price int,
  fetched_at        timestamptz default now()
);
create index if not exists best_price_odds_week_idx on best_price_odds (week);

-- One row per (game, real sportsbook) — the full board behind best_price_odds, so the
-- page can show every book's line, not just the winner.
create table if not exists book_odds (
  game_id           text references game (game_id) on delete cascade,
  book              text,
  week              int,
  spread_home_line  real,
  spread_home_price int,
  spread_away_line  real,
  spread_away_price int,
  ml_home_price     int,
  ml_away_price     int,
  fetched_at        timestamptz default now(),
  primary key (game_id, book)
);
create index if not exists book_odds_week_idx on book_odds (week);

-- "ELWAY" model: home/away average points + win prob, pulled from a personal Google
-- Sheet (nflhub/sources/elway.py). spread_home/total are DERIVED here (avg pts diff /
-- sum) — the sheet's own spread/total columns are intentionally not used, since the
-- point is comparing this model against the real sportsbook line, not the sheet
-- author's own line.
create table if not exists elway_odds (
  game_id        text primary key references game (game_id) on delete cascade,
  week           int,
  home_win_prob  real,
  away_win_prob  real,
  spread_home    real,             -- away_avg_pts - home_avg_pts; negative = home favored
  total          real,             -- home_avg_pts + away_avg_pts
  fetched_at     timestamptz default now()
);
create index if not exists elway_odds_week_idx on elway_odds (week);

create table if not exists reminder_log (
  kind    text,
  key     text,
  sent_at timestamptz default now(),
  primary key (kind, key)
);

create table if not exists refresh_request (
  id           int primary key default 1,
  requested_at timestamptz,
  handled_at   timestamptz
);
insert into refresh_request (id) values (1) on conflict (id) do nothing;

-- RLS: personal single-tenant tool, allow anon full access on every table.
do $$
declare t text;
begin
  foreach t in array array[
    'kv','game','odds','roster_snapshot','news','pickem_pick','ats_pick',
    'survivor_pick','reminder_log','refresh_request','budget_snapshot',
    'edge_national_pct','edge_opponent_pick','edge_bias','edge_season_standing',
    'edge_recommendation_log','best_price_odds','book_odds','elway_odds'
  ]
  loop
    execute format('alter table %I enable row level security', t);
    execute format('drop policy if exists "anon all" on %I', t);
    execute format(
      'create policy "anon all" on %I for all to anon, authenticated using (true) with check (true)', t
    );
  end loop;
end $$;
