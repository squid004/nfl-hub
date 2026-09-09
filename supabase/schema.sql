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
    'survivor_pick','reminder_log','refresh_request'
  ]
  loop
    execute format('alter table %I enable row level security', t);
    execute format('drop policy if exists "anon all" on %I', t);
    execute format(
      'create policy "anon all" on %I for all to anon, authenticated using (true) with check (true)', t
    );
  end loop;
end $$;
