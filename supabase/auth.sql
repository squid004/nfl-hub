-- Owner-only writes. Run in the Supabase SQL editor AFTER Google sign-in works and you have put
-- your user id in app_owner (see docs/auth-setup.md). Reads stay public; every write requires the
-- signed-in owner. The refresh job and other server code use the secret key (SUPABASE_SECRET_KEY),
-- which bypasses RLS, so they are unaffected.
--
-- ROLLBACK (re-opens writes to everyone), if anything breaks:
--   do $$ declare t text; begin
--     foreach t in array array['kv','game','odds','roster_snapshot','news','pickem_pick','ats_pick',
--       'survivor_pick','reminder_log','refresh_request','budget_snapshot','edge_national_pct',
--       'edge_opponent_pick','edge_bias','edge_season_standing','edge_recommendation_log',
--       'best_price_odds','book_odds','elway_odds','spread_history'] loop
--       execute format('drop policy if exists "public read" on %I', t);
--       execute format('drop policy if exists "owner write" on %I', t);
--       execute format('drop policy if exists "anon all" on %I', t);
--       execute format('create policy "anon all" on %I for all to anon, authenticated using (true) with check (true)', t);
--     end loop; end $$;

-- 1. Who the owner is.
create table if not exists app_owner (
  user_id uuid primary key references auth.users (id) on delete cascade
);
alter table app_owner enable row level security;

create or replace function is_owner() returns boolean
language sql stable security definer set search_path = public as $$
  select exists (select 1 from app_owner where user_id = auth.uid());
$$;

-- Seed the owner ONCE, replacing the uuid with your own (Authentication > Users > User UID):
--   insert into app_owner (user_id) values ('00000000-0000-0000-0000-000000000000');

-- app_owner itself: a signed-in user may read only their own row (the page uses this to know it is the owner);
-- nobody can write it through the API, so no one can add themselves. Seed it from this editor (postgres role).
drop policy if exists "owner reads self" on app_owner;
create policy "owner reads self" on app_owner for select to authenticated using (user_id = auth.uid());

-- 2. Every data table: public read, owner-only write.
do $$
declare t text;
begin
  foreach t in array array[
    'kv','game','odds','roster_snapshot','news','pickem_pick','ats_pick',
    'survivor_pick','reminder_log','refresh_request','budget_snapshot',
    'edge_national_pct','edge_opponent_pick','edge_bias','edge_season_standing',
    'edge_recommendation_log','best_price_odds','book_odds','elway_odds','spread_history'
  ]
  loop
    execute format('alter table %I enable row level security', t);
    execute format('drop policy if exists "anon all" on %I', t);
    execute format('drop policy if exists "public read" on %I', t);
    execute format('drop policy if exists "owner write" on %I', t);
    execute format('create policy "public read" on %I for select to anon, authenticated using (true)', t);
    execute format(
      'create policy "owner write" on %I for all to authenticated using (is_owner()) with check (is_owner())', t
    );
  end loop;
end $$;
