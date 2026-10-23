# Project notes for Claude

## Supabase tables

When creating a new table in the Supabase `public` schema, the same SQL must include: explicit `GRANT`s for each role that needs access (`anon`, `authenticated`, `service_role`, only the privileges needed), `alter table ... enable row level security`, and RLS policies. Tables that should stay private get no grants. (Since 2026-10-30 Supabase no longer grants API access to new `public` tables automatically.) Prefer storing small new data as a `kv` row over adding a table.
