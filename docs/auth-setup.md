# Owner sign-in setup (Google through Supabase Auth)

Goal: anyone can view NFL Hub; only the signed-in owner can change picks or request a refresh. The page ships a
sign-in control and a read-only mode; the real lock is the database (`supabase/auth.sql`). Server code (the refresh
job) writes with `SUPABASE_SECRET_KEY`, which bypasses the lock, and that secret is already set.

Do the steps in this order. Steps 1-3 change nothing about who can write; step 7 is the lock.

## 1. Google Cloud: an OAuth client
1. console.cloud.google.com: create or pick a project.
2. APIs & Services > OAuth consent screen: External, app name, your email as support and developer contact. Leave it in
   Testing and add your own Google email under Test users.
3. APIs & Services > Credentials > Create credentials > OAuth client ID > Web application.
4. Authorized JavaScript origins: `https://squid004.github.io`
5. Authorized redirect URIs: `https://ssomwvamhlmozguhosaf.supabase.co/auth/v1/callback`
6. Copy the Client ID and Client secret.

## 2. Supabase dashboard
1. Authentication > Providers > Google: enable, paste the Client ID and Client secret.
2. Authentication > URL Configuration: Site URL `https://squid004.github.io/nfl-hub/`; add the same address under Redirect URLs.
3. Leave "Allow new users to sign up" ON for now.

## 3. First sign-in
Open the hub and press "Sign in with Google" in the header (after this change is deployed). Then, in Authentication >
Users, copy your row's User UID. (If you use the installed home-screen app, do the first sign-in in the browser.)

## 4. (Optional now) turn on read-only mode for visitors
`Auth.ENFORCE` in `js/auth.js` is `false`, so the edit buttons are still visible to everyone. Once you can sign in, set it
to `true` (and bump the `?v=` stamp in `index.html`) so signed-out visitors see the pick and refresh buttons dimmed, with a
"Read-only" note. This is cosmetic; the database is what actually blocks writes.

## 5. Turn off new sign-ups
Authentication > Sign In / Providers (settings): turn off "Allow new users to sign up".

## 6. Seed the owner
In the Supabase SQL editor, run this with your UID from step 3 (it creates the owner table and adds you; the database is not locked yet):

```sql
create table if not exists app_owner (user_id uuid primary key references auth.users (id) on delete cascade);
insert into app_owner (user_id) values ('YOUR-USER-UID') on conflict do nothing;
```

## 7. Lock the database
Run all of `supabase/auth.sql` in the SQL editor. Best on a Tuesday or Wednesday, away from pick deadlines. It makes every
table public-read and owner-only-write. Its header contains the one-statement rollback that re-opens writes.

## Testing checklist
- [ ] Signed out in a private window: the page loads and shows picks; clicking a pick says "Sign in to edit" (once step 4 is on)
- [ ] With only the public key, a direct REST PATCH on `pickem_pick` changes 0 rows (and a POST is rejected)
- [ ] Signed in as the owner: set a Moneyline pick, a gut pick, an ATS pick, and press Refresh now; each saves
- [ ] The next GitHub Actions refresh run succeeds (it uses the secret key)
- [ ] The pool-picks sync and any task that reads `pickem_pick` still reads it (reads are public)

Writes now ask Supabase for the written row back and show an error toast if nothing changed, because a blocked UPDATE under
row-level security can otherwise touch 0 rows without raising an error.
