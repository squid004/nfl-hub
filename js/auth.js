'use strict';

// Owner sign-in (Google, through Supabase Auth). Anyone can still READ the hub; the database
// decides who can write (supabase/auth.sql, applied separately). This file only handles the
// session, the header control, a read-only mode that hides the edit controls for visitors, and
// toasts for blocked or failed writes.
//
// ENFORCE controls the read-only mode (a cosmetic layer -- the database is the real lock). It is
// false until Google sign-in works end to end, otherwise the page would hide its own pick buttons
// from the owner. Flip it to true (and bump the ?v= stamp) once you can sign in, BEFORE running
// supabase/auth.sql.
const Auth = {
  ENFORCE: true,
  session: null,
  isOwner: false,

  async init() {
    const sb = DB._sb;
    if (!sb || !sb.auth) return;
    const { data } = await sb.auth.getSession();
    await this._set(data && data.session);
    sb.auth.onAuthStateChange((_event, session) => { this._set(session).then(() => this.render()); });
    this.render();
    document.body.addEventListener('click', e => {
      const b = e.target.closest('button[data-auth]');
      if (!b) return;
      if (b.dataset.auth === 'signin') this.signIn();
      if (b.dataset.auth === 'signout') this.signOut();
    });
  },

  async _set(session) {
    this.session = session || null;
    this.isOwner = false;
    if (!this.session) return;
    // Once supabase/auth.sql has run, only the user in app_owner may write. Before that the table
    // doesn't exist (the query errors) and a signed-in user is treated as the owner.
    try {
      const { data, error } = await DB._sb.from('app_owner').select('user_id').eq('user_id', this.session.user.id).maybeSingle();
      this.isOwner = error ? true : !!data;
    } catch { this.isOwner = true; }
  },

  // True if the page should offer edit controls.
  canWrite() { return !this.ENFORCE || (!!this.session && this.isOwner); },

  async signIn() {
    const { error } = await DB._sb.auth.signInWithOAuth({
      provider: 'google',
      options: { redirectTo: window.location.origin + window.location.pathname },
    });
    if (error) Toast.show('Sign-in failed: ' + error.message, 'bad');
  },

  async signOut() {
    await DB._sb.auth.signOut();
  },

  render() {
    const box = document.getElementById('auth-box');
    if (box) {
      box.innerHTML = this.session
        ? `<span class="small muted" title="${esc(this.session.user.email || '')}">${this.isOwner ? 'Owner' : 'Signed in (not owner)'}</span>
           <button data-auth="signout" class="small">Sign out</button>`
        : `<button data-auth="signin" class="small">Sign in with Google</button>`;
    }
    document.body.classList.toggle('read-only', !this.canWrite());
    const note = document.getElementById('read-only-note');
    if (note) note.hidden = this.canWrite();
  },

  // Gate for every click that writes. Returns true if allowed; otherwise explains why.
  guard() {
    if (this.canWrite()) return true;
    Toast.show(this.session ? 'This account is not the owner of this hub.' : 'Sign in to edit.', 'warn');
    return false;
  },
};

// Minimal toast: used for blocked/failed writes instead of alert().
const Toast = {
  show(msg, kind = 'info') {
    let host = document.getElementById('toast-host');
    if (!host) {
      host = document.createElement('div');
      host.id = 'toast-host';
      document.body.appendChild(host);
    }
    const t = document.createElement('div');
    t.className = `toast ${kind}`;
    t.textContent = msg;
    host.appendChild(t);
    setTimeout(() => t.remove(), 6000);
  },
};
