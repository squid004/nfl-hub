'use strict';

// Supabase project for nfl-hub. The anon key is meant for client-side use; RLS on every
// table is "anon all" (personal single-tenant tool). Fill these in after creating the
// project and running supabase/schema.sql.
const DB = {
  SUPABASE_URL: 'https://ssomwvamhlmozguhosaf.supabase.co',
  SUPABASE_ANON: 'sb_publishable_ViNynCyQ-0IrksDFyGtqGQ_IrJamDpX',
  _sb: null,

  init() {
    if (this.SUPABASE_URL.startsWith('PASTE')) {
      document.body.insertAdjacentHTML('afterbegin',
        '<div class="banner">Supabase not configured — edit <code>js/db.js</code>.</div>');
      return;
    }
    this._sb = window.supabase.createClient(this.SUPABASE_URL, this.SUPABASE_ANON);
  },
  _c() { if (!this._sb) throw new Error('Supabase not initialized'); return this._sb; },

  async kv(key) {
    const { data } = await this._c().from('kv').select('value').eq('key', key).maybeSingle();
    return data ? data.value : null;
  },

  async weekGames(week) {
    const { data, error } = await this._c().from('game').select('*').eq('week', week).order('kickoff');
    if (error) throw error;
    return data || [];
  },

  async weekOdds(week) {
    const { data, error } = await this._c().from('odds').select('*').eq('week', week);
    if (error) throw error;
    const map = {};
    (data || []).forEach(o => { map[o.game_id] = o; });
    return map;
  },

  async latestRoster(league) {
    const { data } = await this._c().from('roster_snapshot').select('*')
      .eq('league', league).order('captured_at', { ascending: false }).limit(1).maybeSingle();
    return data || null;
  },

  async pickemPicks(week) {
    const { data } = await this._c().from('pickem_pick').select('*').eq('week', week);
    const map = {};
    (data || []).forEach(p => { map[p.game_id] = p; });
    return map;
  },

  async survivorPicks() {
    const { data } = await this._c().from('survivor_pick').select('*').order('week');
    const map = {};
    (data || []).forEach(p => { map[p.week] = p.team; });
    return map;
  },

  async news() {
    const { data } = await this._c().from('news').select('*').order('published', { ascending: false }).limit(20);
    return data || [];
  },

  async refreshRequest() {
    const { data } = await this._c().from('refresh_request').select('*').eq('id', 1).maybeSingle();
    return data || null;
  },

  // --- writes ---
  async setPickemPick(week, gameId, team) {
    const { error } = await this._c().from('pickem_pick')
      .upsert({ week, game_id: gameId, pick: team, created_at: new Date().toISOString() },
              { onConflict: 'week,game_id' });
    if (error) throw error;
  },

  async setSurvivorPick(week, team) {
    const { error } = await this._c().from('survivor_pick')
      .upsert({ week, team, created_at: new Date().toISOString() }, { onConflict: 'week' });
    if (error) throw error;
  },

  async requestRefresh() {
    const { error } = await this._c().from('refresh_request')
      .update({ requested_at: new Date().toISOString(), handled_at: null }).eq('id', 1);
    if (error) throw error;
  },
};
