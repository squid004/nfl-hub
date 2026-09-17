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

  async atsPicks(week) {
    const { data } = await this._c().from('ats_pick').select('*').eq('week', week);
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

  async hist() {
    const { data } = await this._c().from('kv').select('value').eq('key', 'hist_distribution').maybeSingle();
    if (!data) return null;
    try { return JSON.parse(data.value); } catch { return null; }
  },

  // --- pickem-edge ---
  async edgeRecommendationLog(season, week) {
    const { data } = await this._c().from('edge_recommendation_log').select('*')
      .eq('season', season).eq('week', week);
    const map = {};
    (data || []).forEach(r => { map[r.game_id] = r; });
    return map;
  },

  async edgeBiasAll() {
    const { data } = await this._c().from('edge_bias').select('*').order('team');
    return data || [];
  },

  async edgeStandings(season) {
    const { data } = await this._c().from('edge_season_standing').select('*')
      .eq('season', season).order('week');
    return data || [];
  },

  async edgeOpponentPicks(season, week) {
    const { data } = await this._c().from('edge_opponent_pick').select('*')
      .eq('season', season).eq('week', week).order('opponent');
    return data || [];
  },

  async edgeRecommendationLogSeason(season) {
    const { data } = await this._c().from('edge_recommendation_log').select('*')
      .eq('season', season).order('week');
    return data || [];
  },

  async seasonGames(season) {
    const { data } = await this._c().from('game').select('*').eq('season', season);
    return data || [];
  },

  // --- writes ---
  async setPickemPick(week, gameId, team, spread) {
    const { error } = await this._c().from('pickem_pick')
      .upsert({ week, game_id: gameId, pick: team,
                spread_at_pick: (spread == null || spread === '') ? null : Number(spread),
                created_at: new Date().toISOString() },
              { onConflict: 'week,game_id' });
    if (error) throw error;
  },

  async budgetSnapshot(week) {
    const { data } = await this._c().from('budget_snapshot').select('*').eq('week', week);
    return data || [];
  },

  async setAtsPick(week, gameId, team, spread) {
    const { error } = await this._c().from('ats_pick')
      .upsert({ week, game_id: gameId, pick: team,
                spread_at_pick: spread == null ? null : Number(spread),
                created_at: new Date().toISOString() },
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

  async edgeUpsertStanding(row) {
    const { error } = await this._c().from('edge_season_standing')
      .upsert({ ...row, updated_at: new Date().toISOString() }, { onConflict: 'season,week' });
    if (error) throw error;
  },

  async edgeSetBiasOverride(team, biasValue, nObservations) {
    const { error } = await this._c().from('edge_bias').upsert({
      team, bias_value: biasValue, n_observations: nObservations ?? 0,
      overridden: true, last_updated: new Date().toISOString(),
    }, { onConflict: 'team' });
    if (error) throw error;
  },

  async edgeClearBiasOverride(team) {
    const { error } = await this._c().from('edge_bias').update({ overridden: false }).eq('team', team);
    if (error) throw error;
  },

  async edgeInsertOpponentPicks(rows) {
    if (!rows.length) return;
    const payload = rows.map(r => ({ ...r, imported_at: new Date().toISOString() }));
    const { error } = await this._c().from('edge_opponent_pick')
      .upsert(payload, { onConflict: 'season,week,opponent,team_picked' });
    if (error) throw error;
  },

  async edgeDeleteOpponentPick(season, week, opponent, teamPicked) {
    const { error } = await this._c().from('edge_opponent_pick').delete()
      .eq('season', season).eq('week', week).eq('opponent', opponent).eq('team_picked', teamPicked);
    if (error) throw error;
  },

  async edgeSetNationalPct(season, week, team, pct) {
    const { error } = await this._c().from('edge_national_pct').upsert({
      season, week, team, pct, source: 'manual', fetched_at: new Date().toISOString(), is_stale: false,
    }, { onConflict: 'season,week,team,source' });
    if (error) throw error;
  },
};
