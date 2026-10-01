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

  // Enhancement-only reads: a missing table (schema.sql not yet re-run after a new
  // feature) or a transient error here must never blank the whole page — every other
  // section still has everything it needs. Swallow and degrade to "no data" instead of
  // throwing, mirroring how refresh.py treats these same sources as soft-fail extras.
  async _softMap(table, week, key = 'game_id') {
    try {
      const { data, error } = await this._c().from(table).select('*').eq('week', week);
      if (error) throw error;
      const map = {};
      (data || []).forEach(r => { map[r[key]] = r; });
      return map;
    } catch (e) {
      console.warn(`${table} unavailable:`, e.message || e);
      return {};
    }
  },

  async bestPriceOdds(week) {
    return this._softMap('best_price_odds', week);
  },

  async bookOdds(week) {
    try {
      const { data, error } = await this._c().from('book_odds').select('*').eq('week', week);
      if (error) throw error;
      const map = {};
      (data || []).forEach(r => { (map[r.game_id] = map[r.game_id] || []).push(r); });
      return map;
    } catch (e) {
      console.warn('book_odds unavailable:', e.message || e);
      return {};
    }
  },

  async elwayOdds(week) {
    return this._softMap('elway_odds', week);
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

  // { week_winners: [{week, names, correct, dog_pct}], week_winner_dog_pct,
  //   season_leaders: {names, correct, total, dog_pct} }, written by nflhub.refresh
  // (nflhub/sources/edge_core.py pool_leaderboard_summary) -- the "chalk is king" reminder
  // at the bottom of the Upset Budget panel. See js/history.js renderBins().
  async edgeLeaderboard() {
    const { data } = await this._c().from('kv').select('value').eq('key', 'edge_pool_leaderboard').maybeSingle();
    if (!data) return null;
    try { return JSON.parse(data.value); } catch { return null; }
  },

  // team abbr -> [{player, position, status, detail}], written by nflhub.refresh (ESPN's
  // league-wide injury report). Only ever read here for the QB flag — see pickem.js.
  async injuries() {
    const { data } = await this._c().from('kv').select('value').eq('key', 'injuries').maybeSingle();
    if (!data) return {};
    try { return JSON.parse(data.value); } catch { return {}; }
  },

  // team abbr -> {name, qbert}, the QB ELWAY's "Current Rankings" tab currently evaluates
  // that team with at QB1 (written by nflhub.refresh). See pickem.js's elwayStaleQb().
  async elwayQb1() {
    const { data } = await this._c().from('kv').select('value').eq('key', 'elway_qb1').maybeSingle();
    if (!data) return {};
    try { return JSON.parse(data.value); } catch { return {}; }
  },

  // { teams: {TEAM: {rush_off_epa, pass_off_epa, rush_def_epa_allowed, pass_def_epa_allowed}},
  //   matchups: {"AWAY@HOME": {home, away, home_ratings, away_ratings, callouts: [...]}} },
  // written by nflhub.refresh (garbage-time-excluded EPA ratings; see nflhub/sources/team_ratings.py).
  async teamRatings() {
    const { data } = await this._c().from('kv').select('value').eq('key', 'team_ratings').maybeSingle();
    if (!data) return null;
    try { return JSON.parse(data.value); } catch { return null; }
  },

  async spreadHistory(week) {
    try {
      const { data, error } = await this._c().from('spread_history').select('game_id,captured_at,spread_home')
        .eq('week', week).order('captured_at', { ascending: true });
      if (error) throw error;
      const map = {};
      (data || []).forEach(r => { (map[r.game_id] = map[r.game_id] || []).push(r); });
      return map;
    } catch (e) {
      console.warn('spread_history unavailable:', e.message || e);
      return {};
    }
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

  async requestRefresh() {
    const { error } = await this._c().from('refresh_request')
      .update({ requested_at: new Date().toISOString(), handled_at: null }).eq('id', 1);
    if (error) throw error;
  },

  async edgeSetNationalPct(season, week, team, pct) {
    const { error } = await this._c().from('edge_national_pct').upsert({
      season, week, team, pct, source: 'manual', fetched_at: new Date().toISOString(), is_stale: false,
    }, { onConflict: 'season,week,team,source' });
    if (error) throw error;
  },
};
