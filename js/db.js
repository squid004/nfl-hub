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

  // team abbr -> ordered list of QB names (index 0 = current starter), from ESPN's depth
  // chart (written by nflhub.refresh). See pickem.js's qbChip().
  async qbDepthChart() {
    const { data } = await this._c().from('kv').select('value').eq('key', 'qb_depth_chart').maybeSingle();
    if (!data) return {};
    try { return JSON.parse(data.value); } catch { return {}; }
  },

  // { checked_at, errors: [...], ok } for the LATEST refresh run, plus a rolling log of just
  // the runs that had errors ({ checked_at, errors }[], newest last, capped server-side at
  // PIPELINE_HEALTH_LOG_MAX) -- written every refresh_all() run by nflhub/refresh.py. No
  // push/email; this is the page a user checks instead of being notified.
  async pipelineHealth() {
    const { data } = await this._c().from('kv').select('value').eq('key', 'pipeline_health').maybeSingle();
    let current = null;
    if (data) { try { current = JSON.parse(data.value); } catch { current = null; } }
    const { data: logData } = await this._c().from('kv').select('value').eq('key', 'pipeline_health_log').maybeSingle();
    let log = [];
    if (logData) { try { log = JSON.parse(logData.value); } catch { log = []; } }
    return { current, log };
  },

  // { teams: {TEAM: {rush_off_epa, pass_off_epa, rush_def_epa_allowed, pass_def_epa_allowed}},
  //   matchups: {"AWAY@HOME": {home, away, home_ratings, away_ratings, callouts: [...]}} },
  // written by nflhub.refresh (garbage-time-excluded EPA ratings; see nflhub/sources/team_ratings.py).
  async teamRatings() {
    const { data } = await this._c().from('kv').select('value').eq('key', 'team_ratings').maybeSingle();
    if (!data) return null;
    try { return JSON.parse(data.value); } catch { return null; }
  },

  // { season, week, teams: {TEAM: {rush_off_epa, pass_off_epa, rush_def_epa_allowed,
  //   pass_def_epa_allowed}} } -- this week's EPA display-ratings, frozen at first kickoff
  // (written by nflhub.refresh; see team_ratings.py's _refresh_parlay_snapshot). The
  // Parlays tab's matchup identifier reads this INSTEAD of teamRatings() so a call-out made
  // before kickoff doesn't quietly stop qualifying once team_ratings' own live numbers move
  // on from incorporating the very game it described.
  async parlayRatingsSnapshot() {
    const { data } = await this._c().from('kv').select('value').eq('key', 'parlay_ratings_snapshot').maybeSingle();
    if (!data) return null;
    try { return JSON.parse(data.value); } catch { return null; }
  },

  // { "<season>": { "<week>": { generated, teams, power_rankings, parlay_teams } } } --
  // permanent per-week snapshot ("ratings as of the last refresh before that week's first
  // kickoff", frozen forever once computed; see team_ratings.py's refresh_weekly_power_
  // rankings()). Powers the week dropdown's Power Rankings/Parlays history -- the CURRENT
  // week has no entry yet (team_ratings() above is its own live pre-kickoff preview).
  async weeklyPowerRankings() {
    const { data } = await this._c().from('kv').select('value').eq('key', 'power_rankings_by_week').maybeSingle();
    if (!data) return null;
    try { return JSON.parse(data.value); } catch { return null; }
  },

  // { seasons: [2007...2025], rankings: [{team, season, rank, score_display, ratings_display}, ...] }
  // -- every (team, season) from every fully completed regular season, pooled into one
  // dataset (written by nflhub.refresh; see team_ratings.py's refresh_historical()/
  // historical_power_rankings()). Full-season averages, not the live tab's EWMA rating.
  async historicalPower() {
    const { data } = await this._c().from('kv').select('value').eq('key', 'historical_power_rankings').maybeSingle();
    if (!data) return null;
    try { return JSON.parse(data.value); } catch { return null; }
  },

  // The pool Monte Carlo's recommendation for the current week (nflhub/sources/pool_recommend.py):
  // {season, week, generated, pot, p_win, ev_dollars, tiebreak_guess, all_favorites, games: {game_id: {pick,
  // contrarian, gain_dollars, ...}}} or null when it hasn't run (no pool history yet, week 1, etc).
  async poolRecommendation() {
    const raw = await this.kv('pool_recommendation');
    if (!raw) return null;
    try { return JSON.parse(raw); } catch { return null; }
  },

  // {game_id: {season, week, home, away, gameday, delta, p_home, spread_line, ml_home, ml_away}} --
  // the model's pre-kickoff prediction and the market line at that moment, append-only (never
  // rebuilt on a model refit; written by team_ratings._log_prospective). The 2026 holdout.
  async prospectiveLog() {
    const raw = await this.kv('prospective_log');
    if (!raw) return {};
    try { return JSON.parse(raw); } catch { return {}; }
  },

  // [{season, week, home, away, home_score, away_score, elway_favorite, winner}, ...] --
  // every COMPLETED game that has a stored ELWAY pick (written by nflhub.refresh; see
  // nflhub/sources/elway.py's score_history()). Starts wherever this app began recording
  // ELWAY picks (2026 wk2 currently) -- there's no row for an earlier game, by construction,
  // not a bug. Market/Power Rankings already have a full historical record via
  // historicalPower() above; this is the one piece that dataset doesn't have.
  async elwayScoreHistory() {
    const { data } = await this._c().from('kv').select('value').eq('key', 'elway_score_history').maybeSingle();
    if (!data) return [];
    try { return JSON.parse(data.value) || []; } catch { return []; }
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

  // --- writes ---
  async setPickemPick(week, gameId, team, spread) {
    const { error } = await this._c().from('pickem_pick')
      .upsert({ week, game_id: gameId, pick: team,
                spread_at_pick: (spread == null || spread === '') ? null : Number(spread),
                created_at: new Date().toISOString() },
              { onConflict: 'week,game_id' });
    if (error) throw error;
  },

  // Gut picks: the FIRST team clicked on a Moneyline card while it was still locked (nothing
  // but teams/records/QB chip/weather chip visible). Stored as one kv blob,
  // { [game_id]: { pick, week, season, home, away, at } }, so no schema change is needed.
  // Write-once per game: recordGutPick re-reads fresh and never overwrites an existing entry.
  async gutPicks() {
    const { data } = await this._c().from('kv').select('value').eq('key', 'gut_picks').maybeSingle();
    if (!data || !data.value) return {};
    try { return JSON.parse(data.value) || {}; } catch { return {}; }
  },

  async recordGutPick(gameId, entry) {
    const cur = await this.gutPicks();
    if (cur[gameId]) return;
    cur[gameId] = { ...entry, at: new Date().toISOString() };
    const { error } = await this._c().from('kv')
      .upsert({ key: 'gut_picks', value: JSON.stringify(cur), updated_at: new Date().toISOString() },
              { onConflict: 'key' });
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
