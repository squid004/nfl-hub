'use strict';

// ESPN's public scoreboard API, fetched directly from the browser (not through Supabase/
// the refresh cron) ONLY while a game is live -- confirmed CORS-open (Access-Control-Allow-
// Origin: *) before building this, same endpoint nflhub.sources.nfl_schedule.fetch_
// scoreboard() already uses server-side. The 10-minute refresh cron is both too slow for
// "feels live" and, per its own comment, not even reliably on-time under GitHub's own
// scheduling load -- this sidesteps that entirely for just the score/state fields, which
// are the only thing that needs to move faster than everything else on the page.
const ESPN_SCOREBOARD_URL = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard';
const LIVE_POLL_INTERVAL_MS = 20_000;
const DOWN_ORDINAL = { 1: '1st', 2: '2nd', 3: '3rd', 4: '4th' };

// ESPN's situation.yardLine is 0-100 measured from the POSSESSING team's own goal line (0)
// to their opponent's (100) -- not pinned to either team's own "50," so converting it to the
// usual broadcast "TEAM ##" format means picking whichever side of midfield it's actually
// on: <=50 is still the possessing team's own territory, >50 is already past midfield into
// the opponent's. Checked against two different live games before trusting this (DEN@LAC,
// LAC with the ball at yardLine 17 in the 1st quarter -- exactly what you'd expect from a
// drive pinned deep in LAC's own territory, not 17 yards from a LAC touchdown).
function _fieldPositionText(yardLine, possAbbr, otherAbbr) {
  if (yardLine == null || !possAbbr || !otherAbbr) return null;
  return yardLine <= 50 ? `${possAbbr} ${yardLine}` : `${otherAbbr} ${100 - yardLine}`;
}

// {text, redZone} for the "who has the ball and where" line, or null when ESPN hasn't got a
// down/distance/possession reading for this moment (kickoffs, injury stoppages, etc.).
function _situationText(comp, possAbbr, otherAbbr) {
  const sit = comp.situation;
  if (!sit || sit.down == null || sit.distance == null || !possAbbr) return null;
  const field = _fieldPositionText(sit.yardLine, possAbbr, otherAbbr);
  const dd = `${DOWN_ORDINAL[sit.down] || sit.down + 'th'} & ${sit.distance}`;
  return { text: `${possAbbr} ball · ${dd}${field ? ' at ' + field : ''}`, redZone: !!sit.isRedZone };
}

const App = {
  _ctx: null,
  _timer: null,
  _liveTimer: null,
  // null = viewing the live/current week; a number = browsing a past week's Moneyline
  // Pick'em, ATS, Power Rankings, and Parlays. Historical Power Rankings is never affected
  // (it already covers every season, independent of any single week). Deadlines (header
  // chips) and Historical Power always stay on the LIVE week regardless of this -- browsing
  // old data shouldn't make the "locks at" countdown show a date that already passed.
  _viewWeek: null,

  async init() {
    DB.init();
    Auth.init();
    Tabs.init();
    KeepDetailsOpen.init();
    document.body.addEventListener('click', e => {
      const b = e.target.closest('button[data-act]');
      if (!b) return;
      // every click that writes goes through the owner guard (a cosmetic layer; the database enforces it)
      if (['pick', 'gutpick', 'atspick', 'refresh'].includes(b.dataset.act) && !Auth.guard()) return;
      if (b.dataset.act === 'pick') Pickem.pick(b.dataset.week, b.dataset.game, b.dataset.team, b.dataset.spread);
      if (b.dataset.act === 'gutpick') Pickem.gutpick(b.dataset.week, b.dataset.game, b.dataset.team, b.dataset.spread, b.dataset.home, b.dataset.away);
      if (b.dataset.act === 'atspick') Pickem.atspick(b.dataset.week, b.dataset.game, b.dataset.team, b.dataset.spread);
      if (b.dataset.act === 'refresh') App.requestRefresh();
      if (b.dataset.act === 'power-sort') Power.sortBy(b.dataset.col);
      if (b.dataset.act === 'historical-sort') Historical.sortBy(b.dataset.col);
      if (b.dataset.act === 'model-trends-toggle-agree') ModelTrends.toggleAgreement();
      if (b.dataset.act === 'model-trends-toggle-colorby') ModelTrends.toggleColorBy();
      if (b.dataset.act === 'model-trends-reset-filters') ModelTrends.resetAllFilters();
      if (b.dataset.act === 'model-trends-clear-cell') ModelTrends.clearCellFilter();
    });
    document.body.addEventListener('change', e => {
      const t = e.target.closest('input[data-act="tb-input"]');
      if (!t || !Auth.guard()) return;
      Pickem.setTiebreak(t.dataset.week, t.value);
    });
    document.getElementById('week-select').addEventListener('change', e => {
      const v = e.target.value;
      App._viewWeek = v === '' ? null : parseInt(v, 10);
      App.reload();
    });
    await this.reload();
    // Track the cron without a manual reload.
    // Skip the poll while the tab is in the background (a forgotten open tab was a steady source of Supabase egress), and catch
    // up as soon as it is visible again.
    this._timer = setInterval(() => { if (!document.hidden) this.reload(true); }, 90_000);
    document.addEventListener('visibilitychange', () => { if (!document.hidden) this.reload(true); });
  },

  // Rebuilt every reload (liveWeek can itself advance, e.g. Tue/Wed's rollover) but never
  // yanks the user back to "current" out from under them -- only touches the option list and
  // re-applies whatever's currently selected.
  _renderWeekSelect(liveWeek, viewWeek) {
    const el = document.getElementById('week-select');
    const opts = [`<option value="">Current (Week ${liveWeek})</option>`];
    for (let w = liveWeek - 1; w >= 1; w--) opts.push(`<option value="${w}">Week ${w}</option>`);
    el.innerHTML = opts.join('');
    el.value = (viewWeek === liveWeek) ? '' : String(viewWeek);
  },

  async reload(quiet = false) {
    if (!quiet) document.getElementById('status').textContent = 'Loading…';
    try {
      await DB.loadKvMeta();
      const [weekRaw, seasonRaw] = await Promise.all([DB.kv('week'), DB.kv('season')]);
      const liveWeek = parseInt(weekRaw || '1', 10);
      const season = parseInt(seasonRaw || '2025', 10);
      // A stale selection from before a week rollover (or before the season's first week)
      // just falls back to live -- nothing to show for a week that's not live yet.
      if (this._viewWeek != null && (this._viewWeek >= liveWeek || this._viewWeek < 1)) this._viewWeek = null;
      const viewWeek = this._viewWeek ?? liveWeek;
      const isPast = viewWeek !== liveWeek;

      const [games, odds, bestPrice, bookOdds, elway, yRoster, eRoster, pPicks, aPicks, refreshReq, lastRefresh, hist, bSnap,
             edgeLog, edgeBias, edgeLeaderboard, injuries, spreadHist, elwayQb1,
             teamRatingsLive, qbDepthChart, parlaySnapshotLive, historicalPower, weeklyPowerRankings, liveGames, elwayScoreHistory,
             pipelineHealth, gutPicks, prospectiveLog, poolRec, poolSim, poolTb] =
        await Promise.all([
          DB.weekGames(viewWeek), DB.weekOdds(viewWeek), DB.bestPriceOdds(viewWeek), DB.bookOdds(viewWeek), DB.elwayOdds(viewWeek),
          DB.latestRoster('yahoo'), DB.latestRoster('espn'),
          DB.pickemPicks(viewWeek), DB.atsPicks(viewWeek),
          DB.refreshRequest(), DB.kv('last_refresh'), DB.hist(), DB.budgetSnapshot(viewWeek),
          Promise.resolve({}) /* old FADE/CHALK log: no longer shown, no longer fetched */, DB.edgeBiasAll(), DB.edgeLeaderboard(),
          DB.injuries(), DB.spreadHistory(viewWeek), DB.elwayQb1(),
          DB.teamRatings(), DB.qbDepthChart(), DB.parlayRatingsSnapshot(), DB.historicalPower(),
          DB.weeklyPowerRankings(), isPast ? DB.weekGames(liveWeek) : Promise.resolve(null), DB.elwayScoreHistory(),
          DB.pipelineHealth(), DB.gutPicks(), DB.prospectiveLog(), DB.poolRecommendation(), DB.poolSim(), DB.poolTiebreaks(),
        ]);

      // Power Rankings/Parlays: a browsed PAST week reads its own permanently-frozen
      // snapshot (team_ratings.py's refresh_weekly_power_rankings()) instead of the live,
      // still-moving ratings -- same shape either way so Power.render()/Parlays.render()
      // don't need to know which they got.
      let teamRatings = teamRatingsLive, parlaySnapshot = parlaySnapshotLive;
      if (isPast) {
        const snap = weeklyPowerRankings?.[String(season)]?.[String(viewWeek)];
        teamRatings = snap
          ? { generated: snap.generated, season, week: viewWeek, teams: snap.teams, matchups: snap.matchups || {},
              power_rankings: snap.power_rankings, team_records: snap.team_records }
          : null;
        parlaySnapshot = snap ? { season, week: viewWeek, teams: snap.parlay_teams } : null;
      } else if (teamRatings) {
        // Within the CURRENT (still-live) week, a game that's already finished drops out of
        // the live matchups blob entirely -- team_ratings.py's matchup_callouts() only ever
        // computes "upcoming" (no-result-yet) games, by design (recomputing a finished
        // game's own prediction today would use post-game ratings, a hindsight leak). The
        // SAME game's pre-game prediction already exists, correctly frozen, in this week's
        // own power_rankings_by_week snapshot the moment its first kickoff passes (self-
        // healing, same mechanism that backfills past weeks) -- fall back to that for
        // whichever keys the live blob is missing, so Details doesn't just go blank the
        // moment a game ends mid-week. Live still wins for every key it actually has.
        const liveWeekSnap = weeklyPowerRankings?.[String(season)]?.[String(liveWeek)];
        if (liveWeekSnap && liveWeekSnap.matchups) {
          teamRatings = { ...teamRatings, matchups: { ...liveWeekSnap.matchups, ...teamRatings.matchups } };
        }
      }

      const ctx = { week: viewWeek, liveWeek, isPastWeek: isPast, season, games, odds, bestPrice, bookOdds, elway, yRoster, eRoster, pPicks, aPicks,
                    refreshReq, lastRefresh, hist, bSnap,
                    edgeLog, edgeBias, edgeLeaderboard,
                    injuries, spreadHist, elwayQb1, teamRatings, qbDepthChart, parlaySnapshot, historicalPower, elwayScoreHistory,
                    pipelineHealth, gutPicks, prospectiveLog, poolRec, poolSim, poolTb };
      this._ctx = ctx;
      this._renderWeekSelect(liveWeek, viewWeek);
      // Deadlines always reflects the REAL current week/games, never the browsed one -- a
      // "locks at" countdown for a week that already happened would just read as overdue.
      Deadlines.render(isPast ? { ...ctx, week: liveWeek, games: liveGames || [] } : ctx);
      Pickem.render(ctx, 'ml');
      History.renderBins(ctx, 'ml');
      History.render(ctx, 'ml');
      Pickem.render(ctx, 'ats');
      History.renderBins(ctx, 'ats');
      History.render(ctx, 'ats');
      Edge.render(ctx);
      Odds.render(ctx);
      Power.render(ctx);
      Parlays.render(ctx);
      Historical.render(ctx);
      ModelTrends.render(ctx);
      Methodology.render(ctx);
      PipelineHealth.render(ctx);
      this.renderStatus(ctx);
      this._syncLivePolling();
    } catch (e) {
      document.getElementById('status').textContent = 'Error: ' + e.message;
    }
  },

  renderStatus(ctx) {
    const parts = [`Week ${ctx.week}`];
    if (ctx.lastRefresh) parts.push(`data as of ${fmtLocal(ctx.lastRefresh)}`);
    const r = ctx.refreshReq;
    if (r && r.requested_at && (!r.handled_at || r.handled_at < r.requested_at)) {
      parts.push('refresh queued — running within ~10 min');
    }
    document.getElementById('status').textContent = parts.join(' · ');
  },

  // Starts/stops the fast ESPN poll based on whether anything in the CURRENTLY RENDERED
  // week is live right now -- called after every normal reload (both the 90s timer and a
  // manual one), so it turns itself on the moment a game kicks off and back off once the
  // whole slate's final, without needing its own separate polling loop to watch for that.
  _syncLivePolling() {
    const hasLiveGame = (this._ctx?.games || []).some(g => g.state === 'in');
    if (hasLiveGame && !this._liveTimer) {
      this._liveTimer = setInterval(() => this._pollLiveScores(), LIVE_POLL_INTERVAL_MS);
      this._pollLiveScores();  // don't make the first live update wait a full interval
    } else if (!hasLiveGame && this._liveTimer) {
      clearInterval(this._liveTimer);
      this._liveTimer = null;
    }
  },

  // Fetches ESPN's scoreboard directly (bypassing Supabase/the refresh cron entirely) and
  // patches state/home_score/away_score straight onto the SAME game objects `ctx.games`
  // already holds -- mutated in place, not replaced, so this stays in sync with whatever
  // Pickem.render() reads without rebuilding `ctx` itself. Re-renders only the two card
  // grids (not the full reload() pipeline -- Odds/Power/Historical/etc. don't need to
  // refresh every 20s just because a score did), and re-checks afterward whether every
  // game in the slate has gone final, so the fast timer turns itself back off on its own.
  async _pollLiveScores() {
    if (!this._ctx || !this._ctx.games.length) return;
    try {
      const url = `${ESPN_SCOREBOARD_URL}?seasontype=2&week=${this._ctx.week}&dates=${this._ctx.season}`;
      const resp = await fetch(url);
      if (!resp.ok) return;
      const data = await resp.json();
      const live = new Map();
      for (const ev of data.events || []) {
        const comp = (ev.competitions || [])[0];
        if (!comp) continue;
        const state = comp.status?.type?.state || 'pre';
        const home = (comp.competitors || []).find(c => c.homeAway === 'home');
        const away = (comp.competitors || []).find(c => c.homeAway === 'away');
        const homeAbbr = home?.team?.abbreviation, awayAbbr = away?.team?.abbreviation;
        const possId = comp.situation?.possession;
        const possAbbr = possId === home?.team?.id ? homeAbbr : possId === away?.team?.id ? awayAbbr : null;
        const otherAbbr = possAbbr === homeAbbr ? awayAbbr : homeAbbr;
        live.set(String(ev.id), {
          state, home_score: Number(home?.score || 0), away_score: Number(away?.score || 0),
          // "10:00 - OT" / "9:21 - 1st" while live, "Final" once over -- only kept for those
          // two states, not 'pre' (shortDetail there is just the scheduled kickoff time,
          // already shown elsewhere on the card).
          clock: (state === 'in' || state === 'post') ? (comp.status?.type?.shortDetail || null) : null,
          situation: state === 'in' ? _situationText(comp, possAbbr, otherAbbr) : null,
        });
      }
      let changed = false;
      for (const g of this._ctx.games) {
        const upd = live.get(g.game_id);
        if (!upd) continue;
        const sitText = upd.situation ? upd.situation.text : null;
        const sitRedZone = upd.situation ? upd.situation.redZone : false;
        if (g.state !== upd.state || g.home_score !== upd.home_score || g.away_score !== upd.away_score
            || g.live_clock !== upd.clock || g.live_situation !== sitText || g.live_red_zone !== sitRedZone) {
          g.state = upd.state; g.home_score = upd.home_score; g.away_score = upd.away_score;
          g.live_clock = upd.clock; g.live_situation = sitText; g.live_red_zone = sitRedZone;
          changed = true;
        }
      }
      if (changed) {
        Pickem.render(this._ctx, 'ml');
        Pickem.render(this._ctx, 'ats');
      }
    } catch (e) {
      // Best-effort, client-side, every 20s -- a dropped fetch just tries again next tick.
    }
    this._syncLivePolling();
  },

  async requestRefresh() {
    try {
      await DB.requestRefresh();
      document.getElementById('status').textContent =
        'Refresh queued — the updater runs within ~10 min. This page auto-updates.';
    } catch (e) {
      Toast.show('Could not queue refresh: ' + e.message, 'bad');
    }
  },
};

window.addEventListener('DOMContentLoaded', () => App.init());
