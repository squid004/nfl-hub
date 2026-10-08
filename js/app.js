'use strict';

const App = {
  _ctx: null,
  _timer: null,
  // null = viewing the live/current week; a number = browsing a past week's Moneyline
  // Pick'em, ATS, Power Rankings, and Parlays. Historical Power Rankings is never affected
  // (it already covers every season, independent of any single week). Deadlines (header
  // chips) and Historical Power always stay on the LIVE week regardless of this -- browsing
  // old data shouldn't make the "locks at" countdown show a date that already passed.
  _viewWeek: null,

  async init() {
    DB.init();
    Tabs.init();
    document.body.addEventListener('click', e => {
      const b = e.target.closest('button[data-act]');
      if (!b) return;
      if (b.dataset.act === 'pick') Pickem.pick(b.dataset.week, b.dataset.game, b.dataset.team, b.dataset.spread);
      if (b.dataset.act === 'atspick') Pickem.atspick(b.dataset.week, b.dataset.game, b.dataset.team, b.dataset.spread);
      if (b.dataset.act === 'refresh') App.requestRefresh();
      if (b.dataset.act === 'power-sort') Power.sortBy(b.dataset.col);
      if (b.dataset.act === 'historical-sort') Historical.sortBy(b.dataset.col);
      if (b.dataset.act === 'model-trends-toggle-agree') ModelTrends.toggleAgreement();
      if (b.dataset.act === 'model-trends-toggle-colorby') ModelTrends.toggleColorBy();
      if (b.dataset.act === 'model-trends-reset-filters') ModelTrends.resetAllFilters();
      if (b.dataset.act === 'model-trends-clear-cell') ModelTrends.clearCellFilter();
    });
    document.getElementById('week-select').addEventListener('change', e => {
      const v = e.target.value;
      App._viewWeek = v === '' ? null : parseInt(v, 10);
      App.reload();
    });
    await this.reload();
    // Track the cron without a manual reload.
    this._timer = setInterval(() => this.reload(true), 90_000);
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
             teamRatingsLive, qbDepthChart, parlaySnapshotLive, historicalPower, weeklyPowerRankings, liveGames] =
        await Promise.all([
          DB.weekGames(viewWeek), DB.weekOdds(viewWeek), DB.bestPriceOdds(viewWeek), DB.bookOdds(viewWeek), DB.elwayOdds(viewWeek),
          DB.latestRoster('yahoo'), DB.latestRoster('espn'),
          DB.pickemPicks(viewWeek), DB.atsPicks(viewWeek),
          DB.refreshRequest(), DB.kv('last_refresh'), DB.hist(), DB.budgetSnapshot(viewWeek),
          DB.edgeRecommendationLog(season, viewWeek), DB.edgeBiasAll(), DB.edgeLeaderboard(),
          DB.injuries(), DB.spreadHistory(viewWeek), DB.elwayQb1(),
          DB.teamRatings(), DB.qbDepthChart(), DB.parlayRatingsSnapshot(), DB.historicalPower(),
          DB.weeklyPowerRankings(), isPast ? DB.weekGames(liveWeek) : Promise.resolve(null),
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
      }

      const ctx = { week: viewWeek, liveWeek, isPastWeek: isPast, season, games, odds, bestPrice, bookOdds, elway, yRoster, eRoster, pPicks, aPicks,
                    refreshReq, lastRefresh, hist, bSnap,
                    edgeLog, edgeBias, edgeLeaderboard,
                    injuries, spreadHist, elwayQb1, teamRatings, qbDepthChart, parlaySnapshot, historicalPower };
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
      this.renderStatus(ctx);
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

  async requestRefresh() {
    try {
      await DB.requestRefresh();
      document.getElementById('status').textContent =
        'Refresh queued — the updater runs within ~10 min. This page auto-updates.';
    } catch (e) {
      alert('Could not queue refresh: ' + e.message);
    }
  },
};

window.addEventListener('DOMContentLoaded', () => App.init());
