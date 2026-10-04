'use strict';

const App = {
  _ctx: null,
  _timer: null,

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
    });
    await this.reload();
    // Track the cron without a manual reload.
    this._timer = setInterval(() => this.reload(true), 90_000);
  },

  async reload(quiet = false) {
    if (!quiet) document.getElementById('status').textContent = 'Loading…';
    try {
      const [weekRaw, seasonRaw] = await Promise.all([DB.kv('week'), DB.kv('season')]);
      const week = parseInt(weekRaw || '1', 10);
      const season = parseInt(seasonRaw || '2025', 10);
      const [games, odds, bestPrice, bookOdds, elway, yRoster, eRoster, pPicks, aPicks, refreshReq, lastRefresh, hist, bSnap,
             edgeLog, edgeBias, edgeLeaderboard, injuries, spreadHist, elwayQb1,
             teamRatings] =
        await Promise.all([
          DB.weekGames(week), DB.weekOdds(week), DB.bestPriceOdds(week), DB.bookOdds(week), DB.elwayOdds(week),
          DB.latestRoster('yahoo'), DB.latestRoster('espn'),
          DB.pickemPicks(week), DB.atsPicks(week),
          DB.refreshRequest(), DB.kv('last_refresh'), DB.hist(), DB.budgetSnapshot(week),
          DB.edgeRecommendationLog(season, week), DB.edgeBiasAll(), DB.edgeLeaderboard(),
          DB.injuries(), DB.spreadHistory(week), DB.elwayQb1(),
          DB.teamRatings(),
        ]);
      const ctx = { week, season, games, odds, bestPrice, bookOdds, elway, yRoster, eRoster, pPicks, aPicks,
                    refreshReq, lastRefresh, hist, bSnap,
                    edgeLog, edgeBias, edgeLeaderboard,
                    injuries, spreadHist, elwayQb1, teamRatings };
      this._ctx = ctx;
      Deadlines.render(ctx);
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
