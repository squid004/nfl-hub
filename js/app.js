'use strict';

const App = {
  _ctx: null,
  _timer: null,

  async init() {
    DB.init();
    document.body.addEventListener('click', e => {
      const b = e.target.closest('button[data-act]');
      if (!b) return;
      if (b.dataset.act === 'pick') Pickem.pick(b.dataset.week, b.dataset.game, b.dataset.team);
      if (b.dataset.act === 'surv') Survivor.pick(b.dataset.week, b.dataset.team);
      if (b.dataset.act === 'refresh') App.requestRefresh();
    });
    await this.reload();
    // Track the cron without a manual reload.
    this._timer = setInterval(() => this.reload(true), 90_000);
  },

  async reload(quiet = false) {
    if (!quiet) document.getElementById('status').textContent = 'Loading…';
    try {
      const week = parseInt((await DB.kv('week')) || '1', 10);
      const [games, odds, yRoster, eRoster, pPicks, sPicks, news, refreshReq, lastRefresh, hist] =
        await Promise.all([
          DB.weekGames(week), DB.weekOdds(week),
          DB.latestRoster('yahoo'), DB.latestRoster('espn'),
          DB.pickemPicks(week), DB.survivorPicks(),
          DB.news(), DB.refreshRequest(), DB.kv('last_refresh'), DB.hist(),
        ]);
      const ctx = { week, games, odds, yRoster, eRoster, pPicks, sPicks, news, refreshReq, lastRefresh, hist };
      this._ctx = ctx;
      Deadlines.render(ctx);
      Fantasy.render(ctx);
      Pickem.render(ctx);
      History.renderBins(ctx);
      Survivor.render(ctx);
      History.render(ctx);
      Odds.render(ctx);
      News.render(ctx);
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
