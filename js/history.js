'use strict';

// Joins each game's spread to the nflverse historical distribution (kv.hist_distribution,
// built by nflhub/sources/history.py) to estimate how many upsets to expect this week.
const History = {
  weekBucket(w) { return Number(w) === 1 ? 'week1' : 'rest'; },

  bucketLabel(absSpread) {
    const s = Math.abs(absSpread);
    if (s <= 2.5) return '≤2.5';
    if (s === 3) return '3';
    if (s <= 6) return '3.5–6';
    if (s <= 9.5) return '6.5–9.5';
    return '10+';
  },

  // {su, ats, n} for a game, most specific cell available (week x side x bucket -> week x all).
  lookup(dist, absSpread, homeFav, week) {
    if (!dist || !dist.cells) return null;
    const wk = dist.cells[this.weekBucket(week)];
    if (!wk) return null;
    const lab = this.bucketLabel(absSpread);
    const side = homeFav ? 'home' : 'away';
    return (wk[side] && wk[side][lab]) || (wk.all && wk.all[lab]) || null;
  },

  // Poisson-binomial mean/sd of favorite SU wins and ATS covers over this week's games.
  expected(games, odds, dist, week) {
    let suM = 0, suV = 0, atsM = 0, atsV = 0, k = 0;
    for (const g of games) {
      const o = odds[g.game_id];
      if (!o || o.spread == null) continue;
      const cell = this.lookup(dist, Math.abs(o.spread), o.spread <= 0, week);
      if (!cell || cell.su == null) continue;
      suM += cell.su; suV += cell.su * (1 - cell.su);
      if (cell.ats != null) { atsM += cell.ats; atsV += cell.ats * (1 - cell.ats); }
      k++;
    }
    return { k, suMean: suM, suSd: Math.sqrt(suV), atsMean: atsM, atsSd: Math.sqrt(atsV) };
  },

  // One-line lean for the pick'em summary strip.
  summaryLine(ctx) {
    const d = ctx.hist;
    if (!d) return '';
    const wk = this.weekBucket(ctx.week);
    const s = d.summary[wk];
    const e = this.expected(ctx.games, ctx.odds, d, ctx.week);
    if (!e.k || !s) return '';
    const upsets = Math.round(e.k - e.suMean);
    const covLean = Math.round(e.k - e.atsMean); // dogs expected to cover
    const homeNote = (s.home_ats != null && s.home_ats < 0.485) ? ', tilt to home dogs' : '';
    return `History (${d.seasons}, ${wk === 'week1' ? 'Week 1' : 'Weeks 2+'}): ` +
      `favorites expected to win ${e.suMean.toFixed(1)} of ${e.k} straight up → ~${upsets} moneyline upsets (±${e.suSd.toFixed(1)}). ` +
      `Favorites cover ${Math.round(s.ats * 100)}% ATS historically → ~${covLean} dogs cover${homeNote}.`;
  },

  render(ctx) {
    const host = document.getElementById('history');
    if (!host) return;
    const d = ctx.hist;
    if (!d) { host.innerHTML = '<div class="panel"><h2>History</h2><p class="muted">Not built yet — appears after the next daily refresh.</p></div>'; return; }

    const s1 = d.summary.week1, sr = d.summary.rest;
    const row = (lab, s) => `<tr><td>${lab}</td>
      <td class="num">${Math.round(s.su * 100)}%</td>
      <td class="num">${Math.round(s.ats * 100)}%</td>
      <td class="num">${s.home_ats != null ? Math.round(s.home_ats * 100) + '%' : '—'}</td>
      <td class="num muted">${s.avg_miss ?? '—'}</td>
      <td class="num muted">${s.n}</td></tr>`;

    const wkRows = d.byweek.map(w => `<tr>
      <td>Wk ${w.week}</td>
      <td class="num">${w.su != null ? Math.round(w.su * 100) + '%' : '—'}</td>
      <td class="num">${w.ats != null ? Math.round(w.ats * 100) + '%' : '—'}</td>
      <td class="num muted">${w.avg_miss ?? '—'}</td>
      <td class="num muted">${w.n}</td></tr>`).join('');

    // per spread-bucket cells for the current week bucket
    const wkKey = this.weekBucket(ctx.week);
    const cellAll = (d.cells[wkKey] && d.cells[wkKey].all) || {};
    const bktRows = d.buckets.map(b => {
      const c = cellAll[b] || {};
      return `<tr><td>${b}</td>
        <td class="num">${c.su != null ? Math.round(c.su * 100) + '%' : '—'}</td>
        <td class="num">${c.ats != null ? Math.round(c.ats * 100) + '%' : '—'}</td>
        <td class="num muted">${c.n ?? '—'}</td></tr>`;
    }).join('');

    host.innerHTML = `
      <div class="panel">
        <h2>History &mdash; favorite vs. the number (${d.seasons})</h2>
        <p class="muted">Straight-up win and against-the-spread cover rates for closing favorites.
          Shrunk toward each bucket's all-weeks rate (k=${d.shrink_k}); n is the raw sample.</p>
        <div class="grid2">
          <div>
            <h3>Week 1 vs. the rest</h3>
            <table><thead><tr><th>Split</th><th class="num">Fav SU</th><th class="num">Fav ATS</th>
              <th class="num">Home-fav ATS</th><th class="num">Avg miss</th><th class="num">n</th></tr></thead>
              <tbody>${row('Week 1', s1)}${row('Weeks 2+', sr)}</tbody></table>
            <h3 style="margin-top:14px;">By spread size &mdash; ${wkKey === 'week1' ? 'Week 1' : 'Weeks 2+'}</h3>
            <table><thead><tr><th>Spread</th><th class="num">Fav SU</th><th class="num">Fav ATS</th><th class="num">n</th></tr></thead>
              <tbody>${bktRows}</tbody></table>
          </div>
          <div>
            <h3>Week over week</h3>
            <table><thead><tr><th>Week</th><th class="num">Fav SU</th><th class="num">Fav ATS</th>
              <th class="num">Avg miss</th><th class="num">n</th></tr></thead>
              <tbody>${wkRows}</tbody></table>
          </div>
        </div>
      </div>`;
  },
};
