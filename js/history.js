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

  // "Upset budget": this week's games grouped by spread bin, with how many to take as upsets.
  renderBins(ctx) {
    const host = document.getElementById('upsets');
    if (!host) return;
    const d = ctx.hist;
    if (!d) { host.innerHTML = ''; return; }

    const groups = {};
    d.buckets.forEach(b => { groups[b] = []; });
    for (const g of ctx.games) {
      const o = ctx.odds[g.game_id];
      if (!o || o.spread == null) continue;
      const homeFav = o.spread <= 0;
      const cell = this.lookup(d, Math.abs(o.spread), homeFav, ctx.week);
      if (!cell || cell.su == null) continue;
      groups[this.bucketLabel(Math.abs(o.spread))].push({
        fav: homeFav ? g.home : g.away,
        dog: homeFav ? g.away : g.home,
        dogHome: !homeFav,          // home team is the underdog
        su: cell.su,
        ats: cell.ats,
        picked: (ctx.pPicks[g.game_id] || {}).pick,
      });
    }

    let totN = 0, totMlUp = 0, totAtsUp = 0, totPicked = 0, totYourUp = 0;
    const binStats = [];
    const rows = d.buckets.map(lab => {
      const gs = groups[lab].slice().sort((a, b) => a.su - b.su); // weakest favorites first
      const n = gs.length;
      const mlUp = gs.reduce((s, x) => s + (1 - x.su), 0);
      const atsUp = gs.reduce((s, x) => s + (x.ats != null ? 1 - x.ats : 0), 0);
      const take = Math.round(mlUp);
      const pickedN = gs.filter(x => x.picked).length;
      const yourUp = gs.filter(x => x.picked && x.picked === x.dog).length;
      totN += n; totMlUp += mlUp; totAtsUp += atsUp;
      totPicked += pickedN; totYourUp += yourUp;
      binStats.push({ lab, n, take, pickedN, yourUp });
      const list = gs.map((x, i) => {
        const t = i < take;
        const dogHit = x.picked && x.picked === x.dog;
        return `<span class="dogpick${t ? ' take' : ''}${dogHit ? ' on' : ''}" ` +
          `title="${x.fav} favored — wins SU ${Math.round(x.su * 100)}% historically">` +
          `${x.dog} ${x.dogHome ? 'H' : 'A'}${t ? ' ✓' : ''}</span>`;
      }).join(' ');
      return `<tr>
        <td>${lab}</td>
        <td class="num">${n || '—'}</td>
        <td class="num">${n ? Math.round(mlUp / n * 100) + '%' : '—'}</td>
        <td class="num"><strong>${n ? take : 0}</strong></td>
        <td>${list || '<span class="muted">—</span>'}</td>
      </tr>`;
    }).join('');

    const totRate = totN ? Math.round(totMlUp / totN * 100) : 0;

    // "your picks vs. the budget" — only once picks exist this week
    let cmp = '';
    if (totPicked > 0) {
      const budget = Math.round(totMlUp);
      const crows = binStats.filter(b => b.n).map(b => {
        const dlt = b.yourUp - b.take;
        const tag = dlt === 0 ? '<span class="chip good">on budget</span>'
          : dlt > 0 ? `<span class="chip warn">+${dlt}</span>`
          : `<span class="chip">${dlt}</span>`;
        return `<tr><td>${b.lab}</td><td class="num">${b.pickedN}/${b.n}</td>
          <td class="num">${b.yourUp}</td><td class="num">${b.take}</td><td>${tag}</td></tr>`;
      }).join('');
      const dlt = totYourUp - budget;
      const verdict = dlt === 0 ? 'matches the budget'
        : dlt > 0 ? `${dlt} more upset${dlt > 1 ? 's' : ''} than the budget — aggressive`
        : `${-dlt} fewer upset${-dlt > 1 ? 's' : ''} than the budget — chalk-heavy`;
      cmp = `
        <h3 style="margin-top:16px;">Your picks vs. the budget</h3>
        <p class="muted">${totPicked} of ${totN} games picked &middot;
          ${totYourUp} upset pick${totYourUp === 1 ? '' : 's'} &middot; budget ~${budget}
          &rarr; <strong>${verdict}</strong>.</p>
        <table><thead><tr><th>Spread bin</th><th class="num">Picked</th>
          <th class="num">Your upsets</th><th class="num">Budget</th><th>vs budget</th></tr></thead>
          <tbody>${crows}</tbody></table>`;
    }

    host.innerHTML = `
      <div class="panel">
        <h2>Upset budget by spread bin &mdash; Week ${ctx.week}</h2>
        <p class="muted">This week's games grouped by the favorite's spread. <strong>Take as upsets</strong>
          is the bin's historical moneyline upset rate applied to this week's games, rounded.
          The ✓ marks the least-safe favorites in each bin — spend the upset picks there.
          A picked dog is highlighted.</p>
        <table><thead><tr><th>Spread bin</th><th class="num">Games</th>
          <th class="num">Upset rate</th><th class="num">Take as upsets</th><th>Fade the favorite in</th></tr></thead>
          <tbody>${rows}
            <tr class="tot"><td>Total</td><td class="num">${totN}</td>
              <td class="num">${totRate}%</td><td class="num"><strong>${Math.round(totMlUp)}</strong></td><td></td></tr>
          </tbody></table>
        <p class="tablefoot muted">Spread pool: the same bins historically send about
          <strong>${Math.round(totAtsUp)}</strong> of ${totN} favorites to <em>not</em> cover —
          roughly half, tilted to the small-spread bins and home dogs.</p>
        ${cmp}
      </div>`;
  },

  render(ctx) {
    const host = document.getElementById('history');
    if (!host) return;
    const d = ctx.hist;
    if (!d) { host.innerHTML = '<div class="panel"><h2>History</h2><p class="muted">Not built yet — appears after the next daily refresh.</p></div>'; return; }

    const s1 = d.summary.week1, sr = d.summary.rest;
    const up = v => v == null ? '—' : Math.round((1 - v) * 100) + '%'; // 1 - fav SU = dog upset rate

    // "home dog" = home team is the underdog => the AWAY team is favored => away_su cell.
    const row = (lab, s) => `<tr><td>${lab}</td>
      <td class="num">${up(s.su)}</td>
      <td class="num" title="home team as underdog, n=${s.away_n ?? '—'}">${up(s.away_su)}</td>
      <td class="num" title="away team as underdog, n=${s.home_n ?? '—'}">${up(s.home_su)}</td>
      <td class="num muted">${s.ats != null ? Math.round(s.ats * 100) + '%' : '—'}</td>
      <td class="num muted">${s.n}</td></tr>`;

    const wkRows = d.byweek.map(w => `<tr>
      <td>Wk ${w.week}</td>
      <td class="num">${up(w.su)}</td>
      <td class="num">${up(w.away_su)}</td>
      <td class="num">${up(w.home_su)}</td>
      <td class="num muted">${w.n}</td></tr>`).join('');

    const wkKey = this.weekBucket(ctx.week);
    const cells = d.cells[wkKey] || {};
    const bktRows = d.buckets.map(b => {
      const any = (cells.all || {})[b] || {};
      const hd = (cells.away || {})[b] || {};   // away favored -> home dog
      const ad = (cells.home || {})[b] || {};   // home favored -> away dog
      return `<tr><td>${b}</td>
        <td class="num">${up(any.su)}</td>
        <td class="num" title="n=${hd.n ?? '—'}">${up(hd.su)}</td>
        <td class="num" title="n=${ad.n ?? '—'}">${up(ad.su)}</td>
        <td class="num muted">${any.n ?? '—'}</td></tr>`;
    }).join('');

    const wkLabel = wkKey === 'week1' ? 'Week 1' : 'Weeks 2+';
    host.innerHTML = `
      <div class="panel">
        <h2>History &mdash; dog upset rates (${d.seasons})</h2>
        <p class="muted">How often the underdog wins outright, split by whether the dog is at
          home or on the road. Shrunk toward each bucket's all-weeks rate (k=${d.shrink_k});
          n is the raw sample. (Favorite ATS cover % shown for reference.)</p>
        <div class="grid2">
          <div>
            <h3>Week 1 vs. the rest</h3>
            <table><thead><tr><th>Split</th><th class="num">Any dog</th><th class="num">Home dog</th>
              <th class="num">Away dog</th><th class="num">Fav ATS</th><th class="num">n</th></tr></thead>
              <tbody>${row('Week 1', s1)}${row('Weeks 2+', sr)}</tbody></table>
            <h3 style="margin-top:14px;">By spread size &mdash; ${wkLabel}</h3>
            <table><thead><tr><th>Spread</th><th class="num">Any dog</th><th class="num">Home dog</th>
              <th class="num">Away dog</th><th class="num">n</th></tr></thead>
              <tbody>${bktRows}</tbody></table>
          </div>
          <div>
            <h3>Week over week</h3>
            <table><thead><tr><th>Week</th><th class="num">Any dog</th><th class="num">Home dog</th>
              <th class="num">Away dog</th><th class="num">n</th></tr></thead>
              <tbody>${wkRows}</tbody></table>
          </div>
        </div>
      </div>`;
  },
};
