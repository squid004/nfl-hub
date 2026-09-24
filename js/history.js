'use strict';

// Two modes, same nflverse distribution (kv.hist_distribution, built by
// nflhub/sources/history.py):
//   'ml'  -> favorite SU win rate      (dog "upset" = dog wins outright)
//   'ats' -> favorite ATS cover rate   (dog "cover" = dog beats the number)
const HIST_MODES = {
  ml: {
    field: 'su', other: 'ats', picksKey: 'pPicks',
    upsetsId: 'upsets', histId: 'history',
    dogWord: 'upset', dogVerb: 'wins outright', favVerb: 'wins SU',
    budgetTitle: 'Upset budget', budgetCol: 'Upset rate', budgetVerb: 'Take as upsets',
    fadeLabel: 'Fade the favorite in', histTitle: 'dog upset rates',
    histBlurb: 'How often the underdog wins outright',
    refCol: 'Fav ATS',
  },
  ats: {
    field: 'ats', other: 'su', picksKey: 'aPicks',
    upsetsId: 'atsupsets', histId: 'atshistory',
    dogWord: 'dog cover', dogVerb: 'covers', favVerb: 'covers',
    budgetTitle: 'Dog-cover budget', budgetCol: 'Dog cover rate', budgetVerb: 'Take the dog side',
    fadeLabel: 'Take the dog in', histTitle: 'dog cover rates',
    histBlurb: 'How often the underdog beats the spread',
    refCol: 'Fav SU',
  },
};

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

  lookup(dist, absSpread, homeFav, week) {
    if (!dist || !dist.cells) return null;
    const wk = dist.cells[this.weekBucket(week)];
    if (!wk) return null;
    const lab = this.bucketLabel(absSpread);
    const side = homeFav ? 'home' : 'away';
    return (wk[side] && wk[side][lab]) || (wk.all && wk.all[lab]) || null;
  },

  // P(favorite wins SU / covers) from the smooth curve at this exact spread, with the
  // key-number correction at 3/7 applied. Mirror of predict() in nflhub/sources/history.py
  // — see that docstring for why a smooth curve plus an explicit key-number bump, instead
  // of either pure discrete buckets or a pure smooth curve alone.
  predict(dist, absSpread, homeFav, week, field) {
    if (!dist || !dist.curves) return null;
    const wkb = this.weekBucket(week);
    const side = homeFav ? 'home' : 'away';
    const curves = dist.curves[wkb] || {};
    const coef = (curves[side] && curves[side][field]) || (curves.all && curves.all[field]);
    if (!coef) return null;
    const [a, b] = coef;
    let p = 1 / (1 + Math.exp(-(a + b * absSpread)));
    for (const kn of [3, 7]) {
      if (Math.abs(absSpread - kn) < 0.01) {
        const adj = (((dist.key_adjustments || {})[wkb] || {})[side] || {})[field]?.[String(kn)];
        if (adj != null) p += adj;
        break;
      }
    }
    return Math.min(0.995, Math.max(0.005, p));
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

  summaryLine(ctx, mode = 'ml') {
    const d = ctx.hist;
    if (!d) return '';
    const wk = this.weekBucket(ctx.week);
    const s = d.summary[wk];
    const e = this.expected(ctx.games, ctx.odds, d, ctx.week);
    if (!e.k || !s) return '';
    const label = wk === 'week1' ? 'Week 1' : 'Weeks 2+';
    if (mode === 'ml') {
      const upsets = Math.round(e.k - e.suMean);
      const covLean = Math.round(e.k - e.atsMean);
      return `History (${d.seasons}, ${label}): favorites expected to win ${e.suMean.toFixed(1)} of ` +
        `${e.k} straight up → ~${upsets} moneyline upsets (±${e.suSd.toFixed(1)}). ` +
        `Favorites cover ${Math.round(s.ats * 100)}% ATS → ~${covLean} dogs cover.`;
    }
    const covers = Math.round(e.k - e.atsMean);
    let tilt = '';
    if (s.home_ats != null && s.away_ats != null) {
      tilt = s.home_ats <= s.away_ats
        ? ' Home favorites cover least → lean road dogs.'
        : ' Road favorites cover least → lean home dogs.';
    }
    return `History (${d.seasons}, ${label}): favorites expected to cover ${e.atsMean.toFixed(1)} of ` +
      `${e.k} → ~${covers} dog covers (±${e.atsSd.toFixed(1)}).${tilt}`;
  },

  // Shared by renderBins (the Upset Budget table) and Pickem's per-game "budget dog"
  // chip, so both always agree on which games are flagged. Bucket ASSIGNMENT (which row of
  // the table a game groups under) is discrete, for display only. Both the "how many to
  // take" count and which SPECIFIC game gets flagged come from the smooth curve
  // (this.predict(), see its comment) instead: "how many" sums 1-predict() at the game's
  // real market spread (preferring the cross-book average, ctx.bestPrice, over the
  // single-book line when available); "which one" ranks games within a bucket by
  // predict() evaluated at the market spread nudged by how much ELWAY's avg-points model
  // disagrees in the dog's favor (1 ELWAY point == 1 spread point — an even trade with no
  // evidence yet to weight it otherwise). Server-side twin: nflhub/sources/history.py
  // week_budget.
  computeBudget(ctx, mode = 'ml') {
    const M = HIST_MODES[mode];
    const d = ctx.hist;
    if (!d) return null;
    const F = M.field, O = M.other;
    const picks = ctx[M.picksKey] || {};
    const bestPrice = ctx.bestPrice || {};
    const elway = ctx.elway || {};

    const groups = {};
    d.buckets.forEach(b => { groups[b] = []; });
    for (const g of ctx.games) {
      const o = ctx.odds[g.game_id];
      if (!o || o.spread == null) continue;
      const bp = bestPrice[g.game_id];
      const sp = (bp && bp.avg_spread_home != null) ? bp.avg_spread_home : o.spread;
      const homeFav = sp <= 0;
      const v = this.predict(d, Math.abs(sp), homeFav, ctx.week, F);
      if (v == null) continue;

      let dogEdge = 0;
      const el = elway[g.game_id];
      if (el && el.spread_home != null) {
        const dogIsHome = !homeFav;
        const marketDogMargin = -Math.abs(sp);
        const elwayDogMargin = dogIsHome ? -el.spread_home : el.spread_home;
        dogEdge = elwayDogMargin - marketDogMargin;
      }
      const rankV = this.predict(d, Math.max(0, Math.abs(sp) - dogEdge), homeFav, ctx.week, F) ?? v;

      // Display bucket only: BUCKETS/bucketLabel assumes clean half-point spreads (true
      // for a single book's line, not necessarily for a cross-book average, e.g. 2.75).
      const dispSpread = Math.round(Math.abs(sp) * 2) / 2;
      const cell = this.lookup(d, dispSpread, homeFav, ctx.week);

      groups[this.bucketLabel(dispSpread)].push({
        gameId: g.game_id,
        fav: homeFav ? g.home : g.away,
        dog: homeFav ? g.away : g.home,
        dogHome: !homeFav,
        su: cell ? cell.su : null, ats: cell ? cell.ats : null,
        picked: (picks[g.game_id] || {}).pick,
        v, rankV,
      });
    }

    let totN = 0, totPrimary = 0, totOther = 0, totPicked = 0, totYourDog = 0;
    const binStats = [];
    const flaggedIds = new Set();
    const rankByGame = {}; // gameId -> { rank, n, taken } — every game in its bucket, not just flagged
    const bins = d.buckets.map(lab => {
      const gs = groups[lab].slice().sort((a, b) => a.rankV - b.rankV); // most live dog (lowest fav prob) first
      const n = gs.length;
      const primary = gs.reduce((s, x) => s + (1 - x.v), 0);
      const other = gs.reduce((s, x) => s + (x[O] != null ? 1 - x[O] : 0), 0);
      const take = Math.round(primary);
      gs.forEach((x, i) => {
        x.rank = i + 1;            // 1 = most live dog in this bucket, all games ranked
        x.taken = i < take;
        if (x.taken) flaggedIds.add(x.gameId);
        rankByGame[x.gameId] = { rank: x.rank, n, taken: x.taken };
      });
      const pickedN = gs.filter(x => x.picked).length;
      const yourDog = gs.filter(x => x.picked && x.picked === x.dog).length;
      totN += n; totPrimary += primary; totOther += other;
      totPicked += pickedN; totYourDog += yourDog;
      binStats.push({ lab, n, take, pickedN, yourDog });
      return { lab, gs, n, primary, take };
    });

    return { bins, binStats, totN, totPrimary, totOther, totPicked, totYourDog, flaggedIds, rankByGame };
  },

  // "budget": this week's games grouped by spread bin, with how many dog picks to make.
  renderBins(ctx, mode = 'ml') {
    const M = HIST_MODES[mode];
    const host = document.getElementById(M.upsetsId);
    if (!host) return;
    const budget = this.computeBudget(ctx, mode);
    if (!budget) { host.innerHTML = ''; return; }
    const F = M.field;
    const { bins, binStats, totN, totPrimary, totOther, totPicked, totYourDog } = budget;

    const rows = bins.map(({ lab, gs, n, primary, take }) => {
      // Sorted by rankV already (most live first), so rank order = list order.
      const list = gs.map(x => {
        const dogHit = x.picked && x.picked === x.dog;
        return `<span class="dogpick${x.taken ? ' take' : ''}${dogHit ? ' on' : ''}" ` +
          `title="${x.fav} favored — ${M.favVerb} ${Math.round(x.v * 100)}% by this exact spread's smooth curve ` +
          `(bucket average ${x[F] != null ? Math.round(x[F] * 100) + '%' : '—'}); ` +
          `#${x.rank} of ${n} in this bucket by market-spread + ELWAY-adjusted rank">` +
          `#${x.rank} ${x.dog} ${x.dogHome ? 'H' : 'A'}${x.taken ? ' ✓' : ''}</span>`;
      }).join(' ');
      return `<tr>
        <td>${lab}</td>
        <td class="num">${n || '—'}</td>
        <td class="num">${n ? Math.round(primary / n * 100) + '%' : '—'}</td>
        <td class="num"><strong>${n ? take : 0}</strong></td>
        <td>${list || '<span class="muted">—</span>'}</td>
      </tr>`;
    }).join('');

    const totRate = totN ? Math.round(totPrimary / totN * 100) : 0;

    let cmp = '';
    if (totPicked > 0) {
      const budget = Math.round(totPrimary);
      const crows = binStats.filter(b => b.n).map(b => {
        const dlt = b.yourDog - b.take;
        const tag = dlt === 0 ? '<span class="chip good">on budget</span>'
          : dlt > 0 ? `<span class="chip warn">+${dlt}</span>`
          : `<span class="chip">${dlt}</span>`;
        return `<tr><td>${b.lab}</td><td class="num">${b.pickedN}/${b.n}</td>
          <td class="num">${b.yourDog}</td><td class="num">${b.take}</td><td>${tag}</td></tr>`;
      }).join('');
      const dlt = totYourDog - budget;
      const w = M.dogWord;
      const verdict = dlt === 0 ? 'matches the budget'
        : dlt > 0 ? `${dlt} more ${w}${dlt > 1 ? 's' : ''} than the budget — aggressive`
        : `${-dlt} fewer ${w}${-dlt > 1 ? 's' : ''} than the budget — chalk-heavy`;
      cmp = `
        <h3 style="margin-top:16px;">Your picks vs. the budget</h3>
        <p class="muted">${totPicked} of ${totN} games picked &middot;
          ${totYourDog} ${w} pick${totYourDog === 1 ? '' : 's'} &middot; budget ~${budget}
          &rarr; <strong>${verdict}</strong>.</p>
        <table><thead><tr><th>Spread bin</th><th class="num">Picked</th>
          <th class="num">Your ${w}s</th><th class="num">Budget</th><th>vs budget</th></tr></thead>
          <tbody>${crows}</tbody></table>`;
    }

    const footer = mode === 'ml'
      ? `Spread pool: the same bins historically send about <strong>${Math.round(totOther)}</strong>
         of ${totN} favorites to <em>not</em> cover.`
      : `Moneyline: the same bins historically see about <strong>${Math.round(totOther)}</strong>
         of ${totN} favorites lose outright.`;

    const locked = (ctx.bSnap || []).find(r => r.mode === mode && r.bin === 'TOTAL');
    const lockLine = locked
      ? `<p class="muted">Locked for the record: <strong>take ${locked.suggested}</strong>
         (frozen ${fmtLocal(locked.captured_at)}). Live number above may drift as lines move.</p>`
      : '';

    host.innerHTML = `
      <div class="panel">
        <h2>${M.budgetTitle} by spread bin &mdash; Week ${ctx.week}</h2>
        <p class="muted">This week's games grouped by the favorite's spread for display —
          <strong>${M.budgetVerb}</strong> sums each game's own ${M.dogWord} rate from a
          smooth curve fit to spread size (not a shared bucket average), with a small
          correction at key numbers 3/7 where NFL final margins cluster. The ✓ marks the
          games it ranks most live within their bin, in order. Dogs tagged H (home) / A (away).</p>
        ${lockLine}
        <table><thead><tr><th>Spread bin</th><th class="num">Games</th>
          <th class="num">${M.budgetCol}</th><th class="num">${M.budgetVerb}</th><th>${M.fadeLabel}</th></tr></thead>
          <tbody>${rows}
            <tr class="tot"><td>Total</td><td class="num">${totN}</td>
              <td class="num">${totRate}%</td><td class="num"><strong>${Math.round(totPrimary)}</strong></td><td></td></tr>
          </tbody></table>
        <p class="tablefoot muted">${footer}</p>
        ${cmp}
      </div>`;
  },

  render(ctx, mode = 'ml') {
    const M = HIST_MODES[mode];
    const host = document.getElementById(M.histId);
    if (!host) return;
    const d = ctx.hist;
    if (!d) { host.innerHTML = `<div class="panel"><h2>History</h2><p class="muted">Not built yet — appears after the next daily refresh.</p></div>`; return; }
    const F = M.field, O = M.other;

    const s1 = d.summary.week1, sr = d.summary.rest;
    const up = v => v == null ? '—' : Math.round((1 - v) * 100) + '%'; // 1 - fav rate = dog rate
    const refPct = v => v == null ? '—' : Math.round(v * 100) + '%';

    // "home dog" = home team is the underdog => AWAY team favored => away_* cell.
    const row = (lab, s) => `<tr><td>${lab}</td>
      <td class="num">${up(s[F])}</td>
      <td class="num" title="home team as underdog, n=${s.away_n ?? '—'}">${up(s['away_' + F])}</td>
      <td class="num" title="away team as underdog, n=${s.home_n ?? '—'}">${up(s['home_' + F])}</td>
      <td class="num muted">${refPct(s[O])}</td>
      <td class="num muted">${s.n}</td></tr>`;

    const wkRows = d.byweek.map(w => `<tr>
      <td>Wk ${w.week}</td>
      <td class="num">${up(w[F])}</td>
      <td class="num">${up(w['away_' + F])}</td>
      <td class="num">${up(w['home_' + F])}</td>
      <td class="num muted">${w.n}</td></tr>`).join('');

    const wkKey = this.weekBucket(ctx.week);
    const cells = d.cells[wkKey] || {};
    const bktRows = d.buckets.map(b => {
      const any = (cells.all || {})[b] || {};
      const hd = (cells.away || {})[b] || {};   // away favored -> home dog
      const ad = (cells.home || {})[b] || {};   // home favored -> away dog
      return `<tr><td>${b}</td>
        <td class="num">${up(any[F])}</td>
        <td class="num" title="n=${hd.n ?? '—'}">${up(hd[F])}</td>
        <td class="num" title="n=${ad.n ?? '—'}">${up(ad[F])}</td>
        <td class="num muted">${any.n ?? '—'}</td></tr>`;
    }).join('');

    const wkLabel = wkKey === 'week1' ? 'Week 1' : 'Weeks 2+';
    host.innerHTML = `
      <div class="panel">
        <h2>History &mdash; ${M.histTitle} (${d.seasons})</h2>
        <p class="muted">${M.histBlurb}, split by whether the dog is at home or on the road.
          Shrunk toward each bucket's all-weeks rate (k=${d.shrink_k}); n is the raw sample.
          (${M.refCol} shown for reference.)</p>
        <div class="grid2">
          <div>
            <h3>Week 1 vs. the rest</h3>
            <table><thead><tr><th>Split</th><th class="num">Any dog</th><th class="num">Home dog</th>
              <th class="num">Away dog</th><th class="num">${M.refCol}</th><th class="num">n</th></tr></thead>
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
