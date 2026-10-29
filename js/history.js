'use strict';

// Two modes, same nflverse distribution (kv.hist_distribution, built by
// nflhub/sources/history.py):
//   'ml'  -> favorite SU win rate      (dog "upset" = dog wins outright)
//   'ats' -> favorite ATS cover rate   (dog "cover" = dog beats the number)
const HIST_MODES = {
  ml: {
    field: 'su', picksKey: 'pPicks',
    upsetsId: 'upsets', histId: 'history',
    dogWord: 'upset', dogVerb: 'wins outright', favVerb: 'wins SU',
    budgetTitle: 'Upset budget', budgetCol: 'Upset rate', budgetVerb: 'Take as upsets',
    fadeLabel: 'Fade the favorite in', histTitle: 'dog upset rates',
    histBlurb: 'How often the underdog wins outright',
  },
  ats: {
    field: 'ats', picksKey: 'aPicks',
    upsetsId: 'atsupsets', histId: 'atshistory',
    dogWord: 'dog cover', dogVerb: 'covers', favVerb: 'covers',
    budgetTitle: 'Dog-cover budget', budgetCol: 'Dog cover rate', budgetVerb: 'Take the dog side',
    fadeLabel: 'Take the dog in', histTitle: 'dog cover rates',
    histBlurb: 'How often the underdog beats the spread',
  },
};

// Target-setting chart for the bottom of the Upset Budget panel: each week, how many more
// games the pool's weekly winner got right than the market ("always pick the favorite")
// baseline, from nflhub.sources.edge_core.pool_leaderboard_summary (kv 'edge_pool_leaderboard',
// rebuilt once/day). Moneyline only -- "beat the market" is an SU concept here, not an ATS one.
// It's the bar a real pick has to clear, not a reason to pick more dogs for its own sake.
function chalkMarginData(ctx) {
  const lb = ctx.edgeLeaderboard;
  if (!lb || lb.avg_margin_over_market == null) return null;
  const rows = (lb.week_winners || []).filter(w => w.margin_over_market != null);
  return rows.length ? { rows, avg: lb.avg_margin_over_market } : null;
}

function chalkReminderHtml(ctx) {
  const d = chalkMarginData(ctx);
  if (!d) return '';
  const avg = (d.avg > 0 ? '+' : '') + (Math.round(d.avg * 10) / 10);
  return `<div style="margin-top:14px;">
    <div id="chalk-margin-chart" style="height:200px;"></div>
    <p class="muted small" style="margin:4px 0 0;">Correct picks by each week's pool winner over the
      "always take the favorite" baseline. The weekly winner has averaged <strong>${avg}</strong>
      game${Math.abs(d.avg) === 1 ? '' : 's'} per week over the market &mdash; pick your spots.</p>
  </div>`;
}

function drawChalkMarginChart(ctx) {
  const d = chalkMarginData(ctx);
  const el = document.getElementById('chalk-margin-chart');
  if (!d || !el || typeof Plotly === 'undefined') return;
  const cs = getComputedStyle(document.documentElement);
  const cssVar = name => cs.getPropertyValue(name).trim();
  const colors = { text: cssVar('--text'), dim: cssVar('--dim'), line: cssVar('--line'),
    panel: cssVar('--panel2'), good: cssVar('--good'), warn: cssVar('--warn') };
  const weeks = d.rows.map(w => w.week);
  const axisCommon = { gridcolor: colors.line, color: colors.dim, tickfont: { size: 11 } };
  Plotly.react(el, [{
    type: 'bar', x: weeks, y: d.rows.map(w => w.margin_over_market),
    customdata: d.rows.map(w => [w.correct, w.market_correct]),
    marker: { color: colors.good },
    hovertemplate: 'Week %{x}: winner %{customdata[0]} correct vs market %{customdata[1]} ' +
      '(%{y:+d})<extra></extra>',
  }], {
    autosize: true,
    margin: { l: 40, r: 16, t: 8, b: 36 },
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { color: colors.text, size: 12 },
    xaxis: Object.assign({ title: { text: 'NFL week' }, dtick: 1 }, axisCommon),
    yaxis: Object.assign({ title: { text: 'Correct picks over market' }, zeroline: true,
      zerolinecolor: colors.dim, rangemode: 'tozero', dtick: 1 }, axisCommon),
    shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: d.avg, y1: d.avg,
      line: { color: colors.warn, width: 2, dash: 'dash' } }],
    showlegend: false,
    hoverlabel: { bgcolor: colors.panel, bordercolor: colors.line, font: { color: colors.text, size: 12 } },
  }, { responsive: true, displaylogo: false, displayModeBar: false });
}

const History = {
  bucketLabel(absSpread) {
    const s = Math.abs(absSpread);
    if (s <= 2.5) return '≤2.5';
    if (s === 3) return '3';
    if (s <= 6) return '3.5–6';
    if (s <= 9.5) return '6.5–9.5';
    return '10+';
  },

  // One constant historical estimate by spread size, no week-of-season split. A prior version
  // split week 1 from "the rest," but neither that split nor several candidate finale-week
  // splits (tested when investigating week-to-week variation) replicated across independent
  // halves of the 2007-2025 data -- see nflhub/sources/history.py module docstring for the
  // actual numbers. Pooling everything is both simpler and the statistically defensible choice.
  // `neutral` (a neutral-site game, no real home or away side) reads the all-games cell: only 70 neutral REG games since 2007 exist,
  // about 6 per spread bucket, far too few for cells of their own, and the home/away cells would misapply a home edge.
  lookup(dist, absSpread, homeFav, neutral = false) {
    if (!dist || !dist.cells) return null;
    const cells = dist.cells;
    const lab = this.bucketLabel(absSpread);
    if (neutral) return (cells.all && cells.all[lab]) || null;
    const side = homeFav ? 'home' : 'away';
    return (cells[side] && cells[side][lab]) || (cells.all && cells.all[lab]) || null;
  },

  // P(favorite wins SU / covers) from the smooth curve at this exact spread, with the
  // key-number correction at 3/7 applied. Mirror of predict() in nflhub/sources/history.py
  // — see that docstring for why a smooth curve plus an explicit key-number bump, instead
  // of either pure discrete buckets or a pure smooth curve alone.
  predict(dist, absSpread, homeFav, field, neutral = false) {
    if (!dist || !dist.curves) return null;
    const side = neutral ? 'all' : (homeFav ? 'home' : 'away');   // a neutral-site game uses the pooled curve
    const curves = dist.curves;
    const coef = (curves[side] && curves[side][field]) || (curves.all && curves.all[field]);
    if (!coef) return null;
    const [a, b] = coef;
    let p = 1 / (1 + Math.exp(-(a + b * absSpread)));
    for (const kn of [3, 7]) {
      if (Math.abs(absSpread - kn) < 0.01) {
        const adj = ((dist.key_adjustments || {})[side] || {})[field]?.[String(kn)];
        if (adj != null) p += adj;
        break;
      }
    }
    return Math.min(0.995, Math.max(0.005, p));
  },

  // P(the team on homeFavSide, home if true else away, wins/covers) from a SIGNED margin
  // for that side (negative = actually the worse side by that many, evaluated by flipping
  // to the other side's curve). Mirror of _predict_signed in nflhub/sources/history.py —
  // see that docstring for why: a floor-clamped magnitude let an ELWAY/market blend
  // collapse multiple different games to the identical toss-up value and re-tie them.
  predictSigned(dist, homeFavSide, signedMargin, field, neutral = false) {
    if (signedMargin >= 0) return this.predict(dist, signedMargin, homeFavSide, field, neutral);
    const pOther = this.predict(dist, -signedMargin, !homeFavSide, field, neutral);
    return pOther == null ? null : 1 - pOther;
  },

  // Poisson-binomial mean/sd of favorite SU wins and ATS covers over this week's games.
  expected(games, odds, dist, teamRatings = null) {
    let suM = 0, suV = 0, atsM = 0, atsV = 0, k = 0;
    for (const g of games) {
      const o = odds[g.game_id];
      if (!o || o.spread == null) continue;
      // a neutral-site game has no home/away side: read the pooled cell (see lookup())
      const m = typeof matchupFor === 'function' ? matchupFor(teamRatings, g.home, g.away) : null;
      const cell = this.lookup(dist, Math.abs(o.spread), o.spread <= 0, !!(m && m.neutral_site));
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
    const s = d.summary;
    const e = this.expected(ctx.games, ctx.odds, d, ctx.teamRatings);
    if (!e.k || !s) return '';
    if (mode === 'ml') {
      const upsets = Math.round(e.k - e.suMean);
      const covLean = Math.round(e.k - e.atsMean);
      return `History (${d.seasons}): favorites expected to win ${e.suMean.toFixed(1)} of ` +
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
    return `History (${d.seasons}): favorites expected to cover ${e.atsMean.toFixed(1)} of ` +
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
    const F = M.field;
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
      const mu = typeof matchupFor === 'function' ? matchupFor(ctx.teamRatings, g.home, g.away) : null;
      const neutral = !!(mu && mu.neutral_site);   // no home/away side: pooled cell and curve
      const v = this.predict(d, Math.abs(sp), homeFav, F, neutral);
      if (v == null) continue;

      // Blend the market favorite's margin toward ELWAY's margin for that same team
      // (positive = favored by that many; can go negative, meaning that team is now the
      // modeled underdog). 0.65 = more trust than an even split, since ELWAY is Nate
      // Silver's Silver Bulletin NFL forecasting model, not an untested personal formula
      // — see nflhub/sources/history.py week_budget's ELWAY_BLEND_WEIGHT for why 0.65
      // and not full weight.
      let effMargin = Math.abs(sp);
      const el = elway[g.game_id];
      if (el && el.spread_home != null) {
        const elwayFavMargin = homeFav ? -el.spread_home : el.spread_home;
        effMargin = 0.35 * Math.abs(sp) + 0.65 * elwayFavMargin;
      }
      const rankV = this.predictSigned(d, homeFav, effMargin, F, neutral) ?? v;

      // Display bucket only: BUCKETS/bucketLabel assumes clean half-point spreads (true
      // for a single book's line, not necessarily for a cross-book average, e.g. 2.75).
      const dispSpread = Math.round(Math.abs(sp) * 2) / 2;
      const cell = this.lookup(d, dispSpread, homeFav, neutral);

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

    let totN = 0, totPrimary = 0;
    const flaggedIds = new Set();
    const rankByGame = {}; // gameId -> { rank, n, taken, lab } — every game in its bucket, not just flagged
    const bins = d.buckets.map(lab => {
      const gs = groups[lab].slice().sort((a, b) => a.rankV - b.rankV); // most live dog (lowest fav prob) first
      const n = gs.length;
      const primary = gs.reduce((s, x) => s + (1 - x.v), 0);
      const take = Math.round(primary);
      gs.forEach((x, i) => {
        x.rank = i + 1;            // 1 = most live dog in this bucket, all games ranked
        x.taken = i < take;
        if (x.taken) flaggedIds.add(x.gameId);
        rankByGame[x.gameId] = { rank: x.rank, n, taken: x.taken, lab };
      });
      totN += n; totPrimary += primary;
      return { lab, gs, n, primary, take };
    });

    return { bins, totN, totPrimary, flaggedIds, rankByGame };
  },

  // "budget": this week's games grouped by spread bin, with how many dog picks to make.
  renderBins(ctx, mode = 'ml') {
    const M = HIST_MODES[mode];
    const host = document.getElementById(M.upsetsId);
    if (!host) return;
    const budget = this.computeBudget(ctx, mode);
    if (!budget) { host.innerHTML = ''; return; }
    const F = M.field;
    const { bins, totN, totPrimary } = budget;

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

    host.innerHTML = `
      <div class="panel">
        <h2>${M.budgetTitle} by spread bin &mdash; Week ${ctx.week}</h2>
        <table><thead><tr><th>Spread bin</th><th class="num">Games</th>
          <th class="num">${M.budgetCol}</th><th class="num">${M.budgetVerb}</th><th>${M.fadeLabel}</th></tr></thead>
          <tbody>${rows}
            <tr class="tot"><td>Total</td><td class="num">${totN}</td>
              <td class="num">${totRate}%</td><td class="num"><strong>${Math.round(totPrimary)}</strong></td><td></td></tr>
          </tbody></table>
        ${mode === 'ml' ? chalkReminderHtml(ctx) : ''}
      </div>`;
    if (mode === 'ml') drawChalkMarginChart(ctx);
  },

  render(ctx, mode = 'ml') {
    const M = HIST_MODES[mode];
    const host = document.getElementById(M.histId);
    if (!host) return;
    const d = ctx.hist;
    if (!d) { host.innerHTML = `<div class="panel"><h2>History</h2><p class="muted">Not built yet — appears after the next daily refresh.</p></div>`; return; }
    const F = M.field;

    const up = v => v == null ? '—' : Math.round((1 - v) * 100) + '%'; // 1 - fav rate = dog rate

    // "home dog" = home team is the underdog => AWAY team favored => away_* cell.
    const cells = d.cells || {};
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

    host.innerHTML = `
      <div class="panel">
        <h2>History &mdash; ${M.histTitle} (${d.seasons})</h2>
        <details>
          <summary class="muted small">Show breakdown (by spread size)</summary>
          <p class="muted">${M.histBlurb}, split by whether the dog is at home or on the road.
            One pooled estimate across the whole season -- a week-of-season split (week 1,
            and several candidate season-finale splits) was tested and didn't hold up on an
            independent half of the data, so it's not used (see nflhub/sources/history.py).
            Shrunk toward each bucket's all-games rate (k=${d.shrink_k}); n is the raw sample.</p>
          <table><thead><tr><th>Spread</th><th class="num">Any dog</th><th class="num">Home dog</th>
            <th class="num">Away dog</th><th class="num">n</th></tr></thead>
            <tbody>${bktRows}</tbody></table>
        </details>
      </div>`;
  },
};
