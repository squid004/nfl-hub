'use strict';

// Market-vs-model helpers shared by the Moneyline card's "Similar games" table and the Model
// Trends distribution panel. Everything is derived from the backtest scatter (market spread +
// delta + final score for every graded game), so it grows automatically as games are played.
//
// Delta is put on the market's own scale by converting it to points: a regression THROUGH THE
// ORIGIN of actual margin on delta over every graded game (through the origin so the sign of the
// model's pick never changes -- delta already contains the home-field term).
const Agreement = {
  // Similarity windows, tightest first: market |spread| within `m` points, model |delta| (in
  // points) within `d`. The market spread is discrete, so the first level is its exact number;
  // delta is continuous, so its tightest window is +/-0.5 pt. Widen only until MIN_GAMES graded games
  // match in the row that applies to this game (agreed or disagreed).
  LEVELS: [{ m: 0, d: 0.5 }, { m: 0.5, d: 1 }, { m: 1, d: 1.5 }, { m: 1.5, d: 2 }],
  MIN_GAMES: 25,

  _fitFor: null,
  _slope: null,

  // Points of actual margin per 1.0 of delta.
  slope(scatter) {
    if (this._fitFor === scatter) return this._slope;
    let sxy = 0, sxx = 0;
    for (const r of scatter) {
      sxy += r.delta * (r.home_score - r.away_score);
      sxx += r.delta * r.delta;
    }
    this._fitFor = scatter;
    this._slope = sxx > 0 ? sxy / sxx : 7;
    return this._slope;
  },

  percentile(sorted, p) {
    if (!sorted.length) return NaN;
    const idx = (sorted.length - 1) * p;
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
  },

  summarize(values) {
    const v = values.filter(x => Number.isFinite(x)).slice().sort((a, b) => a - b);
    const n = v.length;
    if (!n) return null;
    const mean = v.reduce((a, b) => a + b, 0) / n;
    const sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
    return { n, mean, sd, p5: this.percentile(v, 0.05), p25: this.percentile(v, 0.25),
      p50: this.percentile(v, 0.5), p75: this.percentile(v, 0.75), p95: this.percentile(v, 0.95) };
  },

  _tally(rows) {
    const t = { agreeN: 0, agreeRight: 0, disN: 0, marketRight: 0, powerRight: 0 };
    for (const r of rows) {
      const homeWon = r.home_score > r.away_score;
      if (r.spread === 0 || r.delta === 0 || r.home_score === r.away_score) continue;
      const marketHome = r.spread > 0, modelHome = r.delta > 0;
      if (marketHome === modelHome) {
        t.agreeN++;
        if (marketHome === homeWon) t.agreeRight++;
      } else {
        t.disN++;
        if (marketHome === homeWon) t.marketRight++; else t.powerRight++;
      }
    }
    return t;
  },

  // spreadHome: market spread, HOME-favored positive (the scatter's convention).
  // delta: the model's delta for this game, home-positive. exclude: {season, week, home, away}
  // so a finished game doesn't count itself.
  similarHtml(scatter, spreadHome, delta, exclude) {
    if (!scatter || !scatter.length || spreadHome == null || delta == null || spreadHome === 0) return '';
    const k = this.slope(scatter);
    const mkt = Math.abs(spreadHome), pts = Math.abs(delta) * k;
    const rows = exclude
      ? scatter.filter(r => !(r.season === exclude.season && r.week === exclude.week && r.home === exclude.home && r.away === exclude.away))
      : scatter;
    const EPS = 1e-9;
    const nowAgree = (spreadHome > 0) === (delta > 0);
    let lvl = this.LEVELS[0], t = null;
    for (const L of this.LEVELS) {
      lvl = L;
      t = this._tally(rows.filter(r => Math.abs(Math.abs(r.spread) - mkt) <= L.m + EPS
        && Math.abs(Math.abs(r.delta) * k - pts) <= L.d + EPS));
      if ((nowAgree ? t.agreeN : t.disN) >= this.MIN_GAMES) break;
    }

    // Only the row that applies to this game: when market and model agree they are both right
    // or both wrong; when they disagree exactly one of them is right.
    const cell = (v, n) => n ? `${v} <span class="muted small">(${(100 * v / n).toFixed(0)}%)</span>` : '<span class="muted">—</span>';
    const [n, heads, vals] = nowAgree
      ? [t.agreeN, ['Both right', 'Both wrong'], [t.agreeRight, t.agreeN - t.agreeRight]]
      : [t.disN, ['Market right', 'Power right'], [t.marketRight, t.powerRight]];

    const f1 = x => (+x.toFixed(2)).toString();
    const mktTxt = lvl.m ? `${f1(mkt)} &plusmn;${f1(lvl.m)}` : `exactly ${f1(mkt)}`;
    const widened = lvl !== this.LEVELS[0];
    const enough = n >= this.MIN_GAMES;
    return `<div class="game-card-matchup">
      <div class="stat-label" title="Graded games (2007 on) with a similar market spread and a similar model delta, the delta converted to points at ${k.toFixed(2)} pts per 1.0 delta. The windows widen only until ${this.MIN_GAMES}+ games match in the agreed/disagreed group that applies to this game. Ties excluded.">Similar games</div>
      <div class="muted small">Market spread ${mktTxt} pts &middot; model &Delta; ${f1(Math.abs(delta))} &asymp; ${pts.toFixed(1)} pts (&plusmn;${f1(lvl.d)}) &mdash; they ${nowAgree ? '<strong>agree</strong>' : '<strong>disagree</strong>'} here.${widened ? (enough ? ` Window widened to reach ${this.MIN_GAMES}+ comparable games.` : ` Widest window used and still only ${n} comparable games -- treat with caution.`) : ''}</div>
      <table class="small" style="margin-top:4px;width:100%;">
        <thead><tr><th class="num">Games</th>${heads.map(h => `<th class="num">${h}</th>`).join('')}</tr></thead>
        <tbody><tr><td class="num">${n}</td>${vals.map(v => `<td class="num">${cell(v, n)}</td>`).join('')}</tr></tbody>
      </table>
    </div>`;
  },
};
