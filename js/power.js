'use strict';

// Composite data-driven power ranking: nflhub.sources.team_ratings.power_rankings, a weighted
// z-score across 8 stats (rush/pass offense+defense EPA, points scored/allowed, turnovers
// committed/forced), garbage time excluded. Weights and the methodology blurb below come
// straight from the team_ratings kv payload (power_ranking_meta), not hardcoded here, so this
// page can't drift out of sync with whatever weights are actually running. Descriptive only
// -- see the Moneyline Pick'em matchup blocks and research/ for the backtesting showing this
// doesn't beat the market spread.
//
// Power Score + the 4 EPA columns render a fixed 0-100 scale (*_display fields, 100 = the best
// value EVER recorded across the full 2007-present dataset, not just this season's 32 teams --
// see team_ratings.py compute_ratings' historical_bounds), so a weak season's "best" team
// doesn't read as an inflated 100. Points/turnovers are directly countable already, so those
// columns show the plain per-game average for the window being evaluated, not a 0-100 score.
const EPA_0_100_COLS = new Set(['score', 'sos', 'rush_off_epa', 'pass_off_epa', 'rush_def_epa_allowed', 'pass_def_epa_allowed']);
const POWER_COLS = [
  ['rank', 'Rank'],
  ['team', 'Team'],
  ['score', 'Power Score'],
  ['rush_off_epa', 'Rush Offense'],
  ['pass_off_epa', 'Pass Offense'],
  ['rush_def_epa_allowed', 'Rush Defense'],
  ['pass_def_epa_allowed', 'Pass Defense'],
  ['points_off', 'Points/G'],
  ['points_def_allowed', 'Points Allowed/G'],
  ['turnovers_off', 'Turnovers/G'],
  ['turnovers_def_forced', 'Takeaways/G'],
  ['sos', 'SOS', 'Strength of schedule: an EWMA-weighted average of opponents’ own Power Score AT THE TIME each game was played (older games count less, same decay as every other rating here), on the same 0-100 scale as Power Score. 100 = toughest schedule ever recorded, 0 = easiest. Descriptive only -- not folded into Power Score itself.'],
];

const DIST_METRICS = [
  ['rush_off_epa', 'Rush Offense'],
  ['pass_off_epa', 'Pass Offense'],
  ['rush_def_epa_allowed', 'Rush Defense'],
  ['pass_def_epa_allowed', 'Pass Defense'],
];
// Partition the sorted 0-100 scale into tiers wherever a consecutive gap is large relative
// to the league's typical gap here -- the single LARGEST resulting tier is "the pack" (never
// called out); every other tier is a real, separated group, however many teams it has. This
// generalizes a plain "is this one team isolated" check: a 1-team tier IS that isolated
// outlier, but a 2-3 team tier catches a case nearest-neighbor-only logic would miss entirely
// -- e.g. two teams sitting close to EACH OTHER but both genuinely cut off from the pack
// (each one's nearest neighbor is its tier-mate, not far, so neither alone would ever trip a
// per-team isolation check -- the gap that matters is the one below the pair, not between them).
const TIER_GAP_FLOOR = 4;   // 0-100 scale points -- guards a tight league (tiny median gap)
                             // from reading any gap as "huge" in relative terms alone
const TIER_GAP_MULT = 2.5;  // a cut's gap must also be this many times the league's typical
                             // (median) consecutive gap here

function buildTiers(teamVals) {
  const sorted = [...teamVals].sort((a, b) => b.value - a.value); // best first
  const n = sorted.length;
  if (n < 2) return [sorted];
  const gaps = [];
  for (let i = 0; i < n - 1; i++) gaps.push(sorted[i].value - sorted[i + 1].value);
  const sortedGaps = [...gaps].sort((a, b) => a - b);
  const median = sortedGaps[Math.floor(sortedGaps.length / 2)];
  const threshold = Math.max(TIER_GAP_FLOOR, median * TIER_GAP_MULT);
  const tiers = [[sorted[0]]];
  for (let i = 0; i < n - 1; i++) {
    if (gaps[i] >= threshold) tiers.push([]);
    tiers[tiers.length - 1].push(sorted[i + 1]);
  }
  return tiers;
}

function fmtPowerVal(key, v) {
  if (v == null) return '—';
  if (key === 'team' || key === 'rank') return v;
  if (EPA_0_100_COLS.has(key)) return v.toFixed(1); // fixed 0-100 scale
  if (key.startsWith('points')) return v.toFixed(1); // raw per-game average
  return v.toFixed(2); // turnovers: raw per-game average
}

const Power = {
  _sort: { col: 'rank', dir: 1 },

  sortBy(col) {
    if (this._sort.col === col) this._sort.dir *= -1;
    else this._sort = { col, dir: col === 'rank' ? 1 : -1 };
    this.render(App._ctx);
  },

  render(ctx) {
    const el = document.getElementById('power');
    if (!el) return;
    const teamRatings = ctx.teamRatings || {};
    const pr = teamRatings.power_rankings || {};
    const meta = teamRatings.power_ranking_meta || null;
    const rows = Object.entries(pr).map(([team, info]) => ({ team, rank: info.rank, score: info.score_display, sos: info.sos_display, ...info.ratings_display }));

    if (!rows.length) {
      el.innerHTML = `<div class="panel"><h2>Power Rankings</h2><p class="muted">No rating data yet.</p></div>`;
      return;
    }

    const { col, dir } = this._sort;
    rows.sort((a, b) => {
      if (col === 'team') return dir * a.team.localeCompare(b.team);
      return dir * ((a[col] ?? 0) - (b[col] ?? 0)) || a.team.localeCompare(b.team);
    });

    const header = POWER_COLS.map(([key, label, title]) => {
      const active = col === key;
      const arrow = active ? (dir === 1 ? ' ▲' : ' ▼') : '';
      return `<th class="num"${title ? ` title="${esc(title)}"` : ''}><button data-act="power-sort" data-col="${key}" class="sort-btn${active ? ' active' : ''}">${label}${arrow}</button></th>`;
    }).join('');

    const body = rows.map(r => `<tr>${POWER_COLS.map(([key]) =>
      `<td class="num">${fmtPowerVal(key, r[key])}</td>`).join('')}</tr>`).join('');

    const statLabel = key => (POWER_COLS.find(([k]) => k === key) || [null, key])[1];
    const weightsList = meta
      ? Object.entries(meta.weights)
          .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
          .map(([key, w]) => `<li><strong>${statLabel(key)}:</strong> ${w >= 0 ? '+' : ''}${w.toFixed(4)}${w === 0 ? ' (dropped — redundant once the other 7 stats are known)' : ''}</li>`)
          .join('')
      : '';

    el.innerHTML = `
      <div class="panel">
        <h2>Power Rankings</h2>
        <p class="muted small">${meta ? esc(meta.method) : 'Composite of 8 stats, garbage time excluded.'}
          Power Score, SOS, and the 4 EPA columns are shown on a 0-100 scale anchored to the
          best/worst ever recorded across the full 2007-present dataset (not just this season's
          32 teams), so higher is always better and a weak season's best team won't look
          inflated. Points and turnover columns show the actual per-game average for the window
          evaluated. Click a column header to sort.</p>
        ${meta ? `<details class="power-methodology">
          <summary class="muted small">Weights used (click to expand)</summary>
          <ul class="power-weights">${weightsList}</ul>
        </details>` : ''}
        <table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>
      </div>
      ${this._distributionPanel(rows)}`;
  },

  // One number-line strip per EPA stat, every team positioned at its actual 0-100 value (not
  // just rank order) so clustering vs. real gaps is visible as literal physical distance, not
  // just color or order. See buildTiers() above for what actually counts as a real tier break.
  _distributionPanel(rows) {
    const metricHtml = ([key, label]) => {
      const teamVals = rows.filter(r => r[key] != null).map(r => ({ team: r.team, value: r[key] }));
      if (teamVals.length < 4) return '';
      const avg = teamVals.reduce((s, t) => s + t.value, 0) / teamVals.length;
      const tiers = buildTiers(teamVals).map(teams => ({
        teams, avg: teams.reduce((s, t) => s + t.value, 0) / teams.length,
      }));
      const packIdx = tiers.reduce((best, t, i) => t.teams.length > tiers[best].teams.length ? i : best, 0);
      // Standout tiers ranked outward from the pack on each side: the one furthest from
      // average is "Elite"/"Weak" (the strongest, most-separated group); any other standout
      // tier in between is just "Above average"/"Below average".
      const above = tiers.map((t, i) => ({ ...t, i })).filter(t => t.i !== packIdx && t.avg > avg)
        .sort((a, b) => b.avg - a.avg);
      const below = tiers.map((t, i) => ({ ...t, i })).filter(t => t.i !== packIdx && t.avg < avg)
        .sort((a, b) => a.avg - b.avg);
      const standoutIdx = new Map();
      above.forEach((t, rank) => standoutIdx.set(t.i, { cls: 'outlier-good', label: rank === 0 ? 'Elite' : 'Above average' }));
      below.forEach((t, rank) => standoutIdx.set(t.i, { cls: 'outlier-bad', label: rank === 0 ? 'Weak' : 'Below average' }));

      const ticks = tiers.flatMap((t, i) => {
        const standout = standoutIdx.get(i);
        return t.teams.map(team => `<div class="dist-tick${standout ? ' ' + standout.cls : ''}" style="left:${team.value}%"
            title="${esc(team.team)}: ${team.value.toFixed(1)}/100">${standout && t.teams.length <= 3 ? `<span class="dist-tick-label ${standout.cls}">${esc(team.team)}</span>` : ''}</div>`);
      }).join('');

      const tierLine = t => {
        const standout = standoutIdx.get(t.i);
        const names = t.teams.map(x => x.team).join(', ');
        const diff = Math.abs(t.avg - avg).toFixed(1);
        const dir = t.avg > avg ? 'above' : 'below';
        return `<li><strong class="${standout.cls === 'outlier-good' ? 'result-good' : 'result-bad'}">`
          + `${standout.label}</strong> (${names}) — avg ${t.avg.toFixed(1)}/100, ${diff}pts ${dir} `
          + `the league average, cut off from the rest of the pack by a real gap.</li>`;
      };
      const standoutLines = [...above, ...below].map(tierLine).join('');
      const summary = standoutLines
        ? `<ul class="dist-tiers">${standoutLines}</ul>`
        : `<p class="muted small dist-summary">No real tiers here — the whole league is bunched together with no significant gaps.</p>`;

      return `<div class="dist-metric">
        <div class="dist-label">${esc(label)}</div>
        <div class="dist-strip">
          <div class="dist-mean-line" style="left:${avg}%" title="League average: ${avg.toFixed(1)}/100"></div>
          ${ticks}
        </div>
        ${summary}
      </div>`;
    };
    const metrics = DIST_METRICS.map(metricHtml).filter(Boolean).join('');
    if (!metrics) return '';
    return `
      <div class="panel">
        <h2>EPA distribution &amp; tiers</h2>
        <p class="muted small">Every team's 0-100 EPA score (same scale as the table above),
          positioned on a line so clustering is visible at a glance, not just rank. Teams only
          split into a separate tier when a real gap (not just rank) separates them from the
          main pack -- a pack of similarly-bad teams near the bottom stays one tier, however
          far it sits from average; a smaller group genuinely cut off, elite or weak, gets its
          own tier instead. <span class="result-good">Green</span>/<span class="result-bad">red</span>
          mark standout tiers; the thin vertical line is the league average.</p>
        ${metrics}
      </div>`;
  },
};
