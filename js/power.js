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
// A team only counts as a real outlier when its CLOSEST neighbor on the 0-100 scale is
// still unusually far away -- that's what separates "this team is genuinely isolated" from
// "this team happens to sit next to one isolated team" (the near-side neighbor of a real
// outlier is NOT itself isolated; nearest-neighbor distance, not farthest-neighbor distance,
// is what makes that distinction). Directly answers "if most of the bad ones are close to
// average, where's the real gap" -- a cluster of similarly-bad teams has small gaps between
// them and triggers nothing, no matter how far the whole cluster sits from the league mean.
const OUTLIER_GAP_FLOOR = 4;   // 0-100 scale points -- guards a tight league (tiny median
                                // gap) from reading any gap as "huge" in relative terms alone
const OUTLIER_GAP_MULT = 2.5;  // nearest-neighbor gap must also be this many times the
                                // league's typical (median) nearest-neighbor gap

function findOutliers(teamVals) {
  const sorted = [...teamVals].sort((a, b) => a.value - b.value);
  const n = sorted.length;
  const nearest = sorted.map((row, i) => {
    const below = i > 0 ? row.value - sorted[i - 1].value : Infinity;
    const above = i < n - 1 ? sorted[i + 1].value - row.value : Infinity;
    return Math.min(below, above);
  });
  const finiteSorted = nearest.filter(g => Number.isFinite(g)).sort((a, b) => a - b);
  const median = finiteSorted.length ? finiteSorted[Math.floor(finiteSorted.length / 2)] : 0;
  const threshold = Math.max(OUTLIER_GAP_FLOOR, median * OUTLIER_GAP_MULT);
  return sorted
    .map((row, i) => ({ ...row, nearestGap: nearest[i] }))
    .filter(row => Number.isFinite(row.nearestGap) && row.nearestGap >= threshold);
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
  // just color or order. See findOutliers() above for what actually counts as a flagged gap.
  _distributionPanel(rows) {
    const metricHtml = ([key, label]) => {
      const teamVals = rows.filter(r => r[key] != null).map(r => ({ team: r.team, value: r[key] }));
      if (teamVals.length < 4) return '';
      const outliers = findOutliers(teamVals);
      const outlierTeams = new Map(outliers.map(o => [o.team, o]));
      const avg = teamVals.reduce((s, t) => s + t.value, 0) / teamVals.length;
      const ticks = teamVals.map(t => {
        const out = outlierTeams.get(t.team);
        const cls = out ? (t.value >= avg ? 'outlier-good' : 'outlier-bad') : '';
        return `<div class="dist-tick${cls ? ' ' + cls : ''}" style="left:${t.value}%"
            title="${esc(t.team)}: ${t.value.toFixed(1)}/100">${out ? `<span class="dist-tick-label ${cls}">${esc(t.team)}</span>` : ''}</div>`;
      }).join('');
      const summary = outliers.length
        ? outliers.sort((a, b) => b.nearestGap - a.nearestGap).map(o =>
            `<strong class="${o.value >= avg ? 'result-good' : 'result-bad'}">${esc(o.team)}</strong> `
            + `(${o.value.toFixed(1)}/100) is a real outlier — its nearest neighbor is still `
            + `${o.nearestGap.toFixed(1)}pts away, well past the league's typical gap here.`
          ).join(' ')
        : `No real outliers here — teams differ gradually, not in a clustered-pack-plus-isolated-team pattern.`;
      return `<div class="dist-metric">
        <div class="dist-label">${esc(label)}</div>
        <div class="dist-strip">
          <div class="dist-mean-line" style="left:${avg}%" title="League average: ${avg.toFixed(1)}/100"></div>
          ${ticks}
        </div>
        <p class="muted small dist-summary">${summary}</p>
      </div>`;
    };
    const metrics = DIST_METRICS.map(metricHtml).filter(Boolean).join('');
    if (!metrics) return '';
    return `
      <div class="panel">
        <h2>EPA distribution &amp; outliers</h2>
        <p class="muted small">Every team's 0-100 EPA score (same scale as the table above),
          positioned on a line so clustering is visible at a glance, not just rank. A team is
          only called a real outlier when its CLOSEST neighbor is still unusually far away --
          a pack of similarly-bad teams near the bottom isn't an outlier just because one of
          them ranks dead last; a team genuinely alone out past the rest of the league is.
          <span class="result-good">Green</span>/<span class="result-bad">red</span> labels
          mark the flagged teams; the thin vertical line is the league average.</p>
        ${metrics}
      </div>`;
  },
};
