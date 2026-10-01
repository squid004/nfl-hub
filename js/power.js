'use strict';

// Composite data-driven power ranking: nflhub.sources.team_ratings.power_rankings, a weighted
// z-score across 8 stats (rush/pass offense+defense EPA, points scored/allowed, turnovers
// committed/forced), garbage time excluded. Weights and the methodology blurb below come
// straight from the team_ratings kv payload (power_ranking_meta), not hardcoded here, so this
// page can't drift out of sync with whatever weights are actually running. Descriptive only
// -- see the Moneyline Pick'em matchup blocks and research/ for the backtesting showing this
// doesn't beat the market spread.
//
// Every stat column renders the 0-100 normalized value (*_0_100 fields, 100 = best in the NFL
// this season), not raw signed EPA/points/turnovers -- same scale for every column, nothing
// with a +/- sign to interpret. The underlying fit still runs on the raw values server-side.
const POWER_COLS = [
  ['rank', 'Rank'],
  ['team', 'Team'],
  ['score', 'Power Score'],
  ['rush_off_epa', 'Rush Offense'],
  ['pass_off_epa', 'Pass Offense'],
  ['rush_def_epa_allowed', 'Rush Defense'],
  ['pass_def_epa_allowed', 'Pass Defense'],
  ['points_off', 'Points Scored'],
  ['points_def_allowed', 'Points Allowed'],
  ['turnovers_off', 'Turnover Avoidance'],
  ['turnovers_def_forced', 'Takeaways Forced'],
];

function fmtPowerVal(key, v) {
  if (v == null) return '—';
  if (key === 'team' || key === 'rank') return v;
  return v.toFixed(1); // every column is 0-100 normalized now, same scale throughout
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
    const rows = Object.entries(pr).map(([team, info]) => ({ team, rank: info.rank, score: info.score_0_100, ...info.ratings_0_100 }));

    if (!rows.length) {
      el.innerHTML = `<div class="panel"><h2>Power Rankings</h2><p class="muted">No rating data yet.</p></div>`;
      return;
    }

    const { col, dir } = this._sort;
    rows.sort((a, b) => {
      if (col === 'team') return dir * a.team.localeCompare(b.team);
      return dir * ((a[col] ?? 0) - (b[col] ?? 0)) || a.team.localeCompare(b.team);
    });

    const header = POWER_COLS.map(([key, label]) => {
      const active = col === key;
      const arrow = active ? (dir === 1 ? ' ▲' : ' ▼') : '';
      return `<th class="num"><button data-act="power-sort" data-col="${key}" class="sort-btn${active ? ' active' : ''}">${label}${arrow}</button></th>`;
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
          Every column is shown on a 0-100 scale (100 = best in the NFL this season), including
          defensive/turnover stats, so higher is always better everywhere in the table.
          Click a column header to sort.</p>
        ${meta ? `<details class="power-methodology">
          <summary class="muted small">Weights used (click to expand)</summary>
          <ul class="power-weights">${weightsList}</ul>
        </details>` : ''}
        <table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>
      </div>`;
  },
};
