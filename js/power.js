'use strict';

// Composite data-driven power ranking: nflhub.sources.team_ratings.power_rankings, a weighted
// z-score across 8 stats (rush/pass offense+defense EPA, points scored/allowed, turnovers
// committed/forced), garbage time excluded, weights = each stat's own standalone historical
// predictive value. Descriptive only -- see the Moneyline Pick'em matchup blocks and
// research/ for the backtesting showing this doesn't beat the market spread.
const POWER_COLS = [
  ['rank', 'Rank'],
  ['team', 'Team'],
  ['score', 'Power Score'],
  ['rush_off_epa', 'Rush Off EPA'],
  ['pass_off_epa', 'Pass Off EPA'],
  ['rush_def_epa_allowed', 'Rush Def EPA (allowed)'],
  ['pass_def_epa_allowed', 'Pass Def EPA (allowed)'],
  ['points_off', 'Points Scored'],
  ['points_def_allowed', 'Points Allowed'],
  ['turnovers_off', 'Turnovers Committed'],
  ['turnovers_def_forced', 'Turnovers Forced'],
];

function fmtPowerVal(key, v) {
  if (v == null) return '—';
  if (key === 'team' || key === 'rank') return v;
  if (key.startsWith('points')) return v.toFixed(1);
  if (key.startsWith('turnovers')) return v.toFixed(2);
  return v.toFixed(3); // EPA metrics + composite score
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
    const pr = (ctx.teamRatings || {}).power_rankings || {};
    const rows = Object.entries(pr).map(([team, info]) => ({ team, rank: info.rank, score: info.score, ...info.ratings }));

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

    el.innerHTML = `
      <div class="panel">
        <h2>Power Rankings</h2>
        <p class="muted small">Composite of 8 stats (rush/pass offense &amp; defense EPA,
          points scored/allowed, turnovers committed/forced), garbage time excluded and
          weighted by each stat's own standalone historical predictive value against real
          game outcomes. Descriptive context, not a betting model &mdash; backtesting found
          this does not beat the closing market spread. Click a column header to sort.</p>
        <table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>
      </div>`;
  },
};
