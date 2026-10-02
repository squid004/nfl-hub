'use strict';

// pickem-edge integration: pool-specific leverage/fade math (see nflhub/sources/edge_core.py
// for the ported formulas and SPEC.md in github.com/squid004/pickem-edge for the rationale).
// This module owns the #edge section -- just a read-only per-team bias chart now (opponent
// picks / bias editor / standing / season log panels, and the budget-banner summary line
// above the Moneyline table, were all removed -- the standing/budget info that banner showed
// is already visible per-game via each card's Pool edge chip, which is what actually drives
// picks; the aggregate line was redundant with that, not a separate fact).

const EDGE_TEAMS = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN',
  'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LAC', 'LAR', 'LV', 'MIA', 'MIN', 'NE', 'NO',
  'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WSH'];

const signedPct = v => (v == null ? '—' : (v >= 0 ? '+' : '') + Math.round(v * 100) + '%');

const Edge = {
  render(ctx) {
    const host = document.getElementById('edge');
    if (!host) return;
    host.innerHTML = this._biasChartPanel(ctx);
  },

  // Read-only: learned per-team pool bias (your pool's pick% minus the national pick% for
  // that team), as a diverging bar chart -- green/right = your pool overpicks that team vs.
  // the country, amber/left = underpicks. No edit UI; the value is learned automatically from
  // opponent picks pulled server-side each refresh, not something to hand-tune here.
  _biasChartPanel(ctx) {
    const byTeam = {};
    (ctx.edgeBias || []).forEach(b => { byTeam[b.team] = b; });
    const sorted = EDGE_TEAMS
      .map(team => ({ team, ...(byTeam[team] || { bias_value: 0, n_observations: 0, overridden: false }) }))
      .sort((a, b) => (b.bias_value || 0) - (a.bias_value || 0));
    const maxAbs = Math.max(0.01, ...sorted.map(b => Math.abs(b.bias_value || 0)));
    const rows = sorted.map(b => {
      const v = b.bias_value || 0;
      const widthPct = Math.round(Math.abs(v) / maxAbs * 100);
      const title = `${b.n_observations} observation${b.n_observations === 1 ? '' : 's'}${b.overridden ? ' (manual override)' : ''}`;
      return `<div class="bias-row" title="${esc(title)}">
        <span class="bias-team">${b.team}</span>
        <div class="bias-track">
          <div class="bias-half neg">${v < 0 ? `<div class="bias-bar neg" style="width:${widthPct}%"></div>` : ''}</div>
          <div class="bias-half pos">${v >= 0 ? `<div class="bias-bar pos" style="width:${widthPct}%"></div>` : ''}</div>
        </div>
        <span class="bias-val num">${signedPct(v)}</span>
      </div>`;
    }).join('');
    return `
      <div class="panel">
        <h2>Pool bias by team</h2>
        <p class="muted">Learned per-team pool bias = your pool's pick% minus the national
          pick% for that team, averaged and damped under 3 observations.
          <span class="pick-fav">Green</span>/right = your pool overpicks that team vs. the
          country; <span class="pick-dog">amber</span>/left = underpicks. Learned automatically
          from opponent picks pulled from the pool sheet each refresh.</p>
        <div class="bias-chart">${rows}</div>
      </div>`;
  },
};
