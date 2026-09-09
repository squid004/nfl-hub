'use strict';

const Odds = {
  render(ctx) {
    const rows = ctx.games.map(g => {
      const o = ctx.odds[g.game_id] || {};
      const state = g.state === 'post' ? `<span class="muted">(${g.away_score}-${g.home_score} F)</span>`
        : g.state === 'in' ? '<span class="chip">LIVE</span>' : '';
      return `<tr>
        <td class="muted">${fmtLocal(g.kickoff, false)}</td>
        <td>${g.away} @ ${g.home} ${state}</td>
        <td class="muted">${signed(o.spread)}</td>
        <td class="muted">${o.total ?? '—'}</td>
        <td class="num muted">${o.ml_away ?? '—'}</td>
        <td class="num muted">${o.ml_home ?? '—'}</td>
        <td class="num muted">${pct(o.implied_away)}</td>
        <td class="num muted">${pct(o.implied_home)}</td>
        <td class="muted">${o.book ?? '—'}</td>
      </tr>`;
    }).join('');

    document.getElementById('odds').innerHTML = `
      <div class="panel">
        <h2>Odds board</h2>
        <table><thead><tr><th>Kick</th><th>Matchup</th><th>Spread</th><th>Total</th>
          <th class="num">Away ML</th><th class="num">Home ML</th>
          <th class="num">Away%</th><th class="num">Home%</th><th>Book</th></tr></thead>
          <tbody>${rows}</tbody></table>
      </div>`;
  },
};
