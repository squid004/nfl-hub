'use strict';

const Survivor = {
  render(ctx) {
    const { week, games, odds, sPicks } = ctx;
    const usedPrior = new Set(Object.entries(sPicks)
      .filter(([w]) => +w < week).map(([, t]) => t));
    const thisPick = sPicks[week] || null;

    const opts = [];
    games.forEach(g => {
      const o = odds[g.game_id] || {};
      [['home', g.home, g.away, o.implied_home, o.spread, o.ml_home],
       ['away', g.away, g.home, o.implied_away, o.spread == null ? null : -o.spread, o.ml_away]]
        .forEach(([side, team, opp, wp, spread, ml]) => {
          opts.push({ team, opp, side, kickoff: g.kickoff, wp, spread, total: o.total, ml,
                      used: usedPrior.has(team), isPick: team === thisPick });
        });
    });
    opts.sort((a, b) => (b.wp || 0) - (a.wp || 0));

    const rows = opts.filter(o => o.wp != null && o.wp >= 0.5).map(o => `
      <tr>
        <td class="${o.used ? 'used' : ''}">${o.team}${o.isPick ? ' <span class="chip good">pick</span>' : ''}</td>
        <td class="muted">${o.opp}</td>
        <td class="muted">${o.side}</td>
        <td class="muted">${fmtLocal(o.kickoff, false)}</td>
        <td class="muted">${o.spread == null ? '—' : signed(o.spread)}</td>
        <td class="muted">${o.total ?? '—'}</td>
        <td class="num muted">${o.ml ?? '—'}</td>
        <td class="num">${pct(o.wp)}</td>
        <td>${o.used ? '<span class="muted">used</span>'
          : `<button data-act="surv" data-week="${week}" data-team="${o.team}"
               class="${thisPick ? '' : 'primary'}">Pick</button>`}</td>
      </tr>`).join('');

    document.getElementById('survivor').innerHTML = `
      <div class="panel">
        <h2>Survivor &mdash; week ${week}
          ${thisPick ? `<span class="chip good">picked ${thisPick}</span>`
                     : '<span class="chip bad">no pick</span>'}
          &middot; locks ${games.length ? fmtLocal(games[0].kickoff) : 'TBD'}</h2>
        <p class="muted">Used: ${[...usedPrior].sort().join(', ') || 'none yet'}</p>
        <table><thead><tr><th>Team</th><th>Opp</th><th>H/A</th><th>Kick</th>
          <th>Spread</th><th>O/U</th><th class="num">ML</th><th class="num">Win%</th><th></th></tr></thead>
          <tbody>${rows}</tbody></table>
      </div>`;
  },

  async pick(week, team) {
    try { await DB.setSurvivorPick(+week, team); App.reload(); }
    catch (e) { alert('Could not save pick: ' + e.message); }
  },
};
