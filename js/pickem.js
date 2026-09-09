'use strict';

const Pickem = {
  render(ctx) {
    const { week, games, odds, pPicks } = ctx;
    const rows = games.map(g => {
      const o = odds[g.game_id] || {};
      const fav = o.spread == null ? null : (o.spread < 0 ? g.home : o.spread > 0 ? g.away : 'EVEN');
      const mine = (pPicks[g.game_id] || {}).pick;
      let result = null;
      if (g.state === 'post') result = g.home_score > g.away_score ? g.home
        : g.away_score > g.home_score ? g.away : 'TIE';
      const btn = team => `<button data-act="pick" data-week="${week}" data-game="${g.game_id}"
        data-team="${team}" class="${mine === team ? 'primary' : ''}">${team}</button>`;
      return `<tr>
        <td class="muted">${fmtLocal(g.kickoff, false)}</td>
        <td>${g.away}${result === g.away ? ' <span class="chip good">W</span>' : ''}</td>
        <td>${g.home}${result === g.home ? ' <span class="chip good">W</span>' : ''}</td>
        <td class="muted">${fav ? `${fav} ${o.spread ?? ''}` : '—'}</td>
        <td class="muted">${o.total ?? '—'}</td>
        <td class="num muted">${pct(o.implied_away)}</td>
        <td class="num muted">${pct(o.implied_home)}</td>
        <td>${mine ? `<strong>${mine}</strong>` : '<span class="muted">—</span>'}</td>
        <td class="btns">${btn(g.away)} ${btn(g.home)}</td>
      </tr>`;
    }).join('');

    const made = Object.keys(pPicks).length;
    document.getElementById('pickem').innerHTML = `
      <div class="panel">
        <h2>Pick’em &mdash; ${made}/${games.length} made
          &middot; locks ${games.length ? fmtLocal(games[0].kickoff) : 'TBD'}</h2>
        <table><thead><tr><th>Kick</th><th>Away</th><th>Home</th><th>Spread</th><th>O/U</th>
          <th class="num">Away%</th><th class="num">Home%</th><th>Pick</th><th></th></tr></thead>
          <tbody>${rows}</tbody></table>
      </div>`;
  },

  async pick(week, gameId, team) {
    try { await DB.setPickemPick(+week, gameId, team); App.reload(); }
    catch (e) { alert('Could not save pick: ' + e.message); }
  },
};
