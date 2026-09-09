'use strict';

const Pickem = {
  render(ctx) {
    const { week, games, odds, pPicks, hist } = ctx;
    const rows = games.map(g => {
      const o = odds[g.game_id] || {};
      const favTeam = o.spread == null ? null : (o.spread < 0 ? g.home : o.spread > 0 ? g.away : g.home);
      const favLabel = o.spread == null ? '—' : (o.spread === 0 ? 'PK' : `${favTeam} ${o.spread}`);
      const mine = (pPicks[g.game_id] || {}).pick;
      let result = null;
      if (g.state === 'post') result = g.home_score > g.away_score ? g.home
        : g.away_score > g.home_score ? g.away : 'TIE';

      // historical rates for a favorite of this spread size / side / week
      let suCell = '<td class="num muted">—</td>', atsCell = '<td class="num muted">—</td>';
      if (hist && o.spread != null) {
        const h = History.lookup(hist, Math.abs(o.spread), o.spread <= 0, week);
        if (h) {
          const suW = h.su != null && h.su < 0.60 ? ' warn' : '';
          const atW = h.ats != null && h.ats < 0.48 ? ' warn' : '';
          suCell = `<td class="num${suW}" title="${favTeam} straight up, n=${h.n}">${h.su != null ? Math.round(h.su * 100) + '%' : '—'}</td>`;
          atsCell = `<td class="num${atW}" title="${favTeam} covers, n=${h.n}">${h.ats != null ? Math.round(h.ats * 100) + '%' : '—'}</td>`;
        }
      }

      const btn = team => `<button data-act="pick" data-week="${week}" data-game="${g.game_id}"
        data-team="${team}" class="${mine === team ? 'primary' : ''}">${team}</button>`;
      return `<tr>
        <td class="muted">${fmtLocal(g.kickoff, false)}</td>
        <td>${g.away}${result === g.away ? ' <span class="chip good">W</span>' : ''}</td>
        <td>${g.home}${result === g.home ? ' <span class="chip good">W</span>' : ''}</td>
        <td class="muted">${favLabel}</td>
        <td class="muted">${o.total ?? '—'}</td>
        <td class="num muted">${pct(o.implied_away)}</td>
        <td class="num muted">${pct(o.implied_home)}</td>
        ${suCell}${atsCell}
        <td>${mine ? `<strong>${mine}</strong>` : '<span class="muted">—</span>'}</td>
        <td class="btns">${btn(g.away)} ${btn(g.home)}</td>
      </tr>`;
    }).join('');

    const made = Object.keys(pPicks).length;
    const lean = hist ? History.summaryLine(ctx) : '';
    document.getElementById('pickem').innerHTML = `
      <div class="panel">
        <h2>Pick’em &mdash; ${made}/${games.length} made
          &middot; locks ${games.length ? fmtLocal(games[0].kickoff) : 'TBD'}</h2>
        ${lean ? `<p class="lean">${esc(lean)}</p>` : ''}
        <table><thead><tr><th>Kick</th><th>Away</th><th>Home</th><th>Fav</th><th>O/U</th>
          <th class="num">Away%</th><th class="num">Home%</th>
          <th class="num" title="Historical: favorite of this spread wins straight up">Fav SU</th>
          <th class="num" title="Historical: favorite of this spread covers">Fav ATS</th>
          <th>Pick</th><th></th></tr></thead>
          <tbody>${rows}</tbody></table>
        <p class="tablefoot muted">Away%/Home% are this game's de-vigged market prices.
          Fav SU / Fav ATS are historical rates for <em>any</em> favorite of that spread size
          in this part of the season — a low number is an upset lean.</p>
      </div>`;
  },

  async pick(week, gameId, team) {
    try { await DB.setPickemPick(+week, gameId, team); App.reload(); }
    catch (e) { alert('Could not save pick: ' + e.message); }
  },
};
