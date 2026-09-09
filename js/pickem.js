'use strict';

// Two modes share this renderer:
//   'ml'  -> Moneyline Pick'em: pick the outright winner
//   'ats' -> Against the Spread: pick the side that covers
const PICK_MODES = {
  ml: {
    id: 'pickem', picksKey: 'pPicks', act: 'pick', setter: 'setPickemPick',
    title: 'Moneyline Pick’em', winChip: 'W', dogChip: 'upset',
  },
  ats: {
    id: 'atspickem', picksKey: 'aPicks', act: 'atspick', setter: 'setAtsPick',
    title: 'Against the Spread', winChip: 'C', dogChip: 'dog',
  },
};

const Pickem = {
  render(ctx, mode = 'ml') {
    const M = PICK_MODES[mode];
    const { week, games, odds, hist } = ctx;
    const picks = ctx[M.picksKey] || {};

    const rows = games.map(g => {
      const o = odds[g.game_id] || {};
      const hasLine = o.spread != null;
      const favTeam = !hasLine ? null : (o.spread <= 0 ? g.home : g.away);
      const dogTeam = favTeam == null ? null : (favTeam === g.home ? g.away : g.home);
      const lineFor = t => !hasLine ? '' : (o.spread === 0 ? ' PK'
        : (' ' + signed(t === g.home ? o.spread : -o.spread)));
      const favLabel = !hasLine ? '—' : (o.spread === 0 ? 'PK' : `${favTeam} ${o.spread}`);
      const mine = (picks[g.game_id] || {}).pick;
      const mineIsDog = mine && dogTeam && mine === dogTeam;

      // who "won" this row for the current mode
      let result = null;
      if (g.state === 'post') {
        if (mode === 'ml') {
          result = g.home_score > g.away_score ? g.home
            : g.away_score > g.home_score ? g.away : 'TIE';
        } else if (hasLine) {
          const favMargin = (o.spread <= 0 ? 1 : -1) * (g.home_score - g.away_score);
          const edge = favMargin - Math.abs(o.spread);
          result = Math.abs(edge) < 1e-9 ? 'PUSH' : (edge > 0 ? favTeam : dogTeam);
        }
      }
      const wchip = t => result === t ? ` <span class="chip good">${M.winChip}</span>` : '';

      let suCell = '<td class="num muted">—</td>', atsCell = '<td class="num muted">—</td>';
      if (hist && hasLine) {
        const h = History.lookup(hist, Math.abs(o.spread), o.spread <= 0, week);
        if (h) {
          const suW = h.su != null && h.su < 0.60 ? ' warn' : '';
          const atW = h.ats != null && h.ats < 0.48 ? ' warn' : '';
          suCell = `<td class="num${suW}" title="${favTeam} straight up, n=${h.n}">${h.su != null ? Math.round(h.su * 100) + '%' : '—'}</td>`;
          atsCell = `<td class="num${atW}" title="${favTeam} covers, n=${h.n}">${h.ats != null ? Math.round(h.ats * 100) + '%' : '—'}</td>`;
        }
      }

      const btn = team => `<button data-act="${M.act}" data-week="${week}" data-game="${g.game_id}"
        data-team="${team}" data-spread="${hasLine ? o.spread : ''}"
        class="${mine === team ? 'primary' : ''}">${team}${mode === 'ats' ? lineFor(team) : ''}</button>`;

      return `<tr>
        <td class="muted">${fmtLocal(g.kickoff, false)}</td>
        <td>${g.away}${wchip(g.away)}</td>
        <td>${g.home}${wchip(g.home)}</td>
        <td class="muted">${favLabel}</td>
        <td class="muted">${o.total ?? '—'}</td>
        <td class="num muted">${pct(o.implied_away)}</td>
        <td class="num muted">${pct(o.implied_home)}</td>
        ${suCell}${atsCell}
        <td>${mine ? `<strong>${mine}</strong>${mineIsDog ? ` <span class="chip warn">${M.dogChip}</span>` : ''}` : '<span class="muted">—</span>'}</td>
        <td class="btns">${btn(g.away)} ${btn(g.home)}</td>
      </tr>`;
    }).join('');

    const made = Object.keys(picks).length;
    const lean = hist ? History.summaryLine(ctx, mode) : '';
    const foot = mode === 'ml'
      ? 'Fav SU / Fav ATS are historical rates for <em>any</em> favorite of that spread size in this part of the season — a low number is an upset lean.'
      : 'Pick the side that covers. Fav ATS is the historical cover rate for any favorite of that spread size — below ~50% leans dog. The line is snapshotted when you pick.';
    document.getElementById(M.id).innerHTML = `
      <div class="panel">
        <h2>${M.title} &mdash; ${made}/${games.length} made
          &middot; locks ${games.length ? fmtLocal(games[0].kickoff) : 'TBD'}</h2>
        ${lean ? `<p class="lean">${esc(lean)}</p>` : ''}
        <table><thead><tr><th>Kick</th><th>Away</th><th>Home</th><th>Fav</th><th>O/U</th>
          <th class="num">Away%</th><th class="num">Home%</th>
          <th class="num" title="Historical: favorite of this spread wins straight up">Fav SU</th>
          <th class="num" title="Historical: favorite of this spread covers">Fav ATS</th>
          <th>Pick</th><th></th></tr></thead>
          <tbody>${rows}</tbody></table>
        <p class="tablefoot muted">Away%/Home% are this game's de-vigged market prices. ${foot}</p>
      </div>`;
  },

  async pick(week, gameId, team, spread) {
    try { await DB.setPickemPick(+week, gameId, team, spread === '' ? null : spread); App.reload(); }
    catch (e) { alert('Could not save pick: ' + e.message); }
  },

  async atspick(week, gameId, team, spread) {
    try { await DB.setAtsPick(+week, gameId, team, spread === '' ? null : spread); App.reload(); }
    catch (e) { alert('Could not save pick: ' + e.message); }
  },
};
