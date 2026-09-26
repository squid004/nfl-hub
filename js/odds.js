'use strict';

// Deviation of one book's spread from the week's cross-book average, by total point
// magnitude (not raw signed value — a book giving the favorite MORE points and a book
// giving the underdog MORE points are the same "wider" line). Red = narrower than the
// field, green = wider. Half-point noise gets no highlight.
function spreadDeviationClass(bookHomeLine, avgHomeLine) {
  if (bookHomeLine == null || avgHomeLine == null) return '';
  const diff = Math.abs(bookHomeLine) - Math.abs(avgHomeLine);
  if (diff <= -0.25) return 'line-narrow';
  if (diff >= 0.25) return 'line-wide';
  return '';
}

function bookCell(row, avgHomeLine) {
  if (!row) return '<td class="muted">—</td>';
  const cls = spreadDeviationClass(row.spread_home_line, avgHomeLine);
  const spreadTxt = row.spread_home_line != null
    ? `${signed(row.spread_home_line)} (${row.spread_home_price > 0 ? '+' : ''}${row.spread_home_price ?? '—'})`
    : '—';
  const mlTxt = (row.ml_away_price != null || row.ml_home_price != null)
    ? `${row.ml_away_price ?? '—'} / ${row.ml_home_price ?? '—'}`
    : '—';
  return `<td class="bookcell ${cls}"><div>${spreadTxt}</div><div class="muted small">${mlTxt}</div></td>`;
}

// Best-price cell: "<line/price> · Book", or a plain "—" when nothing was found for
// this game (already started / not yet posted / actionnetwork scrape skipped this run).
function bestCell(line, price, book) {
  if (price == null) return '<span class="muted">—</span>';
  const num = line != null ? `${signed(line)} ` : '';
  return `${num}${price > 0 ? '+' : ''}${price} <span class="muted">· ${book}</span>`;
}

const Odds = {
  bookNames(ctx) {
    const names = new Set();
    Object.values(ctx.bookOdds || {}).forEach(rows => rows.forEach(r => names.add(r.book)));
    return Array.from(names).sort();
  },

  render(ctx) {
    const injuries = ctx.injuries || {};
    const rows = ctx.games.map(g => {
      const o = ctx.odds[g.game_id] || {};
      const el = (ctx.elway || {})[g.game_id] || {};
      const move = lineMovement((ctx.spreadHist || {})[g.game_id]);
      const state = g.state === 'post' ? `<span class="muted">(${g.away_score}-${g.home_score} F)</span>`
        : g.state === 'in' ? '<span class="chip">LIVE</span>' : '';
      const favTeam = o.spread != null ? (o.spread <= 0 ? g.home : g.away) : null;
      const elwayFlip = elwayFullDisagree(el, g.home, g.away, favTeam);
      const elwaySpreadCell = elwayFlip
        ? `<td class="elway-flip" title="ELWAY's model favors the OTHER team entirely">${signed(el.spread_home)}</td>`
        : `<td class="muted">${el.spread_home != null ? signed(el.spread_home) : '—'}</td>`;
      return `<tr>
        <td class="muted">${fmtLocal(g.kickoff, false)}</td>
        <td>${g.away}${qbChip(g.away, injuries)} @ ${g.home}${qbChip(g.home, injuries)} ${state}</td>
        <td class="muted${move && move.steam ? ' warn' : ''}" title="${move ? `opened ${signed(move.open)}, now ${signed(move.cur)}` : ''}">${signed(o.spread)}</td>
        ${elwaySpreadCell}
        <td class="muted">${o.total ?? '—'}</td>
        <td class="muted">${el.total ?? '—'}</td>
        <td class="num muted">${o.ml_away ?? '—'}</td>
        <td class="num muted">${o.ml_home ?? '—'}</td>
        <td class="num muted">${pct(o.implied_away)}</td>
        <td class="num muted">${pct(o.implied_home)}</td>
        <td class="num muted">${pct(el.away_win_prob)}</td>
        <td class="num muted">${pct(el.home_win_prob)}</td>
        <td class="muted">${o.book ?? '—'}</td>
      </tr>`;
    }).join('');

    const bookNames = this.bookNames(ctx);
    const shopRows = ctx.games.map(g => {
      const bp = (ctx.bestPrice || {})[g.game_id] || {};
      const byBook = {};
      ((ctx.bookOdds || {})[g.game_id] || []).forEach(r => { byBook[r.book] = r; });
      const bookCells = bookNames.map(name => bookCell(byBook[name], bp.avg_spread_home)).join('');
      return `<tr>
        <td class="muted">${fmtLocal(g.kickoff, false)}</td>
        <td>${g.away}${qbChip(g.away, injuries)} @ ${g.home}${qbChip(g.home, injuries)}</td>
        <td class="muted">${bp.avg_spread_home != null ? signed(bp.avg_spread_home) : '—'}</td>
        ${bookCells}
        <td>${bestCell(bp.spread_away_line, bp.spread_away_price, bp.spread_away_book)}<br>
            ${bestCell(bp.spread_home_line, bp.spread_home_price, bp.spread_home_book)}</td>
        <td>${bestCell(null, bp.ml_away_price, bp.ml_away_book)}<br>
            ${bestCell(null, bp.ml_home_price, bp.ml_home_book)}</td>
      </tr>`;
    }).join('');
    const bookHeaders = bookNames.map(n => `<th>${n}</th>`).join('');

    document.getElementById('odds').innerHTML = `
      <div class="panel">
        <h2>Odds board</h2>
        <table><thead><tr><th>Kick</th><th>Matchup</th><th>Spread</th>
          <th title="Home-spread equivalent from ELWAY's avg-points model: away avg pts minus home avg pts. Compare against the Spread column, not the sheet's own line.">ELWAY Spread</th>
          <th>Total</th>
          <th title="Sum of ELWAY's home + away avg-points projections. Compare against the Total column, not the sheet's own line.">ELWAY Total</th>
          <th class="num">Away ML</th><th class="num">Home ML</th>
          <th class="num">Away%</th><th class="num">Home%</th>
          <th class="num" title="ELWAY's away win probability, from its avg-points model">ELWAY Away%</th>
          <th class="num" title="ELWAY's home win probability, from its avg-points model">ELWAY Home%</th>
          <th>Book</th></tr></thead>
          <tbody>${rows}</tbody></table>
        <p class="tablefoot muted">ELWAY columns come from Nate Silver's Silver Bulletin
          NFL forecasting model (transcribed weekly into a Google Sheet) — compare them
          against the paired market column, not each other.
          <span class="elway-flip">Highlighted</span> ELWAY Spread = ELWAY's model favors
          the other team entirely, not just by a smaller or larger margin. A
          <span class="chip warn">QB</span> chip is that team's most severe QB-position
          entry from ESPN's injury report. A highlighted (amber) Spread cell means the line
          has moved &ge;1.5 points in one direction this week — hover for the open line.</p>
      </div>
      <div class="panel">
        <h2>Shop the line</h2>
        <table><thead><tr><th>Kick</th><th>Matchup</th><th>Avg spread<br>(home)</th>
          ${bookHeaders}
          <th>Best spread<br>away / home</th><th>Best ML<br>away / home</th></tr></thead>
          <tbody>${shopRows}</tbody></table>
        <p class="tablefoot muted">Each book cell: home spread (price) on top, away/home
          moneyline below. <span class="line-narrow">Red border</span> = narrower than this
          week's average line across these books; <span class="line-wide">green border</span>
          = wider. Sourced from actionnetwork.com (unofficial, best-effort — blank once a
          game has kicked off, and only real sportsbooks count toward the average).</p>
      </div>`;
  },
};
