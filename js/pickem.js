'use strict';

// Two modes:
//   'ml'  -> Moneyline Pick'em: pick the outright winner. Rendered as one card per game
//            (richest data set: ELWAY, history, pool-leverage edge, plus a narrative).
//   'ats' -> Against the Spread: pick the side that covers. Stays a table (no edge/leverage
//            system exists for ATS, so a table is still easy to scan).
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

// gameId -> { rank, n, taken } for every game in its spread bucket, for the per-game
// "budget dog" chip and narrative — delegates to History.computeBudget so this can never
// disagree with the Upset Budget table itself (same ranking, same data).
function computeBucketRanks(ctx) {
  const budget = History.computeBudget(ctx, 'ml');
  return budget ? budget.rankByGame : {};
}

// True when ELWAY's own favorite is the OTHER team entirely — a full side flip, not just
// a smaller/larger margin in the same direction (which the narrative already covers via
// "ELWAY agrees, projecting ... by about ..."). Shared by the ML cards, the ATS table, and
// the Parlays odds board so the same games get flagged everywhere ELWAY is shown.
function elwayFullDisagree(el, home, away, favTeam) {
  if (!el || el.home_win_prob == null || el.away_win_prob == null || !favTeam) return false;
  const elwayFavTeam = el.home_win_prob > el.away_win_prob ? home : away;
  return elwayFavTeam !== favTeam;
}

// Rule-based (not AI-generated) explanation: every clause traces to a real number already
// on the card, so it's reproducible and never invents anything. Deliberately terse.
function buildNarrative({ favTeam, dogTeam, marketMargin, el, histCell, bucketLabel, bucketRank, edgeRec }) {
  const parts = [`${favTeam} favored by ${marketMargin} over ${dogTeam}.`];

  if (el && el.home_win_prob != null && el.away_win_prob != null) {
    const elwayFav = el.home_win_prob > el.away_win_prob ? 'home' : 'away';
    const elwayFavTeam = elwayFav === 'home' ? el._home : el._away;
    const elwayProb = Math.max(el.home_win_prob, el.away_win_prob);
    if (elwayFavTeam === favTeam) {
      parts.push(`ELWAY agrees, projecting ${favTeam} by about ${Math.abs(el.spread_home).toFixed(1)}.`);
    } else {
      parts.push(`ELWAY disagrees — its avg-points model actually likes ${elwayFavTeam} `
        + `(${Math.round(elwayProb * 100)}% win prob) against the market's lean toward ${favTeam}.`);
    }
  }

  if (histCell && histCell.su != null) {
    const suPct = Math.round(histCell.su * 100);
    parts.push(suPct < 55
      ? `Favorites this size (${bucketLabel}) have only won ${suPct}% of the time historically — a live spot for an upset.`
      : `Favorites this size (${bucketLabel}) have won ${suPct}% of the time historically.`);
  }

  if (bucketRank) {
    parts.push(bucketRank.n > 1
      ? `Ranked #${bucketRank.rank} of ${bucketRank.n} live dogs in this bucket (by market spread, ELWAY-adjusted).`
      : `Only dog in this bucket this week.`);
    if (bucketRank.taken) {
      parts.push(`This week's upset-budget model flags ${dogTeam} as one of its picks from this bucket.`);
    }
  }

  if (edgeRec && edgeRec.recommendation === 'FADE') {
    const fPct = edgeRec.f_estimate != null ? Math.round(edgeRec.f_estimate * 100) : null;
    parts.push(`Pool-leverage likes fading ${favTeam} here too — only ~${fPct}% of your pool is expected `
      + `on ${dogTeam}, so it pays off if it hits (leverage ${edgeRec.leverage?.toFixed(2) ?? '—'}).`);
  } else if (edgeRec && edgeRec.recommendation === 'CHALK') {
    parts.push(`Pool-leverage says stick with the chalk here — not enough separation to justify fading ${favTeam}.`);
  }

  return parts.join(' ');
}

const Pickem = {
  render(ctx, mode = 'ml') {
    if (mode === 'ml') this._renderCards(ctx);
    else this._renderTable(ctx, mode);
  },

  _renderCards(ctx) {
    const M = PICK_MODES.ml;
    const { week, games, odds, hist } = ctx;
    const picks = ctx[M.picksKey] || {};
    const edgeLog = ctx.edgeLog || {};
    const elway = ctx.elway || {};
    const bucketRanks = computeBucketRanks(ctx);

    const cards = games.map(g => {
      const o = odds[g.game_id] || {};
      const elRaw = elway[g.game_id];
      const el = elRaw ? { ...elRaw, _home: g.home, _away: g.away } : null;
      const hasLine = o.spread != null;
      const favTeam = !hasLine ? null : (o.spread <= 0 ? g.home : g.away);
      const dogTeam = favTeam == null ? null : (favTeam === g.home ? g.away : g.home);
      const favLabel = !hasLine ? '—' : (o.spread === 0 ? 'Pick’em' : `${favTeam} ${o.spread}`);
      const mine = (picks[g.game_id] || {}).pick;
      const mineIsDog = mine && dogTeam && mine === dogTeam;

      let result = null;
      if (g.state === 'post') {
        result = g.home_score > g.away_score ? g.home
          : g.away_score > g.home_score ? g.away : 'TIE';
      }
      const wchip = t => result === t ? ` <span class="chip good">${M.winChip}</span>` : '';
      const stateChip = g.state === 'post' ? `<span class="muted">(${g.away_score}-${g.home_score} F)</span>`
        : g.state === 'in' ? '<span class="chip">LIVE</span>' : '';

      const histCell = hist && hasLine ? History.lookup(hist, Math.abs(o.spread), o.spread <= 0, week) : null;
      const bucketRank = bucketRanks[g.game_id] || null;
      // Prefer the bucket the ranking itself used (computeBudget prefers the cross-book
      // average spread when available) over recomputing from o.spread alone — otherwise
      // the badge could name a different bucket than the rank next to it was computed in.
      const bucketLabel = bucketRank ? bucketRank.lab : (hasLine ? History.bucketLabel(Math.abs(o.spread)) : null);
      const edgeRec = edgeLog[g.game_id];
      const hasEdge = edgeRec && edgeRec.recommendation && edgeRec.recommendation !== 'NO_DATA';
      const elwayFlip = hasLine && elwayFullDisagree(el, g.home, g.away, favTeam);

      const btn = team => `<button data-act="${M.act}" data-week="${week}" data-game="${g.game_id}"
        data-team="${team}" data-spread="${hasLine ? o.spread : ''}"
        class="${mine === team ? 'primary' : ''}">${team}</button>`;

      const narrative = hasLine
        ? buildNarrative({ favTeam, dogTeam, marketMargin: Math.abs(o.spread), el, histCell,
                           bucketLabel, bucketRank, edgeRec: hasEdge ? edgeRec : null })
        : 'No market line yet for this game.';

      const edgeChip = hasEdge
        ? `<span class="chip ${edgeRec.recommendation === 'FADE' ? 'good' : ''}"
             title="p=${(edgeRec.p_favorite * 100).toFixed(1)}%, f=${edgeRec.f_estimate != null ? Math.round(edgeRec.f_estimate * 100) + '%' : '—'}, leverage=${edgeRec.leverage != null ? edgeRec.leverage.toFixed(3) : '—'}">
             ${edgeRec.recommendation}${edgeRec.recommendation === 'FADE' ? ' ' + edgeRec.underdog_team : ''}</span>`
        : '<span class="muted">—</span>';

      return `<div class="game-card">
        <div class="game-card-head">
          <span class="muted">${fmtLocal(g.kickoff, false)}</span>
          <span class="matchup">${g.away}${wchip(g.away)} @ ${g.home}${wchip(g.home)}</span>
          ${stateChip}
          ${bucketLabel ? `<span class="chip" title="Spread bucket: ${bucketLabel}">${bucketLabel}</span>` : ''}
          ${bucketRank ? `<span class="chip${bucketRank.taken ? ' warn' : ''}" title="Rank ${bucketRank.rank} of ${bucketRank.n} in this spread bucket, by market spread + ELWAY-adjusted rank">${bucketRank.taken ? 'budget dog ' : 'bucket '}#${bucketRank.rank}/${bucketRank.n}</span>` : ''}
          ${elwayFlip ? `<span class="chip warn" title="ELWAY's avg-points model favors the OTHER team entirely, not just by a smaller or larger margin">ELWAY flip</span>` : ''}
        </div>
        <div class="game-card-body">
          <div class="stat-block">
            <div class="stat-label">Market</div>
            <div>${favLabel} &middot; O/U ${o.total ?? '—'}</div>
            <div class="muted">${pct(o.implied_away)} / ${pct(o.implied_home)} &middot; ${o.book ?? '—'}</div>
          </div>
          <div class="stat-block">
            <div class="stat-label">ELWAY</div>
            <div>${el ? `${pct(el.away_win_prob)} / ${pct(el.home_win_prob)}` : '<span class="muted">—</span>'}</div>
            <div class="muted">${el && el.spread_home != null ? `implied ${signed(el.spread_home)}` : ' '}</div>
          </div>
          <div class="stat-block">
            <div class="stat-label">History</div>
            <div>${histCell ? `Fav SU ${pct(histCell.su)} &middot; Fav ATS ${pct(histCell.ats)}` : '<span class="muted">—</span>'}</div>
            <div class="muted">${bucketLabel ? `${bucketLabel} bucket` : ' '}</div>
          </div>
          <div class="stat-block">
            <div class="stat-label">Pool edge</div>
            <div>${edgeChip}</div>
          </div>
        </div>
        <p class="game-card-narrative">${esc(narrative)}</p>
        <div class="game-card-pick">
          ${btn(g.away)} ${btn(g.home)}
          <span class="game-card-mine">${mine ? `Your pick: <strong>${mine}</strong>${mineIsDog ? ` <span class="chip warn">${M.dogChip}</span>` : ''}` : '<span class="muted">No pick yet</span>'}</span>
        </div>
      </div>`;
    }).join('');

    const made = Object.keys(picks).length;
    const lean = hist ? History.summaryLine(ctx, 'ml') : '';
    document.getElementById(M.id).innerHTML = `
      <div class="panel">
        <h2>${M.title} &mdash; ${made}/${games.length} made
          &middot; locks ${games.length ? fmtLocal(games[0].kickoff) : 'TBD'}</h2>
        ${Edge.budgetBannerHtml(ctx)}
        ${lean ? `<p class="lean">${esc(lean)}</p>` : ''}
        <div class="game-card-grid">${cards}</div>
      </div>`;
  },

  _renderTable(ctx, mode) {
    const M = PICK_MODES[mode];
    const { week, games, odds, hist } = ctx;
    const picks = ctx[M.picksKey] || {};
    const elway = ctx.elway || {};

    const rows = games.map(g => {
      const o = odds[g.game_id] || {};
      const el = elway[g.game_id] || {};
      const hasLine = o.spread != null;
      const favTeam = !hasLine ? null : (o.spread <= 0 ? g.home : g.away);
      const dogTeam = favTeam == null ? null : (favTeam === g.home ? g.away : g.home);
      const lineFor = t => !hasLine ? '' : (o.spread === 0 ? ' PK'
        : (' ' + signed(t === g.home ? o.spread : -o.spread)));
      const favLabel = !hasLine ? '—' : (o.spread === 0 ? 'PK' : `${favTeam} ${o.spread}`);
      const mine = (picks[g.game_id] || {}).pick;
      const mineIsDog = mine && dogTeam && mine === dogTeam;

      let result = null;
      if (g.state === 'post' && hasLine) {
        const favMargin = (o.spread <= 0 ? 1 : -1) * (g.home_score - g.away_score);
        const edge = favMargin - Math.abs(o.spread);
        result = Math.abs(edge) < 1e-9 ? 'PUSH' : (edge > 0 ? favTeam : dogTeam);
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
        class="${mine === team ? 'primary' : ''}">${team}${lineFor(team)}</button>`;

      const elwayFlip = hasLine && elwayFullDisagree(el, g.home, g.away, favTeam);
      const elwaySpreadCell = elwayFlip
        ? `<td class="num elway-flip" title="ELWAY's model favors the OTHER team entirely: ${signed(el.spread_home)}">${signed(el.spread_home)}</td>`
        : `<td class="num muted">${el.spread_home != null ? signed(el.spread_home) : '—'}</td>`;

      return `<tr>
        <td class="muted">${fmtLocal(g.kickoff, false)}</td>
        <td>${g.away}${wchip(g.away)}</td>
        <td>${g.home}${wchip(g.home)}</td>
        <td class="muted">${favLabel}</td>
        ${elwaySpreadCell}
        <td class="muted">${o.total ?? '—'}</td>
        <td class="num muted">${pct(o.implied_away)}</td>
        <td class="num muted">${pct(o.implied_home)}</td>
        <td class="num muted">${pct(el.away_win_prob)}</td>
        <td class="num muted">${pct(el.home_win_prob)}</td>
        ${suCell}${atsCell}
        <td>${mine ? `<strong>${mine}</strong>${mineIsDog ? ` <span class="chip warn">${M.dogChip}</span>` : ''}` : '<span class="muted">—</span>'}</td>
        <td class="btns">${btn(g.away)} ${btn(g.home)}</td>
      </tr>`;
    }).join('');

    const made = Object.keys(picks).length;
    const lean = hist ? History.summaryLine(ctx, mode) : '';
    document.getElementById(M.id).innerHTML = `
      <div class="panel">
        <h2>${M.title} &mdash; ${made}/${games.length} made
          &middot; locks ${games.length ? fmtLocal(games[0].kickoff) : 'TBD'}</h2>
        ${lean ? `<p class="lean">${esc(lean)}</p>` : ''}
        <table><thead><tr><th>Kick</th><th>Away</th><th>Home</th><th>Fav</th>
          <th class="num" title="ELWAY's home-spread equivalent: away avg pts minus home avg pts. Compare against the Fav column's market line, not the sheet's own spread.">ELWAY Spread</th>
          <th>O/U</th>
          <th class="num">Away%</th><th class="num">Home%</th>
          <th class="num" title="ELWAY's away win probability, from its avg-points model">ELWAY Away%</th>
          <th class="num" title="ELWAY's home win probability, from its avg-points model">ELWAY Home%</th>
          <th class="num" title="Historical: favorite of this spread wins straight up">Fav SU</th>
          <th class="num" title="Historical: favorite of this spread covers">Fav ATS</th>
          <th>Pick</th><th></th></tr></thead>
          <tbody>${rows}</tbody></table>
        <p class="tablefoot muted">Away%/Home% are this game's de-vigged market prices.
          Pick the side that covers. Fav ATS is the historical cover rate for any favorite
          of that spread size — below ~50% leans dog. The line is snapshotted when you pick.</p>
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
