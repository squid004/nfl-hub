'use strict';

// Rule-based, non-wagering "biggest statistical trends" board -- no payout/odds math, just
// three call-outs per week: the biggest EPA offense-vs-defense mismatches, the biggest
// ELWAY-vs-market total disagreements, and games where weather is worth real points. All
// three reuse data already computed elsewhere (team_ratings.py's matchup_callouts, the
// elway_odds/odds tables) -- nothing new is fetched for this tab.

// Top `n` games this week by |z| for one phase's edge (rush_edge/pass_edge), biggest first.
function biggestEdges(ctx, phase, n = 3) {
  if (!ctx.teamRatings) return [];
  const rows = [];
  (ctx.games || []).forEach(g => {
    const matchup = matchupFor(ctx.teamRatings, g.home, g.away);
    const edge = matchup && matchup[`${phase}_edge`];
    if (edge) rows.push({ g, edge });
  });
  rows.sort((a, b) => Math.abs(b.edge.z) - Math.abs(a.edge.z));
  return rows.slice(0, n);
}

// 1st/2nd/3rd/4th/...th -- same convention as team_ratings.py's _ordinal().
function ordinal(n) {
  if (n == null) return '—';
  const rem100 = n % 100;
  if (rem100 >= 10 && rem100 <= 20) return `${n}th`;
  return `${n}${ { 1: 'st', 2: 'nd', 3: 'rd' }[n % 10] || 'th' }`;
}

// "TEAM phase offense (Nth) vs OPPONENT phase defense (Nth)" -- league rank (of 32), not the
// raw/display EPA number, so it reads the same as the matchup sentences used elsewhere.
function edgeSentence(phase, edge) {
  return `${edge.team} ${phase} offense (${ordinal(edge.off_rank)}) vs `
    + `${edge.opponent} ${phase} defense (${ordinal(edge.def_rank)})`;
}

// Every game this week with a DraftKings total AND an ELWAY total that disagree by >=3
// points, biggest gap first. Gated on book === 'DraftKings' specifically (not just "any
// market total") since that's the one sportsbook line this is meant to be checked against --
// odds.js sets o.book to 'DraftKings' only when ESPN's per-game DK odds enrichment actually
// succeeded for that game, else it's ESPN's own default line (see nflhub/sources/odds.py).
const TOTAL_DISAGREEMENT_MIN = 3;
function totalDisagreements(ctx) {
  const rows = [];
  (ctx.games || []).forEach(g => {
    const o = (ctx.odds || {})[g.game_id] || {};
    const el = (ctx.elway || {})[g.game_id] || {};
    if (o.book !== 'DraftKings' || o.total == null || el.total == null) return;
    const delta = Math.round((el.total - o.total) * 10) / 10;
    if (Math.abs(delta) >= TOTAL_DISAGREEMENT_MIN) rows.push({ g, o, el, delta });
  });
  rows.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
  return rows;
}

// Every game this week where predict_points()'s weather adjustment is worth >=1 combined
// point, biggest first. This is a heads-up, not a confirmed market miss -- there's no total-
// line movement history (only spread_history) to check whether the book already priced the
// weather in, so the copy below is deliberately framed as "worth checking", not "the book is
// wrong".
function weatherCallouts(ctx) {
  if (!ctx.teamRatings) return [];
  const rows = [];
  (ctx.games || []).forEach(g => {
    const matchup = matchupFor(ctx.teamRatings, g.home, g.away);
    const delta = matchup && matchup.weather_points_delta;
    if (delta != null && Math.abs(delta) >= 1) rows.push({ g, matchup, delta });
  });
  rows.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
  return rows;
}

const Parlays = {
  render(ctx) {
    const el = document.getElementById('parlays-trends');
    if (!el) return;

    const edgeRows = (phase, label) => {
      const rows = biggestEdges(ctx, phase, 3);
      if (!rows.length) return `<p class="muted small">No ${label.toLowerCase()} mismatches big enough to call out this week.</p>`;
      return `<ol>${rows.map(({ g, edge }) =>
        `<li>${esc(edgeSentence(phase, edge))}
           <span class="muted small">(${esc(g.away)} @ ${esc(g.home)}, ${fmtLocal(g.kickoff, false)})</span></li>`
      ).join('')}</ol>`;
    };

    const totalRows = totalDisagreements(ctx);
    const totalHtml = totalRows.length
      ? `<ol>${totalRows.map(({ g, o, el: elw, delta }) =>
          `<li>${esc(g.away)} @ ${esc(g.home)}: ELWAY ${elw.total} vs DraftKings ${o.total}
             <span class="${delta > 0 ? 'result-good' : 'result-bad'}">(${signed(delta)})</span>
             <span class="muted small">${fmtLocal(g.kickoff, false)}</span></li>`
        ).join('')}</ol>`
      : `<p class="muted small">No DraftKings/ELWAY total disagreement of ${TOTAL_DISAGREEMENT_MIN}+ points this week.</p>`;

    const wxRows = weatherCallouts(ctx);
    const wxHtml = wxRows.length
      ? `<ol>${wxRows.map(({ g, delta }) =>
          `<li>${esc(g.away)} @ ${esc(g.home)}: weather is worth about
             <span class="${delta < 0 ? 'result-bad' : 'result-good'}">${signed(delta)} total points</span>
             in our model <span class="muted small">(${fmtLocal(g.kickoff, false)}) — worth checking it's
             actually reflected in the posted total.</span></li>`
        ).join('')}</ol>`
      : `<p class="muted small">No game this week where weather is worth a point of note.</p>`;

    el.innerHTML = `
      <div class="panel">
        <h2>Biggest matchups (EPA)</h2>
        <h3>Pass</h3>
        ${edgeRows('pass', 'Pass')}
        <h3>Run</h3>
        ${edgeRows('rush', 'Run')}
        <p class="tablefoot muted">Only genuine mismatches: a top-8 offense against a
          bottom-8 defense in that phase, ranked by combined z-score against the rest of the
          league. Same underlying model as the matchup table on each Moneyline card —
          descriptive only, not a betting edge (see Power Rankings methodology).</p>
      </div>
      <div class="panel">
        <h2>ELWAY vs. the total</h2>
        ${totalHtml}
        <p class="tablefoot muted">Only shown when ELWAY's avg-points total disagrees with
          the DraftKings total by ${TOTAL_DISAGREEMENT_MIN}+ points.
          <span class="result-good">Green</span> = ELWAY projects MORE points than DraftKings;
          <span class="result-bad">red</span> = fewer. Ranked by size of the gap, biggest
          first.</p>
      </div>
      <div class="panel">
        <h2>Weather vs. the total</h2>
        ${wxHtml}
        <p class="tablefoot muted">Our own wind/precip/cold-adjusted points model
          (<code>predict_points</code>) vs. the same model with no weather adjustment applied —
          <span class="result-bad">red</span> = weather likely suppresses the total,
          <span class="result-good">green</span> = likely lifts it. We don't have total-line
          movement history to confirm whether the book already baked this in, so treat this as
          a prompt to go check the number yourself, not a confirmed market miss.</p>
      </div>`;
  },
};
