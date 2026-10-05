'use strict';

// Rule-based, non-wagering "biggest statistical trends" board -- no payout/odds math, just
// three call-outs per week: EPA offense-vs-defense mismatches on opposite sides of the
// league average, ELWAY-vs-market total disagreements, and games where weather is worth
// real points. All three reuse data already computed elsewhere (team_ratings.py's weekly
// snapshot/matchups, the elway_odds/odds tables) -- nothing new is fetched for this tab.

// Reads js/power.js's shared classifyTeams()/buildTiers() -- loaded before this file in
// index.html -- so the Power Rankings tab and this one agree on exactly what "Elite"/
// "Above average"/"Below average"/"Weak" mean. The frozen weekly snapshot
// (ctx.parlaySnapshot, nflhub.sources.team_ratings._refresh_parlay_snapshot) is used here
// INSTEAD of the live ctx.teamRatings numbers specifically so a matchup flagged before
// kickoff doesn't quietly stop qualifying once that same game finishes and gets folded into
// team_ratings' own always-moving EWMA.
const ABOVE_GROUPS = new Set(['elite', 'above']);
const BELOW_GROUPS = new Set(['below', 'weak']);
const GROUP_LABELS = { elite: 'Elite', above: 'Above average', below: 'Below average', weak: 'Weak' };
// Elite/Weak count double for sort purposes -- an Elite-offense-vs-Weak-defense pairing
// reads as a bigger story than an Above-average-vs-Below-average one, even though both
// qualify as "opposite sides of average".
const GROUP_STRENGTH = { elite: 2, above: 1, below: 1, weak: 2 };

// Every one of this week's games where one side's group (from the frozen snapshot) is on
// the opposite side of the league average from the other -- Elite/Above-average offense
// vs. Below-average/Weak defense, or the reverse. No top-N cap: every qualifying game shows,
// same "call out all of them" rule as the ELWAY-vs-total section below. A team in the
// "Average" group never qualifies either side -- that's the whole point of the band.
function phaseMismatches(ctx, phase) {
  const snap = ctx.parlaySnapshot;
  if (!snap || !snap.teams) return [];
  const offMetric = `${phase}_off_epa`, defMetric = `${phase}_def_epa_allowed`;
  const toVals = metric => Object.entries(snap.teams)
    .filter(([, r]) => r[metric] != null)
    .map(([team, r]) => ({ team, value: r[metric] }));
  const offVals = toVals(offMetric), defVals = toVals(defMetric);
  if (offVals.length < 4 || defVals.length < 4) return [];
  const offGroupOf = classifyTeams(offVals).groupOf;
  const defGroupOf = classifyTeams(defVals).groupOf;

  const rows = [];
  (ctx.games || []).forEach(g => {
    [[g.home, g.away], [g.away, g.home]].forEach(([offTeam, defTeam]) => {
      const offGroup = offGroupOf.get(offTeam);
      const defGroup = defGroupOf.get(defTeam);
      if (!offGroup || !defGroup) return;
      const favorsOffense = ABOVE_GROUPS.has(offGroup) && BELOW_GROUPS.has(defGroup);
      const favorsDefense = BELOW_GROUPS.has(offGroup) && ABOVE_GROUPS.has(defGroup);
      if (favorsOffense || favorsDefense) {
        rows.push({ g, offTeam, defTeam, offGroup, defGroup, favorsOffense });
      }
    });
  });
  rows.sort((a, b) => (GROUP_STRENGTH[b.offGroup] + GROUP_STRENGTH[b.defGroup])
    - (GROUP_STRENGTH[a.offGroup] + GROUP_STRENGTH[a.defGroup]));
  return rows;
}

// "TEAM phase offense (Elite) vs OPPONENT phase defense (Weak)" -- the same group vocabulary
// as the Power Rankings distribution panel, not a raw EPA number or rank.
function mismatchSentence(phase, row) {
  return `${row.offTeam} ${phase} offense (${GROUP_LABELS[row.offGroup]}) vs `
    + `${row.defTeam} ${phase} defense (${GROUP_LABELS[row.defGroup]})`;
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
      const rows = phaseMismatches(ctx, phase);
      if (!ctx.parlaySnapshot) return `<p class="muted small">No frozen ratings snapshot yet for this week.</p>`;
      if (!rows.length) return `<p class="muted small">No ${label.toLowerCase()} matchup on opposite sides of the average this week.</p>`;
      return `<ol>${rows.map(row =>
        `<li>${esc(mismatchSentence(phase, row))}
           <span class="muted small">(${esc(row.g.away)} @ ${esc(row.g.home)}, ${fmtLocal(row.g.kickoff, false)})</span></li>`
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
        <p class="tablefoot muted">Every matchup where one side's group is on the opposite
          side of the league average from the other — <strong class="result-good">Elite</strong>/
          <strong class="mild-good">Above average</strong> meeting
          <strong class="mild-bad">Below average</strong>/<strong class="result-bad">Weak</strong>
          — using the same Elite/Above/Average/Below/Weak groups as the Power Rankings
          distribution panel. Frozen at this week's first kickoff, so a matchup called out
          here stays put even after the game it describes is final — descriptive only, not a
          betting edge.</p>
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
