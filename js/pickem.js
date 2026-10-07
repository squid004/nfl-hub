'use strict';

// Two modes, both rendered as one card per game (same visual format -- see _renderCards()/
// _renderAtsCards()):
//   'ml'  -> Moneyline Pick'em: pick the outright winner. Market/ELWAY/History/Power Model/
//            Pool-edge stat blocks (pool-leverage only exists for ML, not ATS).
//   'ats' -> Against the Spread: pick the side that covers. Market/ELWAY/History/Power
//            Ranking stat blocks, each stating that source's own spread + O/U instead of a
//            SU winner -- no pool-edge block (that system is ML-only). Power Ranking's
//            spread/O&U comes from team_ratings.py's score-distribution pipeline (research/
//            edge_signal_test_v32/v33_score_distribution*.py) -- each side's own MODE (most
//            likely single score), not its mean, per explicit user direction; the details
//            dropdown plots the full distribution.
const PICK_MODES = {
  ml: {
    id: 'pickem', picksKey: 'pPicks', act: 'pick', setter: 'setPickemPick',
    title: 'Moneyline Pick’em', dogChip: 'upset',
  },
  ats: {
    id: 'atspickem', picksKey: 'aPicks', act: 'atspick', setter: 'setAtsPick',
    title: 'Against the Spread', dogChip: 'dog',
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

// Just ELWAY's favorite + its win probability — the two-number "away% / home%" display told
// you nothing extra once you already know who's favored (they're complements), so collapse to
// one, same idea as the Market block's own single `favLabel`.
function elwayFavLabel(el, home, away) {
  if (!el || el.home_win_prob == null || el.away_win_prob == null) return null;
  const team = el.home_win_prob > el.away_win_prob ? home : away;
  const p = Math.max(el.home_win_prob, el.away_win_prob);
  return { team, pct: p };
}

// Worst QB-position entry for a team from ESPN's injury report (kv 'injuries', written by
// refresh.py), or null if that team's QB(s) are all "Active"/unlisted. Ranked so "Out"/"IR"
// always outranks "Doubtful" outranks "Questionable" regardless of report order.
const QB_SEVERITY = { Out: 3, 'Injured Reserve': 3, Doubtful: 2, Questionable: 1 };
function qbFlag(team, injuries) {
  const qbs = ((injuries || {})[team] || []).filter(i => i.position === 'QB' && QB_SEVERITY[i.status]
    // "(coach's decision)" inactives are healthy roster scratches (e.g. emergency #3 QB),
    // not real injuries -- ESPN's feed lumps them into the same report, which would
    // otherwise red-flag a fine backup while saying nothing about the actual starter.
    && !/coach'?s decision/i.test(i.detail || ''));
  if (!qbs.length) return null;
  return qbs.reduce((worst, i) => QB_SEVERITY[i.status] > QB_SEVERITY[worst.status] ? i : worst);
}
// A backup QB's injury only actually matters once he's next in line to play -- i.e. every
// QB ranked ahead of him on the REAL depth chart (kv 'qb_depth_chart', ESPN's own ordered
// depth chart, index 0 = starter) is also hurt. Walk the depth chart in order and stop at
// the first healthy (unflagged) name -- everyone before that point is returned, everyone
// after is irrelevant noise (a deep QB3's unrelated tweak behind two healthy QBs ahead of
// him, e.g.). Matches injuries entries to depth-chart names by exact string equality --
// both come from the same ESPN athlete.displayName field (verified live, suffixes like
// "Jr."/"Sr." included on both sides), unlike elwayStaleQb below which has to fall back to
// last-name matching against ELWAY's sheet (a genuinely different, surname-only source).
// Replaced an earlier heuristic that only had ELWAY's single assumed-starter guess to work
// from, not a real depth chart, and still produced wrong calls.
function starterQbFlags(team, injuries, depthChart) {
  const order = (depthChart || {})[team] || [];
  if (!order.length) return [];
  const flagged = new Map();
  ((injuries || {})[team] || []).forEach(i => {
    if (i.position === 'QB' && QB_SEVERITY[i.status] && !/coach'?s decision/i.test(i.detail || '')) {
      flagged.set(i.player, i);
    }
  });
  const chain = [];
  for (const name of order) {
    const hit = flagged.get(name);
    if (!hit) break;
    chain.push(hit);
  }
  return chain;
}
function qbChip(team, injuries, depthChart) {
  const qbs = starterQbFlags(team, injuries, depthChart);
  if (!qbs.length) return '';
  return qbs.map(qb => {
    const cls = QB_SEVERITY[qb.status] >= 3 ? 'bad' : 'warn';
    return ` <span class="chip ${cls}" title="${esc(qb.detail || '')}">QB: ${esc(qb.player)} (${qb.status})</span>`;
  }).join('');
}

// True (with details) when ELWAY's "Current Rankings" tab is still evaluating `team` with
// a QB1 who our own live injury feed now shows as Out/Doubtful/Questionable -- i.e. an
// injury that broke after the user's last weekly sheet transcription, which ELWAY's own
// depth-chart tracking can't self-correct for until next update. Matches by last name
// (the sheet only gives a surname) against qbFlag's own worst-QB-entry pick, so this only
// fires when the flagged player IS the QB ELWAY is actually using, not some other backup.
function elwayStaleQb(team, elwayQb1, injuries) {
  const assumed = (elwayQb1 || {})[team];
  if (!assumed) return null;
  const flagged = qbFlag(team, injuries);
  if (!flagged) return null;
  const flaggedLast = flagged.player.trim().split(/\s+/).pop().toLowerCase();
  if (assumed.name.toLowerCase() !== flaggedLast) return null;
  return { assumedName: assumed.name, status: flagged.status, detail: flagged.detail };
}

// Fallback for when the live odds feed has gone blank for a game that's underway or over
// (ESPN's scoreboard odds field frequently does this once a game starts) -- refresh.py now
// freezes the odds table's spread/total/moneyline at kickoff going forward (stops upserting
// once a game leaves "pre"), but that can't recover a game whose live row already got
// overwritten with nulls before that fix existed. For the spread specifically, we don't need
// an outside source to recover it: spread_history already has our own timestamped snapshots,
// so the latest one at or before kickoff IS the frozen pre-game line.
function frozenSpread(rawSpread, rows, kickoff) {
  if (rawSpread != null) return rawSpread;
  if (!rows || !rows.length) return null;
  const ko = Date.parse(kickoff);
  const before = rows.filter(r => Date.parse(r.captured_at) <= ko);
  return (before.length ? before[before.length - 1] : rows[0]).spread_home;
}

// Net spread movement this week from spread_history (ascending list of {spread_home,
// captured_at}), home-spread signed. "steam" = the line has moved >=1.5 pts net in one
// direction since the week's first snapshot — a sharp-money signal distinct from ELWAY.
function lineMovement(rows) {
  if (!rows || rows.length < 2) return null;
  const open = rows[0].spread_home, cur = rows[rows.length - 1].spread_home;
  const deltaHome = cur - open;
  return { open, cur, deltaHome, steam: Math.abs(deltaHome) >= 1.5 };
}

// Every nonzero net move gets a chip now, not just ones that clear the steam threshold --
// green at/above 1.5 pts (the threshold "steam" means something by), plain/colorless below
// it but still shown, so a small move isn't hidden, just not called out as sharp money.
// Shows the signed delta itself (home-spread convention: negative = home favored) rather
// than just a yes/no flag, so the magnitude is visible at a glance. Pinned to the right edge
// of the card header (.line-move-chip, margin-left:auto) via CSS, last in header markup order.
function lineMoveChip(move) {
  if (!move || move.deltaHome === 0) return '';
  const mag = Math.abs(move.deltaHome);
  const label = (move.deltaHome > 0 ? '+' : '') + move.deltaHome.toFixed(1);
  const steam = mag >= 1.5;
  const title = `Line opened ${signed(move.open)}, now ${signed(move.cur)} — a ${mag.toFixed(1)}-point move this week`;
  return ` <span class="chip line-move-chip${steam ? ' good' : ''}" title="${title}">Steam: ${label}</span>`;
}

// EPA ratings + rule-based mismatch callouts for one game, from the team_ratings kv blob
// (nflhub/sources/team_ratings.py -- rush/pass offense+defense EPA/play, garbage time
// excluded). Backtesting in that repo's research/ found this does NOT beat the closing
// market spread, so it's shown as descriptive context alongside the market/ELWAY/history
// blocks, not as its own prediction.
function matchupFor(teamRatings, home, away) {
  if (!teamRatings) return null;
  return (teamRatings.matchups || {})[`${away}@${home}`] || null;
}

// Phase -> [offense metric key, defense-allowed metric key]. Each matchup row pairs ONE
// team's offense directly against the OTHER team's defense in the same cell, since that's
// the actual matchup -- a table of each team's own 4 stats side by side (the old layout)
// made you jump between rows/columns to compare the two numbers that actually face off.
const PHASES = [['rush', 'Rush'], ['pass', 'Pass']];

// Header chip only fires for weather bad enough to matter -- the projected score already
// carries the weather adjustment (and says so) regardless, this is just the at-a-glance flag,
// so merely breezy/damp conditions with no real scoring impact shouldn't earn a chip.
// Thresholds match this project's own weather research (research/weather_scoring_analysis.py /
// edge_signal_test_v10_weather.py).
function isBadWeather(wx) {
  return !!wx && (wx.wind_mph >= 20 || wx.precip_mm >= 10 || wx.cold_flag || wx.snow_in > 0);
}

function weatherChip(wx) {
  if (!isBadWeather(wx)) return '';
  return ` <span class="chip bad" title="${esc(forecastSummary(wx))}">Bad Weather</span>`;
}

// Full forecast for the card's details layer -- everything fetch_forecast_weather() returns,
// not just the 3 fields the points-prediction adjustment actually uses.
function forecastSummary(wx) {
  if (!wx) return '';
  const parts = [];
  if (wx.conditions) parts.push(wx.conditions);
  if (wx.temp_hi_f != null) parts.push(`${Math.round(wx.temp_lo_f)}–${Math.round(wx.temp_hi_f)}°F`);
  parts.push(`${Math.round(wx.wind_mph)}mph wind`);
  if (wx.precip_mm > 0) parts.push(`${(wx.precip_mm / 25.4).toFixed(2)}in precip${wx.precip_chance != null ? ` (${wx.precip_chance}% chance)` : ''}`);
  else if (wx.precip_chance) parts.push(`${wx.precip_chance}% chance of precip`);
  if (wx.snow_in > 0) parts.push(`${wx.snow_in}in snow`);
  return parts.join(', ');
}

// Not currently rendered anywhere -- EPA table hidden for now, kept intact for an easy
// re-add rather than deleted outright.
function matchupTableHtml(matchup, home, away) {
  if (!matchup) return '<div class="muted small">No rating data yet.</div>';
  const hr = matchup.home_ratings_display || {};
  const ar = matchup.away_ratings_display || {};
  const fmt = v => v != null ? v.toFixed(0) : '—';
  const cell = (offTeam, offR, phase, defTeam, defR) => {
    const off = offR[`${phase}_off_epa`], def = defR[`${phase}_def_epa_allowed`];
    return `<td class="num" title="${offTeam} ${phase} offense ${fmt(off)}/100 vs ${defTeam} ${phase} defense ${fmt(def)}/100 (both 0-100, 100 = best ever recorded in the 2007-present dataset)">${fmt(off)} vs ${fmt(def)}</td>`;
  };
  const row = (offTeam, offR, defTeam, defR) => `<tr>
    <td>${offTeam} off &rarr; ${defTeam} def</td>
    ${PHASES.map(([phase]) => cell(offTeam, offR, phase, defTeam, defR)).join('')}
  </tr>`;
  return `<table class="mini-ratings matchup-matrix">
    <thead><tr><th></th>${PHASES.map(([, label]) => `<th class="num">${label} (off vs def)</th>`).join('')}</tr></thead>
    <tbody>${row(away, ar, home, hr)}${row(home, hr, away, ar)}</tbody>
  </table>`;
}

// Weather-adjusted projected score + the forecast itself -- an O/U-flavored mechanism (it
// only ever moves the predicted POINT TOTAL, never who's favored). Lives on the Against the
// Spread tab's Details only; the "Bad Weather" CHIP (header-level, see weatherChip() above)
// still shows on Moneyline Pick'em too since that's useful SU context on its own.
function forecastHtml(matchup, home, away) {
  if (!matchup) return '<div class="muted small">No forecast data yet.</div>';
  const ap = matchup.predicted_away_points, hp = matchup.predicted_home_points;
  const wx = matchup.weather;
  const wxNote = wx ? ' (weather adjustment applied)' : '';
  const proj = (ap != null && hp != null)
    ? `<div class="muted small" title="Non-negative L2-regularized regression on own offense vs. opponent defense (same 8 stats), walk-forward validated 2007-2025, plus a separate wind/rain/extreme-cold adjustment (also walk-forward validated, applied only for outdoor stadiums within the ~16-day forecast window) when available. Less accurate than the market total at predicting points -- a second data point, not a replacement.">Projected: ${away} ${ap.toFixed(1)} – ${home} ${hp.toFixed(1)}${wxNote}</div>`
    : '';
  const forecast = wx ? `<div class="muted small">Forecast: ${esc(forecastSummary(wx))}</div>` : '';
  return (forecast + proj) || '<div class="muted small">No forecast data yet.</div>';
}

// Rule-based (not AI-generated) explanation: every clause traces to a real number already
// on the card, so it's reproducible and never invents anything. Deliberately terse.
function buildNarrative({ favTeam, dogTeam, marketMargin, el, histCell, bucketLabel, bucketRank, edgeRec, matchupCallouts }) {
  const parts = [`${favTeam} favored by ${marketMargin} over ${dogTeam}.`];

  if (el && el.home_win_prob != null && el.away_win_prob != null) {
    const elwayFav = el.home_win_prob > el.away_win_prob ? 'home' : 'away';
    const elwayFavTeam = elwayFav === 'home' ? el._home : el._away;
    if (elwayFavTeam === favTeam) {
      parts.push(`ELWAY agrees, projecting ${favTeam} by about ${Math.abs(el.spread_home).toFixed(1)}.`);
    } else {
      parts.push(`ELWAY disagrees — its avg-points model actually likes ${elwayFavTeam} `
        + `against the market's lean toward ${favTeam}.`);
    }
  }

  if (histCell && histCell.su != null) {
    const suPct = Math.round(histCell.su * 100);
    parts.push(suPct < 55
      ? `Favorites ${bucketLabel} have only won ${suPct}% of the time historically — a live spot for an upset.`
      : `Favorites ${bucketLabel} have won ${suPct}% of the time historically.`);
  }

  if (bucketRank && bucketRank.taken) {
    parts.push(`This week's upset-budget model flags ${dogTeam} as one of its picks from this bucket.`);
  }

  if (edgeRec && edgeRec.recommendation === 'FADE') {
    const fPct = edgeRec.f_estimate != null ? Math.round(edgeRec.f_estimate * 100) : null;
    parts.push(`Pool-leverage likes fading ${favTeam} here too — only ~${fPct}% of your pool is expected `
      + `on ${dogTeam}, so it pays off if it hits.`);
  } else if (edgeRec && edgeRec.recommendation === 'CHALK') {
    parts.push(`Pool-leverage says stick with the chalk here — not enough separation to justify fading ${favTeam}.`);
  }

  if (matchupCallouts && matchupCallouts.length) {
    parts.push(...matchupCallouts);
  }

  return parts.join(' ');
}

const Pickem = {
  render(ctx, mode = 'ml') {
    if (mode === 'ml') this._renderCards(ctx);
    else this._renderAtsCards(ctx);
  },

  _renderCards(ctx) {
    const M = PICK_MODES.ml;
    const { week, games, odds, hist } = ctx;
    const picks = ctx[M.picksKey] || {};
    const edgeLog = ctx.edgeLog || {};
    const elway = ctx.elway || {};
    const injuries = ctx.injuries || {};
    const spreadHist = ctx.spreadHist || {};
    const elwayQb1 = ctx.elwayQb1 || {};
    const depthChart = ctx.qbDepthChart || {};
    const teamRatings = ctx.teamRatings || null;
    const bucketRanks = computeBucketRanks(ctx);
    const records = (teamRatings && teamRatings.team_records) || {};
    const recordLabel = team => { const r = records[team]; return r ? ` <span class="team-record">(${r.wins}-${r.losses}${r.ties ? '-' + r.ties : ''})</span>` : ''; };

    const cards = games.map(g => {
      const o = odds[g.game_id] || {};
      const matchup = matchupFor(teamRatings, g.home, g.away);
      const pm = matchup && matchup.power_model;
      const move = lineMovement(spreadHist[g.game_id]);
      const staleHome = elwayStaleQb(g.home, elwayQb1, injuries);
      const staleAway = elwayStaleQb(g.away, elwayQb1, injuries);
      const elRaw = elway[g.game_id];
      const el = elRaw ? { ...elRaw, _home: g.home, _away: g.away } : null;
      const elwayFav = elwayFavLabel(el, g.home, g.away);
      const effSpread = frozenSpread(o.spread, spreadHist[g.game_id], g.kickoff);
      const hasLine = effSpread != null;
      const favTeam = !hasLine ? null : (effSpread <= 0 ? g.home : g.away);
      const dogTeam = favTeam == null ? null : (favTeam === g.home ? g.away : g.home);
      // Favorite's own line always shown negative (betting convention), regardless of
      // which side it is -- effSpread itself is HOME-spread-signed (negative = home
      // favored), so an away favorite needs the sign flipped here.
      const favLabel = !hasLine ? '—' : (effSpread === 0 ? 'Pick’em' : `${favTeam} ${-Math.abs(effSpread)}`);
      const mine = (picks[g.game_id] || {}).pick;
      const mineIsDog = mine && dogTeam && mine === dogTeam;

      let result = null;
      if (g.state === 'post') {
        result = g.home_score > g.away_score ? g.home
          : g.away_score > g.home_score ? g.away : 'TIE';
      }
      const stateChip = g.state === 'in' ? '<span class="chip">LIVE</span>' : '';

      const histCell = hist && hasLine ? History.lookup(hist, Math.abs(effSpread), effSpread <= 0) : null;

      // Grading, once the game is final (null = no verdict -- not final yet, or nothing to
      // compare). A tie grades neither side right nor wrong.
      const postGame = g.state === 'post';
      const elwayHit = postGame && elwayFav && result !== 'TIE' ? elwayFav.team === result : null;
      const histHit = postGame && histCell && favTeam && result !== 'TIE' ? favTeam === result : null;
      const pmHit = postGame && pm && pm.favorite && result !== 'TIE' ? pm.favorite === result : null;
      const pickHit = postGame && mine && result !== 'TIE' ? mine === result : null;
      const resultClass = v => v === true ? 'result-good' : v === false ? 'result-bad' : '';
      // Once final, the line/implied-% stop mattering -- show the actual final score instead,
      // winner bolded green.
      const scorePart = (team, score) => result === team
        ? `<span class="result-good">${team} ${score}</span>` : `${team} ${score}`;

      const bucketRank = bucketRanks[g.game_id] || null;
      // Prefer the bucket the ranking itself used (computeBudget prefers the cross-book
      // average spread when available) over recomputing from o.spread alone — otherwise
      // the badge could name a different bucket than the rank next to it was computed in.
      const bucketLabel = bucketRank ? bucketRank.lab : (hasLine ? History.bucketLabel(Math.abs(effSpread)) : null);
      const edgeRec = edgeLog[g.game_id];
      const hasEdge = edgeRec && edgeRec.recommendation && edgeRec.recommendation !== 'NO_DATA';
      const elwayFlip = hasLine && elwayFullDisagree(el, g.home, g.away, favTeam);

      // Which side to highlight as "the suggested pick": prefer the pool-specific
      // leverage call (FADE/CHALK) when it actually has data this week, otherwise fall
      // back to the upset-budget flag (same source as the "budget dog" chip above).
      const suggestedIsDog = hasLine && (hasEdge ? edgeRec.recommendation === 'FADE' : !!(bucketRank && bucketRank.taken));
      const suggestedTeam = hasLine ? (suggestedIsDog ? dogTeam : favTeam) : null;
      const teamSpan = team => {
        if (team !== suggestedTeam) return team;
        return `<span class="${suggestedIsDog ? 'pick-dog' : 'pick-fav'}">${team}</span>`;
      };

      const btn = team => `<button data-act="${M.act}" data-week="${week}" data-game="${g.game_id}"
        data-team="${team}" data-spread="${hasLine ? effSpread : ''}"
        class="${mine === team ? 'primary' : ''}">${team}</button>`;

      const narrative = hasLine
        ? buildNarrative({ favTeam, dogTeam, marketMargin: Math.abs(effSpread), el, histCell,
                           bucketLabel, bucketRank, edgeRec: hasEdge ? edgeRec : null,
                           matchupCallouts: matchup ? matchup.callouts : null })
        : 'No market line yet for this game.';

      const edgeLabel = edgeRec && edgeRec.recommendation === 'NO_PLAY' ? 'NO PLAY' : edgeRec?.recommendation;
      const edgeChip = hasEdge
        ? `<span class="chip ${edgeRec.recommendation === 'FADE' ? 'good' : ''}"
             title="p=${(edgeRec.p_favorite * 100).toFixed(1)}%, f=${edgeRec.f_estimate != null ? Math.round(edgeRec.f_estimate * 100) + '%' : '—'}, leverage=${edgeRec.leverage != null ? edgeRec.leverage.toFixed(3) : '—'}">
             ${edgeLabel}${edgeRec.recommendation === 'FADE' ? ' ' + edgeRec.underdog_team : ''}</span>`
        : '<span class="muted">—</span>';

      return `<div class="game-card${pickHit === true ? ' result-win' : pickHit === false ? ' result-loss' : ''}">
        <div class="game-card-head">
          <span class="muted">${fmtLocal(g.kickoff, false)}</span>
          <span class="matchup">${teamSpan(g.away)}${recordLabel(g.away)}${qbChip(g.away, injuries, depthChart)} @ ${teamSpan(g.home)}${recordLabel(g.home)}${qbChip(g.home, injuries, depthChart)}</span>
          ${stateChip}
          ${elwayFlip ? `<span class="chip warn" title="ELWAY's avg-points model favors the OTHER team entirely, not just by a smaller or larger margin">ELWAY flip</span>` : ''}
          ${[[staleHome, g.home], [staleAway, g.away]].filter(([s]) => s).map(([s, t]) =>
            `<span class="chip bad" title="ELWAY's rating still assumes ${s.assumedName} at QB1 for ${t}, but our injury report lists him ${s.status}: ${esc(s.detail || '')}">ELWAY stale QB (${t})</span>`
          ).join('')}
          ${matchup ? weatherChip(matchup.weather) : ''}
          ${lineMoveChip(move)}
        </div>
        ${hasLine ? `<div class="market-line">${esc(o.book ?? 'Market')}: ${favLabel}</div>` : ''}
        <div class="game-card-body">
          <div class="stat-block">
            <div class="stat-label">Market</div>
            ${postGame
              ? `<div>${scorePart(g.away, g.away_score)} - ${scorePart(g.home, g.home_score)}</div>`
              : `<div>${favTeam} ${pct(favTeam === g.home ? o.implied_home : o.implied_away)}</div>
                 ${move ? `<div class="muted small">opened ${signed(move.open)}</div>` : ''}`}
          </div>
          <div class="stat-block">
            <div class="stat-label">ELWAY</div>
            <div class="${resultClass(elwayHit)}">${elwayFav ? `${elwayFav.team} ${pct(elwayFav.pct)}` : '<span class="muted">—</span>'}</div>
            <div class="muted">${el && el.spread_home != null ? `implied ${signed(el.spread_home)}` : ' '}</div>
          </div>
          <div class="stat-block">
            <div class="stat-label">History</div>
            <div class="${resultClass(histHit)}">${histCell ? `${favTeam} ${pct(histCell.su)}` : '<span class="muted">—</span>'}</div>
            <div class="muted">${bucketLabel ? `${bucketLabel} bucket` : ' '}</div>
          </div>
          <div class="stat-block">
            <div class="stat-label" title="The Power Rankings composite (8 EPA/points/turnover stats) applied to this matchup, calibrated into a win probability from backtested accuracy at that confidence level (QB-injury games excluded from calibration). Backtesting found this does NOT beat the market -- a third data point alongside ELWAY/History, not a replacement.">Power Model</div>
            <div class="${resultClass(pmHit)}">${pm && pm.favorite ? `${pm.favorite} ${pct(pm.prob)}` : '<span class="muted">&mdash;</span>'}</div>
            <div class="muted">${pm ? `&Delta; ${pm.delta >= 0 ? '+' : ''}${pm.delta.toFixed(2)}` : ' '}</div>
          </div>
          <div class="stat-block">
            <div class="stat-label">Pool edge</div>
            <div>${edgeChip}</div>
          </div>
        </div>
        <div class="game-card-pick">
          ${btn(g.away)} ${btn(g.home)}
          <span class="game-card-mine">${mine ? `Your pick: <strong>${mine}</strong>${mineIsDog ? ` <span class="chip warn">${M.dogChip}</span>` : ''}` : '<span class="muted">No pick yet</span>'}</span>
        </div>
        <details class="game-card-details">
          <summary>Details</summary>
          <p class="game-card-narrative">${esc(narrative)}</p>
        </details>
      </div>`;
    }).join('');

    const made = Object.keys(picks).length;
    document.getElementById(M.id).innerHTML = `
      <div class="panel">
        <h2>${M.title} &mdash; ${made}/${games.length} made
          &middot; locks ${games.length ? fmtLocal(games[0].kickoff) : 'TBD'}</h2>
        <div class="game-card-grid">${cards}</div>
        <details>
          <summary class="muted small">Chip legend</summary>
          <p class="tablefoot muted"><span class="pick-fav">Green</span> in the matchup =
            the favorite is the suggested pick; <span class="pick-dog">yellow</span> = the
            dog is — pool-leverage's FADE/CHALK call when it has data this week, otherwise
            the upset-budget flag. ELWAY is Nate Silver's Silver Bulletin NFL forecasting
            model, transcribed weekly into a Google Sheet — not a personal formula. A
            <span class="chip warn">QB</span> chip shows that team's QB injury situation (red
            = Out/IR) regardless of what ELWAY currently assumes at QB1 — if ELWAY's assumed
            starter is the one flagged, another flagged QB for that team shows too (the
            presumptive next man up); if ELWAY has already moved QB1 off an injured player (or
            we don't know who ELWAY has at QB1), this falls back to the worst OTHER flagged QB
            at Doubtful/Out severity only (a merely "Questionable" QB2/QB3 unrelated to a fine
            starter doesn't get a chip; a real starter change does), so a real starter injury
            never goes unshown just because ELWAY's own sheet caught up to it. Any net line
            move this week shows as a "Steam:" chip, pinned to the right of the card header,
            with the signed point move itself (<span class="chip good">Steam: +1.5</span> at/
            above a 1.5-point move in one direction since this week's first snapshot,
            <span class="chip">Steam: +0.5</span> plain below that).
            <span class="chip bad">ELWAY stale QB</span> = ELWAY's weekly sheet is still
            rating that team with a QB1 our live injury feed now shows hurt — a breaking-news
            injury ELWAY's own depth-chart tracking hasn't caught up to yet.
            <span class="chip bad">Bad Weather</span> = live forecast at kickoff (only shown
            for outdoor stadiums within ~16 days out) is bad enough to matter for scoring —
            &ge;20mph wind, &ge;10mm precip, any snow, or sub-20&deg;F highs; shown here as
            context only -- the actual weather-adjusted point forecast lives on the Against
            the Spread tab.
            <span class="stat-label" style="display:inline;text-transform:none;font-weight:600;">Power Model</span>
            is the Power Rankings composite (same 8 EPA/points/turnover stats as the Power
            Rankings tab) applied to this matchup, with its margin (&Delta;) converted into a
            win probability from a calibration curve fit against real backtested accuracy at
            each confidence level (games where either team's QB was Out/Doubtful excluded from
            that fit). Like ELWAY and History, backtesting found it does not beat the market
            spread at picking winners -- a third independent data point, not a replacement.
            Once a game is final, the ELWAY/History/Power Model numbers turn <span class="result-good">green</span>
            if the side they favored actually won or <span class="result-bad">red</span> if it
            didn't, and the whole card gets a light green/red tint if your own pick hit or
            missed. The market line itself is frozen at whatever it was just before kickoff --
            it won't keep changing once the game starts just because the live odds feed does.
            This tab is about picking winners and playing the pool well; point totals and the
            weather-adjusted forecast/ratings breakdown live on the Against the Spread tab
            instead.</p>
        </details>
      </div>`;
  },

  // Score-distribution bar chart for one ATS card's details dropdown -- lazy-rendered on
  // first <details> open (not eagerly for all ~14-16 games every reload): Plotly can't size
  // itself correctly into a hidden (closed <details>) container, and there's no reason to
  // pay for charts nobody expands anyway.
  _drawDistChart(divId, sd, home, away) {
    const el = document.getElementById(divId);
    if (!el || el._distDrawn) return;
    el._distDrawn = true;
    const cs = getComputedStyle(document.documentElement);
    const cssVar = name => cs.getPropertyValue(name).trim();
    const colors = { text: cssVar('--text'), dim: cssVar('--dim'), line: cssVar('--line'),
                      panel: cssVar('--panel2'), home: cssVar('--accent'), away: cssVar('--warn') };
    // Away mirrored onto negative x, home on positive x -- separates the two curves instead
    // of overlapping them. Points can never actually be negative, so the sign here is purely
    // a left/right layout trick; tickvals/ticktext below relabel every tick back to its real
    // (always non-negative) point value so nothing reads as an actual negative score.
    const trace = (pts, label, mode, color, sign) => ({
      name: `${label} (mode ${mode})`,
      x: pts.map(d => sign * d.points), y: pts.map(d => d.pct),
      customdata: pts.map(d => d.points), type: 'bar', opacity: 0.75,
      marker: { color }, hovertemplate: `${label} %{customdata} pts: %{y}%<extra></extra>`,
    });
    const maxPts = Math.max(...sd.home_pct.map(d => d.points), ...sd.away_pct.map(d => d.points));
    const tickMax = Math.ceil((maxPts + 1) / 10) * 10;
    const tickvals = [], ticktext = [];
    for (let v = -tickMax; v <= tickMax; v += 10) { tickvals.push(v); ticktext.push(String(Math.abs(v))); }
    Plotly.react(divId, [trace(sd.away_pct, away, sd.away_mode, colors.away, -1), trace(sd.home_pct, home, sd.home_mode, colors.home, 1)], {
      autosize: true, margin: { l: 44, r: 12, t: 8, b: 36 },
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
      font: { color: colors.text, size: 11 },
      xaxis: { title: { text: `${away} ←  points  → ${home}` }, gridcolor: colors.line, color: colors.dim,
               tickfont: { size: 10 }, tickvals, ticktext, zeroline: true, zerolinecolor: colors.dim },
      yaxis: { title: { text: '% chance' }, gridcolor: colors.line, color: colors.dim, tickfont: { size: 10 } },
      legend: { orientation: 'h', x: 0, y: 1.15, font: { color: colors.text, size: 11 } },
      hoverlabel: { bgcolor: colors.panel, bordercolor: colors.line, font: { color: colors.text, size: 11 } },
    }, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] });
  },

  _renderAtsCards(ctx) {
    const M = PICK_MODES.ats;
    const { week, games, odds, hist } = ctx;
    const picks = ctx[M.picksKey] || {};
    const elway = ctx.elway || {};
    const injuries = ctx.injuries || {};
    const spreadHist = ctx.spreadHist || {};
    const elwayQb1 = ctx.elwayQb1 || {};
    const depthChart = ctx.qbDepthChart || {};
    const teamRatings = ctx.teamRatings || null;
    const records = (teamRatings && teamRatings.team_records) || {};
    const recordLabel = team => { const r = records[team]; return r ? ` <span class="team-record">(${r.wins}-${r.losses}${r.ties ? '-' + r.ties : ''})</span>` : ''; };

    // Favorite's own line always shown negative (betting convention) -- spreadHome is
    // HOME-spread-signed, so an away favorite needs the sign flipped here.
    const spreadFavLabel = (spreadHome, home, away) =>
      spreadHome == null ? null : { team: spreadHome <= 0 ? home : away, label: spreadHome === 0 ? 'Pick’em' : `${spreadHome <= 0 ? home : away} ${-Math.abs(spreadHome)}` };

    const chartsToWire = [];

    const cards = games.map(g => {
      const o = odds[g.game_id] || {};
      const matchup = matchupFor(teamRatings, g.home, g.away);
      const sd = matchup && matchup.score_distribution;
      const move = lineMovement(spreadHist[g.game_id]);
      const staleHome = elwayStaleQb(g.home, elwayQb1, injuries);
      const staleAway = elwayStaleQb(g.away, elwayQb1, injuries);
      const elRaw = elway[g.game_id];
      const el = elRaw ? { ...elRaw, _home: g.home, _away: g.away } : null;
      const effSpread = frozenSpread(o.spread, spreadHist[g.game_id], g.kickoff);
      const hasLine = effSpread != null;
      const favTeam = !hasLine ? null : (effSpread <= 0 ? g.home : g.away);
      const dogTeam = favTeam == null ? null : (favTeam === g.home ? g.away : g.home);
      const favLabel = !hasLine ? '—' : (effSpread === 0 ? 'Pick’em' : `${favTeam} ${-Math.abs(effSpread)}`);
      const mine = (picks[g.game_id] || {}).pick;
      const mineIsDog = mine && dogTeam && mine === dogTeam;

      const stateChip = g.state === 'in' ? '<span class="chip">LIVE</span>' : '';

      // ATS grading is "did this side COVER the market spread," not who won SU. PUSH grades
      // no side right or wrong, same spirit as SU's TIE handling on the Moneyline cards.
      let result = null;
      if (g.state === 'post' && hasLine) {
        const favMargin = (effSpread <= 0 ? 1 : -1) * (g.home_score - g.away_score);
        const edge = favMargin - Math.abs(effSpread);
        result = Math.abs(edge) < 1e-9 ? 'PUSH' : (edge > 0 ? favTeam : dogTeam);
      }
      const postGame = g.state === 'post';

      const elwaySf = el ? spreadFavLabel(el.spread_home, g.home, g.away) : null;
      const prSf = sd ? spreadFavLabel(sd.spread, g.home, g.away) : null;
      const histCell = hist && hasLine ? History.lookup(hist, Math.abs(effSpread), effSpread <= 0) : null;

      const elwayHit = postGame && elwaySf && result !== 'PUSH' ? elwaySf.team === result : null;
      const histHit = postGame && histCell && favTeam && result !== 'PUSH' ? favTeam === result : null;
      const prHit = postGame && prSf && result !== 'PUSH' ? prSf.team === result : null;
      const pickHit = postGame && mine && result !== 'PUSH' ? mine === result : null;
      const resultClass = v => v === true ? 'result-good' : v === false ? 'result-bad' : '';

      const ouLabel = postGame && o.total != null
        ? (g.home_score + g.away_score > o.total ? `Over ${o.total}`
           : g.home_score + g.away_score < o.total ? `Under ${o.total}` : `Push ${o.total}`)
        : `O/U ${o.total ?? '—'}`;
      const scorePart = (team, score) => result === team
        ? `<span class="result-good">${team} ${score}</span>` : `${team} ${score}`;

      const elwayFlip = hasLine && elwayFullDisagree(el, g.home, g.away, favTeam);

      const btn = team => `<button data-act="${M.act}" data-week="${week}" data-game="${g.game_id}"
        data-team="${team}" data-spread="${hasLine ? effSpread : ''}"
        class="${mine === team ? 'primary' : ''}">${team}</button>`;

      const distChartId = `ats-dist-${g.game_id}`;
      if (sd) chartsToWire.push({ id: distChartId, sd, home: g.home, away: g.away });

      return `<div class="game-card${pickHit === true ? ' result-win' : pickHit === false ? ' result-loss' : ''}">
        <div class="game-card-head">
          <span class="muted">${fmtLocal(g.kickoff, false)}</span>
          <span class="matchup">${g.away}${recordLabel(g.away)}${qbChip(g.away, injuries, depthChart)} @ ${g.home}${recordLabel(g.home)}${qbChip(g.home, injuries, depthChart)}</span>
          ${stateChip}
          ${elwayFlip ? `<span class="chip warn" title="ELWAY's avg-points model favors the OTHER team entirely, not just by a smaller or larger margin">ELWAY flip</span>` : ''}
          ${[[staleHome, g.home], [staleAway, g.away]].filter(([s]) => s).map(([s, t]) =>
            `<span class="chip bad" title="ELWAY's rating still assumes ${s.assumedName} at QB1 for ${t}, but our injury report lists him ${s.status}: ${esc(s.detail || '')}">ELWAY stale QB (${t})</span>`
          ).join('')}
          ${matchup ? weatherChip(matchup.weather) : ''}
          ${lineMoveChip(move)}
        </div>
        <div class="game-card-body">
          <div class="stat-block">
            <div class="stat-label">Market</div>
            ${postGame
              ? `<div>${scorePart(g.away, g.away_score)} - ${scorePart(g.home, g.home_score)}</div>
                 <div class="muted">${ouLabel}</div>`
              : `<div>${favLabel}</div>
                 <div class="muted">${ouLabel}</div>`}
          </div>
          <div class="stat-block">
            <div class="stat-label">ELWAY</div>
            <div class="${resultClass(elwayHit)}">${elwaySf ? elwaySf.label : '<span class="muted">—</span>'}</div>
            <div class="muted">${el && el.total != null ? `O/U ${el.total}` : ' '}</div>
          </div>
          <div class="stat-block">
            <div class="stat-label">History</div>
            <div class="${resultClass(histHit)}">${histCell && histCell.ats != null ? `${favTeam} ${pct(histCell.ats)}` : '<span class="muted">—</span>'}</div>
            <div class="muted">${histCell ? 'covers historically' : ' '}</div>
          </div>
          <div class="stat-block">
            <div class="stat-label" title="Each side's most likely single score from team_ratings.py's score-distribution pipeline (predict_points(), bias-corrected for home/away and favorite/underdog, reshaped to match real historical NFL scoring frequency since the 2015 PAT-distance rule change, and calibrated so its implied win probability matches the Power Model pick exactly). Backtesting found predict_points() does NOT beat the market -- descriptive context, not a replacement. Frozen at kickoff, like the market line.">Power Ranking</div>
            <div class="${resultClass(prHit)}">${prSf ? prSf.label : '<span class="muted">—</span>'}</div>
            <div class="muted">${sd ? `O/U ${sd.total}` : ' '}</div>
          </div>
        </div>
        <div class="game-card-pick">
          ${btn(g.away)} ${btn(g.home)}
          <span class="game-card-mine">${mine ? `Your pick: <strong>${mine}</strong>${mineIsDog ? ` <span class="chip warn">${M.dogChip}</span>` : ''}` : '<span class="muted">No pick yet</span>'}</span>
        </div>
        <details class="game-card-details" id="ats-details-${g.game_id}">
          <summary>Details</summary>
          ${matchup ? `<div class="game-card-matchup">
            <div class="stat-label" title="Own-offense-vs-opponent-defense point projection plus the live forecast for outdoor stadiums within the ~16-day window. Descriptive context only -- backtesting found neither beats the market total.">Projected Score + Forecast</div>
            ${forecastHtml(matchup, g.home, g.away)}
          </div>` : ''}
          ${sd ? `<div class="game-card-matchup">
            <div class="stat-label" title="Each side's full predicted score distribution -- raw model curve reshaped to match real historical NFL scoring frequency, calibrated so the implied win probability matches the Power Model pick.">Point Distribution</div>
            <div id="${distChartId}" class="dist-chart"></div>
          </div>` : ''}
        </details>
      </div>`;
    }).join('');

    const made = Object.keys(picks).length;
    document.getElementById(M.id).innerHTML = `
      <div class="panel">
        <h2>${M.title} &mdash; ${made}/${games.length} made
          &middot; locks ${games.length ? fmtLocal(games[0].kickoff) : 'TBD'}</h2>
        <div class="game-card-grid">${cards}</div>
        <details>
          <summary class="muted small">Chip legend</summary>
          <p class="tablefoot muted">Each stat block states that source's own spread + O/U
            (Market/ELWAY/Power Ranking) or historical ATS cover rate (History) instead of a
            straight-up pick. ELWAY is Nate Silver's Silver Bulletin NFL forecasting model,
            transcribed weekly into a Google Sheet. A
            <span class="chip warn">QB</span> chip shows that team's QB injury situation (red
            = Out/IR) regardless of what ELWAY currently assumes at QB1 — if ELWAY's assumed
            starter is the one flagged, another flagged QB for that team shows too (the
            presumptive next man up); if ELWAY has already moved QB1 off an injured player (or
            we don't know who ELWAY has at QB1), this falls back to the worst OTHER flagged QB
            at Doubtful/Out severity only. Any net line move this week shows as a "Steam:"
            chip, pinned to the right of the card header, with the signed point move itself
            (<span class="chip good">Steam: +1.5</span> at/above a 1.5-point move in one
            direction since this week's first snapshot, <span class="chip">Steam: +0.5</span>
            plain below that).
            <span class="chip bad">ELWAY stale QB</span> = ELWAY's weekly sheet is still
            rating that team with a QB1 our live injury feed now shows hurt.
            <span class="chip bad">Bad Weather</span> = live forecast at kickoff is bad enough
            to matter for scoring, shown in each card's Details section regardless.
            <span class="stat-label" style="display:inline;text-transform:none;font-weight:600;">Power Ranking</span>
            states each side's most likely single score (the distribution MODE, not the
            average) from the same score-distribution pipeline charted in each card's Details
            section -- frozen at kickoff, like the market line, so it won't drift during a
            live game. Once a game is final, the ELWAY/History/Power Ranking numbers turn
            <span class="result-good">green</span> if the side they favored actually covered
            or <span class="result-bad">red</span> if it didn't (a push grades neither), the
            whole card gets a light green/red tint if your own pick covered, and O/U swaps to
            the actual Over/Under result. The market line itself is frozen at whatever it was
            just before kickoff.</p>
        </details>
      </div>`;

    // Lazy chart render: a Plotly chart drawn into a closed <details> can't size itself
    // correctly, and there's no reason to render ~15 charts nobody's opened yet. Draw (once)
    // the first time each card's own details element opens.
    chartsToWire.forEach(({ id, sd, home, away }) => {
      const gid = id.replace('ats-dist-', '');
      const details = document.getElementById(`ats-details-${gid}`);
      if (!details) return;
      details.addEventListener('toggle', () => {
        if (details.open) this._drawDistChart(id, sd, home, away);
      });
    });
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
