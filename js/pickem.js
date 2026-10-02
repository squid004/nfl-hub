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
// The injury designation should show whenever the team's own best/starting QB is out,
// independent of whatever ELWAY's sheet currently assumes -- ELWAY updating QB1 to the new
// starter (which it does once its weekly sheet catches up) must NOT make the chip disappear;
// that's a different fact (elwayStaleQb, a separate chip, covers "ELWAY hasn't caught up
// yet"). So: if the assumed starter (elwayQb1) IS one of the flagged QBs, surface that entry
// plus any other flagged QB for the team (the presumptive next man up). Otherwise -- ELWAY's
// assumed starter isn't flagged, either because we don't know who it is or because ELWAY has
// already moved off the injured player -- fall back to the worst OTHER flagged entry, but
// only at Doubtful/Out severity, not Questionable. Checked against real cases: a healthy,
// playing starter commonly has some unrelated QB2/QB3 sitting at "Questionable" for a minor
// or unrelated reason (Aidan O'Connell on personal matters behind a fine Kirk Cousins; Trey
// Lance behind a fine Justin Herbert) -- falling back for those surfaced a meaningless chip.
// Doubtful/Out don't have that problem: both real cases seen (Mayfield Out, C. Williams
// Doubtful) were genuine starter changes, and a bog-standard deep backup rarely earns that
// stronger a tag for no real reason. Questionable still shows when it's the ASSUMED starter
// (branch above) -- it only stops being a trustworthy signal once it's being used as a guess
// about which OTHER flagged QB might matter.
function starterQbFlags(team, injuries, elwayQb1) {
  const qbs = ((injuries || {})[team] || []).filter(i => i.position === 'QB' && QB_SEVERITY[i.status]
    && !/coach'?s decision/i.test(i.detail || ''));
  if (!qbs.length) return [];

  const assumed = (elwayQb1 || {})[team];
  const lastName = p => p.trim().split(/\s+/).pop().toLowerCase();
  const starterEntry = assumed ? qbs.find(i => lastName(i.player) === assumed.name.toLowerCase()) : null;
  if (starterEntry) return [starterEntry, ...qbs.filter(i => i !== starterEntry)];

  const serious = qbs.filter(i => QB_SEVERITY[i.status] >= 2);
  if (!serious.length) return [];
  return [serious.reduce((worst, i) => QB_SEVERITY[i.status] > QB_SEVERITY[worst.status] ? i : worst)];
}
function qbChip(team, injuries, elwayQb1) {
  const qbs = starterQbFlags(team, injuries, elwayQb1);
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

// Net spread movement this week from spread_history (ascending list of {spread_home,
// captured_at}), home-spread signed. "steam" = the line has moved >=1.5 pts net in one
// direction since the week's first snapshot — a sharp-money signal distinct from ELWAY.
function lineMovement(rows) {
  if (!rows || rows.length < 2) return null;
  const open = rows[0].spread_home, cur = rows[rows.length - 1].spread_home;
  const deltaHome = cur - open;
  return { open, cur, deltaHome, steam: Math.abs(deltaHome) >= 1.5 };
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
  const ap = matchup.predicted_away_points, hp = matchup.predicted_home_points;
  const wx = matchup.weather;
  const wxNote = wx ? ' (weather adjustment applied)' : '';
  const proj = (ap != null && hp != null)
    ? `<div class="muted small" title="Non-negative L2-regularized regression on own offense vs. opponent defense (same 8 stats), walk-forward validated 2007-2025, plus a separate wind/rain/extreme-cold adjustment (also walk-forward validated, applied only for outdoor stadiums within the ~16-day forecast window) when available. Less accurate than the market spread at picking winners -- a second data point alongside the market/ELWAY lines, not a replacement.">Projected: ${away} ${ap.toFixed(1)} – ${home} ${hp.toFixed(1)}${wxNote}</div>`
    : '';
  const forecast = wx ? `<div class="muted small">Forecast: ${esc(forecastSummary(wx))}</div>` : '';
  return `<table class="mini-ratings matchup-matrix">
    <thead><tr><th></th>${PHASES.map(([, label]) => `<th class="num">${label} (off vs def)</th>`).join('')}</tr></thead>
    <tbody>${row(away, ar, home, hr)}${row(home, hr, away, ar)}</tbody>
  </table>${forecast}${proj}`;
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
    else this._renderTable(ctx, mode);
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
    const teamRatings = ctx.teamRatings || null;
    const power = (teamRatings && teamRatings.power_rankings) || {};
    const bucketRanks = computeBucketRanks(ctx);
    // College-football-poll-style "#N " prefix from the power ranking (data-driven composite
    // of 8 stats -- see js/power.js / Power Rankings tab). Descriptive only.
    const rankPrefix = team => power[team] ? `<span class="rank-badge" title="Power ranking #${power[team].rank} of ${Object.keys(power).length}">#${power[team].rank}</span> ` : '';

    const cards = games.map(g => {
      const o = odds[g.game_id] || {};
      const matchup = matchupFor(teamRatings, g.home, g.away);
      const move = lineMovement(spreadHist[g.game_id]);
      const staleHome = elwayStaleQb(g.home, elwayQb1, injuries);
      const staleAway = elwayStaleQb(g.away, elwayQb1, injuries);
      const elRaw = elway[g.game_id];
      const el = elRaw ? { ...elRaw, _home: g.home, _away: g.away } : null;
      const elwayFav = elwayFavLabel(el, g.home, g.away);
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

      const histCell = hist && hasLine ? History.lookup(hist, Math.abs(o.spread), o.spread <= 0) : null;
      const bucketRank = bucketRanks[g.game_id] || null;
      // Prefer the bucket the ranking itself used (computeBudget prefers the cross-book
      // average spread when available) over recomputing from o.spread alone — otherwise
      // the badge could name a different bucket than the rank next to it was computed in.
      const bucketLabel = bucketRank ? bucketRank.lab : (hasLine ? History.bucketLabel(Math.abs(o.spread)) : null);
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
        data-team="${team}" data-spread="${hasLine ? o.spread : ''}"
        class="${mine === team ? 'primary' : ''}">${team}</button>`;

      const narrative = hasLine
        ? buildNarrative({ favTeam, dogTeam, marketMargin: Math.abs(o.spread), el, histCell,
                           bucketLabel, bucketRank, edgeRec: hasEdge ? edgeRec : null,
                           matchupCallouts: matchup ? matchup.callouts : null })
        : 'No market line yet for this game.';

      const edgeLabel = edgeRec && edgeRec.recommendation === 'NO_PLAY' ? 'NO PLAY' : edgeRec?.recommendation;
      const edgeChip = hasEdge
        ? `<span class="chip ${edgeRec.recommendation === 'FADE' ? 'good' : ''}"
             title="p=${(edgeRec.p_favorite * 100).toFixed(1)}%, f=${edgeRec.f_estimate != null ? Math.round(edgeRec.f_estimate * 100) + '%' : '—'}, leverage=${edgeRec.leverage != null ? edgeRec.leverage.toFixed(3) : '—'}">
             ${edgeLabel}${edgeRec.recommendation === 'FADE' ? ' ' + edgeRec.underdog_team : ''}</span>`
        : '<span class="muted">—</span>';

      return `<div class="game-card">
        <div class="game-card-head">
          <span class="muted">${fmtLocal(g.kickoff, false)}</span>
          <span class="matchup">${rankPrefix(g.away)}${teamSpan(g.away)}${wchip(g.away)}${qbChip(g.away, injuries, elwayQb1)} @ ${rankPrefix(g.home)}${teamSpan(g.home)}${wchip(g.home)}${qbChip(g.home, injuries, elwayQb1)}</span>
          ${stateChip}
          ${elwayFlip ? `<span class="chip warn" title="ELWAY's avg-points model favors the OTHER team entirely, not just by a smaller or larger margin">ELWAY flip</span>` : ''}
          ${move && move.steam ? `<span class="chip warn" title="Line opened ${signed(move.open)}, now ${signed(move.cur)} — a ${Math.abs(move.deltaHome).toFixed(1)}-point move this week">STEAM</span>` : ''}
          ${[[staleHome, g.home], [staleAway, g.away]].filter(([s]) => s).map(([s, t]) =>
            `<span class="chip bad" title="ELWAY's rating still assumes ${s.assumedName} at QB1 for ${t}, but our injury report lists him ${s.status}: ${esc(s.detail || '')}">ELWAY stale QB (${t})</span>`
          ).join('')}
          ${matchup ? weatherChip(matchup.weather) : ''}
        </div>
        <div class="game-card-body">
          <div class="stat-block">
            <div class="stat-label">Market</div>
            <div>${favLabel} &middot; O/U ${o.total ?? '—'}</div>
            <div class="muted">${pct(o.implied_away)} / ${pct(o.implied_home)} &middot; ${o.book ?? '—'}</div>
            ${move ? `<div class="muted small">opened ${signed(move.open)}</div>` : ''}
          </div>
          <div class="stat-block">
            <div class="stat-label">ELWAY</div>
            <div>${elwayFav ? `${elwayFav.team} ${pct(elwayFav.pct)}` : '<span class="muted">—</span>'}</div>
            <div class="muted">${el && el.spread_home != null ? `implied ${signed(el.spread_home)}` : ' '}</div>
          </div>
          <div class="stat-block">
            <div class="stat-label">History</div>
            <div>${histCell ? `${favTeam} ${pct(histCell.su)}` : '<span class="muted">—</span>'}</div>
            <div class="muted">${bucketLabel ? `${bucketLabel} bucket` : ' '}</div>
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
          <div class="game-card-matchup">
            <div class="stat-label" title="Rush/pass offense and defense, 0-100 scale (100 = best ever recorded in the 2007-present dataset, garbage time excluded), plus a projected score and full forecast. Descriptive context only -- backtesting found neither beats the market spread.">Matchup (0-100) + Projected Score + Forecast</div>
            ${matchupTableHtml(matchup, g.home, g.away)}
          </div>
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
            never goes unshown just because ELWAY's own sheet caught up to it. <span class="chip warn">STEAM</span> = the market spread
            has moved &ge;1.5 points in one direction since this week's first snapshot.
            <span class="chip bad">ELWAY stale QB</span> = ELWAY's weekly sheet is still
            rating that team with a QB1 our live injury feed now shows hurt — a breaking-news
            injury ELWAY's own depth-chart tracking hasn't caught up to yet.
            <span class="chip bad">Bad Weather</span> = live forecast at kickoff (only shown
            for outdoor stadiums within ~16 days out) is bad enough to matter for scoring —
            &ge;20mph wind, &ge;10mm precip, any snow, or sub-20&deg;F highs. Milder forecasts
            don't get a chip, but still silently adjust the projected score (and say so) in
            each card's "Details" section, where the full forecast always shows when available.</p>
        </details>
      </div>`;
  },

  _renderTable(ctx, mode) {
    const M = PICK_MODES[mode];
    const { week, games, odds, hist } = ctx;
    const picks = ctx[M.picksKey] || {};
    const elway = ctx.elway || {};
    const injuries = ctx.injuries || {};
    const spreadHist = ctx.spreadHist || {};
    const elwayQb1 = ctx.elwayQb1 || {};

    const rows = games.map(g => {
      const o = odds[g.game_id] || {};
      const el = elway[g.game_id] || {};
      const move = lineMovement(spreadHist[g.game_id]);
      const staleQb = elwayStaleQb(g.home, elwayQb1, injuries) || elwayStaleQb(g.away, elwayQb1, injuries);
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
        const h = History.lookup(hist, Math.abs(o.spread), o.spread <= 0);
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
      const elwaySpreadCell = staleQb
        ? `<td class="num elway-stale" title="ELWAY's rating still assumes ${staleQb.assumedName} at QB1, but our injury report lists him ${staleQb.status}">${el.spread_home != null ? signed(el.spread_home) : '—'}</td>`
        : elwayFlip
        ? `<td class="num elway-flip" title="ELWAY's model favors the OTHER team entirely: ${signed(el.spread_home)}">${signed(el.spread_home)}</td>`
        : `<td class="num muted">${el.spread_home != null ? signed(el.spread_home) : '—'}</td>`;

      return `<tr>
        <td class="muted">${fmtLocal(g.kickoff, false)}</td>
        <td>${g.away}${wchip(g.away)}${qbChip(g.away, injuries, elwayQb1)}</td>
        <td>${g.home}${wchip(g.home)}${qbChip(g.home, injuries, elwayQb1)}</td>
        <td class="muted${move && move.steam ? ' warn' : ''}" title="${move ? `opened ${signed(move.open)}, now ${signed(move.cur)}` : ''}">${favLabel}</td>
        ${elwaySpreadCell}
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
    document.getElementById(M.id).innerHTML = `
      <div class="panel">
        <h2>${M.title} &mdash; ${made}/${games.length} made
          &middot; locks ${games.length ? fmtLocal(games[0].kickoff) : 'TBD'}</h2>
        ${lean ? `<p class="lean">${esc(lean)}</p>` : ''}
        <table><thead><tr><th>Kick</th><th>Away</th><th>Home</th><th>Fav</th>
          <th class="num" title="ELWAY's home-spread equivalent: away avg pts minus home avg pts. Compare against the Fav column's market line, not the sheet's own spread.">ELWAY Spread</th>
          <th>O/U</th>
          <th class="num">Away%</th><th class="num">Home%</th>
          <th class="num" title="Historical: favorite of this spread wins straight up">Fav SU</th>
          <th class="num" title="Historical: favorite of this spread covers">Fav ATS</th>
          <th>Pick</th><th></th></tr></thead>
          <tbody>${rows}</tbody></table>
        <details>
          <summary class="muted small">Legend</summary>
          <p class="tablefoot muted">Away%/Home% are this game's de-vigged market prices.
            Pick the side that covers. Fav ATS is the historical cover rate for any favorite
            of that spread size — below ~50% leans dog. The line is snapshotted when you pick.
            A <span class="chip warn">QB</span> chip next to a team shows that team's QB injury
            situation regardless of what ELWAY currently assumes at QB1 (plus any other flagged
            QB, if ELWAY's assumed starter is the one hurt; otherwise only a Doubtful/Out backup
            counts, not a merely Questionable one). A highlighted (amber) Fav cell means
            the line has moved &ge;1.5 points in one direction this week — hover for the
            open/current line. A <span class="elway-stale">red</span> ELWAY Spread cell means
            ELWAY's weekly sheet is still rating a team with a QB1 our live injury feed now
            shows hurt.</p>
        </details>
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
