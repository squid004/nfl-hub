'use strict';

// Composite data-driven power ranking: nflhub.sources.team_ratings.power_rankings, a weighted
// z-score across 8 stats (rush/pass offense+defense EPA, points scored/allowed, turnovers
// committed/forced), garbage time excluded. Weights and the methodology blurb below come
// straight from the team_ratings kv payload (power_ranking_meta), not hardcoded here, so this
// page can't drift out of sync with whatever weights are actually running. Descriptive only
// -- see the Moneyline Pick'em matchup blocks and research/ for the backtesting showing this
// doesn't beat the market spread.
//
// Power Score + the 4 EPA columns render a fixed 0-100 scale (*_display fields, 100 = the best
// value EVER recorded across the full 2007-present dataset, not just this season's 32 teams --
// see team_ratings.py compute_ratings' historical_bounds), so a weak season's "best" team
// doesn't read as an inflated 100. Points/turnovers are directly countable already, so those
// columns show the plain per-game average for the window being evaluated, not a 0-100 score.
const EPA_0_100_COLS = new Set(['score', 'sos', 'rush_off_epa', 'pass_off_epa', 'rush_def_epa_allowed', 'pass_def_epa_allowed']);
const POWER_COLS = [
  ['rank', 'Rank'],
  ['team', 'Team'],
  ['record', 'Record'],
  ['score', 'Power Score'],
  ['elo', 'Elo', 'Elo rating (FiveThirtyEight-style, margin-aware, regressed a third toward the mean each new season). The arrow is the change since last week; the composite Power Score uses Elo as its 10th stat.'],
  ['rush_off_epa', 'Rush Offense'],
  ['pass_off_epa', 'Pass Offense'],
  ['rush_def_epa_allowed', 'Rush Defense'],
  ['pass_def_epa_allowed', 'Pass Defense'],
  ['points_off', 'Points/G'],
  ['points_def_allowed', 'Points Allowed/G'],
  ['turnovers_off', 'Turnovers/G'],
  ['turnovers_def_forced', 'Takeaways/G'],
  ['pen_presnap_yds', 'Pre-snap Pen Yds/G', 'Yards lost per game to pre-snap penalties (false start, delay of game, offside, encroachment, neutral zone infraction, illegal formation/shift/substitution, too many men, illegal motion). Fewer is better -- a discipline signal, not a luck one.'],
  ['sos', 'SOS', 'Strength of schedule: an EWMA-weighted average of opponents’ own Power Score AT THE TIME each game was played (older games count less, same decay as every other rating here), on the same 0-100 scale as Power Score. 100 = toughest schedule ever recorded, 0 = easiest. Descriptive only -- not folded into Power Score itself.'],
];

const DIST_METRICS = [
  ['rush_off_epa', 'Rush Offense'],
  ['pass_off_epa', 'Pass Offense'],
  ['rush_def_epa_allowed', 'Rush Defense'],
  ['pass_def_epa_allowed', 'Pass Defense'],
];
// Partition the sorted 0-100 scale into tiers wherever a consecutive gap is large relative
// to the league's typical gap here -- the single LARGEST resulting tier is "the pack" (never
// called out); every other tier is a real, separated group, however many teams it has. This
// generalizes a plain "is this one team isolated" check: a 1-team tier IS that isolated
// outlier, but a 2-3 team tier catches a case nearest-neighbor-only logic would miss entirely
// -- e.g. two teams sitting close to EACH OTHER but both genuinely cut off from the pack
// (each one's nearest neighbor is its tier-mate, not far, so neither alone would ever trip a
// per-team isolation check -- the gap that matters is the one below the pair, not between them).
const TIER_GAP_FLOOR = 4;   // 0-100 scale points -- guards a tight league (tiny median gap)
                             // from reading any gap as "huge" in relative terms alone
const TIER_GAP_MULT = 2.5;  // a cut's gap must also be this many times the league's typical
                             // (median) consecutive gap here

function buildTiers(teamVals) {
  const sorted = [...teamVals].sort((a, b) => b.value - a.value); // best first
  const n = sorted.length;
  if (n < 2) return [sorted];
  const gaps = [];
  for (let i = 0; i < n - 1; i++) gaps.push(sorted[i].value - sorted[i + 1].value);
  const sortedGaps = [...gaps].sort((a, b) => a - b);
  const median = sortedGaps[Math.floor(sortedGaps.length / 2)];
  const threshold = Math.max(TIER_GAP_FLOOR, median * TIER_GAP_MULT);
  const tiers = [[sorted[0]]];
  for (let i = 0; i < n - 1; i++) {
    if (gaps[i] >= threshold) tiers.push([]);
    tiers[tiers.length - 1].push(sorted[i + 1]);
  }
  return tiers;
}

// Within this many 0-100 scale points of the league mean counts as "Average" -- otherwise
// every team would land in either "Above" or "Below" average (ties basically never happen
// with real EPA data), which defeats the point of having a middle group at all.
const AVG_BAND = 3;

// Classifies every team into exactly one of five groups for one metric's 0-100 values:
// Elite/Weak (a real, gap-isolated tier via buildTiers() -- genuinely cut off from
// everyone else, not just best/worst by rank), or, for the rest ("the pack"), Above
// average / Average / Below average by plain distance from the mean. Shared by the Power
// Rankings distribution panel (live data) and the Parlays matchup identifier (a frozen
// weekly snapshot) so both read the exact same definition of "real" -- see power.js's
// render() vs parlays.js's use of ctx.parlaySnapshot.
function classifyTeams(teamVals) {
  const avg = teamVals.reduce((s, t) => s + t.value, 0) / teamVals.length;
  const tiers = buildTiers(teamVals).map(teams => ({
    teams, avg: teams.reduce((s, t) => s + t.value, 0) / teams.length,
  }));
  const packIdx = tiers.reduce((best, t, i) => t.teams.length > tiers[best].teams.length ? i : best, 0);
  const byValueDesc = (a, b) => b.value - a.value;
  const elite = tiers.filter((t, i) => i !== packIdx && t.avg > avg).flatMap(t => t.teams).sort(byValueDesc);
  const weak = tiers.filter((t, i) => i !== packIdx && t.avg < avg).flatMap(t => t.teams).sort(byValueDesc);
  const pack = tiers[packIdx].teams;
  const above = pack.filter(t => t.value - avg > AVG_BAND).sort(byValueDesc);
  const average = pack.filter(t => Math.abs(t.value - avg) <= AVG_BAND).sort(byValueDesc);
  const below = pack.filter(t => avg - t.value > AVG_BAND).sort(byValueDesc);
  const groupOf = new Map();
  [['elite', elite], ['above', above], ['average', average], ['below', below], ['weak', weak]]
    .forEach(([key, teams]) => teams.forEach(t => groupOf.set(t.team, key)));
  return { avg, elite, above, average, below, weak, groupOf };
}

// Shared display metadata per group, used both to color the main table's EPA cells and to
// build the distribution strips below -- one definition so the two can never drift apart.
const GROUP_META = {
  elite:   { label: 'Elite',         dotCls: 'outlier-good', textCls: 'result-good', labelDots: true },
  above:   { label: 'Above average', dotCls: 'above-avg',    textCls: 'mild-good',    labelDots: false },
  average: { label: 'Average',       dotCls: '',              textCls: 'muted',        labelDots: false },
  below:   { label: 'Below average', dotCls: 'below-avg',    textCls: 'mild-bad',     labelDots: false },
  weak:    { label: 'Weak',          dotCls: 'outlier-bad',  textCls: 'result-bad',   labelDots: true },
};
const GROUP_ORDER = ['elite', 'above', 'average', 'below', 'weak'];

// Per-metric tier boundary (the group's own min value) for the radar popup's bullseye
// bands -- same classifyTeams() groups already driving the table's cell colors and the
// distribution panel, just read back as 4 threshold numbers per metric instead of team
// lists. An empty tier (e.g. nobody is "weak" in some metric this week) collapses to the
// next real boundary down rather than leaving a gap.
function tierCutoffs(classifications, key) {
  const c = classifications[key];
  if (!c) return null;
  const minOf = teams => (teams.length ? Math.min(...teams.map(t => t.value)) : null);
  let elite = minOf(c.elite), above = minOf(c.above), average = minOf(c.average), below = minOf(c.below);
  if (elite == null) elite = above ?? average ?? below ?? 100;
  if (above == null) above = average ?? below ?? elite;
  if (average == null) average = below ?? above;
  if (below == null) below = average;
  return { below, average, above, elite };
}

// Remaps a raw 0-100 EPA value into "how far through its own tier," on a SHARED 0-100
// scale across every axis -- Weak maps to 0-20, Below to 20-40, Average to 40-60, Above to
// 60-80, Elite to 80-100. That's what turns the radar's tier boundaries into true
// concentric circles instead of a lopsided 4-point polygon, since every axis's own
// boundary then sits at the same radius regardless of that metric's real cutoffs.
function normalizeToTier(value, cutoffs) {
  if (value == null || !cutoffs) return null;
  const { below, average, above, elite } = cutoffs;
  const bands = [[0, below, 0, 20], [below, average, 20, 40], [average, above, 40, 60], [above, elite, 60, 80], [elite, 100, 80, 100]];
  for (const [lo, hi, nlo, nhi] of bands) {
    if (value <= hi) {
      const t = hi > lo ? (value - lo) / (hi - lo) : 0;
      return Math.max(0, Math.min(100, nlo + t * (nhi - nlo)));
    }
  }
  return 100;
}

function fmtPowerVal(key, v) {
  if (v == null) return '—';
  if (key === 'team' || key === 'rank' || key === 'record') return v;
  if (key === 'elo') return v.toFixed(0);
  if (EPA_0_100_COLS.has(key)) return v.toFixed(1); // fixed 0-100 scale
  if (key.startsWith('points')) return v.toFixed(1); // raw per-game average
  return v.toFixed(2); // turnovers: raw per-game average
}

// Trend marker: green up arrow with the size of the move, red down arrow, yellow dash for no change.
// `delta` is the signed move (positive = up/better); `fmt` formats its magnitude.
function trendHtml(delta, title, fmt = v => v) {
  const t = title ? ` title="${esc(title)}"` : '';
  if (delta == null) return '';
  if (delta > 0) return ` <span class="trend-up"${t}>&#9650;${fmt(delta)}</span>`;
  if (delta < 0) return ` <span class="trend-down"${t}>&#9660;${fmt(-delta)}</span>`;
  return ` <span class="trend-flat"${t}>&ndash;</span>`;
}

const Power = {
  _sort: { col: 'rank', dir: 1 },
  _open: new Set(),   // teams whose dropdown is open; survives the 90-second re-renders

  sortBy(col) {
    if (this._sort.col === col) this._sort.dir *= -1;
    else this._sort = { col, dir: col === 'rank' ? 1 : -1 };
    this.render(App._ctx);
  },

  render(ctx) {
    const el = document.getElementById('power');
    if (!el) return;
    const teamRatings = ctx.teamRatings || {};
    const pr = teamRatings.power_rankings || {};
    const meta = teamRatings.power_ranking_meta || null;
    const records = teamRatings.team_records || {};
    const prev = ctx.prevPower || null;   // last week's snapshot: baseline for every trend arrow
    const rows = Object.entries(pr).map(([team, info]) => {
      const rec = records[team];
      const played = rec ? rec.wins + rec.losses + rec.ties : 0;
      const p = prev && prev[team] ? prev[team] : null;
      const elo = info.ratings && info.ratings.elo != null ? info.ratings.elo : null;
      const prevElo = p && p.ratings && p.ratings.elo != null ? p.ratings.elo : null;
      // Moves are measured on the numbers as displayed (1 decimal for Power Score, whole points for Elo), so a flat
      // dash never hides behind a visible change and an arrow never shows a move of zero.
      const scoreDelta = p && p.score_display != null && info.score_display != null
        ? Math.round((info.score_display - p.score_display) * 10) / 10 : null;
      const eloDelta = elo != null && prevElo != null ? Math.round(elo) - Math.round(prevElo) : null;
      return {
        rankDelta: p ? p.rank - info.rank : null,   // positive = moved up
        scoreDelta, eloDelta, elo,
        team, rank: info.rank, score: info.score_display, sos: info.sos_display, ...info.ratings_display,
        record: rec ? `${rec.wins}-${rec.losses}${rec.ties ? '-' + rec.ties : ''}` : '—',
        recordPct: played ? (rec.wins + rec.ties * 0.5) / played : null,
      };
    });

    if (!rows.length) {
      el.innerHTML = `<div class="panel"><h2>Power Rankings</h2><p class="muted">No rating data yet.</p></div>`;
      return;
    }

    const { col, dir } = this._sort;
    rows.sort((a, b) => {
      if (col === 'team') return dir * a.team.localeCompare(b.team);
      if (col === 'record') return dir * ((a.recordPct ?? -1) - (b.recordPct ?? -1)) || a.team.localeCompare(b.team);
      return dir * ((a[col] ?? 0) - (b[col] ?? 0)) || a.team.localeCompare(b.team);
    });

    // One classification per EPA stat, shared by the table's cell coloring and each team's dropdown strips -- computed once
    // so the two can never disagree about which group a team is in.
    const classifications = {};
    DIST_METRICS.forEach(([key]) => {
      const teamVals = rows.filter(r => r[key] != null).map(r => ({ team: r.team, value: r[key] }));
      if (teamVals.length >= 4) classifications[key] = classifyTeams(teamVals);
    });

    const header = POWER_COLS.map(([key, label, title]) => {
      const active = col === key;
      const arrow = active ? (dir === 1 ? ' ▲' : ' ▼') : '';
      return `<th class="num"${title ? ` title="${esc(title)}"` : ''}><button data-act="power-sort" data-col="${key}" class="sort-btn${active ? ' active' : ''}">${label}${arrow}</button></th>`;
    }).join('');

    const body = rows.map(r => {
      const open = this._open.has(r.team);
      const cells = POWER_COLS.map(([key]) => {
        const c = classifications[key];
        const g = c ? c.groupOf.get(r.team) : null;
        const cls = g ? GROUP_META[g].textCls : '';
        if (key === 'rank') return `<td class="num">${r.rank}${trendHtml(r.rankDelta, 'Change in power rank since last week')}</td>`;
        if (key === 'team') return `<td class="num"><span class="power-chev">${open ? '&#9662;' : '&#9656;'}</span> ${esc(r.team)}</td>`;
        if (key === 'score') return `<td class="num">${fmtPowerVal(key, r.score)}${trendHtml(r.scoreDelta, 'Change in Power Score since last week', v => v.toFixed(1))}</td>`;
        if (key === 'elo') return `<td class="num">${fmtPowerVal(key, r.elo)}${trendHtml(r.eloDelta, 'Change in Elo since last week')}</td>`;
        return `<td class="num${cls ? ' ' + cls : ''}">${fmtPowerVal(key, r[key])}</td>`;
      }).join('');
      const detail = open ? `<tr class="power-detail" data-detail="${esc(r.team)}"><td colspan="${POWER_COLS.length}">
          <div class="power-detail-grid">
            <div class="power-detail-cell"><div class="power-detail-cap">${esc(r.team)} &mdash; EPA profile</div><div class="power-radar-chart" data-radar="${esc(r.team)}"></div></div>
            <div class="power-detail-cell"><div class="power-detail-cap">Where ${esc(r.team)} sits in the league</div>${this._teamStrips(r.team, rows, classifications)}</div>
            <div class="power-detail-cell"><div class="power-detail-cap">${esc(r.team)} &mdash; Power Score and Elo, ratings entering each week</div><div class="power-trend-chart" data-trend="${esc(r.team)}"></div></div>
          </div></td></tr>` : '';
      return `<tr data-team="${esc(r.team)}" class="${open ? 'power-row-open' : ''}">${cells}</tr>${detail}`;
    }).join('');

    const statLabel = key => (POWER_COLS.find(([k]) => k === key) || [null, key])[1];
    const weightsList = meta
      ? Object.entries(meta.weights)
          .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
          .map(([key, w]) => `<li><strong>${statLabel(key)}:</strong> ${w >= 0 ? '+' : ''}${w.toFixed(4)}${w === 0 ? ' (dropped — redundant once the other 7 stats are known)' : ''}</li>`)
          .join('')
      : '';

    el.innerHTML = `
      <div class="panel">
        <h2>Power Rankings</h2>
        <p class="muted small">Click a team for its EPA profile, league position and season trend. Arrows show the change since last week.</p>
        ${meta ? `<details class="power-methodology">
          <summary class="muted small">Weights used (click to expand)</summary>
          <ul class="power-weights">${weightsList}</ul>
        </details>` : ''}
        <table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>
        ${this._tiersHtml(classifications)}
      </div>`;

    this._lastRows = rows;
    this._lastClassifications = classifications;

    // Charts exist only for open teams; drawn after the markup is in place (Plotly needs the node).
    this._open.forEach(team => {
      const row = rows.find(r => r.team === team);
      if (!row) return;
      const radarEl = el.querySelector(`[data-radar="${CSS.escape(team)}"]`);
      if (radarEl) this._drawRadar(radarEl, team, row, classifications);
      const trendEl = el.querySelector(`[data-trend="${CSS.escape(team)}"]`);
      if (trendEl) this._drawTrend(trendEl, team, ctx.powerSeries ? ctx.powerSeries[team] : null);
    });

    // Delegated, wired once -- #power itself survives every re-render (only its innerHTML's children are replaced).
    if (!this._clickWired) {
      el.addEventListener('click', e => {
        if (e.target.closest('button, a, .power-detail')) return;   // sort buttons and open dropdowns handle themselves
        const tr = e.target.closest('tr[data-team]');
        if (!tr) return;
        const team = tr.dataset.team;
        if (this._open.has(team)) this._open.delete(team); else this._open.add(team);
        this.render(App._ctx);
      });
      this._clickWired = true;
    }
  },

  // True-circle bullseye (see tierCutoffs()/normalizeToTier() above): the angular axis is
  // numeric (0-360deg) with the 4 stat names as custom tick labels at 0/90/180/270, instead
  // of a 4-slot category axis -- that's what lets a tier boundary render as a smooth ring
  // instead of a 4-point diamond, now that every axis's own cutoff sits at the same radius.
  _drawRadar(el, team, row, classifications) {
    const cs = getComputedStyle(document.documentElement);
    const cssVar = name => cs.getPropertyValue(name).trim();
    const colors = {
      text: cssVar('--text'), dim: cssVar('--dim'), accent: cssVar('--accent'),
      weak: cssVar('--bad'), below: cssVar('--warn'), average: cssVar('--dim'),
      above: '#8fd9b6', elite: cssVar('--good'),
    };
    const hexA = (hex, opacity) => {
      const h = hex.replace('#', '');
      if (h.length !== 6) return hex; // not a plain hex var (e.g. a color-mix()) -- skip alpha
      return `#${h}${Math.round(opacity * 255).toString(16).padStart(2, '0')}`;
    };
    const steps = 72;
    const circleTheta = Array.from({ length: steps + 1 }, (_, i) => (360 * i) / steps);
    const ringR = v => circleTheta.map(() => v);
    const band = (r, color) => ({
      type: 'scatterpolar', r, theta: circleTheta, fill: 'tonext',
      fillcolor: hexA(color, 0.2), line: { color: 'rgba(255,255,255,.2)', width: 1 },
      hoverinfo: 'skip', showlegend: false,
    });
    const bandTraces = DIST_METRICS.every(([k]) => classifications[k]) ? [
      { type: 'scatterpolar', r: ringR(0), theta: circleTheta, line: { color: 'rgba(0,0,0,0)', width: 0 }, hoverinfo: 'skip', showlegend: false },
      band(ringR(20), colors.weak), band(ringR(40), colors.below),
      band(ringR(60), colors.average), band(ringR(80), colors.above), band(ringR(100), colors.elite),
    ] : [];

    const axisAngles = [0, 90, 180, 270];
    const raw = DIST_METRICS.map(([k]) => row[k]);
    const norm = DIST_METRICS.map(([k], i) => normalizeToTier(raw[i], tierCutoffs(classifications, k)) ?? 0);
    const teamTrace = {
      type: 'scatterpolar',
      r: [...norm, norm[0]], theta: [...axisAngles, axisAngles[0]],
      text: [...raw.map(v => (v == null ? '—' : v.toFixed(0))), raw[0] == null ? '—' : raw[0].toFixed(0)],
      line: { color: colors.accent, width: 2.5 },
      fill: 'toself', fillcolor: hexA(colors.accent, 0.35),
      marker: { color: colors.accent, size: 6 },
      textposition: 'top center', textfont: { color: colors.text, size: 10 },
      mode: 'lines+markers+text', hovertemplate: '%{text}/100<extra></extra>', showlegend: false,
    };

    Plotly.react(el, [...bandTraces, teamTrace], {
      polar: {
        bgcolor: 'transparent',
        radialaxis: { range: [0, 100], showticklabels: false, gridcolor: 'rgba(255,255,255,.12)', linecolor: 'rgba(255,255,255,.12)' },
        angularaxis: {
          color: colors.text, gridcolor: 'rgba(255,255,255,.12)',
          tickmode: 'array', tickvals: axisAngles, ticktext: DIST_METRICS.map(([, label]) => label),
          rotation: 90, direction: 'clockwise',
        },
      },
      paper_bgcolor: 'transparent', font: { color: colors.text, size: 10 },
      margin: { t: 28, b: 30, l: 74, r: 74 }, showlegend: false,
    }, { displayModeBar: false, responsive: true, staticPlot: true });
  },

  // Season trend for one team: Power Score (left axis) and Elo (right axis) at each week's snapshot, plus the live point.
  // series = [{ label, score, elo }, ...] built in app.js from the stored weekly snapshots.
  _drawTrend(el, team, series) {
    if (!series || series.length < 2) {
      el.innerHTML = '<p class="muted small">Needs at least two weekly snapshots.</p>';
      return;
    }
    const cs = getComputedStyle(document.documentElement);
    const cssVar = name => cs.getPropertyValue(name).trim();
    const text = cssVar('--text'), dim = cssVar('--dim'), accent = cssVar('--accent'), warn = cssVar('--warn');
    const x = series.map(p => p.label);
    const traces = [
      { type: 'scatter', mode: 'lines+markers', name: 'Power Score', x, y: series.map(p => p.score), yaxis: 'y',
        line: { color: accent, width: 2.5 }, marker: { size: 6 }, hovertemplate: '%{y:.1f}<extra>Power Score</extra>' },
      { type: 'scatter', mode: 'lines+markers', name: 'Elo', x, y: series.map(p => p.elo), yaxis: 'y2',
        line: { color: warn, width: 2.5, dash: 'dot' }, marker: { size: 6 }, hovertemplate: '%{y:.0f}<extra>Elo</extra>' },
    ];
    const grid = 'rgba(255,255,255,.12)';
    Plotly.react(el, traces, {
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: text, size: 11 },
      margin: { t: 8, b: 40, l: 52, r: 52 }, height: 300,
      legend: { orientation: 'h', y: -0.22, x: 0.5, xanchor: 'center' },
      xaxis: { type: 'category', gridcolor: grid, linecolor: grid, tickfont: { color: dim } },
      yaxis: { title: { text: 'Power Score', font: { color: accent } }, tickfont: { color: accent }, gridcolor: grid, zeroline: false },
      yaxis2: { title: { text: 'Elo', font: { color: warn } }, tickfont: { color: warn }, overlaying: 'y', side: 'right', showgrid: false, zeroline: false, tickformat: ',.0f' },
    }, { displayModeBar: false, responsive: true });
  },

  // The four EPA number-line strips for ONE team: every team is a faint dot at its real 0-100 value, the selected team is
  // ringed and labeled so its position (and the actual gaps to its neighbors) reads at a glance.
  _teamStrips(team, rows, classifications) {
    const strips = DIST_METRICS.map(([key, label]) => {
      const c = classifications[key];
      if (!c) return '';
      const ticks = rows.filter(r => r[key] != null).map(r => {
        const g = c.groupOf.get(r.team);
        const m = g ? GROUP_META[g] : null;
        const self = r.team === team;
        return `<div class="dist-tick${m && m.dotCls ? ' ' + m.dotCls : ''}${self ? ' dist-tick-self' : ' dist-tick-dim'}" style="left:${r[key]}%"
            title="${esc(r.team)}: ${r[key].toFixed(1)}/100 (${m ? m.label.toLowerCase() : ''})">${self ? `<span class="dist-tick-label dist-tick-label-self">${esc(r.team)} ${r[key].toFixed(0)}</span>` : ''}</div>`;
      }).join('');
      return `<div class="dist-metric">
          <div class="dist-label">${esc(label)}</div>
          <div class="dist-strip"><div class="dist-mean-line" style="left:${c.avg}%" title="League average: ${c.avg.toFixed(1)}/100"></div>${ticks}</div>
        </div>`;
    }).join('');
    return strips || '<p class="muted small">Not enough teams to classify.</p>';
  },

  // The tier cutoffs behind the strips and the table's cell colors, kept as one collapsed list under the table.
  _tiersHtml(classifications) {
    const parts = DIST_METRICS.map(([key, label]) => {
      const c = classifications[key];
      if (!c) return '';
      const nonEmpty = GROUP_ORDER.map(k => ({ key: k, teams: c[k], ...GROUP_META[k] })).filter(g => g.teams.length);
      const lines = nonEmpty.map((g, i) => {
        const min = Math.min(...g.teams.map(t => t.value));
        const prevMin = i > 0 ? Math.min(...nonEmpty[i - 1].teams.map(t => t.value)) : null;
        let cond;
        if (nonEmpty.length === 1) cond = 'the whole league';
        else if (i === 0) cond = `v &ge; ${min.toFixed(1)}`;
        else if (i === nonEmpty.length - 1) cond = `v &lt; ${prevMin.toFixed(1)}`;
        else cond = `${min.toFixed(1)} &le; v &lt; ${prevMin.toFixed(1)}`;
        return `<li><strong class="${g.textCls}">${g.label}</strong>: ${cond}</li>`;
      }).join('');
      return `<div class="dist-metric"><div class="dist-label">${esc(label)}</div><ul class="dist-tiers">${lines}</ul></div>`;
    }).filter(Boolean).join('');
    if (!parts) return '';
    return `<details class="power-methodology">
        <summary class="muted small">EPA tier cutoffs (click to expand)</summary>${parts}
      </details>`;
  },
};
