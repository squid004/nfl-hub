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
  ['rush_off_epa', 'Rush Offense'],
  ['pass_off_epa', 'Pass Offense'],
  ['rush_def_epa_allowed', 'Rush Defense'],
  ['pass_def_epa_allowed', 'Pass Defense'],
  ['points_off', 'Points/G'],
  ['points_def_allowed', 'Points Allowed/G'],
  ['turnovers_off', 'Turnovers/G'],
  ['turnovers_def_forced', 'Takeaways/G'],
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
  if (EPA_0_100_COLS.has(key)) return v.toFixed(1); // fixed 0-100 scale
  if (key.startsWith('points')) return v.toFixed(1); // raw per-game average
  return v.toFixed(2); // turnovers: raw per-game average
}

const Power = {
  _sort: { col: 'rank', dir: 1 },

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
    const rows = Object.entries(pr).map(([team, info]) => {
      const rec = records[team];
      const played = rec ? rec.wins + rec.losses + rec.ties : 0;
      return {
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

    // One classification per EPA stat, shared by the table's cell coloring below and the
    // distribution panel -- computed once so the two can never disagree about which group a
    // team is in.
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

    const body = rows.map(r => `<tr data-team="${esc(r.team)}">${POWER_COLS.map(([key]) => {
      const c = classifications[key];
      const g = c ? c.groupOf.get(r.team) : null;
      const cls = g ? GROUP_META[g].textCls : '';
      return `<td class="num${cls ? ' ' + cls : ''}">${fmtPowerVal(key, r[key])}</td>`;
    }).join('')}</tr>`).join('');

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
        <p class="muted small">${meta ? esc(meta.method) : 'Composite of 8 stats, garbage time excluded.'}
          Power Score, SOS, and the 4 EPA columns are shown on a 0-100 scale anchored to the
          best/worst ever recorded across the full 2007-present dataset (not just this season's
          32 teams), so higher is always better and a weak season's best team won't look
          inflated. Points and turnover columns show the actual per-game average for the window
          evaluated. The 4 EPA columns are colored by the same Elite/Above/Average/Below/Weak
          groups as the distribution panel below -- hover a row to highlight that team's dot
          on each strip. Click a column header to sort.</p>
        ${meta ? `<details class="power-methodology">
          <summary class="muted small">Weights used (click to expand)</summary>
          <ul class="power-weights">${weightsList}</ul>
        </details>` : ''}
        <table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>
      </div>
      ${this._distributionPanel(rows, classifications)}`;

    // Read by the radar popup's hover/click handlers below, which are wired ONCE (see
    // _hoverWired) but need whatever the CURRENT render's data is, not a stale closure from
    // whenever they were first attached -- every reload replaces these.
    this._lastRows = rows;
    this._lastClassifications = classifications;

    // Delegated, wired once -- #power itself survives every re-render (only its innerHTML's
    // children get replaced), so this never needs rewiring on sort/reload.
    if (!this._hoverWired) {
      const highlight = (team, on) => {
        el.querySelectorAll(`.dist-tick[data-team="${team}"]`).forEach(t => t.classList.toggle('dist-tick-hover', on));
      };
      el.addEventListener('mouseover', e => {
        const tr = e.target.closest('tr[data-team]');
        if (!tr) return;
        highlight(tr.dataset.team, true);
        if (!this._pinnedTeam) this._showRadar(tr.dataset.team, tr);
      });
      el.addEventListener('mouseout', e => {
        const tr = e.target.closest('tr[data-team]');
        if (!tr) return;
        highlight(tr.dataset.team, false);
        if (!this._pinnedTeam) this._hideRadar();
      });
      el.addEventListener('click', e => {
        const tr = e.target.closest('tr[data-team]');
        if (!tr) return;
        if (this._pinnedTeam === tr.dataset.team) {
          this._pinnedTeam = null;
          this._hideRadar();
        } else {
          this._pinnedTeam = tr.dataset.team;
          this._showRadar(tr.dataset.team, tr);
        }
      });
      // Click anywhere outside the table/popup unpins -- otherwise a pinned popup can only
      // ever be dismissed by re-clicking the exact same row.
      document.addEventListener('click', e => {
        if (!this._pinnedTeam) return;
        if (e.target.closest('#power') || e.target.closest('.power-radar-popup')) return;
        this._pinnedTeam = null;
        this._hideRadar();
      });
      this._hoverWired = true;
    }
  },

  // Built once, appended to <body> (not inside #power's own innerHTML, which gets fully
  // replaced every render/reload -- a child element there would lose its Plotly chart and
  // any open/pinned state every ~90s).
  _ensureRadarPopup() {
    if (this._radarPopup) return this._radarPopup;
    const popup = document.createElement('div');
    popup.className = 'power-radar-popup';
    popup.hidden = true;
    popup.innerHTML = `<div class="power-radar-cap"></div><div class="power-radar-chart"></div>
      <div class="power-radar-pin-hint">Click a row to pin this open</div>`;
    document.body.appendChild(popup);
    this._radarPopup = popup;
    return popup;
  },

  _hideRadar() {
    if (this._radarPopup) this._radarPopup.hidden = true;
  },

  // Positions near the hovered/clicked row, clamped to stay fully on-screen either way.
  _showRadar(team, anchorEl) {
    const rows = this._lastRows, classifications = this._lastClassifications;
    if (!rows || !classifications) return;
    const row = rows.find(r => r.team === team);
    if (!row) return;

    const popup = this._ensureRadarPopup();
    popup.querySelector('.power-radar-cap').textContent = `${team} — EPA profile`;
    popup.querySelector('.power-radar-pin-hint').textContent = this._pinnedTeam === team ? 'Click the row again to unpin' : 'Click a row to pin this open';
    popup.hidden = false;

    const rect = anchorEl.getBoundingClientRect();
    const width = 260, height = popup.offsetHeight || 300;
    let left = rect.right + 10;
    if (left + width > window.innerWidth - 8) left = rect.left - width - 10;
    if (left < 8) left = Math.max(8, Math.min(window.innerWidth - width - 8, rect.left));
    let top = rect.top;
    if (top + height > window.innerHeight - 8) top = Math.max(8, window.innerHeight - height - 8);
    popup.style.left = `${left}px`;
    popup.style.top = `${top}px`;

    this._drawRadar(popup.querySelector('.power-radar-chart'), team, row, classifications);
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
      margin: { t: 24, b: 8, l: 24, r: 24 }, showlegend: false,
    }, { displayModeBar: false, responsive: true, staticPlot: true });
  },

  // One number-line strip per EPA stat, every team positioned at its actual 0-100 value (not
  // just rank order) so clustering vs. real gaps is visible as literal physical distance, not
  // just color or order. See classifyTeams() above for the five groups every team lands in.
  _distributionPanel(rows, classifications) {
    const metricHtml = ([key, label]) => {
      const c = classifications[key];
      if (!c) return '';
      const groupOf = c.groupOf;

      const ticks = rows.filter(r => r[key] != null).map(r => {
        const g = groupOf.get(r.team);
        const meta = g ? GROUP_META[g] : null;
        return `<div class="dist-tick${meta && meta.dotCls ? ' ' + meta.dotCls : ''}" data-team="${esc(r.team)}" style="left:${r[key]}%"
            title="${esc(r.team)}: ${r[key].toFixed(1)}/100 (${meta ? meta.label.toLowerCase() : ''})">${meta && meta.labelDots ? `<span class="dist-tick-label ${meta.dotCls}">${esc(r.team)}</span>` : ''}</div>`;
      }).join('');

      // Piecewise boundaries instead of team lists: each group's own lowest value is its
      // cutoff, chained against the next-better group's cutoff -- exactly the step function
      // buildTiers()/classifyTeams() actually computed, just read back as thresholds instead
      // of membership.
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

      return `<div class="dist-metric">
        <div class="dist-label">${esc(label)}</div>
        <div class="dist-strip">
          <div class="dist-mean-line" style="left:${c.avg}%" title="League average: ${c.avg.toFixed(1)}/100"></div>
          ${ticks}
        </div>
        <ul class="dist-tiers">${lines}</ul>
      </div>`;
    };
    const metrics = DIST_METRICS.map(metricHtml).filter(Boolean).join('');
    if (!metrics) return '';
    return `
      <div class="panel">
        <h2>EPA distribution &amp; tiers</h2>
        <p class="muted small">Every team's 0-100 EPA score (same scale as the table above),
          positioned on a line so clustering is visible at a glance, not just rank. Every team
          falls into one of five groups: <strong class="result-good">Elite</strong>/
          <strong class="result-bad">Weak</strong> when a real gap (not just rank) cuts a group
          off from everyone else, however small; otherwise
          <strong class="mild-good">Above</strong>/<strong class="mild-bad">Below average</strong>
          if it's meaningfully off the mean, or just <strong class="muted">Average</strong> if
          it's within a few points of it -- so a cluster of similarly-bad teams that's
          nonetheless well below average still shows up as a real group instead of
          disappearing into an undifferentiated middle. Each group's own cutoff value is
          listed below it instead of its team list -- hover a team in the table above to find
          it here. The thin vertical line is the league average.</p>
        ${metrics}
      </div>`;
  },
};
