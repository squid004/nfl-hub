'use strict';

// Every (team, season) pair from every FULLY COMPLETED regular season in the dataset
// (nflhub.sources.team_ratings.refresh_historical()/historical_power_rankings()) -- each
// team's own SIMPLE full-season average per stat, not the live Power Rankings tab's EWMA-
// decayed rolling rating, and z-scored/ranked against the WHOLE pooled multi-season dataset
// at once (not per-season) so eras are directly comparable on one scale. Descriptive only,
// same caveat as the live tab. Team codes are whatever that franchise actually went by THAT
// season (e.g. "STL" through 2015, "LAR" from 2016) -- not normalized to today's rebrand.
const HIST_DISPLAY_CAP = 32;
const HIST_COLS = [
  ['rank', 'Rank'],
  ['season', 'Season'],
  ['team', 'Team'],
  ['score', 'Power Score'],
  ['rush_off_epa', 'Rush Offense'],
  ['pass_off_epa', 'Pass Offense'],
  ['rush_def_epa_allowed', 'Rush Defense'],
  ['pass_def_epa_allowed', 'Pass Defense'],
  ['points_off', 'Points/G'],
  ['points_def_allowed', 'Points Allowed/G'],
  ['turnovers_off', 'Turnovers/G'],
  ['turnovers_def_forced', 'Takeaways/G'],
  ['sos', 'SOS', 'Strength of schedule for that season: the average of that team\'s own opponents\' season-long Power Scores, on the same 0-100 pooled scale. 100 = toughest schedule ever recorded, 0 = easiest. Descriptive only -- not folded into Power Score itself.'],
];
const HIST_0_100_COLS = new Set(['score', 'sos', 'rush_off_epa', 'pass_off_epa', 'rush_def_epa_allowed', 'pass_def_epa_allowed']);
const BACKTEST_OUTCOME_LABEL = { hit: 'Hit', miss: 'Miss', tie: 'Tie', push: 'Push' };

// r.outcome (from the backend) is always MODEL-relative. This recomputes the same hit/miss/
// tie/push classification relative to the MARKET favorite instead, from fields already on
// every point (spread, scores) -- no server round-trip needed to re-color by the other side.
function marketOutcome(r) {
  if (r.home_score === r.away_score) return 'tie';
  if (r.spread === 0) return 'push'; // no market favorite to grade
  const marketFavoredHome = r.spread > 0;
  return marketFavoredHome === (r.home_score > r.away_score) ? 'hit' : 'miss';
}

function fmtHistVal(key, v) {
  if (v == null) return '—';
  if (key === 'team' || key === 'rank' || key === 'season') return v;
  if (HIST_0_100_COLS.has(key)) return v.toFixed(1); // fixed 0-100 scale, pooled across every season
  if (key.startsWith('points')) return v.toFixed(1); // raw per-game average
  return v.toFixed(2); // turnovers: raw per-game average
}

const Historical = {
  _sort: { col: 'rank', dir: 1 },
  _shellBuilt: false,
  _scatterGenerated: null,
  _scatterFull: null,
  _hideAgreements: false,
  _colorBy: 'model',

  sortBy(col) {
    if (this._sort.col === col) this._sort.dir *= -1;
    else this._sort = { col, dir: (col === 'rank' || col === 'season') ? 1 : -1 };
    this.render(App._ctx);
  },

  // "Agreement" = market and model favor the SAME side (spread and delta share a sign) --
  // those games can't show the market beating the model or vice versa, since both picked
  // the same team. Hiding them isolates the only games where the two could have differed.
  toggleAgreement() {
    this._hideAgreements = !this._hideAgreements;
    const btn = document.getElementById('backtest-agree-toggle');
    if (btn) btn.textContent = this._hideAgreements
      ? 'Show all games'
      : 'Hide games where market & model agreed';
    if (this._scatterFull) this._drawBacktest(this._scatterFull);
  },

  // Which side's win/loss the dot colors represent -- the geometry (spread vs. delta) never
  // changes, only which favorite each point is graded against.
  toggleColorBy() {
    this._colorBy = this._colorBy === 'model' ? 'market' : 'model';
    const btn = document.getElementById('backtest-colorby-toggle');
    if (btn) btn.textContent = `Color by: ${this._colorBy === 'model' ? "model's picks" : "market's picks"}`;
    if (this._scatterFull) this._drawBacktest(this._scatterFull);
  },

  render(ctx) {
    const el = document.getElementById('historical');
    if (!el) return;
    const data = ctx.historicalPower;
    const all = (data && data.rankings) || [];
    if (!all.length) {
      el.innerHTML = `<div class="panel"><h2>Historical Power Rankings</h2><p class="muted">No historical data yet.</p></div>`;
      this._shellBuilt = false;
      return;
    }

    // Shell (table container + chart container) is built ONCE -- rebuilding it every sort
    // click or 90s auto-reload would tear down the Plotly chart underneath it, resetting any
    // zoom/pan the user had. Only the table's own innerHTML gets replaced on every render.
    if (!this._shellBuilt) {
      const firstSeason = data.seasons[0], lastSeason = data.seasons[data.seasons.length - 1];
      el.innerHTML = `
        <div class="panel">
          <h2>Historical Power Rankings</h2>
          <p class="muted small">Every team-season from every fully completed regular season on
            record (${firstSeason}-${lastSeason}, ${all.length} team-seasons), ranked by the same
            composite Power Score formula as the live Power Rankings tab -- but each team's own
            full-season average rather than an EWMA-decayed rolling rating, and z-scored against
            every team-season ever at once (not just that year's 32 teams), so a 2023 team and a
            2007 team land on the exact same scale. SOS is each team's opponents' own season-long
            Power Scores averaged together -- safe to compute directly here since full-season
            scores are already fully known, unlike the live tab's point-in-time rating. Team codes
            reflect the team that season (e.g. "STL" through 2015, "LAR" from 2016), not today's
            rebrand. Showing the top ${HIST_DISPLAY_CAP} of ${all.length} by the active sort —
            click a column to re-sort the WHOLE dataset, e.g. sort Pass Defense ascending to
            surface the worst pass defenses of the era, not just this week's.</p>
          <div id="historical-table-wrap"></div>
        </div>
        <div class="panel" id="backtest-panel" hidden>
          <h2>Model vs. market, every graded game</h2>
          <p class="muted small">Market spread against the live Power Rankings model's own
            weighted delta for that matchup (today's weights, applied to each game's pre-game
            rating). Positive = home favored by that measure.
            <span class="result-good">Green</span>/<span class="result-bad">red</span>/
            <span class="chip warn" style="padding:1px 7px;">amber</span> = the currently
            selected side's favorite won / lost / the game tied -- toggle which side below.
            Scroll/drag to zoom, hover a point for details, click to pin it below.</p>
          <div class="btns" style="margin-bottom:8px; flex-wrap:wrap;">
            <button data-act="historical-toggle-agree" id="backtest-agree-toggle">Hide games where market &amp; model agreed</button>
            <button data-act="historical-toggle-colorby" id="backtest-colorby-toggle">Color by: model's picks</button>
            <span id="backtest-agree-stat" class="muted small"></span>
          </div>
          <div id="backtest-chart"></div>
          <div id="backtest-selected" class="muted small"></div>
          <h2 style="margin-top:18px;">When those games happened</h2>
          <p class="muted small">Count of the games shown above, by season and week -- follows
            the agree/disagree toggle automatically, so this always reflects whatever's
            currently plotted, not the full dataset.</p>
          <div id="backtest-weekyear"></div>
        </div>`;
      this._shellBuilt = true;
    }

    const rows = all.map(r => ({ team: r.team, season: r.season, rank: r.rank, score: r.score_display, sos: r.sos_display, ...r.ratings_display }));

    // Sorting always runs over the FULL pooled dataset, not just the visible window -- only
    // the display is capped to the top HIST_DISPLAY_CAP of whatever order that produces, so
    // e.g. sorting Pass Defense ascending surfaces the worst pass defenses of the whole era,
    // not just whichever 32 happened to already be on screen.
    const { col, dir } = this._sort;
    const sorted = [...rows].sort((a, b) => {
      if (col === 'team') return dir * a.team.localeCompare(b.team) || a.season - b.season;
      return dir * ((a[col] ?? 0) - (b[col] ?? 0)) || a.team.localeCompare(b.team);
    });
    const shown = sorted.slice(0, HIST_DISPLAY_CAP);

    const header = HIST_COLS.map(([key, label, title]) => {
      const active = col === key;
      const arrow = active ? (dir === 1 ? ' ▲' : ' ▼') : '';
      return `<th class="num"${title ? ` title="${esc(title)}"` : ''}><button data-act="historical-sort" data-col="${key}" class="sort-btn${active ? ' active' : ''}">${label}${arrow}</button></th>`;
    }).join('');

    const body = shown.map(r => `<tr>${HIST_COLS.map(([key]) =>
      `<td class="num">${fmtHistVal(key, r[key])}</td>`).join('')}</tr>`).join('');

    document.getElementById('historical-table-wrap').innerHTML =
      `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;

    const scatter = data.backtest_scatter;
    if (scatter && scatter.length && this._scatterGenerated !== data.generated) {
      this._renderBacktestChart(scatter);
      this._scatterGenerated = data.generated;
    }
  },

  // Market-favorite-won %, over just the games shown when "hide agreements" is on (i.e. the
  // games where market and model picked opposite sides) -- the only games where one of them
  // could have been righter than the other. Excludes spread === 0 (no market favorite to
  // grade) and tie/push outcomes.
  _renderAgreeStat(shown) {
    const el = document.getElementById('backtest-agree-stat');
    if (!el) return;
    if (!this._hideAgreements) { el.textContent = ''; return; }
    const decided = shown.filter(r => (r.outcome === 'hit' || r.outcome === 'miss') && r.spread !== 0);
    if (!decided.length) { el.textContent = `${shown.length} games shown.`; return; }
    const marketRight = decided.filter(r => (r.spread > 0) === (r.home_score > r.away_score)).length;
    const modelRight = decided.filter(r => r.outcome === 'hit').length;
    el.textContent = `${shown.length} games shown — market's pick won ` +
      `${(100 * marketRight / decided.length).toFixed(1)}% · model's pick won ` +
      `${(100 * modelRight / decided.length).toFixed(1)}%`;
  },

  _drawBacktest(scatter) {
    const cs = getComputedStyle(document.documentElement);
    const cssVar = name => cs.getPropertyValue(name).trim();
    const colors = {
      text: cssVar('--text'), dim: cssVar('--dim'), line: cssVar('--line'), panel: cssVar('--panel2'),
      hit: cssVar('--good'), miss: cssVar('--bad'), tie: cssVar('--warn'), accent: cssVar('--accent'),
    };

    // "Agreement" = market and model favor the same side (spread and delta share a sign).
    // Hiding those isolates the only games where the market and the model could have
    // actually differed on the winner.
    const shown = this._hideAgreements ? scatter.filter(r => r.spread * r.delta <= 0) : scatter;
    this._renderAgreeStat(shown);

    // Color mode only changes which favorite each point is graded against for coloring --
    // the geometry (spread vs. delta) and the agreement filter above are unaffected.
    const colorOf = this._colorBy === 'market' ? marketOutcome : r => r.outcome;
    const groups = { hit: [], miss: [], tie: [], push: [] };
    shown.forEach(r => groups[colorOf(r)].push(r));

    const traces = ['hit', 'miss', 'tie'].filter(k => groups[k].length).map(key => {
      const pts = groups[key];
      return {
        name: `${BACKTEST_OUTCOME_LABEL[key]} (${pts.length})`,
        x: pts.map(r => r.spread),
        y: pts.map(r => r.delta),
        customdata: pts.map(r => [r.date, r.away, r.home, r.away_score, r.home_score, r.season, r.week, BACKTEST_OUTCOME_LABEL[key]]),
        mode: 'markers',
        type: 'scattergl',
        marker: { color: colors[key], size: 6, opacity: 0.65, line: { width: 0.5, color: colors[key] } },
        hovertemplate:
          '<b>%{customdata[1]} @ %{customdata[2]}</b><br>' +
          '%{customdata[0]} (wk %{customdata[6]}, %{customdata[5]})<br>' +
          'Final: %{customdata[1]} %{customdata[3]} – %{customdata[2]} %{customdata[4]}<br>' +
          'Spread %{x:+.1f} · Model Δ %{y:+.2f} · %{customdata[7]}' +
          '<extra></extra>',
      };
    });

    const axisCommon = {
      zeroline: true, zerolinecolor: colors.dim, zerolinewidth: 1,
      gridcolor: colors.line, color: colors.dim, tickfont: { size: 11 },
    };
    const layout = {
      autosize: true,
      margin: { l: 52, r: 16, t: 8, b: 48 },
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
      font: { color: colors.text, size: 12 },
      xaxis: Object.assign({ title: { text: 'Market spread → home favored' } }, axisCommon),
      yaxis: Object.assign({ title: { text: 'Power ranking Δ → home favored' } }, axisCommon),
      legend: { orientation: 'h', x: 0, y: 1.08, font: { color: colors.text, size: 12 } },
      hoverlabel: { bgcolor: colors.panel, bordercolor: colors.line, font: { color: colors.text, size: 12 } },
      dragmode: 'zoom',
    };

    // react(), not newPlot(), even on the first call -- it's a safe drop-in that also
    // preserves the user's current zoom/pan on later calls (e.g. toggling the button)
    // instead of resetting the view every time.
    Plotly.react('backtest-chart', traces, layout, {
      responsive: true, scrollZoom: true, displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    });

    this._drawWeekYearDist(shown, colors);
  },

  // Season x week heatmap of `shown` -- always the SAME agree/disagree-toggled set the
  // scatter above is currently plotting, recomputed from scratch alongside it so the two
  // can never fall out of sync. Axis ranges are fixed to the FULL dataset's own season/week
  // span (captured once in _renderBacktestChart), not to whatever's currently shown, so
  // toggling doesn't make the grid itself jump around -- only the counts inside it change.
  _drawWeekYearDist(shown, colors) {
    const { seasons, weeks } = this._weekYearRange;
    const counts = seasons.map(() => weeks.map(() => 0));
    const seasonIdx = new Map(seasons.map((s, i) => [s, i]));
    const weekIdx = new Map(weeks.map((w, i) => [w, i]));
    shown.forEach(r => {
      const si = seasonIdx.get(r.season), wi = weekIdx.get(r.week);
      if (si != null && wi != null) counts[si][wi]++;
    });

    Plotly.react('backtest-weekyear', [{
      type: 'heatmap',
      x: weeks, y: seasons, z: counts,
      colorscale: [[0, colors.panel], [1, colors.accent]],
      showscale: true,
      colorbar: { tickfont: { color: colors.dim, size: 10 }, outlinewidth: 0, len: 1 },
      hovertemplate: 'Season %{y}, week %{x}: %{z} game(s)<extra></extra>',
      xgap: 2, ygap: 2,
    }], {
      autosize: true,
      margin: { l: 52, r: 16, t: 8, b: 40 },
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
      font: { color: colors.text, size: 12 },
      xaxis: { title: { text: 'NFL week' }, dtick: 1, color: colors.dim, tickfont: { size: 11 } },
      yaxis: { title: { text: 'Season' }, dtick: 1, color: colors.dim, tickfont: { size: 11 }, autorange: 'reversed' },
    }, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] });
  },

  _renderBacktestChart(scatter) {
    this._scatterFull = scatter;
    // Fixed once, from the FULL dataset -- the week/year grid's own axes shouldn't resize
    // every time the agree/disagree toggle changes which cells have counts in them.
    const seasonLo = Math.min(...scatter.map(r => r.season)), seasonHi = Math.max(...scatter.map(r => r.season));
    const weekLo = Math.min(...scatter.map(r => r.week)), weekHi = Math.max(...scatter.map(r => r.week));
    this._weekYearRange = {
      seasons: Array.from({ length: seasonHi - seasonLo + 1 }, (_, i) => seasonLo + i),
      weeks: Array.from({ length: weekHi - weekLo + 1 }, (_, i) => weekLo + i),
    };
    const panel = document.getElementById('backtest-panel');
    panel.hidden = false;
    this._drawBacktest(scatter);

    const chartEl = document.getElementById('backtest-chart');
    chartEl.on('plotly_click', ev => {
      if (!ev.points || !ev.points[0]) return;
      const [date, away, home, awayScore, homeScore, season, week, label] = ev.points[0].customdata;
      const cls = label === 'Hit' ? 'result-good' : label === 'Miss' ? 'result-bad' : 'mild-bad';
      // date is a plain "YYYY-MM-DD" string -- parse at noon UTC, not midnight, so a
      // negative-UTC-offset browser timezone can't roll it back to the previous day.
      const d = new Date(`${date}T12:00:00Z`);
      const dateStr = isNaN(d) ? date : d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
      document.getElementById('backtest-selected').innerHTML =
        `<strong>${esc(away)} ${awayScore} @ ${esc(home)} ${homeScore}</strong> — ${esc(dateStr)}, ` +
        `${season} wk ${week} — <span class="${cls}">${esc(label)}</span>`;
    });
  },
};
