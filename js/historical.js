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

  sortBy(col) {
    if (this._sort.col === col) this._sort.dir *= -1;
    else this._sort = { col, dir: (col === 'rank' || col === 'season') ? 1 : -1 };
    this.render(App._ctx);
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
            rating) -- <span class="result-good">green</span> = the model's favorite won,
            <span class="result-bad">red</span> = it lost, <span class="chip warn" style="padding:1px 7px;">amber</span> = the game
            tied. Positive = home favored by that measure. Scroll/drag to zoom, hover a point
            for details, click to pin it below.</p>
          <div id="backtest-chart"></div>
          <div id="backtest-selected" class="muted small"></div>
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

  _renderBacktestChart(scatter) {
    const panel = document.getElementById('backtest-panel');
    panel.hidden = false;
    const cs = getComputedStyle(document.documentElement);
    const cssVar = name => cs.getPropertyValue(name).trim();
    const colors = {
      text: cssVar('--text'), dim: cssVar('--dim'), line: cssVar('--line'), panel: cssVar('--panel2'),
      hit: cssVar('--good'), miss: cssVar('--bad'), tie: cssVar('--warn'),
    };

    const groups = { hit: [], miss: [], tie: [], push: [] };
    scatter.forEach(r => groups[r.outcome].push(r));

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

    Plotly.newPlot('backtest-chart', traces, layout, {
      responsive: true, scrollZoom: true, displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    });

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
