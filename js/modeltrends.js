'use strict';

// Model vs. market backtest -- every graded game, scored against the live Power Rankings
// model's own weighted delta for that matchup (today's weights, applied to each game's
// pre-game rating). Split out of historical.js (which stays focused on the team-season
// ranking table) since this is about auditing the MODEL's own behavior over time, not a
// descriptive ranking of teams.
const BACKTEST_OUTCOME_LABEL = { hit: 'Hit', miss: 'Miss', tie: 'Tie', push: 'Push' };
const BACKTEST_MIN_SPREAD_MAX = 20;  // slider ceiling -- real |spread| tops out well under this
const BACKTEST_MIN_DELTA_MAX = 2;    // slider ceiling -- real |delta| tops out well under this

// r.outcome (from the backend) is always MODEL-relative. This recomputes the same hit/miss/
// tie/push classification relative to the MARKET favorite instead, from fields already on
// every point (spread, scores) -- no server round-trip needed to re-color by the other side.
function marketOutcome(r) {
  if (r.home_score === r.away_score) return 'tie';
  if (r.spread === 0) return 'push'; // no market favorite to grade
  const marketFavoredHome = r.spread > 0;
  return marketFavoredHome === (r.home_score > r.away_score) ? 'hit' : 'miss';
}

// Standard ongoing monitoring metric (added 2026-10) for the QB/skill health adjustment
// baked into `delta` -- see nflhub.sources.team_ratings.compute_calibration_stats()'s own
// docstring for the full reasoning. Hit-rate (old vs. new) is shown for completeness, but
// the number that actually matters is the confidence-reduction RATE comparison: a working
// adjustment reduces |delta| more often on games the old model missed than on games it got
// right (that gap, not the raw hit-rate, is the point of this feature -- honest uncertainty,
// not a better point pick).
function calibrationSummaryHtml(cal) {
  if (!cal) return '';
  const sig = cal.calibration_p != null && cal.calibration_p < 0.05;
  const missPct = cal.miss_confidence_reduced_rate != null ? (cal.miss_confidence_reduced_rate * 100).toFixed(1) : '—';
  const hitPct = cal.hit_confidence_reduced_rate != null ? (cal.hit_confidence_reduced_rate * 100).toFixed(1) : '—';
  const pLabel = cal.calibration_p != null ? (cal.calibration_p < 0.0001 ? 'p<0.0001' : `p=${cal.calibration_p.toFixed(4)}`) : '—';
  return `<p class="lean" title="Computed fresh every refresh from this same chart's delta/delta_raw -- never a stale, separately-run report.">
    <strong>Model calibration check:</strong> hit rate ${(cal.old_hit_rate*100).toFixed(1)}%
    (pre-adjustment) &rarr; ${(cal.new_hit_rate*100).toFixed(1)}% (current), ${cal.net_games >= 0 ? '+' : ''}${cal.net_games}
    games over ${cal.n} graded (${cal.flips} picks changed: ${cal.improvements} improved, ${cal.regressions} regressed).
    On games the pre-adjustment model <strong>missed</strong>, confidence was reduced
    ${missPct}% of the time, vs <strong>${hitPct}%</strong> on games it got right (${pLabel})
    &mdash; ${sig
      ? 'a real, statistically significant tendency to be more honestly uncertain specifically where the model is wrong.'
      : 'not yet a statistically significant difference.'}</p>`;
}

const ModelTrends = {
  _shellBuilt: false,
  _scatterGenerated: null,
  _scatterFull: null,
  _weekYearRange: null,

  // Every backtest-scatter filter/display option lives here, read by _applyFilters()/
  // _drawBacktest() -- see resetAllFilters() for the canonical default state.
  _hideAgreements: false,
  _colorBy: 'model',       // 'model' | 'market' -- which favorite the dot colors are graded against
  _yAxisMode: 'delta',     // 'delta' | 'margin' -- what's actually plotted on the y-axis
  _teamFilter: '',         // '' = all teams, else a team code that must appear as home or away
  _minSpread: 0,           // only show |spread| >= this
  _minDelta: 0,            // only show |delta| >= this
  _seasonFrom: null, _seasonTo: null,  // null = unbounded on that side
  _weekFrom: null, _weekTo: null,
  _cellFilter: null,       // { season, week } set by clicking the heatmap below, or null
  _qbFilter: '',           // '' | 'fav' | 'dog' | 'either' | 'neither' -- QB health of the model's favorite/underdog
  _midgameQbFilter: '',    // '' | 'fav' | 'dog' | 'either' | 'neither' -- confirmed mid-game QB injury (research/edge_signal_test_v34)
  _neutralFilter: '',      // '' | 'exclude' | 'only' -- neutral-site games (is_neutral_site(), no HFA term applied)

  // "Agreement" = market and model favor the SAME side (spread and delta share a sign) --
  // those games can't show the market beating the model or vice versa, since both picked
  // the same team. Hiding them isolates the only games where the two could have differed.
  toggleAgreement() {
    this._hideAgreements = !this._hideAgreements;
    const btn = document.getElementById('backtest-agree-toggle');
    if (btn) btn.textContent = this._hideAgreements
      ? 'Show all games'
      : 'Hide games where market & model agreed';
    this._redraw();
  },

  // Which side's win/loss the dot colors represent -- the geometry (spread vs. delta/margin)
  // never changes, only which favorite each point is graded against.
  toggleColorBy() {
    this._colorBy = this._colorBy === 'model' ? 'market' : 'model';
    const btn = document.getElementById('backtest-colorby-toggle');
    if (btn) btn.textContent = `Color by: ${this._colorBy === 'model' ? "model's picks" : "market's picks"}`;
    this._redraw();
  },

  setYAxisMode(mode) { this._yAxisMode = mode; this._redraw(); },
  setTeamFilter(team) { this._teamFilter = team; this._redraw(); },

  setSeasonRange(from, to) {
    this._seasonFrom = from === '' ? null : Number(from);
    this._seasonTo = to === '' ? null : Number(to);
    this._redraw();
  },
  setWeekRange(from, to) {
    this._weekFrom = from === '' ? null : Number(from);
    this._weekTo = to === '' ? null : Number(to);
    this._redraw();
  },

  setMinSpread(v) {
    this._minSpread = Number(v);
    const lbl = document.getElementById('backtest-min-spread-val');
    if (lbl) lbl.textContent = this._minSpread.toFixed(1);
    this._redraw();
  },
  setMinDelta(v) {
    this._minDelta = Number(v);
    const lbl = document.getElementById('backtest-min-delta-val');
    if (lbl) lbl.textContent = this._minDelta.toFixed(2);
    this._redraw();
  },
  setQbFilter(v) { this._qbFilter = v; this._redraw(); },
  setMidgameQbFilter(v) { this._midgameQbFilter = v; this._redraw(); },
  setNeutralFilter(v) { this._neutralFilter = v; this._redraw(); },

  // Set by clicking a cell in the season/week heatmap below the scatter (cross-filter);
  // cleared by the × on its chip or by Reset all filters.
  setCellFilter(season, week) { this._cellFilter = { season, week }; this._redraw(); },
  clearCellFilter() { this._cellFilter = null; this._redraw(); },

  resetAllFilters() {
    this._hideAgreements = false;
    this._colorBy = 'model';
    this._yAxisMode = 'delta';
    this._teamFilter = '';
    this._minSpread = 0;
    this._minDelta = 0;
    this._seasonFrom = null; this._seasonTo = null;
    this._weekFrom = null; this._weekTo = null;
    this._cellFilter = null;
    this._qbFilter = '';
    this._midgameQbFilter = '';
    this._neutralFilter = '';
    this._syncControlsToState();
    this._redraw();
  },

  _redraw() { if (this._scatterFull) this._drawBacktest(this._scatterFull); },

  // Pushes current state back into the control DOM elements -- only needed after
  // resetAllFilters(), since every other setter is driven BY a control's own event and
  // already reflects what the user just did.
  _syncControlsToState() {
    const set = (id, val) => { const el = document.getElementById(id); if (el) el.value = val; };
    const text = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
    const agreeBtn = document.getElementById('backtest-agree-toggle');
    if (agreeBtn) agreeBtn.textContent = 'Hide games where market & model agreed';
    const colorBtn = document.getElementById('backtest-colorby-toggle');
    if (colorBtn) colorBtn.textContent = "Color by: model's picks";
    set('backtest-yaxis-select', 'delta');
    set('backtest-team-filter', '');
    set('backtest-min-spread', 0); text('backtest-min-spread-val', '0.0');
    set('backtest-min-delta', 0); text('backtest-min-delta-val', '0.00');
    set('backtest-season-from', ''); set('backtest-season-to', '');
    set('backtest-week-from', ''); set('backtest-week-to', '');
    set('backtest-qb-filter', '');
    set('backtest-midgame-qb-filter', '');
    set('backtest-neutral-filter', '');
  },

  render(ctx) {
    const el = document.getElementById('model-trends');
    if (!el) return;
    const data = ctx.historicalPower;
    const scatter = (data && data.backtest_scatter) || [];
    if (!scatter.length) {
      el.innerHTML = `<div class="panel"><h2>Model vs. market, every graded game</h2><p class="muted">No backtest data yet.</p></div>`;
      this._shellBuilt = false;
      return;
    }

    // Shell (chart controls/containers) is built ONCE -- rebuilding it every 90s auto-reload
    // would tear down the Plotly charts underneath it, resetting any zoom/pan the user had,
    // and re-wiring every control's listeners again.
    if (!this._shellBuilt) {
      const seasonOpts = [];
      const weekOpts = [];
      const teamOpts = [];
      const seasonLo = Math.min(...scatter.map(r => r.season)), seasonHi = Math.max(...scatter.map(r => r.season));
      const weekLo = Math.min(...scatter.map(r => r.week)), weekHi = Math.max(...scatter.map(r => r.week));
      this._weekYearRange = {
        seasons: Array.from({ length: seasonHi - seasonLo + 1 }, (_, i) => seasonLo + i),
        weeks: Array.from({ length: weekHi - weekLo + 1 }, (_, i) => weekLo + i),
      };
      this._weekYearRange.seasons.forEach(s => seasonOpts.push(`<option value="${s}">${s}</option>`));
      this._weekYearRange.weeks.forEach(w => weekOpts.push(`<option value="${w}">${w}</option>`));
      const teams = Array.from(new Set(scatter.flatMap(r => [r.home, r.away]))).sort();
      teams.forEach(t => teamOpts.push(`<option value="${esc(t)}">${esc(t)}</option>`));

      el.innerHTML = `
        <div class="panel" id="backtest-panel">
          <h2>Model vs. market, every graded game</h2>
          <p class="muted small">Market spread against the live Power Rankings model's own
            weighted delta for that matchup (today's weights, applied to each game's pre-game
            rating). Positive = home favored by that measure.
            <span class="result-good">Green</span>/<span class="result-bad">red</span>/
            <span class="chip warn" style="padding:1px 7px;">amber</span> = the currently
            selected side's favorite won / lost / the game tied. Scroll/drag to zoom, hover a
            point for details, click to pin it below.</p>
          ${calibrationSummaryHtml(data.calibration)}

          <div class="backtest-controls">
            <div class="backtest-control-row">
              <button data-act="model-trends-toggle-agree" id="backtest-agree-toggle">Hide games where market &amp; model agreed</button>
              <button data-act="model-trends-toggle-colorby" id="backtest-colorby-toggle">Color by: model's picks</button>
              <button data-act="model-trends-reset-filters">Reset all filters</button>
            </div>
            <div class="backtest-control-row">
              <label>Seasons
                <select id="backtest-season-from"><option value="">any</option>${seasonOpts.join('')}</select>
                to
                <select id="backtest-season-to"><option value="">any</option>${seasonOpts.join('')}</select>
              </label>
              <label>Weeks
                <select id="backtest-week-from"><option value="">any</option>${weekOpts.join('')}</select>
                to
                <select id="backtest-week-to"><option value="">any</option>${weekOpts.join('')}</select>
              </label>
              <label>Team <select id="backtest-team-filter"><option value="">All teams</option>${teamOpts.join('')}</select></label>
            </div>
            <div class="backtest-control-row">
              <label>Min |spread| <input type="range" id="backtest-min-spread" min="0" max="${BACKTEST_MIN_SPREAD_MAX}" step="0.5" value="0"><span id="backtest-min-spread-val" class="num">0.0</span></label>
              <label>Min |model &Delta;| <input type="range" id="backtest-min-delta" min="0" max="${BACKTEST_MIN_DELTA_MAX}" step="0.05" value="0"><span id="backtest-min-delta-val" class="num">0.00</span></label>
              <label>Y-axis <select id="backtest-yaxis-select">
                <option value="delta">Power ranking delta</option>
                <option value="margin">Actual margin of victory</option>
              </select></label>
            </div>
            <div class="backtest-control-row">
              <label title="QB listed Out/Doubtful on nflverse's weekly injury report -- not on this chart until a QB1 was actually flagged that week. No data before 2009.">QB health
                <select id="backtest-qb-filter">
                  <option value="">Any QB status</option>
                  <option value="fav">Model favorite's QB out/doubtful</option>
                  <option value="dog">Model underdog's QB out/doubtful</option>
                  <option value="either">Either team's QB out/doubtful</option>
                  <option value="neither">Neither team's QB out/doubtful</option>
                </select>
              </label>
              <label title="Confirmed via nflverse's own play text ('...was injured during the play') AND a real mid-game passer change outside garbage time -- a starter healthy enough to START, hurt partway through. Different signal from QB health above, which is pre-game report status only. See research/edge_signal_test_v34_midgame_qb_injury.py.">Mid-game QB injury
                <select id="backtest-midgame-qb-filter">
                  <option value="">Any game</option>
                  <option value="fav">Model favorite's QB hurt mid-game</option>
                  <option value="dog">Model underdog's QB hurt mid-game</option>
                  <option value="either">Either team's QB hurt mid-game</option>
                  <option value="neither">Neither team's QB hurt mid-game</option>
                </select>
              </label>
              <label title="A neutral-site game (international series, or a venue like the old Buffalo Toronto Series) gets NO home-field term in delta -- see is_neutral_site() in team_ratings.py. Confirmed and fixed 2026-10-07.">Neutral site
                <select id="backtest-neutral-filter">
                  <option value="">Any venue</option>
                  <option value="exclude">Hide neutral-site games</option>
                  <option value="only">Only neutral-site games</option>
                </select>
              </label>
            </div>
            <div class="backtest-control-row">
              <span id="backtest-agree-stat" class="muted small"></span>
              <span id="backtest-cell-chip"></span>
            </div>
          </div>

          <div id="backtest-chart"></div>
          <div id="backtest-selected" class="muted small"></div>
          <h2 style="margin-top:18px;">When those games happened</h2>
          <p class="muted small">Count of the games currently shown above, by season and week --
            every filter and toggle above applies here automatically, and clicking a cell here
            filters the scatter down to just that season/week (clear it with the chip above).</p>
          <div id="backtest-weekyear"></div>
        </div>`;
      this._shellBuilt = true;
      this._wireBacktestControls();
    }

    if (this._scatterGenerated !== data.generated) {
      this._renderBacktestChart(scatter);
      this._scatterGenerated = data.generated;
    }
  },

  // Wired exactly once, right where these elements are created -- the shell (and these
  // controls) only exist for one `_shellBuilt` lifetime, independent of how many times the
  // underlying dataset itself gets (re)rendered.
  _wireBacktestControls() {
    const on = (id, evt, fn) => { const el = document.getElementById(id); if (el) el.addEventListener(evt, fn); };
    const seasonFrom = () => document.getElementById('backtest-season-from').value;
    const seasonTo = () => document.getElementById('backtest-season-to').value;
    const weekFrom = () => document.getElementById('backtest-week-from').value;
    const weekTo = () => document.getElementById('backtest-week-to').value;
    on('backtest-season-from', 'change', () => this.setSeasonRange(seasonFrom(), seasonTo()));
    on('backtest-season-to', 'change', () => this.setSeasonRange(seasonFrom(), seasonTo()));
    on('backtest-week-from', 'change', () => this.setWeekRange(weekFrom(), weekTo()));
    on('backtest-week-to', 'change', () => this.setWeekRange(weekFrom(), weekTo()));
    on('backtest-team-filter', 'change', e => this.setTeamFilter(e.target.value));
    on('backtest-min-spread', 'input', e => this.setMinSpread(e.target.value));
    on('backtest-min-delta', 'input', e => this.setMinDelta(e.target.value));
    on('backtest-yaxis-select', 'change', e => this.setYAxisMode(e.target.value));
    on('backtest-qb-filter', 'change', e => this.setQbFilter(e.target.value));
    on('backtest-midgame-qb-filter', 'change', e => this.setMidgameQbFilter(e.target.value));
    on('backtest-neutral-filter', 'change', e => this.setNeutralFilter(e.target.value));
  },

  // Every active filter composes via AND. Order doesn't affect the result, only performance
  // (cheap either way at ~5k rows), so it just reads top-to-bottom as the controls do.
  _applyFilters(scatter) {
    let rows = scatter;
    if (this._seasonFrom != null) rows = rows.filter(r => r.season >= this._seasonFrom);
    if (this._seasonTo != null) rows = rows.filter(r => r.season <= this._seasonTo);
    if (this._weekFrom != null) rows = rows.filter(r => r.week >= this._weekFrom);
    if (this._weekTo != null) rows = rows.filter(r => r.week <= this._weekTo);
    if (this._teamFilter) rows = rows.filter(r => r.home === this._teamFilter || r.away === this._teamFilter);
    if (this._minSpread > 0) rows = rows.filter(r => Math.abs(r.spread) >= this._minSpread);
    if (this._minDelta > 0) rows = rows.filter(r => Math.abs(r.delta) >= this._minDelta);
    // QB health is always relative to the MODEL's favorite (sign of delta), independent of
    // the colorBy toggle above -- "favorite"/"underdog" here means the model's pick either way.
    if (this._qbFilter) {
      rows = rows.filter(r => {
        const favOut = r.delta >= 0 ? r.home_qb_out : r.away_qb_out;
        const dogOut = r.delta >= 0 ? r.away_qb_out : r.home_qb_out;
        if (this._qbFilter === 'fav') return favOut;
        if (this._qbFilter === 'dog') return dogOut;
        if (this._qbFilter === 'either') return favOut || dogOut;
        if (this._qbFilter === 'neither') return !favOut && !dogOut;
        return true;
      });
    }
    // Same favorite/underdog framing as QB health above, but for a CONFIRMED mid-game
    // injury (research/edge_signal_test_v34) instead of pre-game report status.
    if (this._midgameQbFilter) {
      rows = rows.filter(r => {
        const favHurt = r.delta >= 0 ? r.home_qb_injured_ingame : r.away_qb_injured_ingame;
        const dogHurt = r.delta >= 0 ? r.away_qb_injured_ingame : r.home_qb_injured_ingame;
        if (this._midgameQbFilter === 'fav') return favHurt;
        if (this._midgameQbFilter === 'dog') return dogHurt;
        if (this._midgameQbFilter === 'either') return favHurt || dogHurt;
        if (this._midgameQbFilter === 'neither') return !favHurt && !dogHurt;
        return true;
      });
    }
    // Neutral-site games get no HFA term at all (is_neutral_site(), fixed 2026-10-07) --
    // this isn't favorite/underdog-relative like the filters above, just whether the venue
    // itself had a true home side.
    if (this._neutralFilter === 'exclude') rows = rows.filter(r => !r.neutral);
    else if (this._neutralFilter === 'only') rows = rows.filter(r => r.neutral);
    // "Agreement" = market and model favor the same side (spread and delta share a sign).
    if (this._hideAgreements) rows = rows.filter(r => r.spread * r.delta <= 0);
    if (this._cellFilter) rows = rows.filter(r => r.season === this._cellFilter.season && r.week === this._cellFilter.week);
    return rows;
  },

  _renderCellFilterChip() {
    const el = document.getElementById('backtest-cell-chip');
    if (!el) return;
    if (!this._cellFilter) { el.innerHTML = ''; return; }
    el.innerHTML = `<span class="chip">Season ${this._cellFilter.season}, wk ${this._cellFilter.week} ` +
      `<button data-act="model-trends-clear-cell" style="margin-left:6px; padding:0 6px;">&times;</button></span>`;
  },

  // Market-favorite-won %, over just the games shown when "hide agreements" is on (i.e. the
  // games where market and model picked opposite sides) -- the only games where one of them
  // could have been righter than the other. Excludes spread === 0 (no market favorite to
  // grade) and tie/push outcomes. Reflects every OTHER active filter too, since it's computed
  // from `shown`, not the raw dataset.
  _renderAgreeStat(shown) {
    const el = document.getElementById('backtest-agree-stat');
    if (!el) return;
    if (!this._hideAgreements) { el.textContent = `${shown.length} games shown.`; return; }
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

    const shown = this._applyFilters(scatter);
    this._renderAgreeStat(shown);
    this._renderCellFilterChip();

    // Color mode only changes which favorite each point is graded against for coloring --
    // the geometry and every filter above are unaffected.
    const colorOf = this._colorBy === 'market' ? marketOutcome : r => r.outcome;
    const groups = { hit: [], miss: [], tie: [], push: [] };
    shown.forEach(r => groups[colorOf(r)].push(r));

    const marginMode = this._yAxisMode === 'margin';
    const yOf = marginMode ? (r => r.home_score - r.away_score) : (r => r.delta);
    const yTitle = marginMode ? 'Actual margin → home won by' : 'Power ranking Δ → home favored';
    const yHover = marginMode ? 'Margin %{y:+.0f}' : 'Model Δ %{y:+.2f}';

    const traces = ['hit', 'miss', 'tie'].filter(k => groups[k].length).map(key => {
      const pts = groups[key];
      return {
        name: `${BACKTEST_OUTCOME_LABEL[key]} (${pts.length})`,
        x: pts.map(r => r.spread),
        y: pts.map(yOf),
        customdata: pts.map(r => [r.date, r.away, r.home, r.away_score, r.home_score, r.season, r.week, BACKTEST_OUTCOME_LABEL[key]]),
        mode: 'markers',
        type: 'scattergl',
        marker: { color: colors[key], size: 6, opacity: 0.65, line: { width: 0.5, color: colors[key] } },
        hovertemplate:
          '<b>%{customdata[1]} @ %{customdata[2]}</b><br>' +
          '%{customdata[0]} (wk %{customdata[6]}, %{customdata[5]})<br>' +
          'Final: %{customdata[1]} %{customdata[3]} – %{customdata[2]} %{customdata[4]}<br>' +
          'Spread %{x:+.1f} · ' + yHover + ' · %{customdata[7]}' +
          '<extra></extra>',
      };
    });

    const axisCommon = {
      zeroline: true, zerolinecolor: colors.dim, zerolinewidth: 1,
      gridcolor: colors.line, color: colors.dim, tickfont: { size: 11 },
    };
    const layout = {
      autosize: true,
      margin: { l: 56, r: 16, t: 8, b: 48 },
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
      font: { color: colors.text, size: 12 },
      xaxis: Object.assign({ title: { text: 'Market spread → home favored' } }, axisCommon),
      yaxis: Object.assign({ title: { text: yTitle } }, axisCommon),
      legend: { orientation: 'h', x: 0, y: 1.08, font: { color: colors.text, size: 12 } },
      hoverlabel: { bgcolor: colors.panel, bordercolor: colors.line, font: { color: colors.text, size: 12 } },
      dragmode: 'zoom',
    };

    // react(), not newPlot(), even on the first call -- it's a safe drop-in that also
    // preserves the user's current zoom/pan on later calls (e.g. toggling a filter) instead
    // of resetting the view every time.
    Plotly.react('backtest-chart', traces, layout, {
      responsive: true, scrollZoom: true, displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    });

    this._drawWeekYearDist(shown, colors);
  },

  // Season x week heatmap of `shown` -- always the SAME fully-filtered set the scatter above
  // is currently plotting, recomputed alongside it every time so the two can never fall out
  // of sync. Axis ranges are fixed to the FULL dataset's own season/week span (captured once
  // in render()), not to whatever's currently shown, so filtering doesn't make the grid
  // itself resize -- only the counts (and which cells go to zero) change, which is exactly
  // what makes a season/week range filter visually confirm itself here.
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

    // Cross-filter: clicking a cell in the week/year heatmap narrows the scatter (and the
    // heatmap itself) down to just that season/week.
    const heatmapEl = document.getElementById('backtest-weekyear');
    heatmapEl.on('plotly_click', ev => {
      if (!ev.points || !ev.points[0]) return;
      this.setCellFilter(Number(ev.points[0].y), Number(ev.points[0].x));
    });
  },
};
