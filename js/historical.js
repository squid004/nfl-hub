'use strict';

// Every (team, season) pair from every FULLY COMPLETED regular season in the dataset
// (nflhub.sources.team_ratings.refresh_historical()/historical_power_rankings()) -- each
// team's own SIMPLE full-season average per stat, not the live Power Rankings view's EWMA-
// decayed rolling rating, and z-scored/ranked against the WHOLE pooled multi-season dataset
// at once (not per-season) so eras are directly comparable on one scale. Descriptive only,
// same caveat as the live view. Team codes are whatever that franchise actually went by THAT
// season (e.g. "STL" through 2015, "LAR" from 2016) -- not normalized to today's rebrand,
// except in the season-trend chart, which joins a franchise's seasons across a move.
//
// Rendered by Power.render() into #power-view when the page's toggle is on "Historical
// Rankings" (the two used to be separate tabs). Each row opens the same three-panel dropdown
// as the current-season table, all limited to the row's own season: EPA profile, where the team
// sat in THAT season's league, and the team's weekly Power Score and Elo through THAT season
// ("ratings entering each week", same rolling method as the current season's chart; the
// table's own Power Score is the full-season average, so the two differ).
const HIST_DISPLAY_CAP = 32;
const HIST_COLS = [
  ['rank', 'Rank'],
  ['season', 'Season'],
  ['team', 'Team'],
  ['win_pct', 'Record', 'That season\'s regular-season record. Sorts by win percentage (ties count as half a win).'],
  ['score', 'Power Score'],
  ['elo', 'Elo', 'The team\'s Elo rating at the END of that season (FiveThirtyEight-style, margin-aware, regressed a third toward the mean each new season).'],
  ['rush_off_epa', 'Rush Offense'],
  ['pass_off_epa', 'Pass Offense'],
  ['rush_def_epa_allowed', 'Rush Defense'],
  ['pass_def_epa_allowed', 'Pass Defense'],
  ['points_off', 'Points/G'],
  ['points_def_allowed', 'Points Allowed/G'],
  ['turnovers_off', 'Turnovers/G'],
  ['turnovers_def_forced', 'Takeaways/G'],
  ['pen_presnap_yds', 'Pre-snap Pen Yds/G', 'Yards lost per game to pre-snap penalties (false start, delay of game, offside, encroachment, etc.). Fewer is better.'],
  ['sos', 'SOS', 'Strength of schedule for that season: the average of that team\'s own opponents\' season-long Power Scores, on the same 0-100 pooled scale. 100 = toughest schedule ever recorded, 0 = easiest. Descriptive only -- not folded into Power Score itself.'],
];
const HIST_0_100_COLS = new Set(['score', 'sos', 'rush_off_epa', 'pass_off_epa', 'rush_def_epa_allowed', 'pass_def_epa_allowed']);

// A franchise that moved keeps one line in the season-trend chart.
const FRANCHISE = { STL: 'LAR', SD: 'LAC', OAK: 'LV' };

function fmtHistVal(key, v) {
  if (v == null) return '—';
  if (key === 'team' || key === 'rank' || key === 'season') return v;
  if (key === 'elo') return v.toFixed(0);
  if (HIST_0_100_COLS.has(key)) return v.toFixed(1); // fixed 0-100 scale, pooled across every season
  if (key.startsWith('points')) return v.toFixed(1); // raw per-game average
  return v.toFixed(2); // turnovers: raw per-game average
}

const Historical = {
  _sort: { col: 'rank', dir: 1 },
  _open: new Set(),          // "season-team" keys whose dropdown is open; survives the 90-second re-renders
  _season: 'all',            // 'all' or a season number (as a string)
  _team: 'all',

  sortBy(col) {
    if (this._sort.col === col) this._sort.dir *= -1;
    else this._sort = { col, dir: (col === 'rank' || col === 'season') ? 1 : -1 };
    this.render(App._ctx);
  },

  toggle(key) {
    if (this._open.has(key)) this._open.delete(key); else this._open.add(key);
    this.render(App._ctx);
  },

  setFilter(kind, value) {
    if (kind === 'season') this._season = value; else this._team = value;
    this.render(App._ctx);
  },

  render(ctx) {
    const el = document.getElementById('power-view');
    if (!el) return;
    const data = ctx.historicalPower;
    const all = (data && data.rankings) || [];
    if (!all.length) {
      el.innerHTML = `<p class="muted">No historical data yet.</p>`;
      return;
    }

    const rows = all.map(r => {
      const rec = r.record;
      const games = rec ? rec.wins + rec.losses + rec.ties : 0;
      return {
        key: `${r.season}-${r.team}`,
        team: r.team, season: r.season, rank: r.rank, score: r.score_display, sos: r.sos_display,
        elo: r.ratings && r.ratings.elo != null ? r.ratings.elo : null,
        win_pct: games ? (rec.wins + 0.5 * rec.ties) / games : null,
        record_text: rec ? `${rec.wins}-${rec.losses}${rec.ties ? '-' + rec.ties : ''}` : null,
        ...r.ratings_display,
      };
    });

    // Weekly series for the trend chart (backend `weekly`: {season: {franchise: [[week, score, elo], ...]}}) and ONE shared axis
    // range across every season and team, so any two dropdowns compare on the same scale. Computed once per rebuilt blob.
    const weekly = data.weekly || {};
    if (this._rangesFor !== data.generated) {
      const flat = {};
      Object.entries(weekly).forEach(([sn, teams]) => Object.entries(teams).forEach(([t, pts]) => {
        flat[`${sn}-${t}`] = pts.map(p => ({ score: p[1], elo: p[2] }));
      }));
      this._ranges = Power._trendRanges(flat);
      this._rangesFor = data.generated;
    }
    const ranges = this._ranges;
    const weeklySeries = row => ((weekly[String(row.season)] || {})[FRANCHISE[row.team] || row.team] || [])
      .map(p => ({ label: `Wk ${p[0]}`, score: p[1], elo: p[2] }));

    // Filters narrow the dataset; sorting always runs over what is left, and only the display is capped to the
    // top HIST_DISPLAY_CAP of whatever order that produces -- so with no filter, sorting Pass Defense ascending
    // surfaces the worst pass defenses of the whole era, not just whichever 32 happened to be on screen.
    const seasons = data.seasons.slice().sort((a, b) => b - a);
    const teams = Array.from(new Set(rows.map(r => r.team))).sort();
    let pool = rows;
    if (this._season !== 'all') pool = pool.filter(r => String(r.season) === String(this._season));
    if (this._team !== 'all') pool = pool.filter(r => r.team === this._team);

    const { col, dir } = this._sort;
    const sorted = [...pool].sort((a, b) => {
      if (col === 'team') return dir * a.team.localeCompare(b.team) || a.season - b.season;
      if (col === 'win_pct') return dir * ((a.win_pct ?? -1) - (b.win_pct ?? -1)) || a.team.localeCompare(b.team);
      return dir * ((a[col] ?? 0) - (b[col] ?? 0)) || a.team.localeCompare(b.team);
    });
    const shown = sorted.slice(0, HIST_DISPLAY_CAP);

    const header = HIST_COLS.map(([key, label, title]) => {
      const active = col === key;
      const arrow = active ? (dir === 1 ? ' ▲' : ' ▼') : '';
      return `<th class="num"${title ? ` title="${esc(title)}"` : ''}><button data-act="historical-sort" data-col="${key}" class="sort-btn${active ? ' active' : ''}">${label}${arrow}</button></th>`;
    }).join('');

    const body = shown.map(r => {
      const open = this._open.has(r.key);
      const cells = HIST_COLS.map(([key]) => {
        if (key === 'win_pct') return `<td class="num">${r.record_text ?? '—'}</td>`;
        if (key === 'team') return `<td class="num"><span class="power-chev">${open ? '&#9662;' : '&#9656;'}</span> ${esc(r.team)}</td>`;
        return `<td class="num">${fmtHistVal(key, r[key])}</td>`;
      }).join('');
      const detail = open ? `<tr class="power-detail" data-hdetail="${esc(r.key)}"><td colspan="${HIST_COLS.length}">
          <div class="power-detail-grid">
            <div class="power-detail-cell"><div class="power-detail-cap">${esc(r.team)} ${r.season} &mdash; EPA profile</div><div class="power-radar-chart" data-hradar="${esc(r.key)}"></div></div>
            <div class="power-detail-cell"><div class="power-detail-cap">Where ${esc(r.team)} sat in the ${r.season} league</div>${Power._teamStrips(r.team, rows.filter(x => x.season === r.season))}</div>
            <div class="power-detail-cell"><div class="power-detail-cap">${esc(r.team)} ${r.season} &mdash; Power Score and Elo, ratings entering each week</div><div class="power-trend-chart" data-htrend="${esc(r.key)}"></div></div>
          </div></td></tr>` : '';
      return `<tr data-hkey="${esc(r.key)}" class="${open ? 'power-row-open' : ''}">${cells}</tr>${detail}`;
    }).join('');

    const firstSeason = data.seasons[0], lastSeason = data.seasons[data.seasons.length - 1];
    const filters = `<div class="filters">
        <label>Season <select id="hist-season">
          <option value="all">All seasons</option>
          ${seasons.map(s => `<option value="${s}"${String(this._season) === String(s) ? ' selected' : ''}>${s}</option>`).join('')}
        </select></label>
        <label>Team <select id="hist-team">
          <option value="all">All teams</option>
          ${teams.map(t => `<option value="${esc(t)}"${this._team === t ? ' selected' : ''}>${esc(t)}</option>`).join('')}
        </select></label>
      </div>`;
    const showing = (this._season === 'all' && this._team === 'all')
      ? `Showing the top ${HIST_DISPLAY_CAP} by the active sort — click a column to re-sort the whole dataset.`
      : `${shown.length} team-season${shown.length === 1 ? '' : 's'} match the filters.`;

    el.innerHTML = `
      <p class="muted small">${firstSeason}-${lastSeason}, ${all.length} team-seasons. ${showing} Click a row for its profile and season trend.</p>
      ${filters}
      <table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;

    // Charts exist only for open rows; drawn after the markup is in place (Plotly needs the node).
    this._open.forEach(key => {
      const row = rows.find(r => r.key === key);
      if (!row) return;
      const seasonRows = rows.filter(r => r.season === row.season);
      const classifications = {};
      DIST_METRICS.forEach(([k]) => {
        const vals = seasonRows.filter(r => r[k] != null).map(r => ({ team: r.team, value: r[k] }));
        if (vals.length >= 4) classifications[k] = classifyTeams(vals);
      });
      const radarEl = el.querySelector(`[data-hradar="${CSS.escape(key)}"]`);
      if (radarEl) Power._drawRadar(radarEl, row.team, row, classifications);
      const trendEl = el.querySelector(`[data-htrend="${CSS.escape(key)}"]`);
      if (trendEl) Power._drawTrend(trendEl, row.team, weeklySeries(row), ranges);
    });
  },
};
