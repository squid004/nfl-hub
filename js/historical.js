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

function fmtHistVal(key, v) {
  if (v == null) return '—';
  if (key === 'team' || key === 'rank' || key === 'season') return v;
  if (HIST_0_100_COLS.has(key)) return v.toFixed(1); // fixed 0-100 scale, pooled across every season
  if (key.startsWith('points')) return v.toFixed(1); // raw per-game average
  return v.toFixed(2); // turnovers: raw per-game average
}

const Historical = {
  _sort: { col: 'rank', dir: 1 },

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
      return;
    }

    if (!document.getElementById('historical-table-wrap')) {
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
        </div>`;
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
  },
};
