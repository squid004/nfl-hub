'use strict';

const Fantasy = {
  render(ctx) {
    document.getElementById('fantasy').innerHTML =
      this._panel('Yahoo', ctx.yRoster, ctx.week) +
      this._panel('ESPN', ctx.eRoster, ctx.week);
  },

  _panel(name, snap, week) {
    if (!snap || !snap.payload) {
      return `<div class="panel"><h2>${name} fantasy</h2>
        <p class="muted">No snapshot yet. It appears after the first refresh once
        ${name} credentials are set.</p></div>`;
    }
    const stale = snap.week !== week;
    const p = snap.payload;
    const statusCell = pl => {
      if (pl.on_bye) return '<span class="bye">BYE</span>';
      if (BAD_STATUS.includes(pl.injury_status)) return `<span class="alert">${esc(pl.injury_status.slice(0, 4))}</span>`;
      if (pl.injury_status === 'QUESTIONABLE') return '<span class="warn">Q</span>';
      return '<span class="muted">ok</span>';
    };
    const ptsCell = pl => pl.fp_points != null ? pl.fp_points.toFixed(1)
      : (pl.projected ? pl.projected.toFixed(1) : '—');

    const rows = (p.starters || []).map(pl => `
      <tr>
        <td>${esc(pl.slot)}</td>
        <td>${esc(pl.name)} <span class="muted">${esc(pl.pro_team)}</span></td>
        <td class="muted">${esc(pl.opponent || '—')}</td>
        <td class="muted">${pl.kickoff ? fmtLocal(pl.kickoff, false) : '—'}</td>
        <td>${statusCell(pl)}</td>
        <td class="num">${ptsCell(pl)}</td>
        <td class="num muted">${pl.fp_ecr != null ? pl.fp_ecr : '—'}</td>
        <td>${pl.fp_suggest === 'sit' ? '<span class="chip bad">SIT</span>' : ''}</td>
      </tr>`).join('');

    const bench = (p.bench || []).map(pl => `
      <tr><td>${esc(pl.slot)}</td><td>${esc(pl.name)} <span class="muted">${esc(pl.pro_team)}</span></td>
      <td class="muted">${esc(pl.opponent || '—')}</td>
      <td class="num">${ptsCell(pl)}</td>
      <td>${pl.fp_suggest === 'start' ? '<span class="chip good">START</span>' : ''}</td></tr>`).join('');

    const fp = p.fp;
    const fpBlock = !fp ? '' : `
      <div class="fp">
        <h3>FantasyPros optimal
          <span class="muted">${fp.current_points} &rarr; ${fp.optimal_points}</span>
          ${fp.delta > 0 ? `<span class="chip good">+${fp.delta} pts</span>`
                         : '<span class="chip good">lineup is optimal</span>'}
          ${fp.using_fp ? '' : '<span class="chip warn">native projections</span>'}
        </h3>
        ${(fp.swaps || []).filter(s => s.sit).map(s =>
          `<div>Start <strong>${esc(s.start)}</strong> (${esc(s.start_pos)}) over
           <strong>${esc(s.sit)}</strong> (${esc(s.sit_pos)}) at ${esc(s.slot)}
           <span class="chip good">+${s.gain}</span></div>`).join('')}
      </div>`;

    return `
      <div class="panel">
        <h2>${name} fantasy ${stale ? '<span class="chip warn">stale</span>' : ''}</h2>
        <h3>${esc(p.team_name)} <span class="muted">vs ${esc(p.opponent_name)}</span></h3>
        <div class="scoreline">
          <span>${p.score}</span><span class="muted">&ndash;</span><span>${p.opponent_score}</span>
          <span class="proj">proj ${p.projected} &ndash; ${p.opponent_projected}</span>
        </div>
        ${(p.alerts || []).length ? `<p class="alert">${p.alerts.map(esc).join(' &middot; ')}</p>` : ''}
        <table>
          <thead><tr><th>Slot</th><th>Player</th><th>Opp</th><th>Kick</th><th>St</th>
            <th class="num">Pts</th><th class="num">ECR</th><th></th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
        <details><summary class="muted">Bench (${(p.bench || []).length})</summary>
          <table><tbody>${bench}</tbody></table></details>
        ${fpBlock}
      </div>`;
  },
};
