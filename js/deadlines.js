'use strict';

const Deadlines = {
  compute(ctx) {
    const { week, games, yRoster, eRoster } = ctx;
    if (!games.length) return [];
    const first = new Date(Math.min(...games.map(g => +new Date(g.kickoff))));
    const out = [
      { kind: 'pickem', label: 'Pick’em', when: first,
        detail: `Week ${week} pick'em locks at first kickoff.` },
      { kind: 'survivor', label: 'Survivor', when: first,
        detail: `Week ${week} survivor pick locks at first kickoff.` },
    ];
    for (const [lg, snap] of [['Yahoo', yRoster], ['ESPN', eRoster]]) {
      if (!snap || snap.week !== week) continue;
      const p = snap.payload || {};
      const teams = new Set((p.starters || []).map(s => s.pro_team).filter(Boolean));
      const rel = games.filter(g => teams.has(g.home) || teams.has(g.away))
                       .map(g => +new Date(g.kickoff));
      const lock = rel.length ? new Date(Math.min(...rel)) : first;
      let detail = `${p.team_name || 'My team'} vs ${p.opponent_name || 'opponent'}.`;
      if (p.alerts && p.alerts.length) detail += ' ' + p.alerts.slice(0, 3).join('; ');
      if (p.fp && p.fp.delta >= 0.5 && p.fp.swaps) {
        const tips = p.fp.swaps.filter(s => s.sit).slice(0, 2)
          .map(s => `start ${s.start} over ${s.sit} (+${s.gain})`);
        if (tips.length) detail += ` FantasyPros +${p.fp.delta}: ` + tips.join('; ');
      }
      out.push({ kind: `${lg.toLowerCase()}_lineup`, label: `${lg} lineup`, when: lock, detail });
    }
    return out.sort((a, b) => a.when - b.when);
  },

  render(ctx) {
    document.getElementById('deadline-chips').innerHTML = this.compute(ctx).map(d =>
      `<span class="chip ${severity(d.when)}" title="${esc(d.detail)}">${esc(d.label)}: ${humanize(d.when)}</span>`
    ).join('');
  },
};
