'use strict';

// pickem-edge integration: pool-specific leverage/fade math (see nflhub/sources/edge_core.py
// for the ported formulas and SPEC.md in github.com/squid004/pickem-edge for the rationale).
// This module owns the #edge section (opponent picks, bias editor, standing, season log)
// plus the budget-banner helper pickem.js calls for the Moneyline table.

const EDGE_TEAMS = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN',
  'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LAC', 'LAR', 'LV', 'MIA', 'MIN', 'NE', 'NO',
  'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WSH'];
function deviationBudget(bucket, poolSize) {
  if (bucket === 'LEADING') return 0;
  if (bucket === 'EARLY') return 1;
  if (bucket === 'MIDDLE') return poolSize >= 30 ? 2 : 1;
  return poolSize >= 30 ? 4 : 3; // BEHIND
}

function currentStanding(standings, week) {
  const exact = standings.find(s => s.week === week);
  if (exact) return exact;
  const prior = standings.filter(s => s.week < week).sort((a, b) => b.week - a.week);
  return prior[0] || null;
}

const signedPct = v => (v == null ? '—' : (v >= 0 ? '+' : '') + Math.round(v * 100) + '%');

const Edge = {
  // Called by pickem.js above the Moneyline table.
  budgetBannerHtml(ctx) {
    const standing = currentStanding(ctx.edgeStandings || [], ctx.week);
    if (!standing) {
      return '<p class="lean">Edge: no season standing set yet — enter it in the Edge panel below to turn on fade recommendations.</p>';
    }
    const rows = Object.values(ctx.edgeLog || {});
    const budget = rows.length ? rows[0].budget_at_time : deviationBudget(standing.standing_bucket, standing.pool_size);
    const used = rows.filter(r => r.recommendation === 'FADE').length;
    return `<p class="lean">Edge: standing <strong>${standing.standing_bucket}</strong>
      (pool of ${standing.pool_size}) &rarr; budget <strong>${budget}</strong>
      fade${budget === 1 ? '' : 's'} this week, used ${used}/${budget}.</p>`;
  },

  render(ctx) {
    const host = document.getElementById('edge');
    if (!host) return;
    host.innerHTML = [
      this._opponentPanel(ctx),
      this._biasPanel(ctx),
      this._standingPanel(ctx),
      this._seasonLogPanel(ctx),
    ].join('');
  },

  _opponentPanel(ctx) {
    const rows = (ctx.edgeOpponentPicks || []).map(p => `
      <tr><td>${esc(p.opponent)}</td><td>${esc(p.team_picked)}</td>
        <td><button data-act="edge-pick-del" data-opponent="${esc(p.opponent)}" data-team="${p.team_picked}">remove</button></td></tr>`
    ).join('');
    return `
      <div class="panel">
        <h2>Opponent picks &mdash; pool bias (week ${ctx.week})</h2>
        <p class="muted">Pulled automatically from the pool's Google Sheet each refresh.
          Learns each team's pool bias vs. the national pick % on the next refresh after
          picks land here.</p>
        ${rows ? `<table style="margin-top:12px;"><thead><tr><th>Opponent</th><th>Pick</th><th></th></tr></thead>
          <tbody>${rows}</tbody></table>` : '<p class="muted">No picks pulled for this week yet.</p>'}
      </div>`;
  },

  _biasPanel(ctx) {
    const byTeam = {};
    (ctx.edgeBias || []).forEach(b => { byTeam[b.team] = b; });
    const rows = EDGE_TEAMS.map(team => {
      const b = byTeam[team] || { bias_value: 0, n_observations: 0, overridden: false };
      return `<tr>
        <td>${team}</td>
        <td class="num">${signedPct(b.bias_value)}</td>
        <td class="num muted">${b.n_observations}</td>
        <td>${b.overridden ? '<span class="chip warn">override</span>' : ''}</td>
        <td class="btns">
          <input type="number" step="1" class="edge-bias-input" data-team="${team}"
            style="width:60px" placeholder="%">
          <button data-act="edge-bias-save" data-team="${team}"
            data-n="${b.n_observations}">Save</button>
          ${b.overridden ? `<button data-act="edge-bias-clear" data-team="${team}">Clear</button>` : ''}
        </td>
      </tr>`;
    }).join('');
    return `
      <div class="panel">
        <h2>Bias editor</h2>
        <p class="muted">Learned per-team pool bias = your pool's pick% minus the national
          pick% for that team, averaged and damped under 3 observations. Positive = your
          pool overpicks that team vs. the country; negative = underpicks. Save a value
          (in percentage points, e.g. "8" for +8%) to override until you clear it.</p>
        <table><thead><tr><th>Team</th><th class="num">Bias</th><th class="num">n</th>
          <th></th><th>Override</th></tr></thead><tbody>${rows}</tbody></table>
      </div>`;
  },

  _standingPanel(ctx) {
    const s = currentStanding(ctx.edgeStandings || [], ctx.week) || {};
    const opt = (v, label) => `<option value="${v}"${s.standing_bucket === v ? ' selected' : ''}>${label}</option>`;
    return `
      <div class="panel">
        <h2>Season standing (week ${ctx.week})</h2>
        <p class="muted">Always manual — drives this week's deviation budget (how many
          favorites to fade). Leading late, mirror the field; behind late, you need
          separation, not parallel correct picks.</p>
        <form id="edge-standing-form" class="btns" style="flex-wrap:wrap;gap:8px;">
          <select name="standing_bucket">
            ${opt('LEADING', 'Leading')}${opt('EARLY', 'Early season')}
            ${opt('MIDDLE', 'Middle of pack')}${opt('BEHIND', 'Behind')}
          </select>
          <input type="number" name="pool_size" placeholder="pool size" value="${s.pool_size ?? ''}" style="width:110px">
          <input type="number" name="correct_picks" placeholder="correct (opt)" value="${s.correct_picks ?? ''}" style="width:120px">
          <input type="number" name="total_picks" placeholder="total (opt)" value="${s.total_picks ?? ''}" style="width:110px">
          <input type="number" name="rank" placeholder="rank (opt)" value="${s.rank ?? ''}" style="width:100px">
          <button type="submit">Save standing</button>
        </form>
      </div>`;
  },

  _seasonLogPanel(ctx) {
    const gameById = {};
    (ctx.seasonGames || []).forEach(g => { gameById[g.game_id] = g; });
    let fadeTotal = 0, fadeCorrect = 0, chalkCorrect = 0, graded = 0;
    (ctx.edgeSeasonLog || []).forEach(r => {
      const g = gameById[r.game_id];
      if (!g || g.state !== 'post' || g.home_score === g.away_score) return;
      const favWon = (g.home === r.favorite_team) === (g.home_score > g.away_score);
      graded++;
      if (favWon) chalkCorrect++;
      if (r.recommendation === 'FADE') {
        fadeTotal++;
        if (!favWon) fadeCorrect++;
      }
    });
    const fadeRate = fadeTotal ? Math.round((fadeCorrect / fadeTotal) * 100) : null;
    return `
      <div class="panel">
        <h2>Season log</h2>
        <p class="muted">Graded games: <strong>${graded}</strong> &middot;
          chalk-only baseline: <strong>${chalkCorrect}/${graded}</strong> favorites won &middot;
          fades called: <strong>${fadeTotal}</strong> &middot;
          fade hit rate: <strong>${fadeRate != null ? fadeRate + '%' : '—'}</strong>
          (expect ~40-45% &mdash; well below that means the p estimates are drifting, not bad luck).</p>
      </div>`;
  },

  async deletePick(opponent, team) {
    try {
      await DB.edgeDeleteOpponentPick(App._ctx.season, App._ctx.week, opponent, team);
      App.reload();
    } catch (e) { alert('Could not remove pick: ' + e.message); }
  },

  async saveBiasOverride(team, n) {
    const input = document.querySelector(`.edge-bias-input[data-team="${team}"]`);
    const pct = parseFloat(input.value);
    if (Number.isNaN(pct)) { alert('Enter a number of percentage points, e.g. 8 or -5'); return; }
    try {
      await DB.edgeSetBiasOverride(team, pct / 100, n ? +n : 0);
      App.reload();
    } catch (e) { alert('Could not save override: ' + e.message); }
  },

  async clearBiasOverride(team) {
    try { await DB.edgeClearBiasOverride(team); App.reload(); }
    catch (e) { alert('Could not clear override: ' + e.message); }
  },

  async saveStanding(form) {
    const fd = new FormData(form);
    const num = k => (fd.get(k) ? Number(fd.get(k)) : null);
    try {
      await DB.edgeUpsertStanding({
        season: App._ctx.season, week: App._ctx.week,
        standing_bucket: fd.get('standing_bucket'),
        pool_size: num('pool_size') || 20,
        correct_picks: num('correct_picks'),
        total_picks: num('total_picks'),
        rank: num('rank'),
      });
      App.reload();
    } catch (e) { alert('Could not save standing: ' + e.message); }
  },
};
