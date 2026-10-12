'use strict';

// Pipeline Health tab: shows whether the latest refresh_all() run had any source errors, and
// a rolling log of past failures (nflhub/refresh.py writes both "pipeline_health" and
// "pipeline_health_log" to the kv table every run -- see that file's step 6). No push/email;
// this page is what you check instead of being notified, per explicit user direction
// (2026-10-12): ntfy email forwarding needs a one-time verification step in ntfy.sh's web
// app the user didn't want to do, so alerting was dropped in favor of just a page to look at.
const PipelineHealth = {
  render(ctx) {
    const el = document.getElementById('pipeline-health');
    if (!el) return;
    const health = ctx.pipelineHealth || { current: null, log: [] };
    const current = health.current;
    const log = health.log || [];

    const statusHtml = !current
      ? `<p class="muted">No health data yet -- the refresh pipeline hasn't recorded a run.</p>`
      : current.ok
      ? `<p><span class="chip good">All sources OK</span> as of ${esc(fmtLocal(current.checked_at))}.</p>`
      : `<p><span class="chip bad">${current.errors.length} source(s) failing</span> as of ${esc(fmtLocal(current.checked_at))}.</p>
         <ul>${current.errors.map(e => `<li class="small">${esc(e)}</li>`).join('')}</ul>`;

    const logRows = log.slice().reverse(); // newest first
    const logHtml = !logRows.length
      ? `<p class="muted small">No failed runs recorded.</p>`
      : `<table>
          <thead><tr><th>When</th><th>Errors</th></tr></thead>
          <tbody>${logRows.map(r => `
            <tr>
              <td class="small">${esc(fmtLocal(r.checked_at))}</td>
              <td class="small">${r.errors.map(esc).join('<br>')}</td>
            </tr>`).join('')}</tbody>
        </table>`;

    el.innerHTML = `
      <div class="panel">
        <h2>Pipeline Health</h2>
        <p class="muted small">Every 10-minute refresh run isolates each source (schedule,
          odds, FantasyPros, team ratings, etc.) in its own try/except -- one failing source
          never blocks the rest. This is the record of which sources errored, and when.</p>
        ${statusHtml}
      </div>
      <div class="panel">
        <h2>Failure log</h2>
        <p class="muted small">Last ${log.length} run(s) that had at least one error, newest first.</p>
        ${logHtml}
      </div>`;
  },
};
