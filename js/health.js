'use strict';

// Standalone Pipeline Health page (health.html): reads the same kv records the old tab did and re-checks every 90 s while the
// page is visible. Needs only DB (anon read), util.js and pipelinehealth.js -- none of the main page's scripts.
async function loadHealth() {
  try {
    const ph = await DB.pipelineHealth();
    PipelineHealth.render({ pipelineHealth: ph });
    const el = document.getElementById('health-updated');
    if (el) el.textContent = ph.current ? `checked ${fmtLocal(ph.current.checked_at)}` : '';
  } catch (e) {
    const sec = document.getElementById('pipeline-health');
    if (sec) sec.innerHTML = `<div class="panel"><h2>Pipeline Health</h2><p class="muted">Could not load: ${esc(e.message || String(e))}</p></div>`;
  }
}

DB.init();
loadHealth();
setInterval(() => { if (!document.hidden) loadHealth(); }, 90_000);
document.addEventListener('visibilitychange', () => { if (!document.hidden) loadHealth(); });
