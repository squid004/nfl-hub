'use strict';

const News = {
  render(ctx) {
    const items = (ctx.news || []).map(n => `
      <div class="newsitem">
        ${(n.players && n.players.length) ? `<span class="chip warn">${n.players.map(esc).join(', ')}</span>` : ''}
        <a href="${esc(n.link)}" target="_blank" rel="noopener">${esc(n.headline)}</a>
        <div class="muted">${esc(n.description)}</div>
      </div>`).join('');

    document.getElementById('news').innerHTML = `
      <div class="panel">
        <h2>News &amp; injuries for your players</h2>
        ${items || '<p class="muted">Nothing flagged yet. Fills in once rosters load.</p>'}
      </div>`;
  },
};
