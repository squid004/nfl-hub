'use strict';

// Client-side tabs: show/hide <section data-tab="..."> groups. No routing library.
const Tabs = {
  NAMES: ['pickem', 'ats', 'power', 'parlays', 'model-trends'],
  _store: 'nflhub.tab',

  // Old and combined hashes: Historical Power Rankings is now the second view of the Power Rankings tab, and Methodology / Pipeline
  // Health moved to their own pages (old bookmarks still land in the right place).
  _resolve(h) {
    if (h === 'power-history' || h === 'historical') { Power.setViewQuiet('history'); return 'power'; }
    if (h === 'power') { Power.setViewQuiet('current'); return 'power'; }
    if (h === 'methodology') { location.replace('methodology.html'); return null; }
    if (h === 'pipeline-health') { location.replace('health.html'); return null; }
    return h;
  },

  // Keeps the address bar honest about the Power Rankings view: #power (this season) or #power-history.
  syncHash() {
    const cur = document.querySelector('#tabs button.active');
    if (!cur || cur.dataset.tab !== 'power') return;
    const want = '#' + (Power._view === 'history' ? 'power-history' : 'power');
    if (location.hash !== want) history.replaceState(null, '', want);
  },

  init() {
    document.querySelectorAll('#tabs button[data-tab]').forEach(b => {
      b.addEventListener('click', () => this.show(b.dataset.tab));
    });
    window.addEventListener('hashchange', () => {
      const h = this._resolve(location.hash.slice(1));
      if (h && this.NAMES.includes(h)) { this.show(h, true); if (App._ctx) Power.render(App._ctx); }
    });
    let start = this._resolve(location.hash.slice(1));
    if (start === null) return;
    if (!this.NAMES.includes(start)) {
      try { start = localStorage.getItem(this._store); } catch { start = null; }
    }
    this.show(this.NAMES.includes(start) ? start : 'pickem', true);
  },

  show(name, fromInit) {
    if (!this.NAMES.includes(name)) name = 'pickem';
    document.querySelectorAll('section[data-tab]').forEach(s => {
      s.hidden = s.dataset.tab !== name;
    });
    document.querySelectorAll('#tabs button[data-tab]').forEach(b => {
      b.classList.toggle('active', b.dataset.tab === name);
      b.setAttribute('aria-selected', b.dataset.tab === name ? 'true' : 'false');
    });
    try { localStorage.setItem(this._store, name); } catch { /* private mode */ }
    if (name === 'power') this.syncHash();
    else if (location.hash.slice(1) !== name) {
      history.replaceState(null, '', '#' + name);
    }
    if (!fromInit) window.scrollTo(0, 0);
  },
};
