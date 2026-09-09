'use strict';

// Client-side tabs: show/hide <section data-tab="..."> groups. No routing library.
const Tabs = {
  NAMES: ['pickem', 'ats', 'fantasy', 'survivor', 'parlays'],
  _store: 'nflhub.tab',

  init() {
    document.querySelectorAll('#tabs button[data-tab]').forEach(b => {
      b.addEventListener('click', () => this.show(b.dataset.tab));
    });
    window.addEventListener('hashchange', () => {
      const h = location.hash.slice(1);
      if (this.NAMES.includes(h)) this.show(h, true);
    });
    let start = location.hash.slice(1);
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
    if (location.hash.slice(1) !== name) {
      history.replaceState(null, '', '#' + name);
    }
    if (!fromInit) window.scrollTo(0, 0);
  },
};
