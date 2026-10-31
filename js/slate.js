'use strict';

// The slate: one row per game (status, matchup, each source's headline number, pick buttons)
// with the selected game's full card in a pane beside the list -- docked from 1700px wide, a
// drawer over the page below that, a full-screen sheet on phones. Pickem builds the rows and
// the card for every game (see Pickem._renderCards / _renderAtsCards) and hands them here;
// this module only lays them out and keeps track of which game is showing.
//
// Games still waiting on the owner's blind gut pick get a banner above the list and a big
// "Gut call" control across their row instead of numbers (dimmed for view-only visitors).

const Slate = {
  // Per slate ('ml' / 'ats'): the selected game id and the pane's scroll offset, kept across
  // the 90-second re-renders and live-score polls.
  _sel: {},
  _open: false,            // drawer/sheet open (below 1700px; the docked pane always shows)
  _wired: new Set(),
  docked: window.matchMedia('(min-width: 1700px)'),

  // Status cell text: kickoff before the game, the live clock during it, Final after.
  status(g) {
    if (g.state === 'in') return { label: g.live_clock ? g.live_clock.replace(/\s*-\s*/, ' · ') : 'Live', cls: 'is-live' };
    if (g.state === 'post') return { label: g.live_clock || 'Final', cls: 'is-final' };
    return { label: fmtLocal(g.kickoff, false), cls: '' };
  },

  // A score as "24–30" (away–home, like the matchup), the leader marked.
  score(g) {
    const a = +g.away_score || 0, h = +g.home_score || 0;
    const lead = s => `<span class="${s ? 'result-good' : ''}">`;
    return `<span class="slate-score">${lead(a > h)}${a}</span>&ndash;${lead(h > a)}${h}</span></span>`;
  },

  // One row. cells: {label: html|null}; blind: [awayButtonHtml, homeButtonHtml] while the
  // game waits on the blind gut pick (then cells and pick are left out).
  row({ id, g, matchupHtml, line, tags = [], cells = {}, pickHtml = '', resultCls = '', blind = null }) {
    const st = this.status(g);
    return {
      id, g, resultCls, blind: !!blind, done: g.state === 'post', cells,
      html: columns => `<div class="slate-row ${st.cls} ${resultCls}${blind ? ' is-blind' : ''}" data-game="${esc(id)}" role="row">
        <div class="slate-status" role="cell">${esc(st.label)}</div>
        <div class="slate-game" role="cell">
          <button type="button" class="slate-open" aria-label="${esc(`${g.away} at ${g.home}, ${st.label}. Show this game`)}"><span class="matchup">${matchupHtml}</span></button>
          ${line ? `<div class="slate-sub">${line}</div>` : ''}
          ${tags.filter(Boolean).length ? `<div class="slate-tags">${tags.filter(Boolean).join('')}</div>` : ''}
        </div>
        ${blind
          ? `<div class="blind" role="cell" style="grid-column: span ${Math.max(1, columns.length) + 1};" data-viewer-title="Only the owner can make gut calls">
               <span class="blind-label">Gut call</span>${blind[0]}<span class="blind-at">@</span>${blind[1]}</div>`
          : columns.map(c => `<div class="slate-stat" role="cell" data-label="${esc(c)}">${cells[c] ?? '<span class="muted">&mdash;</span>'}</div>`).join('')
            + `<div class="slate-pick" role="cell">${pickHtml}</div>`}
      </div>`,
    };
  },

  // Above the list while blind picks are pending: the owner is asked to make them first,
  // viewers are told they're pending; both see how many games have unlocked.
  _bannerHtml(items) {
    const blind = items.filter(i => i.blind).length;
    if (!blind) return '';
    const made = items.length - blind;
    return `<div class="blind-banner">
      <div class="blind-copy">
        <div class="blind-owner"><strong>Gut calls first</strong><span>Pick each game on instinct alone: its numbers unlock once you do, and the pick itself stays changeable.</span></div>
        <div class="blind-viewer"><strong>Waiting on blind picks</strong><span>Each game's numbers show once the owner makes a gut call on it.</span></div>
      </div>
      <div class="blind-progress"><div class="blind-bar"><span style="width:${Math.round(100 * made / items.length)}%"></span></div>
        <span>${made} of ${items.length} unlocked &middot; ${blind} to go</span></div>
    </div>`;
  },

  // Lays out a slate into `el` (a <section>): head (title, tiles), banner, rows + pane, foot.
  // columns: the stat columns in order; a column with nothing for any game is left out.
  // items: from row(), each with .pane (the game's full card); onShow(id) draws anything the
  // pane's card needs once it is in place (e.g. ATS's score-distribution chart).
  render(el, key, { head = '', foot = '', columns, items, onShow }) {
    if (!el) return;
    const cols = columns.filter(c => items.some(i => !i.blind && i.cells[c] != null));
    const allBlind = items.length > 0 && items.every(i => i.blind);
    const header = `<div class="slate-row slate-head" role="row"><div role="columnheader"></div><div role="columnheader">Game</div>
      ${(cols.length ? cols : ['']).map(c => `<div role="columnheader">${esc(c)}</div>`).join('')}<div role="columnheader">Pick</div></div>`;
    el.innerHTML = `<div class="panel">
      ${head}
      ${this._bannerHtml(items)}
      <div class="split${allBlind ? ' all-blind' : ''}${this._open ? ' is-open' : ''}">
        <div class="slate${allBlind ? ' all-blind' : ''}" role="table" aria-label="This week's games" style="--cols:${Math.max(1, cols.length)}">
          ${header}${items.map(i => i.html(cols)).join('')}
        </div>
        <aside class="pane" data-stick="${key}" aria-label="Game details">
          <button type="button" class="pane-close" aria-label="Close">&times;</button>
          <div class="pane-body"></div>
        </aside>
      </div>
      ${foot}
    </div>`;
    el._slate = { key, items, onShow };
    this.syncBlind(el);
    this._wire(el);
    // Keep the selection; otherwise the first game still to come that has numbers to show.
    const keep = this._sel[key];
    const first = items.find(i => !i.blind && !i.done) || items.find(i => !i.blind) || items[0];
    const id = keep && items.some(i => i.id === keep) ? keep : (first && first.id);
    if (id) this.show(el, id, { keepScroll: id === keep });
  },

  // Show a game in its slate's pane; with `open`, also opens the drawer/sheet where the pane
  // isn't docked.
  show(el, id, { open = false, keepScroll = false, fromRow = null } = {}) {
    const s = el._slate;
    const item = s && s.items.find(i => i.id === id);
    if (!item) return;
    const body = el.querySelector('.pane-body');
    const changed = this._sel[s.key] !== id;
    this._sel[s.key] = id;
    body.innerHTML = item.pane;
    this.syncBlind(body);
    if (s.onShow) s.onShow(id);
    el.querySelectorAll('.slate-row[data-game]').forEach(r => {
      const on = r.dataset.game === id;
      r.classList.toggle('is-selected', on);
      r.querySelector('.slate-open')?.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
    if (!keepScroll || changed) body.scrollTop = 0;
    if (changed) PaneStick.reset(s.key, fromRow); else PaneStick.refresh();
    if (open) this.setOpen(true);
  },

  // From outside (the sidebar's game list): show a game and bring its row into view.
  select(sectionId, id) {
    const el = document.getElementById(sectionId);
    const row = el && el.querySelector(`.slate-row[data-game="${CSS.escape(id)}"]`);
    if (!row) return false;
    this.show(el, id, { open: true, fromRow: row });
    row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    row.classList.add('flash');
    setTimeout(() => row.classList.remove('flash'), 1600);
    return true;
  },

  setOpen(open) {
    this._open = open;
    document.querySelectorAll('.split').forEach(s => s.classList.toggle('is-open', open));
  },

  // View-only: the gut-call buttons stay visible but inactive, with a note on hover. Follows
  // body.read-only, which Auth.render() lifts for the owner.
  syncBlind(root) {
    const ro = document.body.classList.contains('read-only');
    root.querySelectorAll('.blind').forEach(c => {
      c.title = ro ? c.dataset.viewerTitle : '';
      c.querySelectorAll('button').forEach(b => {
        if (ro) { b.setAttribute('aria-disabled', 'true'); b.tabIndex = -1; }
        else { b.removeAttribute('aria-disabled'); b.removeAttribute('tabindex'); }
      });
    });
  },

  // Delegated, once per section (the section survives every re-render).
  _wire(el) {
    if (this._wired.has(el.id)) return;
    this._wired.add(el.id);
    el.addEventListener('click', e => {
      if (e.target.closest('.pane-close')) { this.setOpen(false); return; }
      const row = e.target.closest('.slate-row[data-game]');
      // Pick buttons do their own thing (App's click handler); anywhere else on a row shows that game.
      if (!row || e.target.closest('button[data-act]')) return;
      this.show(el, row.dataset.game, { open: true, fromRow: row });
    });
    el.addEventListener('keydown', e => {
      if ((e.key !== 'ArrowDown' && e.key !== 'ArrowUp') || !e.target.closest('.slate')) return;
      const ids = (el._slate ? el._slate.items : []).map(i => i.id);
      const i = ids.indexOf(this._sel[el._slate.key]);
      const next = ids[i + (e.key === 'ArrowDown' ? 1 : -1)];
      if (next == null) return;
      e.preventDefault();
      this.show(el, next, { open: this._open });
      el.querySelector(`.slate-row[data-game="${CSS.escape(next)}"] .slate-open`)?.focus();
    });
  },

  init() {
    // The drawer closes on Escape or a click outside it (a click on another game switches it).
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && this._open && !this.docked.matches) this.setOpen(false); });
    document.addEventListener('click', e => {
      if (!this._open || this.docked.matches || !e.target.isConnected) return;
      if (!e.target.closest('.pane, .slate-row[data-game], .rail-games, .team-pane')) this.setOpen(false);
    });
    new MutationObserver(() => this.syncBlind(document)).observe(document.body, { attributes: true, attributeFilter: ['class'] });
  },
};

// Docked panes (Slate's, and Power Rankings' team pane) have no scrollbar of their own: the
// page is the only scroll. A pane that fits the window stays pinned under the top bar; a
// taller one moves with the page until its bottom is in view, then holds there, and scrolling
// back up brings its top back the same way. Positions are kept per pane (data-stick), so a
// re-render of the same game or team doesn't move it; a different one starts from its top.
const PaneStick = {
  GAP: 12,
  _tops: new Map(),
  _lastY: window.scrollY,

  _pinned() {
    return (parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--sticky-h')) || 0) + this.GAP;
  },

  // Its top after the page scrolls by dy: moves with the page, but stays between max (pinned
  // under the top bar) and min (its bottom at the window's bottom).
  top(prev, dy, max, min) {
    return Math.max(Math.min(max, min), Math.min(max, prev - dy));
  },

  _update(dy) {
    if (!Slate.docked.matches) return;
    const max = this._pinned();
    document.querySelectorAll('.pane[data-stick]').forEach(pane => {
      if (!pane.offsetParent) return;   // hidden tab
      const key = pane.dataset.stick;
      const t = this.top(this._tops.has(key) ? this._tops.get(key) : max, dy, max, innerHeight - pane.offsetHeight - this.GAP);
      this._tops.set(key, t);
      pane.style.top = `${t}px`;
    });
  },

  refresh() { this._update(0); },

  // Start a pane from its top again. With `row` (the row just picked): a pane taller than the
  // list beside it scrolls with the page, so bring its top into view -- as far as that row can
  // stay on screen.
  reset(key, row) {
    this._tops.delete(key);
    this._update(0);
    if (!row || !Slate.docked.matches) return;
    const pane = document.querySelector(`.pane[data-stick="${key}"]`);
    if (!pane) return;
    const top = pane.getBoundingClientRect().top, pinned = this._pinned();
    if (top >= pinned - 1) return;
    const by = Math.min(0, Math.max(top - pinned, -(innerHeight - this.GAP - row.getBoundingClientRect().bottom)));
    if (by < 0) window.scrollBy({ top: by, behavior: 'smooth' });
  },

  init() {
    window.addEventListener('scroll', () => {
      const dy = window.scrollY - this._lastY;
      this._lastY = window.scrollY;
      this._update(dy);
    }, { passive: true });
    window.addEventListener('resize', () => this._update(0));
  },
};
