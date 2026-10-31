'use strict';

// The page around the tabs: sidebar (desktop) / bottom tab bar (phones), the top bar's title,
// the sidebar's list of this week's games, status badges on the tabs, the More sheet, and the
// jump links on Model Trends. App.reload() calls render(ctx) after every reload; Tabs.show()
// calls onTab(name) on every tab change.
const Shell = {
  desktop: window.matchMedia('(min-width: 1024px)'),
  phone: window.matchMedia('(max-width: 760px)'),

  init() {
    const nav = document.getElementById('tabs');
    const more = document.querySelector('.more-btn');
    const sheet = document.getElementById('sheet');

    // Keyboard: the tab list is one Tab stop; arrows (down/up in the sidebar, right/left in
    // the strip), Home and End move between the tabs on screen.
    nav.addEventListener('keydown', e => {
      if (e.altKey || e.ctrlKey || e.metaKey) return;
      const shown = Array.from(nav.querySelectorAll('button[data-tab]')).filter(b => b.offsetParent !== null);
      const i = shown.findIndex(b => b.classList.contains('active'));
      const vertical = this.desktop.matches;
      const step = e.key === (vertical ? 'ArrowDown' : 'ArrowRight') ? 1 : e.key === (vertical ? 'ArrowUp' : 'ArrowLeft') ? -1 : 0;
      const next = e.key === 'Home' ? shown[0] : e.key === 'End' ? shown[shown.length - 1]
        : step ? shown[(i + step + shown.length) % shown.length] : null;
      if (!next) return;
      e.preventDefault();
      next.click();
      next.focus();
    });
    const orient = () => nav.setAttribute('aria-orientation', this.desktop.matches ? 'vertical' : 'horizontal');
    this.desktop.addEventListener('change', orient);
    orient();

    // More: a sheet with every section and page, grouped as in the sidebar.
    const items = [];
    nav.querySelectorAll(':scope > .nav-group, :scope > button[data-tab]').forEach(n => {
      if (n.classList.contains('nav-group')) items.push(`<div class="nav-group">${esc(n.textContent)}</div>`);
      else items.push(`<button type="button" data-tab="${n.dataset.tab}">${n.querySelector('.ico').outerHTML}${esc(this._label(n))}</button>`);
    });
    items.push('<div class="nav-group">More</div>');
    document.querySelectorAll('.rail-links a').forEach(a => {
      items.push(`<a href="${a.getAttribute('href')}">${a.querySelector('.ico').outerHTML}${esc(a.querySelector('.nav-label').textContent)}</a>`);
    });
    sheet.innerHTML = items.join('');
    const setSheet = open => { sheet.hidden = !open; more.setAttribute('aria-expanded', String(open)); };
    more.addEventListener('click', () => setSheet(sheet.hidden));
    sheet.addEventListener('click', e => {
      const b = e.target.closest('button[data-tab]');
      if (!b) return;
      setSheet(false);
      nav.querySelector(`button[data-tab="${b.dataset.tab}"]`).click();   // the real tab: history and Tabs follow
    });
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && !sheet.hidden) setSheet(false); });
    document.addEventListener('click', e => { if (!sheet.hidden && !e.target.closest('#sheet, .more-btn')) setSheet(false); });
    this._setSheet = setSheet;

    // Height of whatever is pinned to the top (the top bar on desktop, the tab strip on
    // tablets, nothing on phones), so jumps and pinned panes land below it.
    const topbar = document.querySelector('.topbar');
    const navbar = document.querySelector('.navbar');
    const sticky = () => document.documentElement.style.setProperty('--sticky-h',
      `${this.phone.matches ? 0 : (this.desktop.matches ? topbar : navbar).offsetHeight}px`);
    const ro = new ResizeObserver(sticky);
    ro.observe(topbar);
    ro.observe(navbar);
    this.desktop.addEventListener('change', sticky);
    this.phone.addEventListener('change', sticky);

    // The sidebar's games: show that game on the slate (ATS's when on ATS, else Moneyline's).
    document.getElementById('rail-games').addEventListener('click', e => {
      const b = e.target.closest('button[data-game]');
      if (!b) return;
      const onAts = document.querySelector('#tabs button.active')?.dataset.tab === 'ats';
      if (!onAts) Tabs.show('pickem');
      Slate.select(onAts ? 'atspickem' : 'pickem', b.dataset.game);
    });

    window.addEventListener('scroll', () => requestAnimationFrame(() => this._spy()), { passive: true });
  },

  // A tab's full name (not the short one phones show).
  _label(button) {
    const c = button.cloneNode(true);
    c.querySelectorAll('.short, .badge, .ico').forEach(n => n.remove());
    return c.textContent.replace(/\s+/g, ' ').trim();
  },

  onTab(name) {
    const nav = document.getElementById('tabs');
    const active = nav.querySelector(`button[data-tab="${name}"]`);
    if (!active) return;
    nav.querySelectorAll('button[data-tab]').forEach(b => { b.tabIndex = b === active ? 0 : -1; });
    const label = this._label(active);
    document.title = `${label} · NFL Hub`;
    document.getElementById('page-title').textContent = label;
    document.querySelectorAll('#sheet button[data-tab]').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
    // On the phone bar, More stands in for a section it holds.
    document.querySelector('.more-btn').classList.toggle('active', this.phone.matches && active.offsetParent === null);
    if (this._setSheet) this._setSheet(false);
    if (!this.desktop.matches && !this.phone.matches) {
      nav.scrollTo({ left: Math.max(0, active.offsetLeft - (nav.clientWidth - active.offsetWidth) / 2), behavior: 'smooth' });
    }
    this.toc();
  },

  render(ctx) {
    this.renderGames();
    this.renderBadges(ctx);
    this.toc();
  },

  // This week's games in the sidebar: kickoff, live clock or Final, and a dot for your graded pick.
  renderGames() {
    const list = document.getElementById('rail-games');
    const games = Pickem._mlGames || [];
    list.innerHTML = games.length ? games.map(({ g, id, resultCls }) => {
      const st = Slate.status(g);
      return `<li><button type="button" class="rail-game ${st.cls} ${resultCls}" data-game="${esc(id)}" aria-label="${esc(`${g.away} at ${g.home}, ${st.label}`)}">
        <span class="rail-game-teams">${esc(g.away)}<span class="rail-game-at"> @ </span>${esc(g.home)}</span>
        <span class="rail-game-when">${esc(st.label)}</span><span class="rail-game-mark" aria-hidden="true"></span></button></li>`;
    }).join('') : '<li class="rail-games-empty">No games yet</li>';
  },

  // Picks made on Pick'em and ATS (amber while some are missing in the current week), and the
  // pipeline's health as a dot on its link.
  renderBadges(ctx) {
    const badge = (tab, made, total) => {
      const b = document.querySelector(`#tabs button[data-tab="${tab}"]`);
      let span = b.querySelector('.badge');
      if (!span) { span = document.createElement('span'); b.appendChild(span); }
      span.className = `badge${!ctx.isPastWeek && made < total ? ' todo' : ''}`;
      span.textContent = `${made}/${total}`;
      span.title = `${made} of ${total} picks made`;
      span.hidden = !total;
    };
    const total = (ctx.games || []).length;
    badge('pickem', Object.keys(ctx.pPicks || {}).length, total);
    badge('ats', Object.keys(ctx.aPicks || {}).length, total);

    const chip = document.getElementById('health-chip');
    const cur = ctx.pipelineHealth && ctx.pipelineHealth.current;
    chip.hidden = !cur;
    if (!cur) return;
    chip.className = `badge dot ${cur.ok ? 'good' : 'bad'}`;
    chip.title = cur.ok ? 'All sources OK' : (cur.errors || []).join('\n');
  },

  // Jump links on Model Trends: one chip per panel, pinned under the top bar, the one you're
  // reading highlighted.
  toc() {
    const toc = document.getElementById('toc');
    const tab = document.querySelector('#tabs button.active')?.dataset.tab;
    this._panels = tab === 'model-trends'
      ? Array.from(document.querySelectorAll('#model-trends .panel')).filter(p => p.querySelector(':scope > h2'))
      : [];
    toc.hidden = this._panels.length < 3;
    if (toc.hidden) return;
    toc.innerHTML = this._panels.map((p, i) => {
      const h2 = p.querySelector(':scope > h2').cloneNode(true);
      h2.querySelectorAll('.info-btn').forEach(n => n.remove());
      const label = h2.textContent.replace(/\s+/g, ' ').trim().split(/ — |: |, /)[0];
      return `<button type="button" data-i="${i}">${esc(label)}</button>`;
    }).join('');
    toc.onclick = e => {
      const b = e.target.closest('button[data-i]');
      if (b) this._panels[+b.dataset.i]?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    };
    toc.scrollLeft = 0;
    this._spy();
  },

  _spy() {
    const toc = document.getElementById('toc');
    if (!toc || toc.hidden || !this._panels) return;
    const line = toc.getBoundingClientRect().bottom + 24;
    let at = 0;
    this._panels.forEach((p, i) => { if (p.getBoundingClientRect().top <= line) at = i; });
    toc.querySelectorAll('button').forEach((b, i) => {
      const was = b.classList.contains('active');
      b.classList.toggle('active', i === at);
      if (i === at && !was) toc.scrollTo({ left: Math.max(0, b.offsetLeft - 16), behavior: 'smooth' });
    });
  },
};
