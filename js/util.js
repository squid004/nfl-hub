'use strict';

// Timezone the page displays in. Overridable via ?tz=Area/City.
const TZ = new URLSearchParams(location.search).get('tz') || 'America/New_York';

function parseTs(v) {
  if (!v) return null;
  const d = v instanceof Date ? v : new Date(v);
  return isNaN(d) ? null : d;
}

// "Sun 1:20 PM" (or without the weekday)
function fmtLocal(v, withDay = true) {
  const d = parseTs(v);
  if (!d) return 'TBD';
  const opts = { hour: 'numeric', minute: '2-digit', timeZone: TZ };
  if (withDay) opts.weekday = 'short';
  return new Intl.DateTimeFormat('en-US', opts).format(d);
}

function pct(v) {
  return (v === null || v === undefined) ? '—' : `${Math.round(v * 100)}%`;
}

function signed(v) {
  if (v === null || v === undefined) return '—';
  return v > 0 ? `+${v}` : `${v}`;
}

// "in 3 hr" / "in 45 min" / "in 2 days" / "locked"
function humanize(v) {
  const d = parseTs(v);
  if (!d) return '';
  const secs = (d - Date.now()) / 1000;
  if (secs <= 0) return 'locked';
  const hrs = secs / 3600;
  if (hrs < 1) return `in ${Math.round(secs / 60)} min`;
  if (hrs < 48) return `in ${Math.round(hrs)} hr`;
  return `in ${Math.round(hrs / 24)} days`;
}

function severity(v) {
  const d = parseTs(v);
  if (!d) return '';
  const hrs = (d - Date.now()) / 3600000;
  if (hrs <= 0) return 'muted';
  if (hrs < 3) return 'bad';
  if (hrs < 12) return 'warn';
  return '';
}

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

const BAD_STATUS = ['OUT', 'DOUBTFUL', 'SUSPENSION', 'INJURY_RESERVE'];

// Every tab re-renders by replacing its section's innerHTML (every ~90s reload, plus live-score
// polling), which would snap every <details> shut. This remembers each details element's
// open/closed state by a stable key and re-applies it the moment a re-render inserts the new
// copy. The observer callback is a microtask, so it runs before the browser paints (no
// flicker) and after the render function has finished wiring its own 'toggle' listeners
// (the restored state fires 'toggle', e.g. to lazily draw an ATS card's chart).
const KeepDetailsOpen = {
  _state: new Map(),

  // details.id when present, else owning section + summary text + position among details with
  // that same summary in the section.
  _key(d) {
    if (d.id) return d.id;
    const section = d.closest('section');
    const sum = d.querySelector(':scope > summary');
    const label = sum ? sum.textContent.trim() : '';
    const same = Array.from(section ? section.querySelectorAll('details') : [d])
      .filter(x => { const s = x.querySelector(':scope > summary'); return (s ? s.textContent.trim() : '') === label; });
    return `${section ? section.id : ''}|${label}|${same.indexOf(d)}`;
  },

  init() {
    document.addEventListener('toggle', e => {
      if (e.target instanceof HTMLDetailsElement) this._state.set(this._key(e.target), e.target.open);
    }, true);  // 'toggle' doesn't bubble, so listen in the capture phase

    const main = document.querySelector('main');
    if (!main) return;
    new MutationObserver(muts => {
      for (const m of muts) {
        for (const n of m.addedNodes) {
          if (n.nodeType !== 1) continue;
          const found = n.tagName === 'DETAILS' ? [n] : [];
          found.push(...n.querySelectorAll('details'));
          found.forEach(d => {
            const saved = this._state.get(this._key(d));
            if (saved !== undefined && d.open !== saved) d.open = saved;
          });
        }
      }
    }).observe(main, { childList: true, subtree: true });
  },
};

// Touch devices: Plotly's default drag-to-zoom, pinch/scroll zoom and double-tap-to-reset
// fight page scrolling (a finger that starts on a chart zooms it instead of scrolling), and
// the tiny modebar is hard to hit. On a coarse pointer, charts become view-only: fixed axes,
// no zoom/pan, no modebar, and a wider tap target for hover/click. Taps still show tooltips
// and fire plotly_click (e.g. the Model Trends heatmap cell filter); desktop is untouched.
// Plotly attaches its own non-passive touchstart handler to each drag layer and calls
// preventDefault on it, which blocks scrolling even with dragmode off -- so those handlers
// are removed after every draw; the emulated mouse events from a tap still drive hover/click.
// Charts take their colours from the CSS palette; this fills in what they leave to Plotly's
// defaults -- the page's font and hover labels in the panel colours. Anything a chart sets
// itself wins.
(function plotlyThemeDefaults() {
  if (typeof Plotly === 'undefined') return;
  const css = getComputedStyle(document.documentElement);
  const v = name => css.getPropertyValue(name).trim();
  const font = v('--font') || 'system-ui, sans-serif';
  ['react', 'newPlot'].forEach(fn => {
    const orig = Plotly[fn].bind(Plotly);
    Plotly[fn] = (target, data, layout, config) => {
      const lay = Object.assign({}, layout);
      lay.font = Object.assign({ family: font }, lay.font);
      lay.hoverlabel = Object.assign({ bgcolor: v('--panel2'), bordercolor: v('--line') }, lay.hoverlabel);
      lay.hoverlabel.font = Object.assign({ family: font, color: v('--text') }, lay.hoverlabel.font);
      return orig(target, data, lay, config);
    };
  });
})();

(function makePlotlyTouchFriendly() {
  if (typeof Plotly === 'undefined' || !window.matchMedia || !matchMedia('(pointer: coarse)').matches) return;

  const stripTouchHandlers = gd => {
    gd.querySelectorAll('.draglayer .drag').forEach(el => {
      if (el._ontouchstart) el.removeEventListener('touchstart', el._ontouchstart);
      el.ontouchstart = null;
    });
  };

  const adapt = (gd, layout, config) => {
    if (config && config.staticPlot) return [layout, config];
    const lay = Object.assign({}, layout, { dragmode: false });
    if (lay.hoverdistance == null) lay.hoverdistance = 40;
    ['xaxis', 'yaxis'].forEach(a => { lay[a] = Object.assign({}, lay[a], { fixedrange: true }); });
    const cfg = Object.assign({}, config, { scrollZoom: false, doubleClick: false, displayModeBar: false });
    return [lay, cfg];
  };

  ['react', 'newPlot'].forEach(fn => {
    const orig = Plotly[fn].bind(Plotly);
    Plotly[fn] = (target, data, layout, config) => {
      const [lay, cfg] = adapt(target, layout, config);
      return orig(target, data, lay, cfg).then(gd => {
        const el = typeof target === 'string' ? document.getElementById(target) : target;
        if (el && !cfg.staticPlot) {
          stripTouchHandlers(el);
          if (!el._touchStripWired && el.on) { el.on('plotly_afterplot', () => stripTouchHandlers(el)); el._touchStripWired = true; }
        }
        return gd;
      });
    };
  });
})();

// Explanations behind an ⓘ on a panel's title, so the numbers come first. A renderer puts
// Info.btn(key) in the panel's <h2>, Info.cls(key) on the panel, and class="explain" on each
// explanatory paragraph; the open/closed state survives re-renders.
const Info = {
  _open: new Set(),
  btn(key) {
    const open = this._open.has(key);
    return `<button type="button" class="info-btn" data-info="${esc(key)}" aria-expanded="${open}" title="What this shows" aria-label="About this panel">i</button>`;
  },
  cls(key) { return this._open.has(key) ? ' explain-open' : ''; },
  init() {
    document.addEventListener('click', e => {
      const b = e.target.closest('.info-btn');
      if (!b) return;
      const key = b.dataset.info;
      if (this._open.has(key)) this._open.delete(key); else this._open.add(key);
      const open = this._open.has(key);
      b.setAttribute('aria-expanded', String(open));
      const panel = b.closest('.panel');
      if (panel) panel.classList.toggle('explain-open', open);
    });
  },
};

// Wide tables: a renderer wraps the table as
//   <div class="frame"><div class="frame-scroll"><table class="wide" data-pin="2">...</table></div></div>
// and calls Tables.pin(container) after rendering. The first data-pin columns stay put when
// the table scrolls sideways, a fade shows there's more to see, and a table that fits its
// panel pins its header row to the page instead (a sideways-scrolling box can't do both).
const Tables = {
  pin(root) {
    if (!root) return;
    root.querySelectorAll('table.wide').forEach(table => {
      const frame = table.closest('.frame');
      const scroll = table.closest('.frame-scroll');
      if (!frame || !scroll) return;
      const n = +table.dataset.pin || 0;
      const rows = Array.from(table.rows).filter(r => !r.querySelector('[colspan]'));
      const place = () => {
        let left = 0;
        const head = table.tHead && table.tHead.rows[0];
        for (let i = 0; i < n && head && i < head.cells.length; i++) {
          rows.forEach(r => {
            const c = r.cells[i];
            if (!c) return;
            c.classList.add('pin');
            c.classList.toggle('pin-last', i === n - 1);
            c.style.left = `${left}px`;
          });
          left += head.cells[i].getBoundingClientRect().width;
        }
        frame.style.setProperty('--pin-w', `${left}px`);
        frame.classList.toggle('scrolls-x', table.offsetWidth > scroll.clientWidth + 1);
        frame.classList.toggle('more-right', scroll.scrollLeft + scroll.clientWidth < scroll.scrollWidth - 2);
        frame.classList.toggle('more-left', scroll.scrollLeft > 2);
      };
      if (!scroll._pinWired) {
        scroll.addEventListener('scroll', place, { passive: true });
        scroll._pinWired = true;
      }
      if (table._pinObserver) table._pinObserver.disconnect();
      table._pinObserver = new ResizeObserver(place);
      table._pinObserver.observe(table);
      table._pinObserver.observe(scroll);
      place();
    });
  },
};
