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
