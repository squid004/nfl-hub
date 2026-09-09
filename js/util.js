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
