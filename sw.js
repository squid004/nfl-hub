// Service worker for the "Add to Home Screen" PWA install -- NOT a speed-up cache. Every
// same-origin GET goes NETWORK-FIRST: always try the real network response first (so a
// browser that's actually online never gets served something stale -- the exact problem
// index.html's own ?v= cache-busting just fixed at the HTTP-cache layer), and only fall
// back to whatever's cached when the network request fails outright (offline, or this
// device has no signal). The cache here exists purely so the app still opens to SOMETHING
// instead of a blank "no internet" page when offline, not to avoid hitting the network.
//
// Cross-origin requests (Supabase's API, the Plotly/Supabase-js CDN scripts) are left
// completely alone -- never intercepted, never cached -- so this can't interfere with
// CORS behavior or accidentally serve stale data from a service that's supposed to always
// be live.
//
// CACHE_NAME is bumped (alongside index.html's own ?v= date) whenever this file's own
// strategy changes -- `activate` deletes any differently-named cache left over from a
// previous version, so the offline fallback itself can't go stale forever either.
const CACHE_NAME = 'nfl-hub-v2026-10-11';

self.addEventListener('install', event => {
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(names =>
      Promise.all(names.filter(n => n !== CACHE_NAME).map(n => caches.delete(n)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', event => {
  const req = event.request;
  if (req.method !== 'GET' || new URL(req.url).origin !== self.location.origin) {
    return;  // cross-origin (Supabase, CDNs) -- let the browser handle it normally
  }
  event.respondWith(
    fetch(req)
      .then(resp => {
        const copy = resp.clone();
        caches.open(CACHE_NAME).then(cache => cache.put(req, copy));
        return resp;
      })
      .catch(() => caches.match(req))
  );
});
