const CACHE_NAME = 'klaverjassen-v1';

const STATIC_ASSETS = [
    '/static/style.css',
    '/static/game_app.js',
    '/static/game_render.js',
    '/static/game_state.js',
    '/static/i18n.js',
    '/static/manifest.webmanifest',
    '/static/icons/icon-192.png',
    '/static/icons/icon-512.png',
    '/static/icons/icon-maskable-512.png',
    '/static/icons/icon-180.png',
];

self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME).then(cache => cache.addAll(STATIC_ASSETS))
    );
    self.skipWaiting();
});

self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(keys =>
            Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
        )
    );
    self.clients.claim();
});

self.addEventListener('fetch', event => {
    // Let the browser handle WebSocket upgrades and Socket.IO traffic natively.
    if (event.request.url.includes('/socket.io')) return;

    // Network-first for static assets so deployed updates are picked up,
    // with cache fallback for offline support.
    if (new URL(event.request.url).pathname.startsWith('/static/')) {
        event.respondWith(
            fetch(event.request)
                .then(response => {
                    if (response && response.ok) {
                        const responseClone = response.clone();
                        caches.open(CACHE_NAME).then(cache => {
                            cache.put(event.request, responseClone);
                        });
                    }
                    return response;
                })
                .catch(() => caches.match(event.request))
        );
        return;
    }

    // Network-first for everything else (HTML, game API).
    event.respondWith(fetch(event.request));
});
