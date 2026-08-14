const CACHE_NAME = 'sahel-malavision-v1';
const urlsToCache = [
  '/',
  '/app_pwa.py',
  '/malaria_detector.keras',
  '/manifest.json'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request))
  );
});