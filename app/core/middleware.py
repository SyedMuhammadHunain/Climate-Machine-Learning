"""Compression and cache headers for the static frontend (web performance quick wins).

Nothing here touches the API: only files under the static prefixes get the cache header, and only
text-like files are gzipped (woff2 and images are already compressed, so gzip would only cost CPU).
"""
from starlette.middleware.gzip import GZipMiddleware

STATIC_PREFIXES = ("/vendor/", "/assets/")
# Filenames are not versioned, so no "immutable": a changed file is picked up within a day.
STATIC_CACHE_CONTROL = "public, max-age=86400, stale-while-revalidate=604800"
ALREADY_COMPRESSED = (".woff2", ".png", ".jpg", ".jpeg", ".webp", ".gz")


class SelectiveGZipMiddleware:
    """GZip responses of 500 bytes or more, except for formats that are already compressed."""

    def __init__(self, app, minimum_size: int = 500):
        self.plain = app
        self.gzipped = GZipMiddleware(app, minimum_size=minimum_size)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].lower().endswith(ALREADY_COMPRESSED):
            return await self.plain(scope, receive, send)
        return await self.gzipped(scope, receive, send)
