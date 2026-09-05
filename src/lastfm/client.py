"""
Last.fm API Client — Phase 10

Provides rate-limited, cached access to the Last.fm Web Services 2.0 API
for fetching live artist metadata, bios, tags, top tracks, and similar artists.

Gracefully falls back to offline/mock mode if no API key is configured or
if network requests fail.
"""

import logging
import os
import time
from typing import Any
from urllib.parse import urlencode

import requests

from src.config import (
    LASTFM_API_BASE_URL,
    LASTFM_API_KEY,
    LASTFM_MAX_RETRIES,
    LASTFM_RATE_LIMIT_CALLS,
    LASTFM_REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)


class LastFMClient:
    """Rate-limited, cached client for the Last.fm Web Services 2.0 API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = LASTFM_API_BASE_URL,
        rate_limit_calls: int = LASTFM_RATE_LIMIT_CALLS,
        timeout: int = LASTFM_REQUEST_TIMEOUT,
        max_retries: int = LASTFM_MAX_RETRIES,
        cache_size: int = 500,
    ):
        """Initialize Last.fm API client.

        Args:
            api_key: Optional API key. Defaults to LASTFM_API_KEY config.
            base_url: Base URL for Last.fm API.
            rate_limit_calls: Maximum requests per second.
            timeout: HTTP request timeout in seconds.
            max_retries: Maximum number of retries on transient errors.
            cache_size: Maximum entries in in-memory response cache.
        """
        raw_key = api_key if api_key is not None else LASTFM_API_KEY
        self.api_key = raw_key.strip() if raw_key else ""
        self.base_url = base_url
        self.min_interval = 1.0 / max(1, rate_limit_calls)
        self.timeout = timeout
        self.max_retries = max_retries

        self._last_request_time = 0.0
        self._cache: dict[str, Any] = {}
        self._cache_size = cache_size

        # Is live mode active?
        self.is_live = bool(
            self.api_key and self.api_key.upper() != "YOUR_API_KEY_HERE"
        )
        if self.is_live:
            logger.info("LastFMClient initialized in LIVE mode (API key configured).")
        else:
            logger.info("LastFMClient initialized in OFFLINE/FALLBACK mode (no active API key).")

    def _rate_limit(self) -> None:
        """Enforce rate limit between outgoing HTTP requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.time()

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """Send a rate-limited request to Last.fm API with caching and retries."""
        if not self.is_live:
            return None

        # Build cache key
        query_params = {
            "method": method,
            "api_key": self.api_key,
            "format": "json",
            **params,
        }
        cache_key = f"{method}:{urlencode(sorted(params.items()))}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        for attempt in range(1, self.max_retries + 1):
            try:
                self._rate_limit()
                response = requests.get(
                    self.base_url,
                    params=query_params,
                    timeout=self.timeout,
                    headers={"User-Agent": "MusicRecommenderSystem/1.0"},
                )
                if response.status_code == 200:
                    data = response.json()
                    # Check for API-level errors
                    if "error" in data:
                        logger.warning("Last.fm API returned error code %s: %s", data.get("error"), data.get("message"))
                        return None
                    # Cache result
                    if len(self._cache) >= self._cache_size:
                        # Evict oldest entry
                        first_key = next(iter(self._cache))
                        del self._cache[first_key]
                    self._cache[cache_key] = data
                    return data
                elif response.status_code in (429, 500, 502, 503):
                    # Rate limit or server error: wait with exponential backoff
                    sleep_time = attempt * 0.5
                    logger.debug("Last.fm API status %d, retrying in %.1fs...", response.status_code, sleep_time)
                    time.sleep(sleep_time)
                else:
                    logger.warning("Last.fm API HTTP %d for method %s", response.status_code, method)
                    return None
            except requests.RequestException as e:
                logger.warning("Network error contacting Last.fm API (attempt %d/%d): %s", attempt, self.max_retries, e)
                time.sleep(attempt * 0.5)

        return None

    def get_artist_info(self, artist_name: str) -> dict[str, Any]:
        """Fetch artist metadata, bio, tags, and stats.

        Args:
            artist_name: Canonical artist name.

        Returns:
            Dictionary with name, bio, tags, listeners, playcount, url, image_url.
        """
        clean_name = artist_name.strip()
        data = self._request("artist.getInfo", {"artist": clean_name, "autocorrect": 1})

        if data and "artist" in data:
            art = data["artist"]
            # Extract bio summary (stripping HTML link if present)
            bio_text = art.get("bio", {}).get("summary", "")
            if "<a href" in bio_text:
                bio_text = bio_text.split("<a href")[0].strip()

            # Extract tags
            tags_data = art.get("tags", {}).get("tag", [])
            if isinstance(tags_data, dict):
                tags_data = [tags_data]
            tags = [t.get("name", "").lower() for t in tags_data if t.get("name")]

            # Extract image (prefer large/extralarge)
            images = art.get("image", [])
            image_url = ""
            if images:
                for img in reversed(images):
                    if img.get("#text"):
                        image_url = img.get("#text")
                        break

            listeners = int(art.get("stats", {}).get("listeners", 0))
            playcount = int(art.get("stats", {}).get("playcount", 0))

            return {
                "name": art.get("name", clean_name),
                "bio": bio_text,
                "tags": tags,
                "listeners": listeners,
                "playcount": playcount,
                "url": art.get("url", f"https://www.last.fm/music/{clean_name.replace(' ', '+')}"),
                "image_url": image_url,
                "is_live": True,
            }

        # Offline fallback
        return {
            "name": clean_name,
            "bio": f"{clean_name.title()} is an artist in the music catalog.",
            "tags": [],
            "listeners": 0,
            "playcount": 0,
            "url": f"https://www.last.fm/music/{clean_name.replace(' ', '+')}",
            "image_url": "",
            "is_live": False,
        }

    def get_top_tracks(self, artist_name: str, limit: int = 5) -> list[dict[str, Any]]:
        """Fetch top popular tracks for an artist.

        Args:
            artist_name: Artist name.
            limit: Maximum tracks to return.

        Returns:
            List of track dicts with name, listeners, playcount, url.
        """
        clean_name = artist_name.strip()
        data = self._request("artist.getTopTracks", {"artist": clean_name, "limit": limit, "autocorrect": 1})

        if data and "toptracks" in data and "track" in data["toptracks"]:
            raw_tracks = data["toptracks"]["track"]
            if isinstance(raw_tracks, dict):
                raw_tracks = [raw_tracks]

            tracks = []
            for t in raw_tracks[:limit]:
                tracks.append({
                    "title": t.get("name", ""),
                    "artist": clean_name,
                    "listeners": int(t.get("listeners", 0)),
                    "playcount": int(t.get("playcount", 0)),
                    "url": t.get("url", ""),
                    "is_live": True,
                })
            return tracks

        # Offline fallback: generic essential tracks
        return [
            {
                "title": f"Top Track #{i}",
                "artist": clean_name,
                "listeners": 0,
                "playcount": 0,
                "url": f"https://www.last.fm/music/{clean_name.replace(' ', '+')}",
                "is_live": False,
            }
            for i in range(1, limit + 1)
        ]

    def get_similar_artists(self, artist_name: str, limit: int = 10) -> list[dict[str, Any]]:
        """Fetch artists similar to the given artist from Last.fm's crowd graph.

        Args:
            artist_name: Artist name.
            limit: Maximum similar artists to return.

        Returns:
            List of dicts with name, match_score, url.
        """
        clean_name = artist_name.strip()
        data = self._request("artist.getSimilar", {"artist": clean_name, "limit": limit, "autocorrect": 1})

        if data and "similarartists" in data and "artist" in data["similarartists"]:
            raw_artists = data["similarartists"]["artist"]
            if isinstance(raw_artists, dict):
                raw_artists = [raw_artists]

            similar = []
            for a in raw_artists[:limit]:
                similar.append({
                    "name": a.get("name", ""),
                    "match": float(a.get("match", 0.0)),
                    "url": a.get("url", ""),
                    "is_live": True,
                })
            return similar

        return []
