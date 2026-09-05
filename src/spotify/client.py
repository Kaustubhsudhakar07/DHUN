"""
Spotify API & OAuth 2.0 Client for DhunDNA (Phase 7)

Complies strictly with Spotify's 2024–2026 Web API developer policies:
- Uses official Authorization Code / Client Credentials flows
- Queries permitted endpoints:
    GET /v1/me (Profile)
    GET /v1/me/playlists (User Playlists)
    GET /v1/playlists/{id}/tracks (Tracks in a playlist)
    GET /v1/me/top/artists & GET /v1/me/top/tracks (Top affinity items)
- Does NOT call deprecated / forbidden endpoints (/audio-features, /recommendations)
- Includes offline fallback with demonstration playlists when Spotify credentials are not set.
"""

import base64
import hashlib
import logging
import os
import secrets
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple
import requests

logger = logging.getLogger("dhundna.spotify")

SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"

PERMITTED_SCOPES = [
    "playlist-read-private",
    "playlist-read-collaborative",
    "user-read-private",
    "user-top-read",
]


class SpotifyClient:
    """Client for official Spotify Web API integration."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: str = "http://127.0.0.1:8501",
    ):
        self.client_id = client_id or os.getenv("SPOTIFY_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET", "")
        self.redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", redirect_uri)
        self.access_token: Optional[str] = None
        self.is_authenticated = False

    @property
    def has_credentials(self) -> bool:
        """Check whether valid Spotify API credentials are configured in environment."""
        return bool(self.client_id and self.client_id != "your_spotify_client_id_here")

    @staticmethod
    def generate_pkce_pair() -> Tuple[str, str]:
        """Generate a random cryptographic PKCE code_verifier and S256 code_challenge."""
        verifier = secrets.token_urlsafe(64)
        digest = hashlib.sha256(verifier.encode("utf-8")).digest()
        challenge = base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")
        return verifier, challenge

    def get_authorization_url(
        self,
        state: str = "dhundna_auth",
        code_challenge: Optional[str] = None,
    ) -> str:
        """Generate Spotify OAuth 2.0 authorization URL with optional PKCE."""
        if not self.has_credentials:
            return ""

        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(PERMITTED_SCOPES),
            "state": state,
            "show_dialog": "true",
        }
        if code_challenge:
            params["code_challenge_method"] = "S256"
            params["code_challenge"] = code_challenge

        return f"{SPOTIFY_AUTH_URL}?{urllib.parse.urlencode(params)}"

    def exchange_code_for_token(
        self,
        code: str,
        code_verifier: Optional[str] = None,
    ) -> bool:
        """Exchange authorization code for an access token using PKCE or Client Secret."""
        if not self.has_credentials:
            return False

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }
        payload = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
        }

        if code_verifier:
            payload["code_verifier"] = code_verifier
        elif self.client_secret:
            auth_header = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
            headers["Authorization"] = f"Basic {auth_header}"
        else:
            logger.error("Neither code_verifier nor client_secret provided for Spotify token exchange.")
            return False

        try:
            resp = requests.post(SPOTIFY_TOKEN_URL, data=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                self.access_token = data.get("access_token")
                self.is_authenticated = True
                logger.info("Successfully authenticated with Spotify via OAuth 2.0.")
                return True
            else:
                logger.error("Spotify token exchange failed: %s %s", resp.status_code, resp.text)
                return False
        except Exception as e:
            logger.error("Error exchanging code for Spotify token: %s", e)
            return False

    def get_user_playlists(self) -> List[Dict[str, Any]]:
        """Fetch accessible user playlists via GET /v1/me/playlists."""
        if not self.is_authenticated or not self.access_token:
            return self._get_demo_playlists()

        headers = {"Authorization": f"Bearer {self.access_token}"}
        try:
            resp = requests.get(f"{SPOTIFY_API_BASE}/me/playlists?limit=20", headers=headers, timeout=10)
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                return [
                    {
                        "id": p["id"],
                        "name": p["name"],
                        "description": p.get("description", ""),
                        "total_tracks": p.get("tracks", {}).get("total", 0),
                        "image_url": p.get("images", [{}])[0].get("url", "") if p.get("images") else "",
                    }
                    for p in items
                ]
        except Exception as e:
            logger.warning("Error fetching Spotify playlists: %s", e)

        return self._get_demo_playlists()

    def get_playlist_tracks(self, playlist_id: str) -> List[Dict[str, Any]]:
        """Fetch tracks from a playlist via GET /v1/playlists/{id}/tracks."""
        if not self.is_authenticated or not self.access_token:
            return self._get_demo_playlist_tracks(playlist_id)

        headers = {"Authorization": f"Bearer {self.access_token}"}
        try:
            resp = requests.get(f"{SPOTIFY_API_BASE}/playlists/{playlist_id}/tracks?limit=50", headers=headers, timeout=10)
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                tracks = []
                for item in items:
                    t = item.get("track")
                    if not t:
                        continue
                    artist_names = [a.get("name", "") for a in t.get("artists", [])]
                    tracks.append({
                        "id": t.get("id", ""),
                        "name": t.get("name", ""),
                        "artist": ", ".join(artist_names),
                        "album": t.get("album", {}).get("name", ""),
                        "release_date": t.get("album", {}).get("release_date", ""),
                        "duration_ms": t.get("duration_ms", 0),
                        "popularity": t.get("popularity", 50),
                    })
                return tracks
        except Exception as e:
            logger.warning("Error fetching tracks for playlist %s: %s", playlist_id, e)

    def get_top_tracks(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch user's top/history tracks via GET /v1/me/top/tracks."""
        if not self.is_authenticated or not self.access_token:
            return self._get_demo_top_tracks()

        headers = {"Authorization": f"Bearer {self.access_token}"}
        try:
            resp = requests.get(f"{SPOTIFY_API_BASE}/me/top/tracks?limit={limit}", headers=headers, timeout=10)
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                tracks = []
                for t in items:
                    artist_names = [a.get("name", "") for a in t.get("artists", [])]
                    tracks.append({
                        "id": t.get("id", ""),
                        "name": t.get("name", ""),
                        "artist": ", ".join(artist_names),
                        "album": t.get("album", {}).get("name", ""),
                        "release_date": t.get("album", {}).get("release_date", ""),
                        "duration_ms": t.get("duration_ms", 0),
                        "popularity": t.get("popularity", 50),
                    })
                if tracks:
                    return tracks
        except Exception as e:
            logger.warning("Error fetching top tracks from Spotify: %s", e)

        return self._get_demo_top_tracks()

    def _get_demo_top_tracks(self) -> List[Dict[str, Any]]:
        """Return demonstration listening history tracks."""
        return [
            {"name": "Kesariya", "artist": "Pritam, Arijit Singh", "album": "Brahmastra"},
            {"name": "Kasoor", "artist": "Prateek Kuhad", "album": "Kasoor"},
            {"name": "Yellow", "artist": "Coldplay", "album": "Parachutes"},
            {"name": "Tum Se Hi", "artist": "Pritam, Mohit Chauhan", "album": "Jab We Met"},
            {"name": "Choo Lo", "artist": "The Local Train", "album": "Aalas Ka Pedh"},
            {"name": "Karma Police", "artist": "Radiohead", "album": "OK Computer"},
        ]

    # =========================================================================
    # Realistic Demo Fallbacks (Zero-Fake, Real Catalog Curations)
    # =========================================================================

    def _get_demo_playlists(self) -> List[Dict[str, Any]]:
        """Return curated real playlist options for demonstration."""
        return [
            {
                "id": "demo_bollywood_favorites",
                "name": "My Bollywood Favorites",
                "description": "Top Hindi romantic and soulful cinema hits",
                "total_tracks": 8,
                "image_url": "",
            },
            {
                "id": "demo_desi_indie",
                "name": "Desi Indie & Acoustic Vibes",
                "description": "Prateek Kuhad, Ritviz, Anuv Jain, and soulful independent tracks",
                "total_tracks": 6,
                "image_url": "",
            },
            {
                "id": "demo_global_hits",
                "name": "Global 2000s Pop-Rock",
                "description": "Radiohead, Coldplay, Daft Punk, and global classics",
                "total_tracks": 6,
                "image_url": "",
            },
        ]

    def _get_demo_playlist_tracks(self, playlist_id: str) -> List[Dict[str, Any]]:
        """Return real catalog tracks for the selected demo playlist."""
        if playlist_id == "demo_bollywood_favorites":
            return [
                {"name": "Kesariya", "artist": "Pritam, Arijit Singh", "album": "Brahmastra"},
                {"name": "Tum Se Hi", "artist": "Pritam, Mohit Chauhan", "album": "Jab We Met"},
                {"name": "Doobey", "artist": "OAFF, Savera, Lothika", "album": "Gehraiyaan"},
                {"name": "Kalank", "artist": "Pritam, Arijit Singh", "album": "Kalank"},
                {"name": "Channa Mereya", "artist": "Pritam, Arijit Singh", "album": "Ae Dil Hai Mushkil"},
                {"name": "Agar Tum Saath Ho", "artist": "A.R. Rahman, Alka Yagnik, Arijit Singh", "album": "Tamasha"},
                {"name": "Deva Deva", "artist": "Pritam, Arijit Singh", "album": "Brahmastra"},
                {"name": "O Re Piya", "artist": "Rahat Fateh Ali Khan", "album": "Aaja Nachle"},
            ]
        elif playlist_id == "demo_desi_indie":
            return [
                {"name": "Kasoor", "artist": "Prateek Kuhad", "album": "Kasoor"},
                {"name": "Gul", "artist": "Anuv Jain", "album": "Gul"},
                {"name": "Udd Gaye", "artist": "Ritviz", "album": "VED"},
                {"name": "Choo Lo", "artist": "The Local Train", "album": "Aalas Ka Pedh"},
                {"name": "cold/mess", "artist": "Prateek Kuhad", "album": "cold/mess"},
                {"name": "Liggi", "artist": "Ritviz", "album": "Liggi"},
            ]
        else:
            return [
                {"name": "Karma Police", "artist": "Radiohead", "album": "OK Computer"},
                {"name": "Yellow", "artist": "Coldplay", "album": "Parachutes"},
                {"name": "One More Time", "artist": "Daft Punk", "album": "Discovery"},
                {"name": "Fix You", "artist": "Coldplay", "album": "X&Y"},
                {"name": "High and Dry", "artist": "Radiohead", "album": "The Bends"},
                {"name": "Harder Better Faster Stronger", "artist": "Daft Punk", "album": "Discovery"},
            ]
