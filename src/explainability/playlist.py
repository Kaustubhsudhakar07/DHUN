"""
Personalized Playlist Generator — Phase 10

Curates actionable track playlists from Top-K recommended artists:
- Queries top tracks per recommended artist via Last.fm API
- Interleaves tracks across artists for listening variety
- Attaches recommendation rationale to each track
- Aggregates dominant genres/tags into a playlist overview
"""

import datetime
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from src.lastfm.client import LastFMClient
from src.recommenders.base import Recommendation

logger = logging.getLogger(__name__)


@dataclass
class PlaylistItem:
    """Individual track in a personalized playlist."""
    track_title: str
    artist_name: str
    artist_id: int
    explanation: str = ""
    listeners: int = 0
    url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Playlist:
    """Personalized playlist curated from recommendations."""
    title: str
    description: str
    tracks: list[PlaylistItem] = field(default_factory=list)
    total_tracks: int = 0
    genres: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "tracks": [t.to_dict() for t in self.tracks],
            "total_tracks": self.total_tracks,
            "genres": self.genres,
            "created_at": self.created_at,
        }


class PlaylistGenerator:
    """Generates structured playlists from recommendation lists."""

    def __init__(self, lastfm_client: LastFMClient | None = None):
        """Initialize playlist generator with Last.fm API client."""
        self.lastfm = lastfm_client or LastFMClient()

    def generate_playlist(
        self,
        recommendations: Sequence[Recommendation],
        tracks_per_artist: int = 2,
        max_tracks: int = 20,
        title: str = "Personalized Discovery Mix",
        description: str | None = None,
    ) -> Playlist:
        """Construct an interleaved playlist from recommended artists.

        Args:
            recommendations: Sequence of Recommendation objects.
            tracks_per_artist: Number of top tracks to select per artist.
            max_tracks: Maximum total tracks in the playlist.
            title: Playlist title.
            description: Optional custom description.

        Returns:
            Playlist object.
        """
        if not recommendations:
            return Playlist(
                title=title,
                description=description or "No recommendations available to build playlist.",
                tracks=[],
                total_tracks=0,
            )

        # 1. Fetch tracks per artist
        artist_track_buckets = []
        dominant_tags = set()

        for rec in recommendations:
            artist_name = rec.item_name
            reason = rec.reasons[0] if rec.reasons else "Personalized recommendation"

            tracks_info = self.lastfm.get_top_tracks(artist_name, limit=tracks_per_artist)
            items = []
            for t in tracks_info:
                items.append(PlaylistItem(
                    track_title=t.get("title", "Popular Track"),
                    artist_name=artist_name,
                    artist_id=rec.item_id,
                    explanation=reason,
                    listeners=t.get("listeners", 0),
                    url=t.get("url", ""),
                ))
            artist_track_buckets.append(items)

        # 2. Interleave tracks across artists for smooth variety
        # E.g., [Artist 1 Track 1, Artist 2 Track 1, ..., Artist 1 Track 2, Artist 2 Track 2]
        interleaved_tracks: list[PlaylistItem] = []
        max_bucket_len = max((len(b) for b in artist_track_buckets), default=0)

        for track_idx in range(max_bucket_len):
            for bucket in artist_track_buckets:
                if track_idx < len(bucket):
                    interleaved_tracks.append(bucket[track_idx])
                    if len(interleaved_tracks) >= max_tracks:
                        break
            if len(interleaved_tracks) >= max_tracks:
                break

        # 3. Build description
        artist_names = [r.item_name.title() for r in recommendations[:5]]
        default_desc = (
            f"Curated mix featuring {', '.join(artist_names)} and more, "
            f"tailored to your unique listening tastes."
        )

        return Playlist(
            title=title,
            description=description or default_desc,
            tracks=interleaved_tracks,
            total_tracks=len(interleaved_tracks),
            genres=sorted(list(dominant_tags)),
        )
