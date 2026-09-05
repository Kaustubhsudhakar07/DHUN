"""
User Music DNA Profiling Engine for DhunDNA (Phase 5)

Provides a unified, source-agnostic user taste representation ('UserMusicDNA')
that synthesizes:
1. Manual user inputs (favorite songs, artists, genres, languages, eras, mood)
2. Connected Spotify playlist analysis (parsed playlist tracks & artists)
3. Historical catalog interactions (from Last.fm scrobble histories)

Guarantees that regardless of how a user onboarded (Manual vs Spotify vs Scrobble History),
the same downstream hybrid candidate retrieval and ranking pipeline is invoked.
"""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from src.data.entity_resolution import EntityResolver, normalize_artist, normalize_title

logger = logging.getLogger("dhundna.profile")


@dataclass
class UserMusicDNA:
    """Unified user taste profile for personalized music discovery."""

    user_id: str
    source_type: str  # "manual", "spotify", "catalog_history", "blended"
    favorite_songs: List[str] = field(default_factory=list)
    favorite_artists: List[str] = field(default_factory=list)
    preferred_genres: List[str] = field(default_factory=list)
    preferred_languages: List[str] = field(default_factory=list)
    preferred_eras: List[str] = field(default_factory=list)
    mood: Optional[str] = None
    activity: Optional[str] = None

    # Resolved internal catalog mappings
    resolved_track_ids: List[str] = field(default_factory=list)
    resolved_artist_indices: List[int] = field(default_factory=list)

    # Normalized weight dictionaries
    artist_weights: Dict[str, float] = field(default_factory=dict)
    genre_weights: Dict[str, float] = field(default_factory=dict)
    language_weights: Dict[str, float] = field(default_factory=dict)
    era_weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert DNA profile to dictionary."""
        return asdict(self)

    def get_dominant_genres(self, top_n: int = 3) -> List[str]:
        """Return top N dominant genres by weight."""
        sorted_g = sorted(self.genre_weights.items(), key=lambda x: x[1], reverse=True)
        return [g[0] for g in sorted_g[:top_n]]

    def get_dominant_languages(self, top_n: int = 3) -> List[str]:
        """Return top N dominant languages by weight."""
        sorted_l = sorted(self.language_weights.items(), key=lambda x: x[1], reverse=True)
        return [l[0] for l in sorted_l[:top_n]]

    def get_dominant_artists(self, top_n: int = 5) -> List[str]:
        """Return top N dominant artists by weight."""
        sorted_a = sorted(self.artist_weights.items(), key=lambda x: x[1], reverse=True)
        return [a[0] for a in sorted_a[:top_n]]

    def summary(self) -> str:
        """Generate an explainable taste profile summary."""
        artists = ", ".join(self.get_dominant_artists(3)) or "Eclectic tastes"
        genres = ", ".join(self.get_dominant_genres(2)) or "Multi-genre"
        langs = ", ".join(self.get_dominant_languages(2)) or "All languages"
        mood_str = f" • Mood: {self.mood.title()}" if self.mood else ""
        return f"Taste: {genres} ({langs}) • Anchors: {artists}{mood_str}"

    # =========================================================================
    # FACTORY CONSTRUCTORS
    # =========================================================================

    @classmethod
    def from_manual(
        cls,
        favorite_songs: Optional[List[str]] = None,
        favorite_artists: Optional[List[str]] = None,
        preferred_genres: Optional[List[str]] = None,
        preferred_languages: Optional[List[str]] = None,
        preferred_eras: Optional[List[str]] = None,
        mood: Optional[str] = None,
        activity: Optional[str] = None,
        artist2idx: Optional[Dict[str, int]] = None,
        entity_resolver: Optional[EntityResolver] = None,
        user_id: str = "manual_user",
    ) -> "UserMusicDNA":
        """Build UserMusicDNA from interactive manual onboarding form."""
        songs = [s.strip() for s in (favorite_songs or []) if s.strip()]
        artists = [a.strip() for a in (favorite_artists or []) if a.strip()]
        genres = [g.strip().lower() for g in (preferred_genres or []) if g.strip()]
        languages = [l.strip() for l in (preferred_languages or []) if l.strip()]
        eras = [e.strip() for e in (preferred_eras or []) if e.strip()]

        # Uniform initial weights
        a_weights = {a: 1.0 for a in artists}
        g_weights = {g: 1.0 for g in genres}
        l_weights = {l: 1.0 for l in languages}
        e_weights = {e: 1.0 for e in eras}

        # Resolve artists to catalog indices
        resolved_artist_indices = []
        if artist2idx:
            for a in artists:
                clean_a, _ = normalize_artist(a)
                if clean_a.lower() in artist2idx:
                    resolved_artist_indices.append(artist2idx[clean_a.lower()])

        # Resolve songs to catalog track IDs
        resolved_track_ids = []
        if entity_resolver and songs:
            for s in songs:
                rec, conf, _ = entity_resolver.resolve(query_title=s, threshold=0.80)
                if rec and conf >= 0.80:
                    resolved_track_ids.append(rec["track_id"])

        return cls(
            user_id=user_id,
            source_type="manual",
            favorite_songs=songs,
            favorite_artists=artists,
            preferred_genres=genres,
            preferred_languages=languages,
            preferred_eras=eras,
            mood=mood,
            activity=activity,
            resolved_track_ids=resolved_track_ids,
            resolved_artist_indices=resolved_artist_indices,
            artist_weights=a_weights,
            genre_weights=g_weights,
            language_weights=l_weights,
            era_weights=e_weights,
        )

    @classmethod
    def from_spotify_playlist(
        cls,
        playlist_name: str,
        tracks: List[Dict[str, Any]],
        entity_resolver: Optional[EntityResolver] = None,
        artist2idx: Optional[Dict[str, int]] = None,
        user_id: str = "spotify_user",
    ) -> "UserMusicDNA":
        """Build UserMusicDNA from an authorized Spotify playlist."""
        fav_songs = []
        fav_artists = []
        a_counts: Dict[str, float] = {}
        g_counts: Dict[str, float] = {}
        l_counts: Dict[str, float] = {}
        e_counts: Dict[str, float] = {}
        resolved_tracks = []
        resolved_artist_idxs = []

        for t in tracks:
            title = t.get("name", "")
            artist = t.get("artist", "")
            fav_songs.append(title)
            fav_artists.append(artist)

            # Accumulate artist weights
            primary_a, _ = normalize_artist(artist)
            if primary_a:
                a_counts[primary_a] = a_counts.get(primary_a, 0.0) + 1.0
                if artist2idx and primary_a.lower() in artist2idx:
                    idx = artist2idx[primary_a.lower()]
                    if idx not in resolved_artist_idxs:
                        resolved_artist_idxs.append(idx)

            # Match against master catalog for language, genre, era
            if entity_resolver:
                rec, conf, _ = entity_resolver.resolve(query_title=title, query_artist=artist)
                if rec and conf >= 0.75:
                    resolved_tracks.append(rec["track_id"])
                    lang = rec.get("language", "English / Global")
                    genre = rec.get("genre", "pop")
                    era = rec.get("era", "Unknown")
                    l_counts[lang] = l_counts.get(lang, 0.0) + 1.0
                    g_counts[genre] = g_counts.get(genre, 0.0) + 1.0
                    if era != "Unknown":
                        e_counts[era] = e_counts.get(era, 0.0) + 1.0

        # Normalize counts to [0.0, 1.0]
        def _norm_dict(d: Dict[str, float]) -> Dict[str, float]:
            total = sum(d.values())
            return {k: round(v / total, 3) for k, v in d.items()} if total > 0 else {}

        return cls(
            user_id=user_id,
            source_type="spotify",
            favorite_songs=fav_songs[:20],
            favorite_artists=list(set(fav_artists))[:20],
            preferred_genres=list(g_counts.keys())[:5],
            preferred_languages=list(l_counts.keys())[:3],
            preferred_eras=list(e_counts.keys())[:3],
            mood=f"Curated mix based on {playlist_name}",
            resolved_track_ids=resolved_tracks,
            resolved_artist_indices=resolved_artist_idxs,
            artist_weights=_norm_dict(a_counts),
            genre_weights=_norm_dict(g_counts),
            language_weights=_norm_dict(l_counts),
            era_weights=_norm_dict(e_counts),
        )

    @classmethod
    def from_catalog_user(
        cls,
        user_idx: int,
        interactions_matrix: Any,
        idx2artist: List[str],
        top_n: int = 15,
    ) -> "UserMusicDNA":
        """Build UserMusicDNA from existing Last.fm matrix user index."""
        user_row = interactions_matrix[user_idx]
        artist_indices = user_row.indices
        play_counts = user_row.data

        top_pairs = sorted(zip(artist_indices, play_counts), key=lambda x: x[1], reverse=True)[:top_n]
        top_artists = [idx2artist[idx] for idx, _ in top_pairs if idx < len(idx2artist)]
        total_plays = sum(pc for _, pc in top_pairs) or 1.0
        a_weights = {idx2artist[idx]: round(float(pc) / total_plays, 3) for idx, pc in top_pairs if idx < len(idx2artist)}

        return cls(
            user_id=str(user_idx),
            source_type="catalog_history",
            favorite_artists=top_artists,
            resolved_artist_indices=[idx for idx, _ in top_pairs],
            artist_weights=a_weights,
        )
