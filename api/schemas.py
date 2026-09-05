"""
Pydantic Schemas for Music Recommender REST API — Phase 11
"""

from typing import Any, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = "healthy"
    version: str = "1.0.0"
    models_loaded: dict[str, bool] = Field(default_factory=dict)
    n_users: int = 0
    n_artists: int = 0


class AnchorArtistItem(BaseModel):
    """Anchor artist attribution."""
    artist_id: int
    artist_name: str
    play_count: int = 0
    shared_tags: List[str] = Field(default_factory=list)


class RecommendationItem(BaseModel):
    """Individual recommended artist with score and explanation."""
    artist_id: int
    artist_name: str
    score: float
    explanation: str
    shared_tags: List[str] = Field(default_factory=list)
    anchor_artists: List[AnchorArtistItem] = Field(default_factory=list)
    signal_breakdown: dict[str, float] = Field(default_factory=dict)
    image_url: Optional[str] = ""
    lastfm_url: Optional[str] = ""


class RecommendationResponse(BaseModel):
    """Response containing personalized recommendations."""
    user_id: int
    model: str
    count: int
    recommendations: List[RecommendationItem]


class ColdStartRequest(BaseModel):
    """Request payload for cold-start onboarding recommendations."""
    seed_artists: List[str] = Field(
        default_factory=list,
        description="List of artist names the new user likes.",
    )
    seed_genres: List[str] = Field(
        default_factory=list,
        description="List of preferred genres/tags (e.g. rock, indie, electronic).",
    )
    n: int = Field(default=10, ge=1, le=50, description="Number of recommendations.")


class PlaylistItemSchema(BaseModel):
    """Single track in a personalized playlist."""
    track_title: str
    artist_name: str
    artist_id: int
    explanation: str = ""
    listeners: int = 0
    url: str = ""


class PlaylistResponse(BaseModel):
    """Personalized playlist response schema."""
    title: str
    description: str
    total_tracks: int
    tracks: List[PlaylistItemSchema]
    genres: List[str] = Field(default_factory=list)
    created_at: str = ""


class SimilarArtistItem(BaseModel):
    """Similar artist item schema."""
    name: str
    match: float = 0.0
    url: str = ""


class ArtistInfoResponse(BaseModel):
    """Detailed artist metadata from Last.fm API."""
    name: str
    bio: str = ""
    tags: List[str] = Field(default_factory=list)
    listeners: int = 0
    playcount: int = 0
    image_url: str = ""
    url: str = ""
    similar_artists: List[SimilarArtistItem] = Field(default_factory=list)
    is_live: bool = False


class FeedbackRequest(BaseModel):
    """User interaction feedback event."""
    user_id: Optional[int] = None
    artist_id: int
    artist_name: str
    feedback_type: str = Field(
        ...,
        description="One of: thumbs_up, thumbs_down, listen, skip, favorite",
    )
    timestamp: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Feedback acknowledgement."""
    status: str = "recorded"
    message: str


# =========================================================================
# DHUNDNA EXTENDED SCHEMAS (TRACK-LEVEL, DNA, SEMANTIC, SPOTIFY)
# =========================================================================

class TrackRecommendationItem(BaseModel):
    """Recommended song / track item with rich metadata."""
    track_id: str
    track_name: str
    artist_name: str
    album_name: Optional[str] = ""
    release_year: int = 0
    era: str = "Unknown"
    language: str = "English / Global"
    genre: str = "pop"
    popularity: float = 0.0
    score: float = 0.0
    spotify_id: Optional[str] = ""
    explanation: str = ""


class TrackRecommendationResponse(BaseModel):
    """Response containing Top-K recommended tracks."""
    query_summary: str
    count: int
    recommendations: List[TrackRecommendationItem]


class DnaRecommendationRequest(BaseModel):
    """Request payload for personalized recommendations based on UserMusicDNA."""
    favorite_songs: List[str] = Field(default_factory=list)
    favorite_artists: List[str] = Field(default_factory=list)
    preferred_genres: List[str] = Field(default_factory=list)
    preferred_languages: List[str] = Field(default_factory=list)
    preferred_eras: List[str] = Field(default_factory=list)
    mood: Optional[str] = None
    activity: Optional[str] = None
    n: int = Field(default=10, ge=1, le=50)
    language_filter: Optional[str] = None
    era_filter: Optional[str] = None


class SimilarSongRequest(BaseModel):
    """Request payload to discover similar songs to a seed song."""
    song_title: str
    artist_name: Optional[str] = ""
    n: int = Field(default=10, ge=1, le=50)


class SemanticSearchRequest(BaseModel):
    """Request payload for natural language semantic song retrieval."""
    query: str
    n: int = Field(default=10, ge=1, le=50)
    language_filter: Optional[str] = None
    era_filter: Optional[str] = None


class SpotifyPlaylistRecommendRequest(BaseModel):
    """Request payload to generate recommendations from a Spotify playlist."""
    playlist_id: str
    playlist_name: Optional[str] = "Imported Playlist"
    n: int = Field(default=10, ge=1, le=50)
