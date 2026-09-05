"""
FastAPI Application — Music Recommendation System API (Phase 11)

Provides REST endpoints for personalized music recommendations, cold-start
onboarding, explainability, live artist metadata, and playlist generation.
"""

import datetime
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from scipy.sparse import csr_matrix, load_npz

from api.schemas import (
    AnchorArtistItem,
    ArtistInfoResponse,
    ColdStartRequest,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    PlaylistItemSchema,
    PlaylistResponse,
    RecommendationItem,
    RecommendationResponse,
    SimilarArtistItem,
    DnaRecommendationRequest,
    SimilarSongRequest,
    SemanticSearchRequest,
    SpotifyPlaylistRecommendRequest,
    TrackRecommendationItem,
    TrackRecommendationResponse,
)
from src.config import (
    ALS_MODEL,
    ARTIST_MAPPING,
    BPR_MODEL,
    DATA_DIR,
    DEFAULT_HYBRID_WEIGHTS,
    DEFAULT_TOP_K,
    FASTAPI_HOST,
    FASTAPI_PORT,
    HYBRID_WEIGHTS,
    MASTER_CATALOG,
    MODELS_DIR,
    POPULARITY_SCORES,
    TFIDF_MATRIX,
    TFIDF_VECTORIZER,
    TRAIN_INTERACTIONS,
    USER_MAPPING,
)
from src.explainability.explainer import ExplanationEngine
from src.explainability.playlist import PlaylistGenerator
from src.features.tfidf import load_tfidf_features
from src.lastfm.client import LastFMClient
from src.recommenders.als import ALSRecommender
from src.recommenders.bpr import BPRRecommender
from src.recommenders.content import ContentBasedRecommender
from src.recommenders.hybrid import HybridRecommender
from src.recommenders.popularity import PopularityRecommender
from src.data.entity_resolution import EntityResolver
from src.features.semantic import SemanticMusicEngine
from src.profiles.music_dna import UserMusicDNA
from src.recommenders.track_recommender import TrackRecommender
from src.spotify.client import SpotifyClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api")


# =========================================================================
# LIFESPAN & APPLICATION STATE
# =========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models, data mappings, and engines into app.state on startup."""
    logger.info("Initializing Music Recommender API services...")

    # 1. Load mappings
    idx2artist = []
    artist2idx = {}
    if ARTIST_MAPPING.exists():
        with open(ARTIST_MAPPING, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
            idx2artist = mapping_data.get("idx2artist", [])
            artist2idx = mapping_data.get("artist2idx", {})
        logger.info("Loaded artist mapping (%d artists).", len(idx2artist))

    n_users = 0
    if USER_MAPPING.exists():
        with open(USER_MAPPING, "r", encoding="utf-8") as f:
            user_mapping = json.load(f)
            n_users = len(user_mapping.get("idx2user", []))
        logger.info("Loaded user mapping (%d users).", n_users)

    # 2. Load train interaction matrix
    train_matrix = None
    if TRAIN_INTERACTIONS.exists():
        train_matrix = load_npz(TRAIN_INTERACTIONS)
        if n_users == 0:
            n_users = train_matrix.shape[0]
        logger.info("Loaded interaction matrix: %s.", train_matrix.shape)

    # 3. Load ML models
    models_loaded = {}

    # Popularity
    pop_model = None
    if POPULARITY_SCORES.exists() and POPULARITY_SCORES.stat().st_size > 0:
        try:
            pop_model = PopularityRecommender().load(POPULARITY_SCORES, idx2artist=idx2artist)
            models_loaded["popularity"] = True
        except Exception as e:
            logger.warning("Could not load PopularityRecommender: %s", e)
    elif train_matrix is not None:
        pop_model = PopularityRecommender().fit(train_matrix, idx2artist=idx2artist)
        models_loaded["popularity"] = True

    # Content-Based
    content_model = None
    tfidf_matrix = None
    vectorizer = None
    if TFIDF_MATRIX.exists() and TFIDF_VECTORIZER.exists() and train_matrix is not None:
        try:
            tfidf_matrix, vectorizer = load_tfidf_features()
            content_model = ContentBasedRecommender().fit(
                tfidf_matrix=tfidf_matrix,
                train_interactions=train_matrix,
                idx2artist=idx2artist,
                vectorizer=vectorizer,
            )
            models_loaded["content"] = True
        except Exception as e:
            logger.warning("Could not load ContentBasedRecommender: %s", e)

    # ALS
    als_model = None
    if ALS_MODEL.exists() and train_matrix is not None:
        try:
            als_model = ALSRecommender(use_gpu=False).load(
                ALS_MODEL, train_interactions=train_matrix, idx2artist=idx2artist
            )
            models_loaded["als"] = True
        except Exception as e:
            logger.warning("Could not load ALSRecommender: %s", e)

    # BPR
    bpr_model = None
    if BPR_MODEL.exists() and train_matrix is not None:
        try:
            bpr_model = BPRRecommender(use_gpu=False).load(
                BPR_MODEL, train_interactions=train_matrix, idx2artist=idx2artist
            )
            models_loaded["bpr"] = True
        except Exception as e:
            logger.warning("Could not load BPRRecommender: %s", e)

    # Hybrid
    hybrid_model = None
    if als_model and content_model and pop_model:
        try:
            hybrid_model = HybridRecommender(
                cf_recommender=als_model,
                content_recommender=content_model,
                popularity_recommender=pop_model,
            )
            hybrid_model.fit(train_interactions=train_matrix, idx2artist=idx2artist)
            if HYBRID_WEIGHTS.exists():
                hybrid_model.load(HYBRID_WEIGHTS)
            models_loaded["hybrid"] = True
        except Exception as e:
            logger.warning("Could not initialize HybridRecommender: %s", e)

    # 4. Last.fm client, Explainability & Playlist Engines
    lastfm_client = LastFMClient()
    explainer = ExplanationEngine(
        idx2artist=idx2artist,
        tfidf_matrix=tfidf_matrix,
        vectorizer=vectorizer,
        train_interactions=train_matrix,
        lastfm_client=lastfm_client,
    )
    playlist_gen = PlaylistGenerator(lastfm_client=lastfm_client)

    # 5. DhunDNA Master Catalog, Semantic Search & Spotify
    master_df = None
    resolver = None
    semantic_engine = None
    track_recommender = None
    spotify_client = SpotifyClient()

    if MASTER_CATALOG.exists():
        try:
            import pandas as pd
            master_df = pd.read_parquet(MASTER_CATALOG)
            resolver = EntityResolver()
            resolver.index_catalog(master_df.to_dict(orient="records"))
            semantic_engine = SemanticMusicEngine(use_neural=False)  # Fast, lightweight startup
            semantic_engine.fit(master_df, max_tracks=20000)
            track_recommender = TrackRecommender(
                master_catalog=master_df,
                semantic_engine=semantic_engine,
                entity_resolver=resolver,
                cf_recommender=als_model,
                artist2idx=artist2idx,
                idx2artist=idx2artist,
            )
            logger.info("DhunDNA master catalog initialized (%d tracks).", len(master_df))
        except Exception as e:
            logger.warning("Could not initialize DhunDNA master catalog: %s", e)

    # Store state
    app.state.master_catalog = master_df
    app.state.resolver = resolver
    app.state.semantic_engine = semantic_engine
    app.state.track_recommender = track_recommender
    app.state.spotify_client = spotify_client
    app.state.idx2artist = idx2artist
    app.state.artist2idx = artist2idx
    app.state.n_users = n_users
    app.state.train_matrix = train_matrix
    app.state.tfidf_matrix = tfidf_matrix
    app.state.vectorizer = vectorizer
    app.state.models = {
        "hybrid": hybrid_model,
        "als": als_model,
        "content": content_model,
        "popularity": pop_model,
        "bpr": bpr_model,
    }
    app.state.models_loaded = models_loaded
    app.state.lastfm = lastfm_client
    app.state.explainer = explainer
    app.state.playlist_gen = playlist_gen
    app.state.feedback_file = DATA_DIR / "feedback.json"

    logger.info("API services initialized successfully. Models loaded: %s", models_loaded)
    yield
    logger.info("Shutting down API services...")


# =========================================================================
# FASTAPI APP
# =========================================================================

app = FastAPI(
    title="🎵 Personalized Music Recommendation API",
    description=(
        "Hybrid music recommendation engine combining collaborative filtering (ALS, BPR), "
        "content-based tag profiles (TF-IDF), and popularity signals with explainability "
        "and automated playlist curation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================================
# ENDPOINTS
# =========================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check & loaded models status",
)
def get_health():
    """Returns the operational status of the service, loaded models, and catalog scale."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=getattr(app.state, "models_loaded", {}),
        n_users=getattr(app.state, "n_users", 0),
        n_artists=len(getattr(app.state, "idx2artist", [])),
    )


@app.get(
    "/api/v1/recommend/{user_id}",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    summary="Get Top-K personalized recommendations for an existing user",
)
def get_recommendations(
    user_id: int,
    n: int = Query(default=DEFAULT_TOP_K, ge=1, le=50, description="Number of recommendations"),
    model: str = Query(
        default="hybrid",
        enum=["hybrid", "als", "content", "popularity", "bpr"],
        description="Recommendation algorithm to use",
    ),
    explain: bool = Query(default=True, description="Whether to generate explanations"),
):
    """Generate personalized recommendations for a specific user ID."""
    n_users = getattr(app.state, "n_users", 0)
    if user_id < 0 or user_id >= n_users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User ID {user_id} not found. Valid range is 0 to {n_users - 1}.",
        )

    models_dict = getattr(app.state, "models", {})
    recommender = models_dict.get(model)
    if recommender is None or not getattr(recommender, "is_fitted", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{model}' is not loaded or fitted.",
        )

    try:
        raw_recs = recommender.recommend(user_id=user_id, n=n, exclude_known=True)
    except Exception as e:
        logger.exception("Recommendation error for user %d: %s", user_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating recommendations: {str(e)}",
        )

    explainer = getattr(app.state, "explainer", None)
    rec_items = []

    for r in raw_recs:
        item_explanation = r.reasons[0] if r.reasons else "Personalized match"
        shared_tags = []
        anchors = []
        signal_breakdown = {}
        image_url = ""
        lastfm_url = f"https://www.last.fm/music/{r.item_name.replace(' ', '+')}"

        if explain and explainer:
            try:
                exp = explainer.explain(
                    user_id=user_id,
                    item_id=r.item_id,
                    metadata=r.metadata,
                    fetch_online_metadata=False,  # Keep API latency low
                )
                item_explanation = exp.summary
                shared_tags = exp.shared_tags
                signal_breakdown = exp.signal_breakdown
                anchors = [
                    AnchorArtistItem(
                        artist_id=a["artist_id"],
                        artist_name=a["artist_name"],
                        play_count=a["play_count"],
                        shared_tags=a["shared_tags"],
                    )
                    for a in exp.anchor_artists
                ]
            except Exception as e:
                logger.debug("Explanation error for item %d: %s", r.item_id, e)

        rec_items.append(RecommendationItem(
            artist_id=r.item_id,
            artist_name=r.item_name,
            score=round(r.score, 4),
            explanation=item_explanation,
            shared_tags=shared_tags,
            anchor_artists=anchors,
            signal_breakdown=signal_breakdown,
            image_url=image_url,
            lastfm_url=lastfm_url,
        ))

    return RecommendationResponse(
        user_id=user_id,
        model=model,
        count=len(rec_items),
        recommendations=rec_items,
    )


@app.post(
    "/api/v1/recommend/cold-start",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    summary="Get recommendations for a new user based on seed artists or genres",
)
def get_cold_start_recommendations(req: ColdStartRequest):
    """Generate recommendations for onboarding users using seed preferences."""
    artist2idx = getattr(app.state, "artist2idx", {})
    idx2artist = getattr(app.state, "idx2artist", [])
    tfidf_matrix = getattr(app.state, "tfidf_matrix", None)
    vectorizer = getattr(app.state, "vectorizer", None)
    pop_model = getattr(app.state, "models", {}).get("popularity")

    seed_indices = []
    for name in req.seed_artists:
        clean = name.strip().lower()
        if clean in artist2idx:
            seed_indices.append(artist2idx[clean])

    rec_items = []

    # If seed artists exist and TF-IDF is loaded, score candidates via content centroid
    if seed_indices and tfidf_matrix is not None:
        seed_vecs = tfidf_matrix[seed_indices].toarray()
        user_profile = seed_vecs.mean(axis=0)

        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(user_profile.reshape(1, -1), tfidf_matrix).ravel()

        # Zero out seed artists
        for idx in seed_indices:
            sims[idx] = -1.0

        top_indices = np.argsort(sims)[::-1][:req.n]
        for rank_idx in top_indices:
            score = float(sims[rank_idx])
            name = idx2artist[rank_idx] if rank_idx < len(idx2artist) else f"artist_{rank_idx}"
            rec_items.append(RecommendationItem(
                artist_id=int(rank_idx),
                artist_name=name,
                score=round(score, 4),
                explanation=f"Recommended based on your seed artists ({', '.join(req.seed_artists[:3])})",
                signal_breakdown={"content": 100.0},
                lastfm_url=f"https://www.last.fm/music/{name.replace(' ', '+')}",
            ))

    # Fallback to popularity
    elif pop_model and pop_model.is_fitted:
        raw_recs = pop_model.recommend(user_id=0, n=req.n, exclude_known=False)
        for r in raw_recs:
            rec_items.append(RecommendationItem(
                artist_id=r.item_id,
                artist_name=r.item_name,
                score=round(r.score, 4),
                explanation="Popular artist to start your music discovery",
                signal_breakdown={"popularity": 100.0},
                lastfm_url=f"https://www.last.fm/music/{r.item_name.replace(' ', '+')}",
            ))
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neither Content nor Popularity models are available for cold-start.",
        )

    return RecommendationResponse(
        user_id=-1,
        model="cold-start",
        count=len(rec_items),
        recommendations=rec_items,
    )


@app.get(
    "/api/v1/artist/{artist_name}",
    response_model=ArtistInfoResponse,
    tags=["Metadata"],
    summary="Get artist biography, stats, and similar artists",
)
def get_artist(artist_name: str):
    """Retrieve detailed metadata and similar artists via Last.fm API."""
    lastfm: LastFMClient = getattr(app.state, "lastfm", None)
    if not lastfm:
        raise HTTPException(status_code=503, detail="Last.fm client unavailable.")

    info = lastfm.get_artist_info(artist_name)
    similars = lastfm.get_similar_artists(artist_name, limit=5)

    similar_items = [
        SimilarArtistItem(name=s["name"], match=s["match"], url=s["url"])
        for s in similars
    ]

    return ArtistInfoResponse(
        name=info["name"],
        bio=info["bio"],
        tags=info["tags"],
        listeners=info["listeners"],
        playcount=info["playcount"],
        image_url=info["image_url"],
        url=info["url"],
        similar_artists=similar_items,
        is_live=info["is_live"],
    )


@app.get(
    "/api/v1/playlist/{user_id}",
    response_model=PlaylistResponse,
    tags=["Playlists"],
    summary="Generate an interleaved personalized track playlist for a user",
)
def get_playlist(
    user_id: int,
    n_artists: int = Query(default=5, ge=2, le=20, description="Number of recommended artists to sample tracks from"),
    tracks_per_artist: int = Query(default=2, ge=1, le=5, description="Tracks per artist"),
    title: str = Query(default="Personalized Discovery Mix", description="Playlist title"),
):
    """Generate an interleaved track playlist derived from the user's hybrid recommendations."""
    n_users = getattr(app.state, "n_users", 0)
    if user_id < 0 or user_id >= n_users:
        raise HTTPException(status_code=404, detail=f"User ID {user_id} not found.")

    hybrid_model = getattr(app.state, "models", {}).get("hybrid")
    if not hybrid_model:
        raise HTTPException(status_code=503, detail="Hybrid model unavailable.")

    recs = hybrid_model.recommend(user_id=user_id, n=n_artists, exclude_known=True)
    playlist_gen: PlaylistGenerator = getattr(app.state, "playlist_gen", None)

    playlist = playlist_gen.generate_playlist(
        recommendations=recs,
        tracks_per_artist=tracks_per_artist,
        max_tracks=n_artists * tracks_per_artist,
        title=title,
    )

    track_schemas = [
        PlaylistItemSchema(
            track_title=t.track_title,
            artist_name=t.artist_name,
            artist_id=t.artist_id,
            explanation=t.explanation,
            listeners=t.listeners,
            url=t.url,
        )
        for t in playlist.tracks
    ]

    return PlaylistResponse(
        title=playlist.title,
        description=playlist.description,
        total_tracks=playlist.total_tracks,
        tracks=track_schemas,
        genres=playlist.genres,
        created_at=playlist.created_at,
    )


@app.post(
    "/api/v1/feedback",
    response_model=FeedbackResponse,
    tags=["Feedback"],
    summary="Record user interaction feedback",
)
def post_feedback(feedback: FeedbackRequest):
    """Append a user feedback event to the local feedback store."""
    feedback_file: Path = getattr(app.state, "feedback_file", DATA_DIR / "feedback.json")
    feedback_file.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "user_id": feedback.user_id,
        "artist_id": feedback.artist_id,
        "artist_name": feedback.artist_name,
        "feedback_type": feedback.feedback_type,
        "timestamp": feedback.timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    try:
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        return FeedbackResponse(
            status="recorded",
            message=f"Feedback '{feedback.feedback_type}' for {feedback.artist_name} recorded.",
        )
    except Exception as e:
        logger.error("Failed to write feedback: %s", e)
        raise HTTPException(status_code=500, detail="Failed to record feedback.")


# =========================================================================
# DHUNDNA EXTENDED REST ENDPOINTS
# =========================================================================

@app.post(
    "/api/v1/recommend/dna",
    response_model=TrackRecommendationResponse,
    tags=["DhunDNA"],
    summary="Generate personalized track recommendations from a UserMusicDNA profile",
)
def recommend_for_dna(req: DnaRecommendationRequest):
    """Personalized track recommendations for manual onboarding preferences."""
    track_rec: TrackRecommender = getattr(app.state, "track_recommender", None)
    if not track_rec:
        raise HTTPException(status_code=503, detail="DhunDNA Track Recommender is not loaded.")

    artist2idx = getattr(app.state, "artist2idx", {})
    resolver = getattr(app.state, "resolver", None)

    dna = UserMusicDNA.from_manual(
        favorite_songs=req.favorite_songs,
        favorite_artists=req.favorite_artists,
        preferred_genres=req.preferred_genres,
        preferred_languages=req.preferred_languages,
        preferred_eras=req.preferred_eras,
        mood=req.mood,
        activity=req.activity,
        artist2idx=artist2idx,
        entity_resolver=resolver,
    )

    results = track_rec.recommend_for_dna(
        dna=dna,
        top_k=req.n,
        language_filter=req.language_filter,
        era_filter=req.era_filter,
    )

    items = [TrackRecommendationItem(**r) for r in results]
    return TrackRecommendationResponse(
        query_summary=dna.summary(),
        count=len(items),
        recommendations=items,
    )


@app.post(
    "/api/v1/recommend/similar-song",
    response_model=TrackRecommendationResponse,
    tags=["DhunDNA"],
    summary="Find similar songs given a seed track (e.g. 'Kesariya')",
)
def recommend_similar_song(req: SimilarSongRequest):
    """Find Top-K similar songs matching content, genre, language, and artist style."""
    track_rec: TrackRecommender = getattr(app.state, "track_recommender", None)
    if not track_rec:
        raise HTTPException(status_code=503, detail="DhunDNA Track Recommender is not loaded.")

    matched, recs = track_rec.recommend_similar_song(
        song_query=req.song_title,
        artist_query=req.artist_name or "",
        top_k=req.n,
    )

    summary = f"Similar to '{matched['track_name']}' by {matched['artist_name']}" if matched else f"Similar to '{req.song_title}'"
    items = [TrackRecommendationItem(**r) for r in recs]

    return TrackRecommendationResponse(
        query_summary=summary,
        count=len(items),
        recommendations=items,
    )


@app.post(
    "/api/v1/search/semantic",
    response_model=TrackRecommendationResponse,
    tags=["DhunDNA"],
    summary="Natural language semantic song search (e.g. '90s Hindi romantic songs about heartbreak')",
)
def search_semantic(req: SemanticSearchRequest):
    """Retrieve songs matching a natural language descriptive query."""
    sem_engine: SemanticMusicEngine = getattr(app.state, "semantic_engine", None)
    if not sem_engine:
        raise HTTPException(status_code=503, detail="Semantic search engine is not loaded.")

    results = sem_engine.search(
        query=req.query,
        top_k=req.n,
        language_filter=req.language_filter,
        era_filter=req.era_filter,
    )

    items = [
        TrackRecommendationItem(
            track_id=r["track_id"],
            track_name=r["track_name"],
            artist_name=r["artist_name"],
            album_name=r.get("album_name", ""),
            release_year=r.get("release_year", 0),
            era=r.get("era", "Unknown"),
            language=r.get("language", "English / Global"),
            genre=r.get("genre", "pop"),
            popularity=r.get("popularity", 0.0),
            score=r.get("score", 0.0),
            explanation=f"Matches semantic query: '{req.query}'",
        )
        for r in results
    ]

    return TrackRecommendationResponse(
        query_summary=f"Semantic query: '{req.query}'",
        count=len(items),
        recommendations=items,
    )


@app.get(
    "/api/v1/spotify/playlists",
    tags=["DhunDNA Spotify"],
    summary="Get user accessible Spotify playlists (or demo playlists if unauthenticated)",
)
def get_spotify_playlists():
    """Retrieve accessible Spotify playlists."""
    spotify: SpotifyClient = getattr(app.state, "spotify_client", None)
    if not spotify:
        raise HTTPException(status_code=503, detail="Spotify client unavailable.")
    return {
        "is_authenticated": spotify.is_authenticated,
        "has_credentials": spotify.has_credentials,
        "playlists": spotify.get_user_playlists(),
    }


@app.post(
    "/api/v1/recommend/spotify-playlist",
    response_model=TrackRecommendationResponse,
    tags=["DhunDNA Spotify"],
    summary="Generate personalized recommendations from an analyzed Spotify playlist",
)
def recommend_from_spotify_playlist(req: SpotifyPlaylistRecommendRequest):
    """Analyze a Spotify playlist and recommend novel discovery tracks."""
    spotify: SpotifyClient = getattr(app.state, "spotify_client", None)
    track_rec: TrackRecommender = getattr(app.state, "track_recommender", None)
    resolver = getattr(app.state, "resolver", None)
    artist2idx = getattr(app.state, "artist2idx", {})

    if not spotify or not track_rec:
        raise HTTPException(status_code=503, detail="Required recommendation services unavailable.")

    # 1. Fetch playlist or history tracks
    is_history = req.playlist_id in ("user_history", "top_tracks", "listening_history")
    if is_history:
        tracks = spotify.get_top_tracks(limit=30)
    else:
        tracks = spotify.get_playlist_tracks(req.playlist_id)

    if not tracks:
        raise HTTPException(status_code=404, detail="No tracks found for the selected playlist or listening history.")

    # 2. Build UserMusicDNA from playlist or history
    dna = UserMusicDNA.from_spotify_playlist(
        playlist_name=req.playlist_name or ("Listening History" if is_history else "Playlist"),
        tracks=tracks,
        entity_resolver=resolver,
        artist2idx=artist2idx,
        user_id=f"spotify_{req.playlist_id}",
    )

    # 3. Generate recommendations
    results = track_rec.recommend_for_dna(dna=dna, top_k=req.n)
    items = [TrackRecommendationItem(**r) for r in results]

    return TrackRecommendationResponse(
        query_summary=f"Curated mix derived from playlist '{req.playlist_name}' ({len(tracks)} seed tracks analyzed)",
        count=len(items),
        recommendations=items,
    )
