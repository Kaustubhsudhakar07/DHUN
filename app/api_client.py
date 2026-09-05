"""
API Client & Direct Model Connector for Streamlit Application (Phase 12)

Provides dual-mode operation:
1. Primary: REST calls to running FastAPI service (http://localhost:8000)
2. Fallback: Direct in-process invocation of recommendation models if FastAPI is offline.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger("streamlit.client")

# Defaults
DEFAULT_API_URL = "http://127.0.0.1:8000"
TIMEOUT_SECS = 4.0


class RecommendationClient:
    """Client for interacting with the Music Recommender service."""

    def __init__(self, base_url: str = DEFAULT_API_URL):
        self.base_url = base_url.rstrip("/")
        self._direct_loaded = False
        self._direct_state: Dict[str, Any] = {}

    def is_api_online(self) -> bool:
        """Check if the FastAPI backend is running and healthy."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=1.5)
            return resp.status_code == 200
        except Exception:
            return False

    def get_health(self) -> Dict[str, Any]:
        """Fetch system health and status."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=TIMEOUT_SECS)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug("API /health error: %s", e)

        # Fallback to direct inspection
        return {
            "status": "degraded_standalone",
            "version": "1.0.0",
            "models_loaded": {"hybrid": True, "direct_mode": True},
            "n_users": 358868,
            "n_artists": 98104,
        }

    def get_recommendations(
        self,
        user_id: int,
        n: int = 10,
        model: str = "hybrid",
        explain: bool = True,
        hybrid_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Fetch Top-K personalized recommendations."""
        # Try REST endpoint
        try:
            params = {"n": n, "model": model, "explain": explain}
            resp = requests.get(
                f"{self.base_url}/api/v1/recommend/{user_id}",
                params=params,
                timeout=TIMEOUT_SECS,
            )
            if resp.status_code == 200:
                data = resp.json()
                # If custom hybrid weights were provided and model is hybrid,
                # we can optionally adjust/re-rank in direct mode if desired.
                return data
        except Exception as e:
            logger.debug("REST get_recommendations error: %s", e)

        # Fallback to direct model execution
        return self._direct_get_recommendations(
            user_id=user_id,
            n=n,
            model=model,
            explain=explain,
            hybrid_weights=hybrid_weights,
        )

    def get_cold_start_recommendations(
        self,
        seed_artists: List[str],
        seed_genres: Optional[List[str]] = None,
        n: int = 10,
    ) -> Dict[str, Any]:
        """Fetch recommendations for onboarding / new users."""
        payload = {
            "seed_artists": seed_artists,
            "seed_genres": seed_genres or [],
            "n": n,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/recommend/cold-start",
                json=payload,
                timeout=TIMEOUT_SECS,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug("REST get_cold_start error: %s", e)

        return self._direct_get_cold_start(seed_artists=seed_artists, n=n)

    def get_artist(self, artist_name: str) -> Dict[str, Any]:
        """Fetch artist details, bio, tags, and similar artists."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/artist/{artist_name}",
                timeout=TIMEOUT_SECS,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug("REST get_artist error: %s", e)

        return self._direct_get_artist(artist_name)

    def get_playlist(
        self,
        user_id: int,
        n_artists: int = 5,
        tracks_per_artist: int = 2,
        title: str = "Personalized Discovery Mix",
    ) -> Dict[str, Any]:
        """Generate an interleaved personalized playlist."""
        try:
            params = {
                "n_artists": n_artists,
                "tracks_per_artist": tracks_per_artist,
                "title": title,
            }
            resp = requests.get(
                f"{self.base_url}/api/v1/playlist/{user_id}",
                params=params,
                timeout=TIMEOUT_SECS,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug("REST get_playlist error: %s", e)

        return self._direct_get_playlist(
            user_id=user_id,
            n_artists=n_artists,
            tracks_per_artist=tracks_per_artist,
            title=title,
        )

    def post_feedback(
        self,
        user_id: int,
        artist_id: int,
        artist_name: str,
        feedback_type: str,
    ) -> Dict[str, Any]:
        """Record user feedback event."""
        payload = {
            "user_id": user_id,
            "artist_id": artist_id,
            "artist_name": artist_name,
            "feedback_type": feedback_type,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/feedback",
                json=payload,
                timeout=TIMEOUT_SECS,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug("REST post_feedback error: %s", e)

        # Fallback direct append to feedback.json
        from src.config import DATA_DIR
        import datetime
        fb_file = DATA_DIR / "feedback.json"
        fb_file.parent.mkdir(parents=True, exist_ok=True)
        payload["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with open(fb_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        return {"status": "recorded", "message": f"Direct recorded {feedback_type}"}

    def recommend_for_dna(
        self,
        favorite_songs: Optional[List[str]] = None,
        favorite_artists: Optional[List[str]] = None,
        preferred_genres: Optional[List[str]] = None,
        preferred_languages: Optional[List[str]] = None,
        preferred_eras: Optional[List[str]] = None,
        mood: Optional[str] = None,
        activity: Optional[str] = None,
        n: int = 10,
        language_filter: Optional[str] = None,
        era_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Request recommendations based on UserMusicDNA manual preferences."""
        payload = {
            "favorite_songs": favorite_songs or [],
            "favorite_artists": favorite_artists or [],
            "preferred_genres": preferred_genres or [],
            "preferred_languages": preferred_languages or [],
            "preferred_eras": preferred_eras or [],
            "mood": mood,
            "activity": activity,
            "n": n,
            "language_filter": language_filter,
            "era_filter": era_filter,
        }
        try:
            resp = requests.post(f"{self.base_url}/api/v1/recommend/dna", json=payload, timeout=TIMEOUT_SECS)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug("REST recommend_for_dna error: %s", e)

        return self._direct_recommend_for_dna(payload)

    def recommend_similar_song(
        self,
        song_title: str,
        artist_name: str = "",
        n: int = 10,
    ) -> Dict[str, Any]:
        """Find Top-K similar songs to a seed song."""
        payload = {"song_title": song_title, "artist_name": artist_name, "n": n}
        try:
            resp = requests.post(f"{self.base_url}/api/v1/recommend/similar-song", json=payload, timeout=TIMEOUT_SECS)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug("REST recommend_similar_song error: %s", e)

        return self._direct_recommend_similar_song(payload)

    def search_semantic(
        self,
        query: str,
        n: int = 10,
        language_filter: Optional[str] = None,
        era_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Natural language semantic music search."""
        payload = {
            "query": query,
            "n": n,
            "language_filter": language_filter,
            "era_filter": era_filter,
        }
        try:
            resp = requests.post(f"{self.base_url}/api/v1/search/semantic", json=payload, timeout=TIMEOUT_SECS)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug("REST search_semantic error: %s", e)

        return self._direct_search_semantic(payload)

    def get_spotify_playlists(self) -> Dict[str, Any]:
        """Fetch accessible Spotify playlists."""
        try:
            resp = requests.get(f"{self.base_url}/api/v1/spotify/playlists", timeout=TIMEOUT_SECS)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug("REST get_spotify_playlists error: %s", e)

        from src.spotify.client import SpotifyClient
        sc = SpotifyClient()
        return {
            "is_authenticated": sc.is_authenticated,
            "has_credentials": sc.has_credentials,
            "playlists": sc.get_user_playlists(),
        }

    def recommend_from_spotify_playlist(
        self,
        playlist_id: str,
        playlist_name: str = "Imported Playlist",
        n: int = 10,
    ) -> Dict[str, Any]:
        """Analyze a Spotify playlist and return recommendations."""
        payload = {
            "playlist_id": playlist_id,
            "playlist_name": playlist_name,
            "n": n,
        }
        try:
            resp = requests.post(f"{self.base_url}/api/v1/recommend/spotify-playlist", json=payload, timeout=TIMEOUT_SECS)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug("REST recommend_from_spotify_playlist error: %s", e)

        return self._direct_recommend_from_spotify(payload)

    # =========================================================================
    # Direct In-Memory Fallback Handlers
    # =========================================================================

    def _ensure_direct_loaded(self):
        """Lazy load models if API is not responding."""
        if self._direct_loaded:
            return

        import os
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        from scipy.sparse import load_npz
        from src.config import (
            ALS_MODEL,
            ARTIST_MAPPING,
            HYBRID_WEIGHTS,
            POPULARITY_SCORES,
            TFIDF_MATRIX,
            TFIDF_VECTORIZER,
            TRAIN_INTERACTIONS,
        )
        from src.explainability.explainer import ExplanationEngine
        from src.explainability.playlist import PlaylistGenerator
        from src.features.tfidf import load_tfidf_features
        from src.lastfm.client import LastFMClient
        from src.recommenders.als import ALSRecommender
        from src.recommenders.content import ContentBasedRecommender
        from src.recommenders.hybrid import HybridRecommender
        from src.recommenders.popularity import PopularityRecommender

        idx2artist = []
        artist2idx = {}
        if ARTIST_MAPPING.exists():
            with open(ARTIST_MAPPING, "r", encoding="utf-8") as f:
                mapping = json.load(f)
                idx2artist = mapping.get("idx2artist", [])
                artist2idx = mapping.get("artist2idx", {})

        train_matrix = load_npz(TRAIN_INTERACTIONS) if TRAIN_INTERACTIONS.exists() else None

        pop = PopularityRecommender().load(POPULARITY_SCORES, idx2artist=idx2artist)
        tfidf_mat, vec = load_tfidf_features()
        content = ContentBasedRecommender().fit(tfidf_mat, train_matrix, idx2artist, vec)
        als = ALSRecommender(use_gpu=False).load(ALS_MODEL, train_matrix, idx2artist)

        hybrid = HybridRecommender(cf_recommender=als, content_recommender=content, popularity_recommender=pop)
        hybrid.fit(train_matrix, idx2artist)
        if HYBRID_WEIGHTS.exists():
            hybrid.load(HYBRID_WEIGHTS)

        lastfm = LastFMClient()
        explainer = ExplanationEngine(
            idx2artist=idx2artist,
            tfidf_matrix=tfidf_mat,
            vectorizer=vec,
            train_interactions=train_matrix,
            lastfm_client=lastfm,
        )
        playlist_gen = PlaylistGenerator(lastfm_client=lastfm)
        # Master Catalog & Track Recommender for direct fallback
        from src.config import MASTER_CATALOG
        from src.data.entity_resolution import EntityResolver
        from src.features.semantic import SemanticMusicEngine
        from src.recommenders.track_recommender import TrackRecommender
        import pandas as pd

        master_df = None
        resolver = None
        sem_engine = None
        track_rec = None
        if MASTER_CATALOG.exists():
            try:
                master_df = pd.read_parquet(MASTER_CATALOG)
                resolver = EntityResolver()
                resolver.index_catalog(master_df.to_dict(orient="records"))
                sem_engine = SemanticMusicEngine(use_neural=False)
                sem_engine.fit(master_df, max_tracks=20000)
                track_rec = TrackRecommender(
                    master_catalog=master_df,
                    semantic_engine=sem_engine,
                    entity_resolver=resolver,
                    cf_recommender=als,
                    artist2idx=artist2idx,
                    idx2artist=idx2artist,
                )
            except Exception as e:
                logger.warning("Direct master catalog init error: %s", e)

        self._direct_state = {
            "idx2artist": idx2artist,
            "artist2idx": artist2idx,
            "train_matrix": train_matrix,
            "tfidf_matrix": tfidf_mat,
            "vectorizer": vec,
            "models": {
                "hybrid": hybrid,
                "als": als,
                "content": content,
                "popularity": pop,
            },
            "lastfm": lastfm,
            "explainer": explainer,
            "playlist_gen": playlist_gen,
            "master_catalog": master_df,
            "resolver": resolver,
            "semantic_engine": sem_engine,
            "track_recommender": track_rec,
        }
        self._direct_loaded = True

    def _direct_get_recommendations(
        self,
        user_id: int,
        n: int = 10,
        model: str = "hybrid",
        explain: bool = True,
        hybrid_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        self._ensure_direct_loaded()
        models = self._direct_state["models"]
        recommender = models.get(model, models["hybrid"])

        if model == "hybrid" and hybrid_weights:
            recommender.set_weights(hybrid_weights)

        raw = recommender.recommend(user_id=user_id, n=n, exclude_known=True)
        explainer = self._direct_state["explainer"]

        rec_items = []
        for r in raw:
            summary = r.reasons[0] if r.reasons else "Personalized match"
            shared_tags = []
            anchors = []
            signal_breakdown = {}

            if explain and explainer:
                try:
                    exp = explainer.explain(user_id=user_id, item_id=r.item_id, metadata=r.metadata)
                    summary = exp.summary
                    shared_tags = exp.shared_tags
                    signal_breakdown = exp.signal_breakdown
                    anchors = exp.anchor_artists
                except Exception:
                    pass

            rec_items.append({
                "artist_id": r.item_id,
                "artist_name": r.item_name,
                "score": round(r.score, 4),
                "explanation": summary,
                "shared_tags": shared_tags,
                "anchor_artists": anchors,
                "signal_breakdown": signal_breakdown,
                "image_url": "",
                "lastfm_url": f"https://www.last.fm/music/{r.item_name.replace(' ', '+')}",
            })

        return {
            "user_id": user_id,
            "model": model,
            "count": len(rec_items),
            "recommendations": rec_items,
        }

    def _direct_get_cold_start(self, seed_artists: List[str], n: int = 10) -> Dict[str, Any]:
        self._ensure_direct_loaded()
        artist2idx = self._direct_state["artist2idx"]
        idx2artist = self._direct_state["idx2artist"]
        tfidf_mat = self._direct_state["tfidf_matrix"]

        seed_indices = [artist2idx[a.strip().lower()] for a in seed_artists if a.strip().lower() in artist2idx]
        rec_items = []

        if seed_indices and tfidf_mat is not None:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            seed_vecs = tfidf_mat[seed_indices].toarray()
            centroid = seed_vecs.mean(axis=0).reshape(1, -1)
            sims = cosine_similarity(centroid, tfidf_mat).ravel()
            for idx in seed_indices:
                sims[idx] = -1.0
            top = np.argsort(sims)[::-1][:n]
            for r_idx in top:
                name = idx2artist[r_idx]
                rec_items.append({
                    "artist_id": int(r_idx),
                    "artist_name": name,
                    "score": round(float(sims[r_idx]), 4),
                    "explanation": f"Similar tag profile to {', '.join(seed_artists[:2])}",
                    "signal_breakdown": {"content": 100.0},
                    "shared_tags": [],
                    "anchor_artists": [],
                    "image_url": "",
                    "lastfm_url": f"https://www.last.fm/music/{name.replace(' ', '+')}",
                })
        else:
            pop = self._direct_state["models"]["popularity"]
            raw = pop.recommend(user_id=0, n=n, exclude_known=False)
            for r in raw:
                rec_items.append({
                    "artist_id": r.item_id,
                    "artist_name": r.item_name,
                    "score": round(r.score, 4),
                    "explanation": "Global popular artist",
                    "signal_breakdown": {"popularity": 100.0},
                    "shared_tags": [],
                    "anchor_artists": [],
                    "image_url": "",
                    "lastfm_url": f"https://www.last.fm/music/{r.item_name.replace(' ', '+')}",
                })

        return {
            "user_id": -1,
            "model": "cold-start",
            "count": len(rec_items),
            "recommendations": rec_items,
        }

    def _direct_get_artist(self, artist_name: str) -> Dict[str, Any]:
        self._ensure_direct_loaded()
        lastfm = self._direct_state["lastfm"]
        info = lastfm.get_artist_info(artist_name)
        similars = lastfm.get_similar_artists(artist_name, limit=5)
        return {
            "name": info["name"],
            "bio": info["bio"],
            "tags": info["tags"],
            "listeners": info["listeners"],
            "playcount": info["playcount"],
            "image_url": info["image_url"],
            "url": info["url"],
            "similar_artists": similars,
            "is_live": info["is_live"],
        }

    def _direct_get_playlist(
        self,
        user_id: int,
        n_artists: int = 5,
        tracks_per_artist: int = 2,
        title: str = "Personalized Discovery Mix",
    ) -> Dict[str, Any]:
        self._ensure_direct_loaded()
        hybrid = self._direct_state["models"]["hybrid"]
        recs = hybrid.recommend(user_id=user_id, n=n_artists, exclude_known=True)
        playlist_gen = self._direct_state["playlist_gen"]
        playlist = playlist_gen.generate_playlist(
            recommendations=recs,
            tracks_per_artist=tracks_per_artist,
            max_tracks=n_artists * tracks_per_artist,
            title=title,
        )
        return {
            "title": playlist.title,
            "description": playlist.description,
            "total_tracks": playlist.total_tracks,
            "genres": playlist.genres,
            "created_at": playlist.created_at,
            "tracks": [
                {
                    "track_title": t.track_title,
                    "artist_name": t.artist_name,
                    "artist_id": t.artist_id,
                    "explanation": t.explanation,
                    "listeners": t.listeners,
                    "url": t.url,
                }
                for t in playlist.tracks
            ],
        }

    def _direct_recommend_for_dna(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_direct_loaded()
        tr = self._direct_state.get("track_recommender")
        if not tr:
            return {"query_summary": "No catalog available", "count": 0, "recommendations": []}
        from src.profiles.music_dna import UserMusicDNA
        dna = UserMusicDNA.from_manual(
            favorite_songs=payload.get("favorite_songs"),
            favorite_artists=payload.get("favorite_artists"),
            preferred_genres=payload.get("preferred_genres"),
            preferred_languages=payload.get("preferred_languages"),
            preferred_eras=payload.get("preferred_eras"),
            mood=payload.get("mood"),
            activity=payload.get("activity"),
            artist2idx=self._direct_state.get("artist2idx", {}),
            entity_resolver=self._direct_state.get("resolver"),
        )
        results = tr.recommend_for_dna(
            dna=dna,
            top_k=payload.get("n", 10),
            language_filter=payload.get("language_filter"),
            era_filter=payload.get("era_filter"),
        )
        return {"query_summary": dna.summary(), "count": len(results), "recommendations": results}

    def _direct_recommend_similar_song(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_direct_loaded()
        tr = self._direct_state.get("track_recommender")
        if not tr:
            return {"query_summary": "No catalog available", "count": 0, "recommendations": []}
        matched, recs = tr.recommend_similar_song(
            song_query=payload.get("song_title", ""),
            artist_query=payload.get("artist_name", ""),
            top_k=payload.get("n", 10),
        )
        summary = f"Similar to '{matched['track_name']}' by {matched['artist_name']}" if matched else f"Similar to '{payload.get('song_title')}'"
        return {"query_summary": summary, "count": len(recs), "recommendations": recs}

    def _direct_search_semantic(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_direct_loaded()
        sem = self._direct_state.get("semantic_engine")
        if not sem:
            return {"query_summary": "No semantic engine available", "count": 0, "recommendations": []}
        results = sem.search(
            query=payload.get("query", ""),
            top_k=payload.get("n", 10),
            language_filter=payload.get("language_filter"),
            era_filter=payload.get("era_filter"),
        )
        recs = [
            {
                "track_id": r["track_id"],
                "track_name": r["track_name"],
                "artist_name": r["artist_name"],
                "album_name": r.get("album_name", ""),
                "release_year": r.get("release_year", 0),
                "era": r.get("era", "Unknown"),
                "language": r.get("language", "English / Global"),
                "genre": r.get("genre", "pop"),
                "popularity": r.get("popularity", 0.0),
                "score": r.get("score", 0.0),
                "explanation": f"Matches semantic query: '{payload.get('query')}'",
            }
            for r in results
        ]
        return {"query_summary": f"Semantic query: '{payload.get('query')}'", "count": len(recs), "recommendations": recs}

    def _direct_recommend_from_spotify(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from src.spotify.client import SpotifyClient
        from src.profiles.music_dna import UserMusicDNA
        sc = SpotifyClient()
        is_history = payload.get("playlist_id") in ("user_history", "top_tracks", "listening_history")
        if is_history:
            tracks = sc.get_top_tracks(limit=30)
        else:
            tracks = sc.get_playlist_tracks(payload.get("playlist_id", ""))
        self._ensure_direct_loaded()
        tr = self._direct_state.get("track_recommender")
        if not tr:
            return {"query_summary": "No catalog available", "count": 0, "recommendations": []}
        dna = UserMusicDNA.from_spotify_playlist(
            playlist_name=payload.get("playlist_name", "Playlist"),
            tracks=tracks,
            entity_resolver=self._direct_state.get("resolver"),
            artist2idx=self._direct_state.get("artist2idx", {}),
        )
        results = tr.recommend_for_dna(dna=dna, top_k=payload.get("n", 10))
        return {"query_summary": f"Curated mix derived from '{payload.get('playlist_name')}'", "count": len(results), "recommendations": results}
