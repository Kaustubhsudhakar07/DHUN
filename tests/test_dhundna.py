"""
Tests for DhunDNA: Indian + Global Music Discovery Engine
Covers Entity Resolution, Master Catalog, UserMusicDNA, Semantic Search, Track Recommender, Spotify Client, and API Endpoints.
"""

import pytest
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

from src.config import DATA_DIR
from src.data.entity_resolution import EntityResolver, normalize_artist, normalize_title
from src.profiles.music_dna import UserMusicDNA
from src.features.semantic import SemanticMusicEngine
from src.recommenders.track_recommender import TrackRecommender
from src.spotify.client import SpotifyClient


# =========================================================================
# 1. ENTITY RESOLUTION TESTS
# =========================================================================

def test_entity_resolver_clean_title():
    assert normalize_title('Kesariya (From "Brahmastra")') == "kesariya"
    assert normalize_title("Tum Se Hi - Remastered 2021") == "tum se hi"
    assert normalize_title("Doobey [Remix]") == "doobey"
    assert normalize_title("Yellow (Live in Buenos Aires)") == "yellow"


def test_entity_resolver_normalize_artist():
    primary, artists = normalize_artist("Pritam, Arijit Singh & Amitabh Bhattacharya")
    assert primary == "pritam"
    assert "pritam" in artists
    assert "arijit singh" in artists
    assert "amitabh bhattacharya" in artists


def test_entity_resolver_matching():
    catalog = [
        {"track_id": "tr1", "track_name": "Kesariya", "artist_name": "Pritam, Arijit Singh", "genre": "bollywood"},
        {"track_id": "tr2", "track_name": "Tum Se Hi", "artist_name": "Pritam, Mohit Chauhan", "genre": "bollywood"},
        {"track_id": "tr3", "track_name": "Yellow", "artist_name": "Coldplay", "genre": "rock"},
    ]
    resolver = EntityResolver()
    resolver.index_catalog(catalog)

    match, score, match_type = resolver.resolve("Kesariya (Brahmastra)", "Arijit", threshold=0.6)
    assert match is not None
    assert match["track_id"] == "tr1"
    assert score >= 0.6

    match_coldplay, _, _ = resolver.resolve("Yellow", "Coldplay")
    assert match_coldplay is not None
    assert match_coldplay["track_id"] == "tr3"


# =========================================================================
# 2. MASTER CATALOG VERIFICATION
# =========================================================================

def test_master_catalog_schema_and_content():
    catalog_path = DATA_DIR / "processed" / "master_catalog.parquet"
    assert catalog_path.exists(), "master_catalog.parquet must exist"
    df = pd.read_parquet(catalog_path)
    assert len(df) >= 100000, f"Expected at least 100,000 tracks, got {len(df)}"

    required_cols = [
        "track_id", "track_name", "artist_name", "language", "genre",
        "release_year", "era", "popularity"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    languages = set(df["language"].dropna().unique())
    assert "Hindi" in languages
    assert "English / Global" in languages


# =========================================================================
# 3. USER MUSIC DNA TESTS
# =========================================================================

def test_user_music_dna_from_manual():
    dna = UserMusicDNA.from_manual(
        favorite_songs=["Kesariya", "Tum Se Hi"],
        favorite_artists=["Pritam", "Arijit Singh", "Coldplay"],
        preferred_genres=["bollywood", "pop", "rock"],
        preferred_languages=["Hindi", "English / Global"],
        preferred_eras=["2000s", "2020s"],
        mood="Romantic / Soulful",
        activity="Late Night Drive",
    )
    assert len(dna.favorite_songs) == 2
    assert "Pritam" in dna.artist_weights
    assert "bollywood" in dna.genre_weights
    assert "Hindi" in dna.language_weights
    summary = dna.summary()
    assert "Anchors" in summary
    assert "Pritam" in summary


def test_user_music_dna_from_spotify_playlist():
    sample_tracks = [
        {"name": "Kesariya", "artist": "Pritam", "language": "Hindi", "genre": "bollywood", "era": "2020s"},
        {"name": "Tum Se Hi", "artist": "Pritam", "language": "Hindi", "genre": "bollywood", "era": "2000s"},
        {"name": "Yellow", "artist": "Coldplay", "language": "English / Global", "genre": "rock", "era": "2000s"},
    ]
    dna = UserMusicDNA.from_spotify_playlist(
        playlist_name="My Desi & Rock Mix",
        tracks=sample_tracks,
    )
    assert len(dna.favorite_songs) == 3
    assert "pritam" in dna.artist_weights
    assert "coldplay" in dna.artist_weights
    assert dna.source_type == "spotify"


# =========================================================================
# 4. SEMANTIC MUSIC ENGINE TESTS
# =========================================================================

def test_semantic_music_engine_tfidf_search():
    sample_catalog = pd.DataFrame([
        {
            "track_id": "s1",
            "track_name": "Tujhe Dekha To",
            "artist_name": "Kumar Sanu, Lata Mangeshkar",
            "album_name": "Dilwale Dulhania Le Jayenge",
            "genre": "bollywood",
            "language": "Hindi",
            "release_year": 1995,
            "era": "1990s",
            "popularity": 75,
            "danceability": 0.5,
            "energy": 0.6,
            "valence": 0.7,
            "clean_title": "tujhe dekha to",
        },
        {
            "track_id": "s2",
            "track_name": "Smells Like Teen Spirit",
            "artist_name": "Nirvana",
            "album_name": "Nevermind",
            "genre": "grunge rock",
            "language": "English / Global",
            "release_year": 1991,
            "era": "1990s",
            "popularity": 85,
            "danceability": 0.5,
            "energy": 0.9,
            "valence": 0.4,
            "clean_title": "smells like teen spirit",
        },
        {
            "track_id": "s3",
            "track_name": "Kesariya",
            "artist_name": "Pritam, Arijit Singh",
            "album_name": "Brahmastra",
            "genre": "bollywood love romantic",
            "language": "Hindi",
            "release_year": 2022,
            "era": "2020s",
            "popularity": 80,
            "danceability": 0.6,
            "energy": 0.7,
            "valence": 0.8,
            "clean_title": "kesariya",
        },
    ])
    engine = SemanticMusicEngine(use_neural=False)
    engine.fit(sample_catalog)

    results = engine.search("romantic love song hindi bollywood", top_k=2)
    assert len(results) > 0
    top_result = results[0]
    assert top_result["language"] == "Hindi"

    # Test language filter
    eng_results = engine.search("90s song", top_k=2, language_filter="English / Global")
    assert all(r["language"] == "English / Global" for r in eng_results)


# =========================================================================
# 5. TRACK RECOMMENDER TESTS
# =========================================================================

def test_track_recommender_scoring():
    sample_df = pd.DataFrame([
        {
            "track_id": "t1",
            "track_name": "Kesariya",
            "artist_name": "Pritam",
            "album_name": "Brahmastra",
            "genre": "bollywood",
            "language": "Hindi",
            "release_year": 2022,
            "era": "2020s",
            "popularity": 80.0,
            "catalog_artist_idx": -1,
            "danceability": 0.6,
            "energy": 0.7,
            "valence": 0.8,
            "tempo": 120.0,
            "acousticness": 0.3,
            "clean_title": "kesariya",
        },
        {
            "track_id": "t2",
            "track_name": "Tum Se Hi",
            "artist_name": "Pritam",
            "album_name": "Jab We Met",
            "genre": "bollywood",
            "language": "Hindi",
            "release_year": 2007,
            "era": "2000s",
            "popularity": 75.0,
            "catalog_artist_idx": -1,
            "danceability": 0.55,
            "energy": 0.65,
            "valence": 0.75,
            "tempo": 115.0,
            "acousticness": 0.35,
            "clean_title": "tum se hi",
        },
        {
            "track_id": "t3",
            "track_name": "Yellow",
            "artist_name": "Coldplay",
            "album_name": "Parachutes",
            "genre": "rock",
            "language": "English / Global",
            "release_year": 2000,
            "era": "2000s",
            "popularity": 85.0,
            "catalog_artist_idx": -1,
            "danceability": 0.4,
            "energy": 0.5,
            "valence": 0.4,
            "tempo": 88.0,
            "acousticness": 0.4,
            "clean_title": "yellow",
        },
    ])
    resolver = EntityResolver()
    resolver.index_catalog(sample_df.to_dict(orient="records"))
    sem_engine = SemanticMusicEngine(use_neural=False)
    sem_engine.fit(sample_df)

    recommender = TrackRecommender(
        master_catalog=sample_df,
        semantic_engine=sem_engine,
        entity_resolver=resolver,
    )

    dna = UserMusicDNA.from_manual(
        favorite_artists=["Pritam"],
        preferred_genres=["bollywood"],
        preferred_languages=["Hindi"],
    )
    recs = recommender.recommend_for_dna(dna=dna, top_k=2)
    assert len(recs) > 0
    assert recs[0]["artist_name"] == "Pritam"

    # Similar song test
    seed, similar_recs = recommender.recommend_similar_song("Kesariya", "Pritam", top_k=2)
    assert seed is not None
    assert seed["track_name"] == "Kesariya"
    assert len(similar_recs) > 0
    # Tum Se Hi should rank top because it shares artist Pritam, language Hindi, and genre bollywood
    assert similar_recs[0]["track_name"] == "Tum Se Hi"


# =========================================================================
# 6. SPOTIFY CLIENT TESTS
# =========================================================================

def test_spotify_client_demo_mode():
    client = SpotifyClient()
    playlists = client.get_user_playlists()
    assert len(playlists) >= 3
    names = [p["name"] for p in playlists]
    assert "My Bollywood Favorites" in names
    assert "Desi Indie & Acoustic Vibes" in names

    tracks = client.get_playlist_tracks("demo_bollywood_favorites")
    assert len(tracks) >= 3
    track_names = [t["name"] for t in tracks]
    assert "Kesariya" in track_names
    assert "Tum Se Hi" in track_names


def test_spotify_client_pkce_url():
    client = SpotifyClient(client_id="test_client_id_123")
    url = client.get_authorization_url()
    assert "accounts.spotify.com/authorize" in url
    assert "client_id=test_client_id_123" in url


# =========================================================================
# 7. FASTAPI DHUNDNA ENDPOINTS
# =========================================================================

def test_fastapi_dhundna_endpoints():
    from api.main import app
    with TestClient(app) as tc:
        # 1. Spotify Playlists
        resp = tc.get("/api/v1/spotify/playlists")
        assert resp.status_code == 200
        data = resp.json()
        assert "playlists" in data
        assert len(data["playlists"]) >= 3

        # 2. Similar Song Recommendation
        resp = tc.post("/api/v1/recommend/similar-song", json={"song_title": "Kesariya", "artist_name": "Pritam", "n": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] > 0
        assert len(data["recommendations"]) > 0

        # 3. DNA Recommendation
        resp = tc.post("/api/v1/recommend/dna", json={
            "favorite_songs": ["Kesariya"],
            "favorite_artists": ["Arijit Singh", "Pritam"],
            "preferred_languages": ["Hindi"],
            "n": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] > 0
        assert len(data["recommendations"]) > 0

        # 4. Semantic Search
        resp = tc.post("/api/v1/search/semantic", json={
            "query": "romantic love song hindi",
            "n": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] > 0
        assert len(data["recommendations"]) > 0
