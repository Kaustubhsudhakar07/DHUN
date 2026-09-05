"""
Unit and Integration Tests for FastAPI Backend — Phase 11
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with lifespan context entered."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verify /health returns 200 with operational model status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data
    assert data["n_users"] > 0
    assert data["n_artists"] > 0


def test_recommend_existing_user(client):
    """Verify recommendations for User 0 return top-K items with explanations."""
    response = client.get("/api/v1/recommend/0?n=5&model=hybrid")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 0
    assert data["model"] == "hybrid"
    assert data["count"] == 5
    assert len(data["recommendations"]) == 5

    first_rec = data["recommendations"][0]
    assert "artist_id" in first_rec
    assert "artist_name" in first_rec
    assert "score" in first_rec
    assert "explanation" in first_rec
    assert len(first_rec["explanation"]) > 0


def test_recommend_model_selection(client):
    """Verify selecting different recommendation algorithms works."""
    for model_name in ["als", "popularity"]:
        response = client.get(f"/api/v1/recommend/0?n=3&model={model_name}")
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == model_name
        assert len(data["recommendations"]) == 3


def test_recommend_invalid_user(client):
    """Verify querying an out-of-bounds user returns HTTP 404."""
    response = client.get("/api/v1/recommend/99999999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_cold_start_recommendations(client):
    """Verify cold-start onboarding recommendations given seed artists."""
    payload = {
        "seed_artists": ["radiohead", "nirvana"],
        "n": 4,
    }
    response = client.post("/api/v1/recommend/cold-start", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == -1
    assert data["model"] == "cold-start"
    assert len(data["recommendations"]) == 4


def test_artist_info_endpoint(client):
    """Verify artist metadata endpoint returns bio, tags, and stats."""
    response = client.get("/api/v1/artist/radiohead")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "bio" in data
    assert "tags" in data
    assert "url" in data


def test_playlist_endpoint(client):
    """Verify playlist generation endpoint creates an interleaved track playlist."""
    response = client.get("/api/v1/playlist/0?n_artists=3&tracks_per_artist=2")
    assert response.status_code == 200
    data = response.json()
    assert "title" in data
    assert "tracks" in data
    assert data["total_tracks"] > 0
    assert len(data["tracks"]) == data["total_tracks"]
    first_track = data["tracks"][0]
    assert "track_title" in first_track
    assert "artist_name" in first_track


def test_feedback_endpoint(client):
    """Verify recording feedback returns success acknowledgement."""
    payload = {
        "user_id": 0,
        "artist_id": 1,
        "artist_name": "radiohead",
        "feedback_type": "thumbs_up",
    }
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "recorded"
