"""
Unit & Integration Tests for Streamlit Frontend Helpers (Phase 12)

Tests api_client.py (direct fallback and REST pathways), components.py
(Plotly chart generation, HTML escaping), and CSS definitions.
"""

import json
import os
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from app.api_client import RecommendationClient
from app.components import (
    plot_beyond_accuracy_radar,
    plot_ndcg_comparison,
    plot_signal_donut,
    plot_top_tags,
)
from app.styles import CUSTOM_CSS


@pytest.fixture(scope="module")
def client():
    """Create a client instance for testing direct fallback modes."""
    # Point to an unused port to guarantee fallback testing
    return RecommendationClient(base_url="http://127.0.0.1:59999")


def test_client_initialization(client):
    """Verify RecommendationClient initializes correctly."""
    assert client.base_url == "http://127.0.0.1:59999"
    assert client.is_api_online() is False


def test_client_fallback_health(client):
    """Verify health returns valid structure in fallback mode."""
    health = client.get_health()
    assert "status" in health
    assert "n_users" in health
    assert "n_artists" in health


def test_client_fallback_recommendations(client):
    """Verify recommendations can be generated via direct ML fallback."""
    res = client.get_recommendations(user_id=0, n=5, model="hybrid", explain=True)
    assert res["user_id"] == 0
    assert len(res["recommendations"]) == 5
    rec0 = res["recommendations"][0]
    assert "artist_name" in rec0
    assert "score" in rec0
    assert "explanation" in rec0
    assert "signal_breakdown" in rec0


def test_client_fallback_cold_start(client):
    """Verify cold-start recommendations work with seed artists."""
    res = client.get_cold_start_recommendations(seed_artists=["Radiohead", "Coldplay"], n=5)
    assert len(res["recommendations"]) == 5
    assert res["model"] == "cold-start"
    assert res["recommendations"][0]["score"] > 0


def test_client_fallback_artist_info(client):
    """Verify artist info retrieval in direct mode."""
    info = client.get_artist("Radiohead")
    assert info["name"] == "Radiohead"
    assert "bio" in info
    assert "tags" in info
    assert "similar_artists" in info


def test_client_fallback_playlist(client):
    """Verify playlist generation in direct mode."""
    playlist = client.get_playlist(user_id=0, n_artists=3, tracks_per_artist=2, title="Test Mix")
    assert playlist["title"] == "Test Mix"
    assert len(playlist["tracks"]) == 6
    assert "track_title" in playlist["tracks"][0]
    assert "explanation" in playlist["tracks"][0]


def test_client_fallback_feedback(client, tmp_path, monkeypatch):
    """Verify feedback recording to file."""
    import src.config as cfg
    fake_feedback = tmp_path / "test_feedback.json"
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    res = client.post_feedback(user_id=0, artist_id=1, artist_name="Test Artist", feedback_type="thumbs_up")
    assert res["status"] == "recorded"
    assert (tmp_path / "feedback.json").exists()


def test_ndcg_comparison_chart():
    """Verify Plotly NDCG chart generation."""
    sample_eval = {
        "ranking": {
            "Popularity": {"ndcg": {"5": 0.01, "10": 0.02, "20": 0.03}},
            "Hybrid": {"ndcg": {"5": 0.06, "10": 0.07, "20": 0.09}},
        }
    }
    fig = plot_ndcg_comparison(sample_eval)
    assert fig is not None
    assert len(fig.data) == 2


def test_beyond_accuracy_radar_chart():
    """Verify Plotly radar chart generation."""
    sample_eval = {
        "beyond_accuracy": {
            "ALS": {"coverage": 0.02, "diversity": 0.8, "novelty": 6.0, "personalization": 0.99},
            "Hybrid": {"coverage": 0.01, "diversity": 0.67, "novelty": 4.9, "personalization": 0.97},
        }
    }
    fig = plot_beyond_accuracy_radar(sample_eval)
    assert fig is not None
    assert len(fig.data) == 2


def test_signal_donut_chart():
    """Verify Plotly signal donut chart generation."""
    signals = {"collaborative": 40.0, "content": 30.0, "popularity": 30.0}
    fig = plot_signal_donut(signals)
    assert fig is not None
    assert len(fig.data) == 1


def test_top_tags_chart():
    """Verify Plotly top tags bar chart generation."""
    tags = ["rock", "electronic", "indie"]
    weights = [0.9, 0.6, 0.4]
    fig = plot_top_tags(tags, weights, "Test Band")
    assert fig is not None
    assert len(fig.data) == 1


def test_custom_css_definitions():
    """Verify that CSS contains key design classes."""
    assert ".hero-container" in CUSTOM_CSS
    assert ".rec-card" in CUSTOM_CSS
    assert ".rec-rank" in CUSTOM_CSS
    assert ".badge-tag" in CUSTOM_CSS
    assert ".stat-card" in CUSTOM_CSS


def test_render_track_card_html_no_code_blocks(monkeypatch):
    """Verify that render_track_card does not generate indented code blocks with literal </div>."""
    captured = []
    import streamlit as st
    from markdown_it import MarkdownIt
    from app.components import render_track_card

    monkeypatch.setattr(st, "markdown", lambda html_str, **kwargs: captured.append(html_str))

    track = {
        "track_name": 'Deva Deva (From "Brahmastra")',
        "artist_name": "Pritam",
        "album_name": 'Deva Deva (From "Brahmastra")',
        "score": 0.95,
        "explanation": "Because you love Pritam",
        "language": "Hindi",
        "genre": "bollywood",
    }
    render_track_card(track=track, rank=1, client=None, show_feedback=False)
    assert len(captured) == 1
    raw_html = captured[0]

    md = MarkdownIt()
    rendered = md.render(raw_html)
    assert "<pre>" not in rendered, "Indented code block detected!"
    assert "<code>" not in rendered, "Code tag detected in card rendering!"
    assert "&lt;/div&gt;" not in rendered, "Literal </div> detected!"


def test_render_recommendation_card_no_code_blocks(monkeypatch):
    """Verify that render_recommendation_card renders clean HTML without code blocks."""
    captured = []
    import streamlit as st
    from markdown_it import MarkdownIt
    from app.components import render_recommendation_card

    monkeypatch.setattr(st, "markdown", lambda html_str, **kwargs: captured.append(html_str))

    rec = {
        "artist_name": "Pritam",
        "score": 0.95,
        "explanation": "Similar to artists you enjoy",
        "shared_tags": ["bollywood", "hindi"],
        "anchor_artists": ["Arijit Singh"],
        "signal_breakdown": {"collaborative": 50, "content": 50},
    }
    render_recommendation_card(rec=rec, rank=1, user_id=0, client=None, show_feedback=False)
    assert len(captured) == 1
    raw_html = captured[0]

    md = MarkdownIt()
    rendered = md.render(raw_html)
    assert "<pre>" not in rendered
    assert "<code>" not in rendered
    assert "&lt;/div&gt;" not in rendered

