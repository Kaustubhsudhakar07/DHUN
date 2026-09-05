"""
Unit tests for Explainability Engine & Playlist Generation (Phase 10).
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.explainability.explainer import Explanation, ExplanationEngine
from src.explainability.playlist import Playlist, PlaylistGenerator
from src.lastfm.client import LastFMClient
from src.recommenders.base import Recommendation


@pytest.fixture
def toy_train_matrix():
    data = [100, 50, 10, 5, 80, 60, 20]
    rows = [0, 0, 0, 0, 1, 1, 1]
    cols = [0, 1, 2, 3, 0, 4, 5]
    return csr_matrix((data, (rows, cols)), shape=(2, 6), dtype=np.float32)


@pytest.fixture
def toy_idx2artist():
    return ["radiohead", "muse", "coldplay", "portishead", "nirvana", "pearl jam"]


@pytest.fixture
def toy_tfidf():
    documents = [
        "rock alternative indie",
        "rock alternative space_rock",
        "pop rock alternative",
        "trip_hop electronic downtempo",
        "grunge rock alternative",
        "grunge rock alternative",
    ]
    vectorizer = TfidfVectorizer(sublinear_tf=True, norm="l2")
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix, vectorizer


class TestExplanationEngine:

    def test_explain_returns_valid_object(self, toy_train_matrix, toy_idx2artist, toy_tfidf):
        tfidf_matrix, vectorizer = toy_tfidf
        engine = ExplanationEngine(
            idx2artist=toy_idx2artist,
            tfidf_matrix=tfidf_matrix,
            vectorizer=vectorizer,
            train_interactions=toy_train_matrix,
            lastfm_client=LastFMClient(api_key=""),  # Offline
        )

        meta = {
            "cf_score": 0.8,
            "content_score": 0.7,
            "popularity_score": 0.5,
            "active_weights": {"collaborative": 0.4, "content": 0.3, "popularity": 0.3},
        }

        exp = engine.explain(user_id=0, item_id=4, metadata=meta, fetch_online_metadata=False)
        assert isinstance(exp, Explanation)
        assert exp.item_id == 4
        assert exp.item_name == "nirvana"
        assert len(exp.summary) > 0
        assert isinstance(exp.anchor_artists, list)

    def test_signal_breakdown_percentages(self, toy_train_matrix, toy_idx2artist, toy_tfidf):
        tfidf_matrix, vectorizer = toy_tfidf
        engine = ExplanationEngine(
            idx2artist=toy_idx2artist,
            tfidf_matrix=tfidf_matrix,
            vectorizer=vectorizer,
            train_interactions=toy_train_matrix,
            lastfm_client=LastFMClient(api_key=""),
        )

        meta = {
            "cf_score": 0.6,
            "content_score": 0.6,
            "popularity_score": 0.6,
            "active_weights": {"collaborative": 0.5, "content": 0.3, "popularity": 0.2},
        }
        exp = engine.explain(user_id=0, item_id=4, metadata=meta, fetch_online_metadata=False)
        breakdown = exp.signal_breakdown
        total = sum(breakdown.values())
        assert pytest.approx(total, abs=0.5) == 100.0
        assert breakdown["collaborative"] > breakdown["content"]
        assert breakdown["content"] > breakdown["popularity"]

    def test_anchor_artists_identification(self, toy_train_matrix, toy_idx2artist, toy_tfidf):
        tfidf_matrix, vectorizer = toy_tfidf
        engine = ExplanationEngine(
            idx2artist=toy_idx2artist,
            tfidf_matrix=tfidf_matrix,
            vectorizer=vectorizer,
            train_interactions=toy_train_matrix,
            lastfm_client=LastFMClient(api_key=""),
        )

        # For user 0 (listened to radiohead, muse, coldplay, portishead)
        # Recommend nirvana (item 4, rock/alternative/grunge)
        exp = engine.explain(user_id=0, item_id=4, fetch_online_metadata=False)
        anchor_ids = [a["artist_id"] for a in exp.anchor_artists]
        # Rock artists like radiohead (0) or muse (1) should be anchors
        assert any(aid in anchor_ids for aid in [0, 1])

    def test_shared_tags_extracted(self, toy_train_matrix, toy_idx2artist, toy_tfidf):
        tfidf_matrix, vectorizer = toy_tfidf
        engine = ExplanationEngine(
            idx2artist=toy_idx2artist,
            tfidf_matrix=tfidf_matrix,
            vectorizer=vectorizer,
            train_interactions=toy_train_matrix,
            lastfm_client=LastFMClient(api_key=""),
        )

        exp = engine.explain(user_id=0, item_id=4, fetch_online_metadata=False)
        # nirvana shares "rock" and "alternative" with user 0's rock artists
        assert any(t in ["rock", "alternative"] for t in exp.shared_tags)

    def test_to_dict_serialization(self, toy_train_matrix, toy_idx2artist, toy_tfidf):
        tfidf_matrix, vectorizer = toy_tfidf
        engine = ExplanationEngine(
            idx2artist=toy_idx2artist,
            tfidf_matrix=tfidf_matrix,
            vectorizer=vectorizer,
            train_interactions=toy_train_matrix,
            lastfm_client=LastFMClient(api_key=""),
        )
        exp = engine.explain(user_id=0, item_id=4, fetch_online_metadata=False)
        d = exp.to_dict()
        assert isinstance(d, dict)
        assert d["item_name"] == "nirvana"
        assert "signal_breakdown" in d
        assert "anchor_artists" in d


class TestPlaylistGenerator:

    def test_empty_recommendations_playlist(self):
        gen = PlaylistGenerator(lastfm_client=LastFMClient(api_key=""))
        playlist = gen.generate_playlist([])
        assert playlist.total_tracks == 0
        assert len(playlist.tracks) == 0

    def test_interleaved_playlist_generation(self):
        gen = PlaylistGenerator(lastfm_client=LastFMClient(api_key=""))
        recs = [
            Recommendation(item_id=0, item_name="radiohead", score=0.9, reasons=["Similar to Muse"]),
            Recommendation(item_id=1, item_name="daft punk", score=0.8, reasons=["Electronic"]),
        ]
        playlist = gen.generate_playlist(recs, tracks_per_artist=2, max_tracks=4)

        assert playlist.total_tracks == 4
        # Verify tracks interleave between artists
        artist_sequence = [t.artist_name for t in playlist.tracks]
        assert artist_sequence[0] == "radiohead"
        assert artist_sequence[1] == "daft punk"
        assert artist_sequence[2] == "radiohead"
        assert artist_sequence[3] == "daft punk"

    def test_playlist_to_dict(self):
        gen = PlaylistGenerator(lastfm_client=LastFMClient(api_key=""))
        recs = [Recommendation(item_id=0, item_name="radiohead", score=0.9)]
        playlist = gen.generate_playlist(recs, tracks_per_artist=1)
        d = playlist.to_dict()
        assert isinstance(d, dict)
        assert "tracks" in d
        assert len(d["tracks"]) == 1
        assert d["tracks"][0]["artist_name"] == "radiohead"
