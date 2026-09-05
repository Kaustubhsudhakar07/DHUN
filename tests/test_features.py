"""
Unit tests for TF-IDF feature engineering (Phase 6).
"""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.tfidf import get_top_tags


@pytest.fixture
def sample_tfidf():
    """Build a small TF-IDF matrix from sample tag documents."""
    documents = [
        "rock alternative_rock indie_rock rock rock",
        "pop electronic dance pop electronic",
        "rock metal heavy_metal metal rock",
        "jazz blues soul jazz",
        "",  # Artist with no tags
    ]

    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        norm="l2",
        token_pattern=r"(?u)\b\w[\w]*\b",
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix, vectorizer, documents


class TestTfidfMatrix:
    """Tests for TF-IDF matrix construction."""

    def test_matrix_shape(self, sample_tfidf):
        tfidf_matrix, vectorizer, docs = sample_tfidf
        assert tfidf_matrix.shape[0] == 5  # 5 artists
        assert tfidf_matrix.shape[1] > 0   # non-zero vocabulary

    def test_sparse_type(self, sample_tfidf):
        tfidf_matrix, _, _ = sample_tfidf
        assert hasattr(tfidf_matrix, "toarray")  # sparse matrix

    def test_l2_normalized(self, sample_tfidf):
        """Each non-zero row should be L2-normalized to unit length."""
        tfidf_matrix, _, _ = sample_tfidf
        dense = tfidf_matrix.toarray()
        for i in range(dense.shape[0]):
            row_norm = np.linalg.norm(dense[i])
            if row_norm > 0:
                np.testing.assert_almost_equal(row_norm, 1.0, decimal=5)

    def test_empty_document_zero_vector(self, sample_tfidf):
        """Artist with empty tags should have all-zero TF-IDF vector."""
        tfidf_matrix, _, _ = sample_tfidf
        empty_row = tfidf_matrix[4].toarray().ravel()
        assert np.all(empty_row == 0)

    def test_nonzero_rows_have_features(self, sample_tfidf):
        """Artists with tags should have non-zero feature vectors."""
        tfidf_matrix, _, _ = sample_tfidf
        for i in range(4):  # First 4 artists have tags
            row = tfidf_matrix[i].toarray().ravel()
            assert row.sum() > 0, f"Artist {i} should have non-zero features"


class TestGetTopTags:
    """Tests for the get_top_tags utility."""

    def test_returns_tags(self, sample_tfidf):
        tfidf_matrix, vectorizer, _ = sample_tfidf
        tags = get_top_tags(vectorizer, tfidf_matrix, artist_idx=0, n=5)
        assert len(tags) > 0
        assert all(isinstance(t, tuple) and len(t) == 2 for t in tags)

    def test_tags_sorted_descending(self, sample_tfidf):
        tfidf_matrix, vectorizer, _ = sample_tfidf
        tags = get_top_tags(vectorizer, tfidf_matrix, artist_idx=0, n=10)
        scores = [score for _, score in tags]
        assert scores == sorted(scores, reverse=True)

    def test_empty_artist_returns_empty(self, sample_tfidf):
        tfidf_matrix, vectorizer, _ = sample_tfidf
        tags = get_top_tags(vectorizer, tfidf_matrix, artist_idx=4, n=5)
        assert tags == []

    def test_rock_artist_has_rock_tag(self, sample_tfidf):
        """Artist 0 (rock) should have 'rock' as a top tag."""
        tfidf_matrix, vectorizer, _ = sample_tfidf
        tags = get_top_tags(vectorizer, tfidf_matrix, artist_idx=0, n=10)
        tag_names = [name for name, _ in tags]
        assert "rock" in tag_names
