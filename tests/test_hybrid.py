"""
Unit tests for Candidate Ranking & Hybrid Recommendation (Phase 9).
"""

import json
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.ranking.ranker import CandidatePool, PostFilter, ScoreNormalizer
from src.recommenders.als import ALSRecommender
from src.recommenders.base import Recommendation
from src.recommenders.content import ContentBasedRecommender
from src.recommenders.hybrid import HybridRecommender
from src.recommenders.popularity import PopularityRecommender


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def toy_train_matrix():
    data = [100, 50, 10, 5, 80, 60, 20, 200, 100, 30, 40, 30, 10, 90, 70, 50, 40, 20]
    rows = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4]
    cols = [0, 1, 2, 3, 0, 4, 5, 1, 2, 6, 3, 4, 7, 0, 1, 5, 6, 7]
    return csr_matrix((data, (rows, cols)), shape=(5, 8), dtype=np.float32)


@pytest.fixture
def toy_idx2artist():
    return [f"artist_{i}" for i in range(8)]


@pytest.fixture
def toy_tfidf():
    documents = [
        "rock alternative indie",
        "rock metal heavy",
        "pop electronic dance",
        "jazz blues soul",
        "rock punk hardcore",
        "electronic ambient chill",
        "pop indie dream",
        "jazz fusion experimental",
    ]
    vectorizer = TfidfVectorizer(sublinear_tf=True, norm="l2")
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix, vectorizer


@pytest.fixture
def fitted_submodels(toy_train_matrix, toy_idx2artist, toy_tfidf):
    tfidf_matrix, vectorizer = toy_tfidf

    pop = PopularityRecommender()
    pop.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)

    cb = ContentBasedRecommender()
    cb.fit(
        tfidf_matrix=tfidf_matrix,
        train_interactions=toy_train_matrix,
        idx2artist=toy_idx2artist,
        vectorizer=vectorizer,
    )

    als = ALSRecommender(factors=4, iterations=5, use_gpu=False)
    als.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)

    return pop, cb, als


# =========================================================================
# ScoreNormalizer Tests
# =========================================================================

class TestScoreNormalizer:

    def test_min_max_basic(self):
        scores = np.array([10.0, 20.0, 30.0])
        norm = ScoreNormalizer.min_max(scores)
        np.testing.assert_allclose(norm, [0.0, 0.5, 1.0], atol=1e-5)

    def test_min_max_constant(self):
        scores = np.array([5.0, 5.0, 5.0])
        norm = ScoreNormalizer.min_max(scores)
        np.testing.assert_allclose(norm, [0.5, 0.5, 0.5])

    def test_min_max_zeros(self):
        scores = np.array([0.0, 0.0, 0.0])
        norm = ScoreNormalizer.min_max(scores)
        np.testing.assert_allclose(norm, [0.0, 0.0, 0.0])

    def test_min_max_empty(self):
        norm = ScoreNormalizer.min_max(np.array([]))
        assert len(norm) == 0

    def test_rank_percentile(self):
        scores = np.array([100.0, 10.0, 50.0])
        norm = ScoreNormalizer.rank_percentile(scores)
        assert norm[0] == 1.0   # Highest
        assert norm[1] == 0.0   # Lowest
        assert norm[2] == 0.5   # Middle


# =========================================================================
# CandidatePool & PostFilter Tests
# =========================================================================

class TestRankingUtils:

    def test_candidate_pool_merge(self):
        l1 = [1, 2, 3]
        l2 = [2, 4, 5]
        l3 = [5, 6, 1]
        merged = CandidatePool.merge(l1, l2, l3)
        assert list(merged) == [1, 2, 3, 4, 5, 6]

    def test_candidate_pool_max_cap(self):
        l1 = [1, 2, 3, 4, 5]
        merged = CandidatePool.merge(l1, max_candidates=3)
        assert len(merged) == 3
        assert list(merged) == [1, 2, 3]

    def test_post_filter_known(self):
        candidates = np.array([0, 1, 2, 3, 4])
        known = {1, 3}
        filtered = PostFilter.filter_known(candidates, known)
        assert list(filtered) == [0, 2, 4]

    def test_post_filter_by_min_score(self):
        candidates = np.array([10, 20, 30])
        scores = np.array([0.8, 0.2, 0.5])
        c_filt, s_filt = PostFilter.filter_by_min_score(candidates, scores, min_score=0.4)
        assert list(c_filt) == [10, 30]
        np.testing.assert_allclose(s_filt, [0.8, 0.5])


# =========================================================================
# HybridRecommender Tests
# =========================================================================

class TestHybridRecommender:

    def test_init_and_fit(self, fitted_submodels, toy_train_matrix, toy_idx2artist):
        pop, cb, als = fitted_submodels
        hybrid = HybridRecommender(
            cf_recommender=als,
            content_recommender=cb,
            popularity_recommender=pop,
        )
        assert hybrid.is_fitted
        hybrid.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)
        assert hybrid.is_fitted

    def test_recommend_returns_top_n(self, fitted_submodels, toy_train_matrix, toy_idx2artist):
        pop, cb, als = fitted_submodels
        hybrid = HybridRecommender(
            cf_recommender=als,
            content_recommender=cb,
            popularity_recommender=pop,
        )
        hybrid.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)

        recs = hybrid.recommend(user_id=0, n=3, exclude_known=True)
        assert len(recs) == 3
        assert all(isinstance(r, Recommendation) for r in recs)
        # Verify scores are sorted descending
        scores = [r.score for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_excludes_known(self, fitted_submodels, toy_train_matrix, toy_idx2artist):
        pop, cb, als = fitted_submodels
        hybrid = HybridRecommender(
            cf_recommender=als,
            content_recommender=cb,
            popularity_recommender=pop,
        )
        hybrid.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)

        known = set(toy_train_matrix[0].indices.tolist())
        recs = hybrid.recommend(user_id=0, n=4, exclude_known=True)
        rec_ids = {r.item_id for r in recs}
        assert rec_ids.isdisjoint(known)

    def test_scores_in_unit_interval(self, fitted_submodels, toy_train_matrix, toy_idx2artist):
        pop, cb, als = fitted_submodels
        hybrid = HybridRecommender(
            cf_recommender=als,
            content_recommender=cb,
            popularity_recommender=pop,
        )
        hybrid.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)

        recs = hybrid.recommend(user_id=0, n=5, exclude_known=False)
        for r in recs:
            assert 0.0 <= r.score <= 1.0 + 1e-6

    def test_cold_start_cf_rebalancing(self, fitted_submodels):
        pop, cb, als = fitted_submodels
        hybrid = HybridRecommender(
            cf_recommender=als,
            content_recommender=cb,
            popularity_recommender=pop,
            weights={"collaborative": 0.6, "content": 0.3, "popularity": 0.1},
        )
        # CF has no signal
        active = hybrid._get_active_weights(user_id=999, has_cf_signal=False, has_content_signal=True)
        assert active["collaborative"] == 0.0
        # Remaining 0.3 content and 0.1 pop rebalance to 0.75 and 0.25
        assert pytest.approx(active["content"], rel=1e-3) == 0.75
        assert pytest.approx(active["popularity"], rel=1e-3) == 0.25

    def test_cold_start_total_rebalancing(self, fitted_submodels):
        pop, cb, als = fitted_submodels
        hybrid = HybridRecommender(
            cf_recommender=als,
            content_recommender=cb,
            popularity_recommender=pop,
        )
        # Neither CF nor content has signal
        active = hybrid._get_active_weights(user_id=999, has_cf_signal=False, has_content_signal=False)
        assert active["collaborative"] == 0.0
        assert active["content"] == 0.0
        assert active["popularity"] == 1.0

    def test_get_scores_batch(self, fitted_submodels, toy_train_matrix, toy_idx2artist):
        pop, cb, als = fitted_submodels
        hybrid = HybridRecommender(
            cf_recommender=als,
            content_recommender=cb,
            popularity_recommender=pop,
        )
        hybrid.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)

        item_ids = np.array([4, 5, 6])
        scores = hybrid.get_scores(user_id=0, item_ids=item_ids)
        assert len(scores) == 3
        assert scores.dtype == np.float32

    def test_explain_composite(self, fitted_submodels, toy_train_matrix, toy_idx2artist):
        pop, cb, als = fitted_submodels
        hybrid = HybridRecommender(
            cf_recommender=als,
            content_recommender=cb,
            popularity_recommender=pop,
        )
        hybrid.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)

        recs = hybrid.recommend(user_id=0, n=2)
        for r in recs:
            assert len(r.reasons) > 0
            assert isinstance(r.metadata, dict)
            assert "cf_score" in r.metadata

    def test_save_and_load_weights(self, fitted_submodels, tmp_path):
        pop, cb, als = fitted_submodels
        custom_weights = {"collaborative": 0.5, "content": 0.4, "popularity": 0.1}
        hybrid = HybridRecommender(
            cf_recommender=als,
            content_recommender=cb,
            popularity_recommender=pop,
            weights=custom_weights,
        )
        weight_file = tmp_path / "test_hybrid_weights.json"
        hybrid.save(path=weight_file)
        assert weight_file.exists()

        hybrid2 = HybridRecommender(
            cf_recommender=als,
            content_recommender=cb,
            popularity_recommender=pop,
        )
        hybrid2.load(path=weight_file)
        assert pytest.approx(hybrid2.weights["collaborative"]) == 0.5
        assert pytest.approx(hybrid2.weights["content"]) == 0.4
        assert pytest.approx(hybrid2.weights["popularity"]) == 0.1
