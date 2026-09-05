"""
Unit tests for recommendation models (Phase 7).

Tests use small synthetic data to verify the BaseRecommender interface
is correctly implemented by each model.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.recommenders.base import Recommendation
from src.recommenders.popularity import PopularityRecommender
from src.recommenders.content import ContentBasedRecommender
from src.recommenders.als import ALSRecommender
from src.recommenders.bpr import BPRRecommender


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def toy_train_matrix():
    """Create a small training interaction matrix.

    5 users × 8 artists:
        u0: plays a0(100), a1(50), a2(10), a3(5)
        u1: plays a0(80), a4(60), a5(20)
        u2: plays a1(200), a2(100), a6(30)
        u3: plays a3(40), a4(30), a7(10)
        u4: plays a0(90), a1(70), a5(50), a6(40), a7(20)
    """
    data = [
        100, 50, 10, 5,         # u0
        80, 60, 20,              # u1
        200, 100, 30,            # u2
        40, 30, 10,              # u3
        90, 70, 50, 40, 20,      # u4
    ]
    rows = [
        0, 0, 0, 0,
        1, 1, 1,
        2, 2, 2,
        3, 3, 3,
        4, 4, 4, 4, 4,
    ]
    cols = [
        0, 1, 2, 3,
        0, 4, 5,
        1, 2, 6,
        3, 4, 7,
        0, 1, 5, 6, 7,
    ]
    return csr_matrix(
        (data, (rows, cols)),
        shape=(5, 8),
        dtype=np.float32,
    )


@pytest.fixture
def toy_idx2artist():
    return [f"artist_{i}" for i in range(8)]


@pytest.fixture
def toy_tfidf():
    """Build TF-IDF from toy tag documents for 8 artists."""
    documents = [
        "rock alternative_rock indie",
        "rock metal heavy_metal",
        "pop electronic dance",
        "jazz blues soul",
        "rock punk hardcore",
        "electronic ambient chill",
        "pop indie_pop dream_pop",
        "jazz fusion experimental",
    ]
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        norm="l2",
        token_pattern=r"(?u)\b\w[\w]*\b",
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix, vectorizer


# =========================================================================
# PopularityRecommender Tests
# =========================================================================

class TestPopularityRecommender:

    def test_fit_sets_fitted(self, toy_train_matrix, toy_idx2artist):
        pop = PopularityRecommender()
        assert not pop.is_fitted
        pop.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)
        assert pop.is_fitted

    def test_recommend_returns_recommendations(self, toy_train_matrix, toy_idx2artist):
        pop = PopularityRecommender()
        pop.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)
        recs = pop.recommend(user_id=0, n=5, exclude_known=False)
        assert len(recs) == 5
        assert all(isinstance(r, Recommendation) for r in recs)

    def test_recommend_excludes_known(self, toy_train_matrix, toy_idx2artist):
        pop = PopularityRecommender()
        pop.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)
        known = toy_train_matrix[0].indices
        recs = pop.recommend(
            user_id=0,
            n=4,
            exclude_known=True,
            known_items=known,
        )
        rec_ids = {r.item_id for r in recs}
        assert rec_ids.isdisjoint(set(known.tolist()))

    def test_scores_descending(self, toy_train_matrix, toy_idx2artist):
        pop = PopularityRecommender()
        pop.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)
        recs = pop.recommend(user_id=0, n=8, exclude_known=False)
        scores = [r.score for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_most_popular_is_a1(self, toy_train_matrix, toy_idx2artist):
        """Artist 1 has the most total plays (50+200+70 = 320)."""
        pop = PopularityRecommender()
        pop.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)
        recs = pop.recommend(user_id=3, n=1, exclude_known=False)
        assert recs[0].item_id == 1

    def test_get_scores_batch(self, toy_train_matrix, toy_idx2artist):
        pop = PopularityRecommender()
        pop.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)
        item_ids = np.array([0, 1, 7])
        scores = pop.get_scores(user_id=0, item_ids=item_ids)
        assert len(scores) == 3
        assert scores[0] > scores[2]  # a0 more popular than a7

    def test_explain_has_content(self, toy_train_matrix, toy_idx2artist):
        pop = PopularityRecommender()
        pop.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)
        reasons = pop.explain(user_id=0, item_id=0)
        assert len(reasons) > 0
        assert "popular" in reasons[0].lower() or "listened" in reasons[0].lower()

    def test_not_fitted_raises(self):
        pop = PopularityRecommender()
        with pytest.raises(RuntimeError, match="not fitted"):
            pop.recommend(user_id=0, n=5)


# =========================================================================
# ContentBasedRecommender Tests
# =========================================================================

class TestContentBasedRecommender:

    def test_fit_sets_fitted(self, toy_tfidf, toy_train_matrix, toy_idx2artist):
        tfidf_matrix, vectorizer = toy_tfidf
        cb = ContentBasedRecommender()
        cb.fit(
            tfidf_matrix=tfidf_matrix,
            train_interactions=toy_train_matrix,
            idx2artist=toy_idx2artist,
            vectorizer=vectorizer,
        )
        assert cb.is_fitted

    def test_recommend_returns_recs(self, toy_tfidf, toy_train_matrix, toy_idx2artist):
        tfidf_matrix, vectorizer = toy_tfidf
        cb = ContentBasedRecommender()
        cb.fit(
            tfidf_matrix=tfidf_matrix,
            train_interactions=toy_train_matrix,
            idx2artist=toy_idx2artist,
            vectorizer=vectorizer,
        )
        recs = cb.recommend(user_id=0, n=4)
        assert len(recs) > 0
        assert all(isinstance(r, Recommendation) for r in recs)

    def test_recommend_excludes_known(self, toy_tfidf, toy_train_matrix, toy_idx2artist):
        tfidf_matrix, vectorizer = toy_tfidf
        cb = ContentBasedRecommender()
        cb.fit(
            tfidf_matrix=tfidf_matrix,
            train_interactions=toy_train_matrix,
            idx2artist=toy_idx2artist,
            vectorizer=vectorizer,
        )
        known = set(toy_train_matrix[0].indices.tolist())
        recs = cb.recommend(user_id=0, n=4, exclude_known=True)
        rec_ids = {r.item_id for r in recs}
        assert rec_ids.isdisjoint(known)

    def test_scores_in_valid_range(self, toy_tfidf, toy_train_matrix, toy_idx2artist):
        """Cosine similarity scores should be in [-1, 1]."""
        tfidf_matrix, vectorizer = toy_tfidf
        cb = ContentBasedRecommender()
        cb.fit(
            tfidf_matrix=tfidf_matrix,
            train_interactions=toy_train_matrix,
            idx2artist=toy_idx2artist,
            vectorizer=vectorizer,
        )
        recs = cb.recommend(user_id=0, n=8, exclude_known=False)
        for r in recs:
            assert -1.0 <= r.score <= 1.0 + 1e-6

    def test_rock_user_gets_rock_recs(self, toy_tfidf, toy_train_matrix, toy_idx2artist):
        """User 0 listens to rock artists (a0, a1). Should recommend
        similar rock artists (a4=rock/punk) over jazz/electronic."""
        tfidf_matrix, vectorizer = toy_tfidf
        cb = ContentBasedRecommender()
        cb.fit(
            tfidf_matrix=tfidf_matrix,
            train_interactions=toy_train_matrix,
            idx2artist=toy_idx2artist,
            vectorizer=vectorizer,
        )
        recs = cb.recommend(user_id=0, n=4, exclude_known=True)
        # a4 (rock/punk) should rank higher than a5 (electronic/ambient)
        rec_ids = [r.item_id for r in recs]
        if 4 in rec_ids and 5 in rec_ids:
            assert rec_ids.index(4) < rec_ids.index(5)

    def test_get_similar_items(self, toy_tfidf, toy_train_matrix, toy_idx2artist):
        tfidf_matrix, vectorizer = toy_tfidf
        cb = ContentBasedRecommender()
        cb.fit(
            tfidf_matrix=tfidf_matrix,
            train_interactions=toy_train_matrix,
            idx2artist=toy_idx2artist,
        )
        similar = cb.get_similar_items(item_id=0, n=3)
        assert len(similar) > 0
        # a0 (rock) should be similar to a1 (rock/metal) and a4 (rock/punk)
        sim_ids = {r.item_id for r in similar}
        assert 0 not in sim_ids  # Self not included

    def test_explain_has_content(self, toy_tfidf, toy_train_matrix, toy_idx2artist):
        tfidf_matrix, vectorizer = toy_tfidf
        cb = ContentBasedRecommender()
        cb.fit(
            tfidf_matrix=tfidf_matrix,
            train_interactions=toy_train_matrix,
            idx2artist=toy_idx2artist,
            vectorizer=vectorizer,
        )
        reasons = cb.explain(user_id=0, item_id=4)
        assert len(reasons) > 0

    def test_not_fitted_raises(self):
        cb = ContentBasedRecommender()
        with pytest.raises(RuntimeError, match="not fitted"):
            cb.recommend(user_id=0, n=5)


# =========================================================================
# ALSRecommender Tests
# =========================================================================

class TestALSRecommender:

    def test_fit_sets_fitted(self, toy_train_matrix, toy_idx2artist):
        als = ALSRecommender(factors=4, iterations=5, use_gpu=False)
        assert not als.is_fitted
        als.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)
        assert als.is_fitted

    def test_recommend_returns_recs(self, toy_train_matrix, toy_idx2artist):
        als = ALSRecommender(factors=4, iterations=5, use_gpu=False)
        als.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)
        recs = als.recommend(user_id=0, n=3)
        assert len(recs) == 3
        assert all(isinstance(r, Recommendation) for r in recs)

    def test_recommend_excludes_known(self, toy_train_matrix, toy_idx2artist):
        als = ALSRecommender(factors=4, iterations=5, use_gpu=False)
        als.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)
        known = set(toy_train_matrix[0].indices.tolist())
        recs = als.recommend(user_id=0, n=3, exclude_known=True)
        rec_ids = {r.item_id for r in recs}
        assert rec_ids.isdisjoint(known)

    def test_get_scores_batch(self, toy_train_matrix, toy_idx2artist):
        als = ALSRecommender(factors=4, iterations=5, use_gpu=False)
        als.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)
        item_ids = np.array([4, 5, 6])
        scores = als.get_scores(user_id=0, item_ids=item_ids)
        assert len(scores) == 3
        assert scores.dtype == np.float32

    def test_get_similar_items(self, toy_train_matrix, toy_idx2artist):
        als = ALSRecommender(factors=4, iterations=5, use_gpu=False)
        als.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)
        similar = als.get_similar_items(item_id=0, n=3)
        assert len(similar) <= 3
        sim_ids = {r.item_id for r in similar}
        assert 0 not in sim_ids

    def test_explain_has_content(self, toy_train_matrix, toy_idx2artist):
        als = ALSRecommender(factors=4, iterations=5, use_gpu=False)
        als.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)
        reasons = als.explain(user_id=0, item_id=4)
        assert len(reasons) > 0

    def test_save_and_load(self, toy_train_matrix, toy_idx2artist, tmp_path):
        als = ALSRecommender(factors=4, iterations=5, use_gpu=False)
        als.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)
        save_file = tmp_path / "als_test.npz"
        als.save(path=save_file)
        assert save_file.exists()

        als_loaded = ALSRecommender(use_gpu=False)
        als_loaded.load(
            path=save_file,
            train_interactions=toy_train_matrix,
            idx2artist=toy_idx2artist,
        )
        assert als_loaded.is_fitted
        recs = als_loaded.recommend(user_id=0, n=3)
        assert len(recs) == 3

    def test_not_fitted_raises(self):
        als = ALSRecommender()
        with pytest.raises(RuntimeError, match="not fitted"):
            als.recommend(user_id=0, n=5)


# =========================================================================
# BPRRecommender Tests
# =========================================================================

class TestBPRRecommender:

    def test_fit_sets_fitted(self, toy_train_matrix, toy_idx2artist):
        bpr = BPRRecommender(factors=4, iterations=10, use_gpu=False)
        assert not bpr.is_fitted
        bpr.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)
        assert bpr.is_fitted

    def test_recommend_returns_recs(self, toy_train_matrix, toy_idx2artist):
        bpr = BPRRecommender(factors=4, iterations=10, use_gpu=False)
        bpr.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)
        recs = bpr.recommend(user_id=0, n=3)
        assert len(recs) == 3
        assert all(isinstance(r, Recommendation) for r in recs)

    def test_recommend_excludes_known(self, toy_train_matrix, toy_idx2artist):
        bpr = BPRRecommender(factors=4, iterations=10, use_gpu=False)
        bpr.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)
        known = set(toy_train_matrix[0].indices.tolist())
        recs = bpr.recommend(user_id=0, n=3, exclude_known=True)
        rec_ids = {r.item_id for r in recs}
        assert rec_ids.isdisjoint(known)

    def test_get_scores_batch(self, toy_train_matrix, toy_idx2artist):
        bpr = BPRRecommender(factors=4, iterations=10, use_gpu=False)
        bpr.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)
        item_ids = np.array([4, 5, 6])
        scores = bpr.get_scores(user_id=0, item_ids=item_ids)
        assert len(scores) == 3
        assert scores.dtype == np.float32

    def test_get_similar_items(self, toy_train_matrix, toy_idx2artist):
        bpr = BPRRecommender(factors=4, iterations=10, use_gpu=False)
        bpr.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)
        similar = bpr.get_similar_items(item_id=0, n=3)
        assert len(similar) <= 3
        sim_ids = {r.item_id for r in similar}
        assert 0 not in sim_ids

    def test_explain_has_content(self, toy_train_matrix, toy_idx2artist):
        bpr = BPRRecommender(factors=4, iterations=10, use_gpu=False)
        bpr.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)
        reasons = bpr.explain(user_id=0, item_id=4)
        assert len(reasons) > 0

    def test_save_and_load(self, toy_train_matrix, toy_idx2artist, tmp_path):
        bpr = BPRRecommender(factors=4, iterations=10, use_gpu=False)
        bpr.fit(train_interactions=toy_train_matrix, idx2artist=toy_idx2artist)
        save_file = tmp_path / "bpr_test.npz"
        bpr.save(path=save_file)
        assert save_file.exists()

        bpr_loaded = BPRRecommender(use_gpu=False)
        bpr_loaded.load(
            path=save_file,
            train_interactions=toy_train_matrix,
            idx2artist=toy_idx2artist,
        )
        assert bpr_loaded.is_fitted
        recs = bpr_loaded.recommend(user_id=0, n=3)
        assert len(recs) == 3

    def test_not_fitted_raises(self):
        bpr = BPRRecommender()
        with pytest.raises(RuntimeError, match="not fitted"):
            bpr.recommend(user_id=0, n=5)

