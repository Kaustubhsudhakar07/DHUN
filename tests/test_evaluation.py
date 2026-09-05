"""
Unit tests for evaluation metrics (Phase 8).

Tests verify metric functions against hand-computed values to ensure
correctness before trusting them for model evaluation.
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.evaluation.beyond_accuracy import (
    catalog_coverage,
    diversity,
    novelty,
    personalization,
)


# =========================================================================
# Ranking Metric Tests
# =========================================================================

class TestPrecisionAtK:

    def test_perfect_precision(self):
        """All recommendations are relevant."""
        recommended = [1, 2, 3]
        relevant = {1, 2, 3}
        assert precision_at_k(recommended, relevant, k=3) == 1.0

    def test_zero_precision(self):
        """No recommendations are relevant."""
        recommended = [4, 5, 6]
        relevant = {1, 2, 3}
        assert precision_at_k(recommended, relevant, k=3) == 0.0

    def test_partial_precision(self):
        """2 of 4 recommendations are relevant."""
        recommended = [1, 4, 2, 5]
        relevant = {1, 2, 3}
        assert precision_at_k(recommended, relevant, k=4) == 0.5

    def test_k_truncation(self):
        """Only consider first K items."""
        recommended = [1, 4, 5, 2, 3]
        relevant = {1, 2, 3}
        # K=2: only [1, 4], precision = 1/2
        assert precision_at_k(recommended, relevant, k=2) == 0.5

    def test_k_zero(self):
        assert precision_at_k([1], {1}, k=0) == 0.0


class TestRecallAtK:

    def test_perfect_recall(self):
        recommended = [1, 2, 3]
        relevant = {1, 2, 3}
        assert recall_at_k(recommended, relevant, k=3) == 1.0

    def test_zero_recall(self):
        recommended = [4, 5, 6]
        relevant = {1, 2, 3}
        assert recall_at_k(recommended, relevant, k=3) == 0.0

    def test_partial_recall(self):
        """1 of 3 relevant items found in top-2."""
        recommended = [1, 4]
        relevant = {1, 2, 3}
        assert recall_at_k(recommended, relevant, k=2) == pytest.approx(1 / 3)

    def test_empty_relevant(self):
        recommended = [1, 2]
        relevant = set()
        assert recall_at_k(recommended, relevant, k=2) == 0.0


class TestHitRateAtK:

    def test_hit(self):
        recommended = [4, 1, 5]
        relevant = {1}
        assert hit_rate_at_k(recommended, relevant, k=3) == 1.0

    def test_miss(self):
        recommended = [4, 5, 6]
        relevant = {1}
        assert hit_rate_at_k(recommended, relevant, k=3) == 0.0

    def test_hit_at_position_1(self):
        recommended = [1, 4, 5]
        relevant = {1}
        assert hit_rate_at_k(recommended, relevant, k=1) == 1.0

    def test_miss_beyond_k(self):
        """Relevant item exists but beyond K."""
        recommended = [4, 5, 1]
        relevant = {1}
        assert hit_rate_at_k(recommended, relevant, k=2) == 0.0


class TestAveragePrecisionAtK:

    def test_perfect_ap(self):
        """Relevant item at position 1."""
        recommended = [1, 2, 3]
        relevant = {1}
        # AP@3 = (1/1 * 1) / min(1, 3) = 1.0
        assert average_precision_at_k(recommended, relevant, k=3) == 1.0

    def test_ap_relevant_at_end(self):
        """Relevant item at position 3."""
        recommended = [4, 5, 1]
        relevant = {1}
        # AP@3 = (1/3) / 1 = 0.333...
        assert average_precision_at_k(recommended, relevant, k=3) == pytest.approx(1 / 3)

    def test_ap_two_relevant(self):
        """Two relevant items at positions 1 and 3."""
        recommended = [1, 4, 2]
        relevant = {1, 2}
        # AP@3 = (1/1 + 2/3) / min(2,3) = (1 + 0.6667) / 2 = 0.8333
        expected = (1 / 1 + 2 / 3) / 2
        assert average_precision_at_k(recommended, relevant, k=3) == pytest.approx(expected)

    def test_empty_relevant(self):
        recommended = [1, 2]
        relevant = set()
        assert average_precision_at_k(recommended, relevant, k=2) == 0.0


class TestNDCGAtK:

    def test_perfect_ndcg(self):
        """Relevant item at position 1 → DCG = IDCG."""
        recommended = [1]
        relevant = {1}
        assert ndcg_at_k(recommended, relevant, k=1) == 1.0

    def test_ndcg_at_position_2(self):
        """Relevant item at position 2."""
        recommended = [4, 1]
        relevant = {1}
        # DCG = 1/log2(3), IDCG = 1/log2(2)
        expected = (1 / np.log2(3)) / (1 / np.log2(2))
        assert ndcg_at_k(recommended, relevant, k=2) == pytest.approx(expected)

    def test_ndcg_zero(self):
        recommended = [4, 5, 6]
        relevant = {1}
        assert ndcg_at_k(recommended, relevant, k=3) == 0.0

    def test_empty_relevant(self):
        recommended = [1, 2]
        relevant = set()
        assert ndcg_at_k(recommended, relevant, k=2) == 0.0

    def test_ndcg_two_relevant_perfect(self):
        """Two relevant items at top positions."""
        recommended = [1, 2, 4]
        relevant = {1, 2}
        # DCG = 1/log2(2) + 1/log2(3) = IDCG
        assert ndcg_at_k(recommended, relevant, k=3) == pytest.approx(1.0)


# =========================================================================
# Beyond-Accuracy Metric Tests
# =========================================================================

class TestCatalogCoverage:

    def test_full_coverage(self):
        recs = [[0, 1], [2, 3], [4]]
        assert catalog_coverage(recs, n_items=5) == 1.0

    def test_partial_coverage(self):
        recs = [[0, 1], [0, 2]]
        assert catalog_coverage(recs, n_items=5) == pytest.approx(3 / 5)

    def test_zero_coverage(self):
        assert catalog_coverage([], n_items=5) == 0.0

    def test_empty_catalog(self):
        assert catalog_coverage([[1]], n_items=0) == 0.0


class TestDiversity:

    def test_identical_items_zero_diversity(self):
        """If no feature matrix, diversity = uniqueness ratio."""
        recs = [0, 0, 0]
        div = diversity(recs, item_features=None)
        assert div == pytest.approx(1 / 3)

    def test_single_item(self):
        assert diversity([0], item_features=None) == 0.0

    def test_empty(self):
        assert diversity([], item_features=None) == 0.0


class TestNovelty:

    def test_popular_items_low_novelty(self):
        """Recommending popular items → low novelty."""
        pop = np.array([100, 50, 10, 1])
        n1 = novelty([0], pop, n_users=100)     # Item 0 is very popular
        n2 = novelty([3], pop, n_users=100)     # Item 3 is very rare
        assert n2 > n1

    def test_zero_popularity_max_novelty(self):
        pop = np.array([0, 0])
        nov = novelty([0], pop, n_users=100)
        assert nov == pytest.approx(np.log2(100))

    def test_empty_recommendations(self):
        assert novelty([], np.array([10, 20]), n_users=100) == 0.0


class TestPersonalization:

    def test_identical_lists(self):
        """Same list for everyone → zero personalization."""
        recs = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
        assert personalization(recs) == 0.0

    def test_completely_different_lists(self):
        """No overlap → personalization = 1.0."""
        recs = [[0, 1], [2, 3], [4, 5]]
        assert personalization(recs) == 1.0

    def test_single_user(self):
        assert personalization([[0, 1]]) == 0.0

    def test_partial_overlap(self):
        recs = [[0, 1, 2], [1, 2, 3]]
        p = personalization(recs)
        assert 0.0 < p < 1.0
