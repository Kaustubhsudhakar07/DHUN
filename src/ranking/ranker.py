"""
Candidate Ranking, Normalization, and Filtering — Phase 9

Provides reusable components for the recommendation ranking pipeline:
- ScoreNormalizer: Per-user score scaling (min-max, rank) across models
- CandidatePool: Merging candidate sets from collaborative, content, and popularity sources
- PostFilter: Known-item filtering and score thresholding
"""

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


class ScoreNormalizer:
    """Scales disparate model scores to comparable ranges [0, 1]."""

    @staticmethod
    def min_max(scores: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """Min-max normalize a 1D array of scores to [0, 1].

        If all scores are equal, returns 0.5 (or 0.0 if all zero).

        Args:
            scores: 1D array of raw scores.
            epsilon: Small float to prevent division by zero.

        Returns:
            Normalized 1D float32 array in [0, 1].
        """
        if len(scores) == 0:
            return np.array([], dtype=np.float32)

        scores_arr = np.asarray(scores, dtype=np.float32)
        min_val = float(scores_arr.min())
        max_val = float(scores_arr.max())

        spread = max_val - min_val
        if spread < epsilon:
            if max_val == 0.0:
                return np.zeros_like(scores_arr)
            return np.full_like(scores_arr, 0.5)

        norm = (scores_arr - min_val) / (spread + epsilon)
        return np.clip(norm, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def rank_percentile(scores: np.ndarray) -> np.ndarray:
        """Convert scores to rank percentiles in [0, 1].

        Highest score gets 1.0, lowest gets 0.0 (or 1/N).
        Robust against outliers and non-linear scale differences.

        Args:
            scores: 1D array of raw scores.

        Returns:
            Rank percentile array in [0, 1].
        """
        n = len(scores)
        if n <= 1:
            return np.ones(n, dtype=np.float32)

        scores_arr = np.asarray(scores)
        # argsort twice gives ranks: 0 for smallest, n-1 for largest
        ranks = np.argsort(np.argsort(scores_arr)).astype(np.float32)
        return (ranks / (n - 1)).astype(np.float32)


class CandidatePool:
    """Combines and deduplicates candidate items from multiple recommendation sources."""

    @staticmethod
    def merge(
        *candidate_lists: Sequence[int] | np.ndarray,
        max_candidates: int | None = None,
    ) -> np.ndarray:
        """Merge candidate ID lists preserving relative order.

        Earlier lists and earlier elements in lists take priority.

        Args:
            *candidate_lists: Variable number of candidate sequences.
            max_candidates: Optional cap on total candidates.

        Returns:
            1D numpy array of unique item IDs.
        """
        seen = set()
        merged = []

        for clist in candidate_lists:
            for item in clist:
                item_id = int(item)
                if item_id not in seen:
                    seen.add(item_id)
                    merged.append(item_id)
                    if max_candidates is not None and len(merged) >= max_candidates:
                        return np.array(merged, dtype=np.int64)

        return np.array(merged, dtype=np.int64)


class PostFilter:
    """Post-scoring candidate filtering."""

    @staticmethod
    def filter_known(
        item_ids: np.ndarray,
        known_items: set[int] | np.ndarray,
    ) -> np.ndarray:
        """Filter out items already interacted with by the user.

        Args:
            item_ids: Candidate item IDs array.
            known_items: Set or array of known/interacted item IDs.

        Returns:
            Filtered item IDs array.
        """
        if len(item_ids) == 0:
            return item_ids

        known_set = known_items if isinstance(known_items, set) else set(known_items.tolist())
        if not known_set:
            return item_ids

        mask = [iid not in known_set for iid in item_ids]
        return item_ids[mask]

    @staticmethod
    def filter_by_min_score(
        item_ids: np.ndarray,
        scores: np.ndarray,
        min_score: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter candidates below a minimum score threshold.

        Args:
            item_ids: Array of item IDs.
            scores: Array of corresponding scores.
            min_score: Minimum allowable score.

        Returns:
            Tuple of (filtered_item_ids, filtered_scores).
        """
        mask = scores >= min_score
        return item_ids[mask], scores[mask]
