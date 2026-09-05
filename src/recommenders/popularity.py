"""
Popularity-Based Recommender — Phase 7

Non-personalized baseline that recommends the most globally popular artists.
Provides a lower bound for evaluating personalized models.

Popularity score = log1p(total_play_count) across all training users.
Using log1p instead of raw counts prevents a handful of mega-popular
artists from dominating all recommendation lists.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from src.config import POPULARITY_SCORES, MODELS_DIR
from src.recommenders.base import BaseRecommender, Recommendation

logger = logging.getLogger(__name__)


class PopularityRecommender(BaseRecommender):
    """Recommend the most globally popular artists.

    This is a non-personalized baseline: every user gets the same ranked
    list (minus their already-known artists). Essential as a sanity check —
    any personalized model that can't beat this is adding no value.
    """

    def __init__(self):
        super().__init__(name="Popularity")
        self._scores: np.ndarray | None = None
        self._sorted_indices: np.ndarray | None = None
        self._listener_counts: np.ndarray | None = None
        self._idx2artist: list[str] | None = None

    def fit(
        self,
        train_interactions: csr_matrix,
        idx2artist: list[str] | None = None,
        **kwargs,
    ) -> "PopularityRecommender":
        """Compute popularity scores from the training interaction matrix.

        Args:
            train_interactions: Sparse CSR matrix [n_users × n_artists]
                with play counts as values.
            idx2artist: Optional list mapping artist index → name.

        Returns:
            self
        """
        logger.info("Fitting PopularityRecommender...")

        # Total play count per artist (column-wise sum), smoothed by log1p
        total_plays = np.asarray(train_interactions.sum(axis=0)).ravel()
        self._scores = np.log1p(total_plays).astype(np.float32)

        # Number of distinct listeners per artist
        binary = train_interactions.copy()
        binary.data = np.ones_like(binary.data)
        self._listener_counts = np.asarray(binary.sum(axis=0)).ravel().astype(np.int32)

        # Pre-sort for fast retrieval
        self._sorted_indices = np.argsort(self._scores)[::-1]

        self._idx2artist = idx2artist
        self._is_fitted = True

        logger.info(
            "PopularityRecommender fitted: %d artists, top artist score=%.2f, "
            "max listeners=%d.",
            len(self._scores),
            self._scores[self._sorted_indices[0]],
            self._listener_counts.max(),
        )
        return self

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_known: bool = True,
        known_items: np.ndarray | None = None,
        **kwargs,
    ) -> list[Recommendation]:
        """Return top-N most popular artists not already known to the user.

        Args:
            user_id: Internal user ID (used to look up known items).
            n: Number of recommendations.
            exclude_known: Whether to filter out already-interacted items.
            known_items: Optional array of item indices already known.
                If None and exclude_known is True, caller should provide
                this or pass train_interactions in kwargs.

        Returns:
            List of Recommendation objects.
        """
        self._check_fitted()

        # Determine items to exclude
        excluded = set()
        if exclude_known:
            if known_items is not None:
                excluded = set(known_items.tolist())
            elif "train_interactions" in kwargs:
                user_row = kwargs["train_interactions"][user_id]
                excluded = set(user_row.indices.tolist())

        recs = []
        for idx in self._sorted_indices:
            if len(recs) >= n:
                break
            idx_int = int(idx)
            if idx_int in excluded:
                continue

            name = (
                self._idx2artist[idx_int]
                if self._idx2artist is not None
                else f"artist_{idx_int}"
            )
            recs.append(Recommendation(
                item_id=idx_int,
                item_name=name,
                score=float(self._scores[idx_int]),
                reasons=self.explain(user_id, idx_int),
                metadata={"listeners": int(self._listener_counts[idx_int])},
            ))

        return recs

    def explain(self, user_id: int, item_id: int) -> list[str]:
        """Explain why this item was recommended.

        Args:
            user_id: Internal user ID.
            item_id: Internal item ID.

        Returns:
            List of explanation strings.
        """
        if self._listener_counts is not None and item_id < len(self._listener_counts):
            n_listeners = int(self._listener_counts[item_id])
            return [f"Popular artist — listened to by {n_listeners:,} users"]
        return ["Popular artist based on global listening data"]

    def get_scores(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        """Return popularity scores for a batch of candidate items.

        Args:
            user_id: Internal user ID (ignored — scores are global).
            item_ids: Array of item indices to score.

        Returns:
            Array of popularity scores.
        """
        self._check_fitted()
        return self._scores[item_ids]

    def save(self, path: Path | None = None) -> None:
        """Save popularity scores to disk."""
        path = path or POPULARITY_SCORES
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "scores": self._scores.tolist(),
            "listener_counts": self._listener_counts.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f)
        logger.info("Saved popularity scores to %s.", path.name)

    def load(
        self,
        path: Path | None = None,
        idx2artist: list[str] | None = None,
    ) -> "PopularityRecommender":
        """Load popularity scores from disk."""
        path = path or POPULARITY_SCORES
        with open(path, "r") as f:
            data = json.load(f)
        self._scores = np.array(data["scores"], dtype=np.float32)
        self._listener_counts = np.array(data["listener_counts"], dtype=np.int32)
        self._sorted_indices = np.argsort(self._scores)[::-1]
        self._idx2artist = idx2artist
        self._is_fitted = True
        logger.info("Loaded popularity scores from %s.", path.name)
        return self

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "PopularityRecommender is not fitted. Call fit() first."
            )
