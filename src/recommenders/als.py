"""
ALS Recommender — Phase 7

Alternating Least Squares for implicit feedback collaborative filtering.

Uses the `implicit` library's ALS implementation which follows the
Hu et al. (2008) formulation:
    Minimize Σ c_ui (p_ui - x_u^T y_i)^2 + λ(||x_u||^2 + ||y_i||^2)

where:
    p_ui = 1 if user u interacted with item i, else 0
    c_ui = 1 + α * log1p(r_ui)  — confidence from play count r_ui

The α parameter controls how much raw play counts influence confidence.
Higher α means the model trusts high play counts more.
"""

import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy.sparse import csr_matrix

from src.config import ALS_DEFAULTS, ALS_MODEL, MODELS_DIR
from src.recommenders.base import BaseRecommender, Recommendation

logger = logging.getLogger(__name__)


class ALSRecommender(BaseRecommender):
    """Implicit feedback ALS collaborative filtering recommender.

    Wraps `implicit.als.AlternatingLeastSquares` with confidence-weighted
    play count transformation and the BaseRecommender interface.
    """

    def __init__(
        self,
        factors: int = ALS_DEFAULTS["factors"],
        regularization: float = ALS_DEFAULTS["regularization"],
        iterations: int = ALS_DEFAULTS["iterations"],
        alpha: float = 40.0,
        use_gpu: bool = ALS_DEFAULTS["use_gpu"],
        random_state: int = 42,
    ):
        """Initialize ALS recommender.

        Args:
            factors: Number of latent factors.
            regularization: L2 regularization weight.
            iterations: Number of ALS iterations.
            alpha: Confidence scaling parameter for play counts.
            use_gpu: Whether to use GPU acceleration (requires CUDA).
            random_state: Random seed for reproducibility.
        """
        super().__init__(name="ALS")
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.use_gpu = use_gpu
        self.random_state = random_state

        self._model = None
        self._train_interactions: csr_matrix | None = None
        self._confidence_matrix: csr_matrix | None = None
        self._idx2artist: list[str] | None = None

    def _build_confidence_matrix(self, interactions: csr_matrix) -> csr_matrix:
        """Transform play counts to confidence weights.

        c_ui = 1 + alpha * log1p(play_count)

        Args:
            interactions: Sparse matrix with raw play counts.

        Returns:
            Sparse matrix with confidence values.
        """
        confidence = interactions.copy().astype(np.float32)
        confidence.data = 1.0 + self.alpha * np.log1p(confidence.data)
        return confidence

    def fit(
        self,
        train_interactions: csr_matrix,
        idx2artist: list[str] | None = None,
        **kwargs,
    ) -> "ALSRecommender":
        """Train ALS model on the training interaction matrix.

        Args:
            train_interactions: Sparse CSR matrix [n_users × n_artists]
                with raw play counts.
            idx2artist: Optional list mapping artist index → name.

        Returns:
            self
        """
        from implicit.als import AlternatingLeastSquares

        logger.info(
            "Fitting ALS (factors=%d, reg=%.4f, iter=%d, alpha=%.1f)...",
            self.factors,
            self.regularization,
            self.iterations,
            self.alpha,
        )

        self._train_interactions = train_interactions
        self._idx2artist = idx2artist

        # Build confidence matrix
        self._confidence_matrix = self._build_confidence_matrix(train_interactions)

        # Initialize and fit the implicit library model
        # implicit expects item-user matrix (CSC or transposed CSR)
        self._model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=self.use_gpu,
            random_state=self.random_state,
        )

        # implicit expects user-item matrix in CSR format
        self._model.fit(self._confidence_matrix)

        self._is_fitted = True
        logger.info(
            "ALS fitted: user factors shape=%s, item factors shape=%s.",
            self._model.user_factors.shape,
            self._model.item_factors.shape,
        )
        return self

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_known: bool = True,
        **kwargs,
    ) -> list[Recommendation]:
        """Generate top-N recommendations using ALS latent factors.

        Args:
            user_id: Internal user ID.
            n: Number of recommendations.
            exclude_known: Whether to exclude already-interacted items.

        Returns:
            List of Recommendation objects.
        """
        self._check_fitted()

        # Use implicit's recommend method
        user_items = self._confidence_matrix if exclude_known else None

        ids, scores = self._model.recommend(
            user_id,
            user_items[user_id] if user_items is not None else None,
            N=n,
            filter_already_liked_items=exclude_known,
        )

        recs = []
        for item_id, score in zip(ids, scores):
            item_id_int = int(item_id)
            name = (
                self._idx2artist[item_id_int]
                if self._idx2artist is not None
                else f"artist_{item_id_int}"
            )
            recs.append(Recommendation(
                item_id=item_id_int,
                item_name=name,
                score=float(score),
                reasons=self.explain(user_id, item_id_int),
            ))

        return recs

    def explain(self, user_id: int, item_id: int) -> list[str]:
        """Explain recommendation based on similar user tastes.

        Args:
            user_id: Internal user ID.
            item_id: Internal item ID.

        Returns:
            List of explanation strings.
        """
        reasons = ["Users with similar listening patterns also enjoy this artist"]

        if self._model is not None and self._train_interactions is not None:
            # Find which of the user's liked items contributed most to this score
            user_row = self._train_interactions[user_id]
            if len(user_row.indices) > 0:
                user_factor = self._model.user_factors[user_id]
                item_factor = self._model.item_factors[item_id]

                # Score the user's known items to find similar ones
                known_factors = self._model.item_factors[user_row.indices]
                known_sims = known_factors @ item_factor
                top_k = min(3, len(known_sims))
                top_idx = np.argsort(known_sims)[::-1][:top_k]

                similar_liked = []
                for ki in top_idx:
                    artist_idx = user_row.indices[ki]
                    name = (
                        self._idx2artist[artist_idx]
                        if self._idx2artist is not None
                        else f"artist_{artist_idx}"
                    )
                    similar_liked.append(name)

                if similar_liked:
                    reasons.append(
                        f"Related to your artists: {', '.join(similar_liked)}"
                    )

        return reasons

    def get_scores(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        """Score a batch of candidate items using ALS factors.

        Args:
            user_id: Internal user ID.
            item_ids: Array of item indices to score.

        Returns:
            Array of dot-product scores.
        """
        self._check_fitted()

        user_factor = self._model.user_factors[user_id]
        item_factors = self._model.item_factors[item_ids]
        scores = item_factors @ user_factor

        return scores.astype(np.float32)

    def get_similar_items(
        self, item_id: int, n: int = 10
    ) -> list[Recommendation]:
        """Find similar items using ALS item factors.

        Args:
            item_id: Internal item ID.
            n: Number of similar items.

        Returns:
            List of Recommendation objects.
        """
        self._check_fitted()

        ids, scores = self._model.similar_items(item_id, N=n + 1)

        recs = []
        for iid, score in zip(ids, scores):
            iid_int = int(iid)
            if iid_int == item_id:
                continue  # Skip self
            name = (
                self._idx2artist[iid_int]
                if self._idx2artist is not None
                else f"artist_{iid_int}"
            )
            recs.append(Recommendation(
                item_id=iid_int,
                item_name=name,
                score=float(score),
            ))

        return recs[:n]

    def save(self, path: Path | None = None) -> None:
        """Save ALS model factors to disk."""
        self._check_fitted()
        path = path or ALS_MODEL
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            user_factors=self._model.user_factors,
            item_factors=self._model.item_factors,
        )
        logger.info("Saved ALS model to %s.", path.name)

    def load(
        self,
        path: Path | None = None,
        train_interactions: csr_matrix | None = None,
        idx2artist: list[str] | None = None,
    ) -> "ALSRecommender":
        """Load ALS model factors from disk.

        Args:
            path: Path to the saved NPZ file.
            train_interactions: Training matrix (needed for recommend).
            idx2artist: Artist name mapping.

        Returns:
            self
        """
        from implicit.als import AlternatingLeastSquares

        path = path or ALS_MODEL

        data = np.load(path)
        user_factors = data["user_factors"]
        item_factors = data["item_factors"]

        self._model = AlternatingLeastSquares(
            factors=user_factors.shape[1],
            use_gpu=self.use_gpu,
        )
        self._model.user_factors = user_factors
        self._model.item_factors = item_factors

        if train_interactions is not None:
            self._train_interactions = train_interactions
            self._confidence_matrix = self._build_confidence_matrix(train_interactions)
        self._idx2artist = idx2artist
        self._is_fitted = True

        logger.info(
            "Loaded ALS model from %s (factors=%d).",
            path.name,
            user_factors.shape[1],
        )
        return self

    def _check_fitted(self) -> None:
        if not self._is_fitted or self._model is None:
            raise RuntimeError(
                "ALSRecommender is not fitted. Call fit() or load() first."
            )
