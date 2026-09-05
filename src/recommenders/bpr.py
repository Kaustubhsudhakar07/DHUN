"""
BPR Recommender — Phase 7

Bayesian Personalized Ranking for implicit feedback collaborative filtering.

Uses the `implicit` library's BPR implementation which follows the
Rendle et al. (2009) formulation:
    Maximize Σ ln σ(x_u^T y_i - x_u^T y_j) - λ(||x_u||^2 + ||y_i||^2 + ||y_j||^2)

where (u, i, j) are triples such that user u interacted with item i
but not item j.

BPR optimizes the ranking order directly (unlike ALS which optimizes
point-wise reconstruction), making it particularly well-suited for
Top-K recommendation tasks.
"""

import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy.sparse import csr_matrix

from src.config import BPR_DEFAULTS, BPR_MODEL, MODELS_DIR
from src.recommenders.base import BaseRecommender, Recommendation

logger = logging.getLogger(__name__)


class BPRRecommender(BaseRecommender):
    """Implicit feedback BPR collaborative filtering recommender.

    Wraps `implicit.bpr.BayesianPersonalizedRanking` with the
    BaseRecommender interface.
    """

    def __init__(
        self,
        factors: int = BPR_DEFAULTS["factors"],
        learning_rate: float = BPR_DEFAULTS["learning_rate"],
        regularization: float = BPR_DEFAULTS["regularization"],
        iterations: int = BPR_DEFAULTS["iterations"],
        alpha: float = 40.0,
        use_gpu: bool = BPR_DEFAULTS["use_gpu"],
        random_state: int = 42,
    ):
        """Initialize BPR recommender.

        Args:
            factors: Number of latent factors.
            learning_rate: SGD learning rate.
            regularization: L2 regularization weight.
            iterations: Number of training epochs.
            alpha: Confidence scaling for play count transformation.
            use_gpu: Whether to use GPU acceleration.
            random_state: Random seed.
        """
        super().__init__(name="BPR")
        self.factors = factors
        self.learning_rate = learning_rate
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

        Same transformation as ALS for consistency:
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
    ) -> "BPRRecommender":
        """Train BPR model on the training interaction matrix.

        Args:
            train_interactions: Sparse CSR matrix [n_users × n_artists]
                with raw play counts.
            idx2artist: Optional list mapping artist index → name.

        Returns:
            self
        """
        from implicit.bpr import BayesianPersonalizedRanking

        logger.info(
            "Fitting BPR (factors=%d, lr=%.4f, reg=%.4f, iter=%d, alpha=%.1f)...",
            self.factors,
            self.learning_rate,
            self.regularization,
            self.iterations,
            self.alpha,
        )

        self._train_interactions = train_interactions
        self._idx2artist = idx2artist

        # Build confidence matrix
        self._confidence_matrix = self._build_confidence_matrix(train_interactions)

        # Initialize and fit
        self._model = BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=self.use_gpu,
            random_state=self.random_state,
        )

        self._model.fit(self._confidence_matrix)

        self._is_fitted = True
        logger.info(
            "BPR fitted: user factors shape=%s, item factors shape=%s.",
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
        """Generate top-N recommendations using BPR latent factors.

        Args:
            user_id: Internal user ID.
            n: Number of recommendations.
            exclude_known: Whether to exclude already-interacted items.

        Returns:
            List of Recommendation objects.
        """
        self._check_fitted()

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
        """Explain recommendation based on pairwise preference learning.

        Args:
            user_id: Internal user ID.
            item_id: Internal item ID.

        Returns:
            List of explanation strings.
        """
        reasons = [
            "Ranked high by learning from your listening preferences"
        ]

        if self._model is not None and self._train_interactions is not None:
            user_row = self._train_interactions[user_id]
            if len(user_row.indices) > 0:
                item_factor = self._model.item_factors[item_id]

                # Find most similar listened items via factor dot product
                known_factors = self._model.item_factors[user_row.indices]
                sims = known_factors @ item_factor
                top_k = min(3, len(sims))
                top_idx = np.argsort(sims)[::-1][:top_k]

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
        """Score a batch of candidate items using BPR factors.

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
        """Find similar items using BPR item factors.

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
                continue
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
        """Save BPR model factors to disk."""
        self._check_fitted()
        path = path or BPR_MODEL
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            user_factors=self._model.user_factors,
            item_factors=self._model.item_factors,
        )
        logger.info("Saved BPR model to %s.", path.name)

    def load(
        self,
        path: Path | None = None,
        train_interactions: csr_matrix | None = None,
        idx2artist: list[str] | None = None,
    ) -> "BPRRecommender":
        """Load BPR model factors from disk.

        Args:
            path: Path to the saved NPZ file.
            train_interactions: Training matrix (needed for recommend).
            idx2artist: Artist name mapping.

        Returns:
            self
        """
        from implicit.bpr import BayesianPersonalizedRanking

        path = path or BPR_MODEL

        data = np.load(path)
        user_factors = data["user_factors"]
        item_factors = data["item_factors"]

        self._model = BayesianPersonalizedRanking(
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
            "Loaded BPR model from %s (factors=%d).",
            path.name,
            user_factors.shape[1],
        )
        return self

    def _check_fitted(self) -> None:
        if not self._is_fitted or self._model is None:
            raise RuntimeError(
                "BPRRecommender is not fitted. Call fit() or load() first."
            )
