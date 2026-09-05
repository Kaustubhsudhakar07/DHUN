"""
Hybrid Recommender — Phase 9

Combines collaborative filtering (ALS or BPR), content-based (TF-IDF),
and popularity signals into a unified ranking system with:
- Multi-source candidate generation and pooling
- Per-user score normalization (min-max / rank-percentile)
- Dynamic weight rebalancing for cold-start safety
- Multi-perspective composite explanations
- Tunable weights persisted to models/hybrid_weights.json
"""

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from src.config import (
    CANDIDATE_POOL_SIZE,
    DEFAULT_HYBRID_WEIGHTS,
    DEFAULT_TOP_K,
    HYBRID_WEIGHTS,
    MODELS_DIR,
)
from src.ranking.ranker import CandidatePool, PostFilter, ScoreNormalizer
from src.recommenders.base import BaseRecommender, Recommendation
from src.recommenders.content import ContentBasedRecommender
from src.recommenders.popularity import PopularityRecommender

logger = logging.getLogger(__name__)


class HybridRecommender(BaseRecommender):
    """Hybrid recommendation engine combining CF, Content, and Popularity signals."""

    def __init__(
        self,
        cf_recommender: BaseRecommender | None = None,
        content_recommender: BaseRecommender | None = None,
        popularity_recommender: BaseRecommender | None = None,
        weights: dict[str, float] | None = None,
        candidate_pool_size: int = CANDIDATE_POOL_SIZE,
        normalization: str = "min_max",
    ):
        """Initialize HybridRecommender.

        Args:
            cf_recommender: Collaborative filtering recommender (ALS or BPR).
            content_recommender: ContentBasedRecommender instance.
            popularity_recommender: PopularityRecommender instance.
            weights: Dict mapping 'collaborative', 'content', 'popularity' -> float weights.
            candidate_pool_size: Maximum size of the merged candidate pool.
            normalization: 'min_max' or 'rank_percentile'.
        """
        super().__init__(name="Hybrid")
        self.cf_model = cf_recommender
        self.content_model = content_recommender
        self.popularity_model = popularity_recommender

        self.weights = self._normalize_weights(weights or DEFAULT_HYBRID_WEIGHTS)
        self.candidate_pool_size = candidate_pool_size
        self.normalization = normalization

        self._train_interactions: csr_matrix | None = None
        self._idx2artist: list[str] | None = None

        # Check if sub-models are already fitted
        self._check_and_set_fitted()

    def _normalize_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """Ensure weights sum to 1.0."""
        cf_w = float(weights.get("collaborative", 0.6))
        cnt_w = float(weights.get("content", 0.3))
        pop_w = float(weights.get("popularity", 0.1))

        total = cf_w + cnt_w + pop_w
        if total <= 0:
            return {"collaborative": 0.6, "content": 0.3, "popularity": 0.1}

        return {
            "collaborative": cf_w / total,
            "content": cnt_w / total,
            "popularity": pop_w / total,
        }

    def _check_and_set_fitted(self) -> None:
        """Mark hybrid as fitted if at least one sub-model is fitted."""
        sub_models = [self.cf_model, self.content_model, self.popularity_model]
        fitted_models = [m for m in sub_models if m is not None and getattr(m, "is_fitted", False)]
        self._is_fitted = len(fitted_models) > 0

    def fit(
        self,
        train_interactions: csr_matrix | None = None,
        idx2artist: list[str] | None = None,
        cf_recommender: BaseRecommender | None = None,
        content_recommender: BaseRecommender | None = None,
        popularity_recommender: BaseRecommender | None = None,
        **kwargs,
    ) -> "HybridRecommender":
        """Connect sub-models and metadata to the hybrid engine.

        Args:
            train_interactions: Sparse interaction matrix.
            idx2artist: Artist name mapping.
            cf_recommender: Optional CF model to attach.
            content_recommender: Optional Content model to attach.
            popularity_recommender: Optional Popularity model to attach.

        Returns:
            self
        """
        if cf_recommender is not None:
            self.cf_model = cf_recommender
        if content_recommender is not None:
            self.content_model = content_recommender
        if popularity_recommender is not None:
            self.popularity_model = popularity_recommender

        self._train_interactions = train_interactions
        self._idx2artist = idx2artist

        # Pass metadata to sub-models if they don't already have it
        for m in [self.cf_model, self.content_model, self.popularity_model]:
            if m is not None and getattr(m, "_idx2artist", None) is None and idx2artist is not None:
                m._idx2artist = idx2artist

        self._check_and_set_fitted()
        if not self._is_fitted:
            raise RuntimeError(
                "HybridRecommender requires at least one fitted sub-model."
            )

        logger.info(
            "HybridRecommender initialized with weights: CF=%.2f, Content=%.2f, Pop=%.2f.",
            self.weights["collaborative"],
            self.weights["content"],
            self.weights["popularity"],
        )
        return self

    def _get_active_weights(
        self,
        user_id: int,
        has_cf_signal: bool,
        has_content_signal: bool,
    ) -> dict[str, float]:
        """Dynamically rebalance weights if user lacks CF history or content profile.

        Args:
            user_id: Internal user ID.
            has_cf_signal: Whether CF returned meaningful non-zero scores.
            has_content_signal: Whether user has listening history with matching tags.

        Returns:
            Dict of adjusted weights summing to 1.0.
        """
        w_cf = self.weights["collaborative"] if (self.cf_model and has_cf_signal) else 0.0
        w_cnt = self.weights["content"] if (self.content_model and has_content_signal) else 0.0
        w_pop = self.weights["popularity"] if self.popularity_model else 0.0

        total = w_cf + w_cnt + w_pop
        if total <= 0:
            # Fallback 100% popularity
            return {"collaborative": 0.0, "content": 0.0, "popularity": 1.0}

        return {
            "collaborative": w_cf / total,
            "content": w_cnt / total,
            "popularity": w_pop / total,
        }

    def _retrieve_candidates(
        self,
        user_id: int,
        n_candidates_per_source: int,
        exclude_known: bool,
        known_items: set[int],
    ) -> np.ndarray:
        """Retrieve candidate IDs from CF, Content, and Popularity sources."""
        cf_cands = []
        content_cands = []
        pop_cands = []

        # 1. CF candidates
        if self.cf_model is not None and self.cf_model.is_fitted:
            try:
                cf_recs = self.cf_model.recommend(
                    user_id=user_id,
                    n=n_candidates_per_source,
                    exclude_known=exclude_known,
                )
                cf_cands = [r.item_id for r in cf_recs]
            except Exception as e:
                logger.debug("CF candidate retrieval error for user %d: %s", user_id, e)

        # 2. Content candidates
        if self.content_model is not None and self.content_model.is_fitted:
            try:
                cnt_recs = self.content_model.recommend(
                    user_id=user_id,
                    n=n_candidates_per_source,
                    exclude_known=exclude_known,
                )
                content_cands = [r.item_id for r in cnt_recs]
            except Exception as e:
                logger.debug("Content candidate retrieval error for user %d: %s", user_id, e)

        # 3. Popularity candidates (always reliable fallback)
        if self.popularity_model is not None and self.popularity_model.is_fitted:
            try:
                pop_recs = self.popularity_model.recommend(
                    user_id=user_id,
                    n=n_candidates_per_source,
                    exclude_known=exclude_known,
                    known_items=np.array(list(known_items)) if known_items else None,
                )
                pop_cands = [r.item_id for r in pop_recs]
            except Exception as e:
                logger.debug("Popularity candidate retrieval error for user %d: %s", user_id, e)

        # Merge candidate pools
        merged = CandidatePool.merge(
            cf_cands,
            content_cands,
            pop_cands,
            max_candidates=self.candidate_pool_size,
        )

        # Exclude known items
        if exclude_known and len(known_items) > 0:
            merged = PostFilter.filter_known(merged, known_items)

        return merged

    def recommend(
        self,
        user_id: int,
        n: int = DEFAULT_TOP_K,
        exclude_known: bool = True,
        known_items: np.ndarray | set[int] | None = None,
        train_interactions: csr_matrix | None = None,
        **kwargs,
    ) -> list[Recommendation]:
        """Generate personalized hybrid recommendations.

        Args:
            user_id: Internal user ID.
            n: Number of recommendations to return.
            exclude_known: Whether to filter out known/listened items.
            known_items: Optional set or array of items already known.
            train_interactions: Optional interaction matrix for looking up known items.

        Returns:
            List of Recommendation objects ordered by hybrid score descending.
        """
        self._check_fitted()

        # Identify known items
        known_set = set()
        if known_items is not None:
            known_set = set(known_items if isinstance(known_items, set) else known_items.tolist())
        elif exclude_known:
            interactions = train_interactions or self._train_interactions
            if interactions is not None and user_id < interactions.shape[0]:
                known_set = set(interactions[user_id].indices.tolist())

        # Retrieve candidate pool
        per_source_n = max(n * 4, min(self.candidate_pool_size // 2, 50))
        candidates = self._retrieve_candidates(
            user_id=user_id,
            n_candidates_per_source=per_source_n,
            exclude_known=exclude_known,
            known_items=known_set,
        )

        if len(candidates) == 0:
            # Ultimate fallback if candidate retrieval yielded nothing
            if self.popularity_model is not None:
                return self.popularity_model.recommend(user_id=user_id, n=n, exclude_known=exclude_known)
            return []

        # Score candidates across each sub-model
        n_cands = len(candidates)
        cf_raw = np.zeros(n_cands, dtype=np.float32)
        cnt_raw = np.zeros(n_cands, dtype=np.float32)
        pop_raw = np.zeros(n_cands, dtype=np.float32)

        has_cf = False
        if self.cf_model is not None and self.cf_model.is_fitted:
            try:
                cf_raw = self.cf_model.get_scores(user_id, candidates)
                has_cf = bool(cf_raw.max() > cf_raw.min()) if len(cf_raw) > 0 else False
            except Exception:
                has_cf = False

        has_cnt = False
        if self.content_model is not None and self.content_model.is_fitted:
            try:
                cnt_raw = self.content_model.get_scores(user_id, candidates)
                has_cnt = bool(cnt_raw.max() > 0.0) if len(cnt_raw) > 0 else False
            except Exception:
                has_cnt = False

        has_pop = False
        if self.popularity_model is not None and self.popularity_model.is_fitted:
            try:
                pop_raw = self.popularity_model.get_scores(user_id, candidates)
                has_pop = True
            except Exception:
                has_pop = False

        # Normalize scores to [0, 1]
        if self.normalization == "rank_percentile":
            norm_cf = ScoreNormalizer.rank_percentile(cf_raw)
            norm_cnt = ScoreNormalizer.rank_percentile(cnt_raw)
            norm_pop = ScoreNormalizer.rank_percentile(pop_raw)
        else:
            norm_cf = ScoreNormalizer.min_max(cf_raw)
            norm_cnt = ScoreNormalizer.min_max(cnt_raw)
            norm_pop = ScoreNormalizer.min_max(pop_raw)

        # Dynamic weight adjustment for cold-start safety
        active_weights = self._get_active_weights(
            user_id=user_id,
            has_cf_signal=has_cf,
            has_content_signal=has_cnt,
        )

        w_cf = active_weights["collaborative"]
        w_cnt = active_weights["content"]
        w_pop = active_weights["popularity"]

        hybrid_scores = (w_cf * norm_cf) + (w_cnt * norm_cnt) + (w_pop * norm_pop)

        # Sort descending
        ranked_order = np.argsort(hybrid_scores)[::-1][:n]

        # Resolve idx2artist
        idx2artist = self._idx2artist
        if idx2artist is None and self.popularity_model is not None:
            idx2artist = getattr(self.popularity_model, "_idx2artist", None)

        recommendations = []
        for rank_idx in ranked_order:
            item_id = int(candidates[rank_idx])
            score = float(hybrid_scores[rank_idx])
            name = idx2artist[item_id] if idx2artist is not None and item_id < len(idx2artist) else f"artist_{item_id}"

            # Component score breakdown
            meta = {
                "cf_score": float(norm_cf[rank_idx]),
                "content_score": float(norm_cnt[rank_idx]),
                "popularity_score": float(norm_pop[rank_idx]),
                "active_weights": active_weights,
            }

            reasons = self.explain(
                user_id=user_id,
                item_id=item_id,
                cf_contrib=float(w_cf * norm_cf[rank_idx]),
                cnt_contrib=float(w_cnt * norm_cnt[rank_idx]),
                pop_contrib=float(w_pop * norm_pop[rank_idx]),
            )

            recommendations.append(Recommendation(
                item_id=item_id,
                item_name=name,
                score=score,
                reasons=reasons,
                metadata=meta,
            ))

        return recommendations

    def explain(
        self,
        user_id: int,
        item_id: int,
        cf_contrib: float = 0.0,
        cnt_contrib: float = 0.0,
        pop_contrib: float = 0.0,
    ) -> list[str]:
        """Synthesize composite explanations from top contributing signals.

        Args:
            user_id: Internal user ID.
            item_id: Internal item ID.
            cf_contrib: Weighted CF score contribution.
            cnt_contrib: Weighted Content score contribution.
            pop_contrib: Weighted Popularity score contribution.

        Returns:
            List of explanation statements.
        """
        reasons = []

        # 1. Content explanation (tag similarity)
        if cnt_contrib > 0.1 and self.content_model is not None:
            try:
                cnt_reasons = self.content_model.explain(user_id, item_id)
                if cnt_reasons:
                    reasons.append(cnt_reasons[0])
            except Exception:
                pass

        # 2. CF explanation (collaborative pattern)
        if cf_contrib > 0.1 and self.cf_model is not None:
            try:
                cf_reasons = self.cf_model.explain(user_id, item_id)
                for r in cf_reasons:
                    if r not in reasons:
                        reasons.append(r)
                        break
            except Exception:
                pass

        # 3. Popularity explanation (if primary or strong driver)
        if not reasons and self.popularity_model is not None:
            try:
                pop_reasons = self.popularity_model.explain(user_id, item_id)
                reasons.extend(pop_reasons)
            except Exception:
                pass

        if not reasons:
            reasons.append("Recommended based on personalized hybrid match")

        return reasons

    def get_scores(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        """Batch score candidate items using weighted hybrid signals.

        Args:
            user_id: Internal user ID.
            item_ids: 1D array of item IDs to score.

        Returns:
            1D array of hybrid scores in [0, 1].
        """
        self._check_fitted()
        n_items = len(item_ids)
        if n_items == 0:
            return np.array([], dtype=np.float32)

        cf_raw = np.zeros(n_items, dtype=np.float32)
        cnt_raw = np.zeros(n_items, dtype=np.float32)
        pop_raw = np.zeros(n_items, dtype=np.float32)

        has_cf = False
        if self.cf_model is not None and self.cf_model.is_fitted:
            try:
                cf_raw = self.cf_model.get_scores(user_id, item_ids)
                has_cf = bool(cf_raw.max() > cf_raw.min()) if len(cf_raw) > 0 else False
            except Exception:
                has_cf = False

        has_cnt = False
        if self.content_model is not None and self.content_model.is_fitted:
            try:
                cnt_raw = self.content_model.get_scores(user_id, item_ids)
                has_cnt = bool(cnt_raw.max() > 0.0) if len(cnt_raw) > 0 else False
            except Exception:
                has_cnt = False

        if self.popularity_model is not None and self.popularity_model.is_fitted:
            try:
                pop_raw = self.popularity_model.get_scores(user_id, item_ids)
            except Exception:
                pass

        norm_cf = ScoreNormalizer.min_max(cf_raw)
        norm_cnt = ScoreNormalizer.min_max(cnt_raw)
        norm_pop = ScoreNormalizer.min_max(pop_raw)

        weights = self._get_active_weights(user_id, has_cf, has_cnt)
        hybrid_scores = (
            weights["collaborative"] * norm_cf
            + weights["content"] * norm_cnt
            + weights["popularity"] * norm_pop
        )

        return hybrid_scores.astype(np.float32)

    def save(self, path: Path | None = None) -> None:
        """Save hybrid configuration and tuned weights to JSON."""
        path = path or HYBRID_WEIGHTS
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "weights": self.weights,
            "candidate_pool_size": self.candidate_pool_size,
            "normalization": self.normalization,
            "cf_model_name": self.cf_model.name if self.cf_model else None,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved hybrid weights to %s.", path.name)

    def load(self, path: Path | None = None) -> "HybridRecommender":
        """Load tuned weights and configuration from JSON."""
        path = path or HYBRID_WEIGHTS
        if not path.exists():
            logger.warning("Hybrid weights file not found at %s. Keeping current weights.", path)
            return self

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "weights" in data:
            self.weights = self._normalize_weights(data["weights"])
        if "candidate_pool_size" in data:
            self.candidate_pool_size = int(data["candidate_pool_size"])
        if "normalization" in data:
            self.normalization = str(data["normalization"])

        logger.info("Loaded hybrid weights: %s from %s.", self.weights, path.name)
        return self

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "HybridRecommender is not fitted. Connect fitted sub-models first."
            )
