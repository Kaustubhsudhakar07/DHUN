"""
Content-Based Recommender — Phase 7

Recommends artists similar to a user's listening history based on TF-IDF
tag similarity. The user profile is a weighted average of TF-IDF vectors
for all artists they've listened to, weighted by log1p(play_count).

This recommender is especially valuable for:
- Cold-start collaborative filtering (new users with some tag-matching artists)
- Explaining recommendations via shared tags
- Providing diversity in hybrid combinations (different signal from CF)
"""

import logging
from typing import Any, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.recommenders.base import BaseRecommender, Recommendation

logger = logging.getLogger(__name__)


class ContentBasedRecommender(BaseRecommender):
    """Recommend artists based on TF-IDF tag similarity to user profile.

    The user profile vector is the weighted centroid of TF-IDF vectors
    for the user's listened artists. Candidates are scored by cosine
    similarity between the user profile and each candidate's TF-IDF vector.
    """

    def __init__(self):
        super().__init__(name="ContentBased")
        self._tfidf_matrix: csr_matrix | None = None
        self._train_interactions: csr_matrix | None = None
        self._idx2artist: list[str] | None = None
        self._vectorizer: Any = None  # TfidfVectorizer, for tag name lookup

    def fit(
        self,
        tfidf_matrix: csr_matrix,
        train_interactions: csr_matrix,
        idx2artist: list[str] | None = None,
        vectorizer: Any = None,
        **kwargs,
    ) -> "ContentBasedRecommender":
        """Store the TF-IDF matrix and training interactions.

        No model training needed — scoring is done at recommend time.

        Args:
            tfidf_matrix: Sparse TF-IDF matrix [n_artists × n_features].
            train_interactions: Sparse CSR matrix [n_users × n_artists].
            idx2artist: Optional list mapping artist index → name.
            vectorizer: Optional fitted TfidfVectorizer for tag name lookup.

        Returns:
            self
        """
        logger.info("Fitting ContentBasedRecommender...")

        self._tfidf_matrix = tfidf_matrix
        self._train_interactions = train_interactions
        self._idx2artist = idx2artist
        self._vectorizer = vectorizer
        self._is_fitted = True

        # Stats
        n_artists_with_features = (tfidf_matrix.getnnz(axis=1) > 0).sum()
        logger.info(
            "ContentBasedRecommender fitted: %d artists, %d with tag features (%.1f%%).",
            tfidf_matrix.shape[0],
            n_artists_with_features,
            (n_artists_with_features / tfidf_matrix.shape[0]) * 100,
        )
        return self

    def _build_user_profile(
        self, user_id: int
    ) -> np.ndarray:
        """Construct a user's TF-IDF profile vector.

        Profile = weighted average of the user's listened artists' TF-IDF
        vectors, where weights = log1p(play_count).

        Args:
            user_id: Internal user ID.

        Returns:
            Dense 1D array [n_features] representing the user's content profile.
        """
        user_row = self._train_interactions[user_id]
        item_indices = user_row.indices
        play_counts = user_row.data

        if len(item_indices) == 0:
            return np.zeros(self._tfidf_matrix.shape[1], dtype=np.float32)

        # Weights from log1p(play_count)
        weights = np.log1p(play_counts).astype(np.float32)
        weight_sum = weights.sum()
        if weight_sum == 0:
            return np.zeros(self._tfidf_matrix.shape[1], dtype=np.float32)

        # Weighted sum of TF-IDF vectors
        item_vectors = self._tfidf_matrix[item_indices].toarray()  # [k × n_features]
        profile = (item_vectors * weights[:, np.newaxis]).sum(axis=0) / weight_sum

        return profile.astype(np.float32)

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_known: bool = True,
        **kwargs,
    ) -> list[Recommendation]:
        """Generate top-N content-based recommendations for a user.

        Args:
            user_id: Internal user ID.
            n: Number of recommendations.
            exclude_known: Whether to exclude already-interacted items.

        Returns:
            List of Recommendation objects sorted by similarity descending.
        """
        self._check_fitted()

        # Build user profile
        profile = self._build_user_profile(user_id)

        # If user has no content profile, return empty
        if profile.sum() == 0:
            logger.debug("User %d has no content profile (no tagged artists).", user_id)
            return []

        # Score all artists
        scores = cosine_similarity(
            profile.reshape(1, -1), self._tfidf_matrix
        ).ravel()

        # Exclude known items
        if exclude_known:
            known = self._train_interactions[user_id].indices
            scores[known] = -1.0

        # Get top-N
        top_indices = np.argsort(scores)[::-1][:n]

        recs = []
        for idx in top_indices:
            idx_int = int(idx)
            score = float(scores[idx_int])
            if score <= 0:
                break

            name = (
                self._idx2artist[idx_int]
                if self._idx2artist is not None
                else f"artist_{idx_int}"
            )

            recs.append(Recommendation(
                item_id=idx_int,
                item_name=name,
                score=score,
                reasons=self.explain(user_id, idx_int),
            ))

        return recs

    def explain(self, user_id: int, item_id: int) -> list[str]:
        """Explain recommendation based on shared tags.

        Args:
            user_id: Internal user ID.
            item_id: Internal item ID.

        Returns:
            List of explanation strings referencing shared tags and
            similar artists from the user's history.
        """
        reasons = []

        if self._tfidf_matrix is None or self._train_interactions is None:
            return ["Recommended based on tag similarity"]

        # Find which listened artists are most similar to this item
        user_row = self._train_interactions[user_id]
        listened_indices = user_row.indices
        listened_plays = user_row.data

        if len(listened_indices) == 0:
            return ["Recommended based on tag similarity"]

        # Compute similarity between this item and each listened artist
        item_vec = self._tfidf_matrix[item_id]
        listened_vecs = self._tfidf_matrix[listened_indices]
        sims = cosine_similarity(item_vec, listened_vecs).ravel()

        # Top 3 most similar listened artists
        top_k = min(3, len(sims))
        top_sim_idx = np.argsort(sims)[::-1][:top_k]

        similar_artists = []
        for si in top_sim_idx:
            if sims[si] > 0.1:  # Only include meaningfully similar
                artist_idx = listened_indices[si]
                artist_name = (
                    self._idx2artist[artist_idx]
                    if self._idx2artist is not None
                    else f"artist_{artist_idx}"
                )
                similar_artists.append(artist_name)

        if similar_artists:
            reasons.append(
                f"Similar to artists you listen to: {', '.join(similar_artists)}"
            )

        # Get top tags for this item (if vectorizer available)
        if self._vectorizer is not None:
            feature_names = self._vectorizer.get_feature_names_out()
            item_row = self._tfidf_matrix[item_id].toarray().ravel()
            top_tag_idx = np.argsort(item_row)[::-1][:5]
            tags = [
                feature_names[i].replace("_", " ")
                for i in top_tag_idx
                if item_row[i] > 0
            ]
            if tags:
                reasons.append(f"Tags: {', '.join(tags)}")

        if not reasons:
            reasons.append("Recommended based on tag similarity")

        return reasons

    def get_scores(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        """Score a batch of candidate items for a user.

        Args:
            user_id: Internal user ID.
            item_ids: Array of item indices to score.

        Returns:
            Array of cosine similarity scores.
        """
        self._check_fitted()

        profile = self._build_user_profile(user_id)
        if profile.sum() == 0:
            return np.zeros(len(item_ids), dtype=np.float32)

        candidate_vecs = self._tfidf_matrix[item_ids]
        scores = cosine_similarity(
            profile.reshape(1, -1), candidate_vecs
        ).ravel()

        return scores.astype(np.float32)

    def get_similar_items(
        self, item_id: int, n: int = 10
    ) -> list[Recommendation]:
        """Find items with similar TF-IDF tag profiles.

        Args:
            item_id: Internal item ID.
            n: Number of similar items to return.

        Returns:
            List of Recommendation objects.
        """
        self._check_fitted()

        item_vec = self._tfidf_matrix[item_id]
        sims = cosine_similarity(item_vec, self._tfidf_matrix).ravel()
        sims[item_id] = -1.0  # Exclude self

        top_indices = np.argsort(sims)[::-1][:n]

        recs = []
        for idx in top_indices:
            idx_int = int(idx)
            score = float(sims[idx_int])
            if score <= 0:
                break
            name = (
                self._idx2artist[idx_int]
                if self._idx2artist is not None
                else f"artist_{idx_int}"
            )
            recs.append(Recommendation(
                item_id=idx_int,
                item_name=name,
                score=score,
            ))

        return recs

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "ContentBasedRecommender is not fitted. Call fit() first."
            )
