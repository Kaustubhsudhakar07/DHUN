"""
Recommendation Explainability Engine — Phase 10

Generates transparent, multi-perspective explanations for recommendations:
- Anchor artist attribution (which listened artists most influenced the recommendation)
- Shared musical tag / genre overlaps
- Signal breakdown (collaborative vs. content vs. popularity contribution percentages)
- Natural language explanation summaries
- Live artist biography and imagery via Last.fm API
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from src.features.tfidf import get_top_tags
from src.lastfm.client import LastFMClient

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    """Structured explanation for a recommended artist."""
    item_id: int
    item_name: str
    summary: str
    anchor_artists: list[dict[str, Any]] = field(default_factory=list)
    shared_tags: list[str] = field(default_factory=list)
    signal_breakdown: dict[str, float] = field(default_factory=dict)
    bio_snippet: str = ""
    image_url: str = ""
    lastfm_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert explanation to dictionary for API responses."""
        return {
            "item_id": self.item_id,
            "item_name": self.item_name,
            "summary": self.summary,
            "anchor_artists": self.anchor_artists,
            "shared_tags": self.shared_tags,
            "signal_breakdown": self.signal_breakdown,
            "bio_snippet": self.bio_snippet,
            "image_url": self.image_url,
            "lastfm_url": self.lastfm_url,
        }


class ExplanationEngine:
    """Engine for decomposing recommendation scores and generating transparent explanations."""

    def __init__(
        self,
        idx2artist: list[str] | None = None,
        tfidf_matrix: csr_matrix | None = None,
        vectorizer: Any = None,
        train_interactions: csr_matrix | None = None,
        lastfm_client: LastFMClient | None = None,
    ):
        """Initialize ExplanationEngine.

        Args:
            idx2artist: List mapping internal index -> artist name.
            tfidf_matrix: Sparse TF-IDF matrix [n_artists × n_features].
            vectorizer: Fitted TfidfVectorizer for tag lookup.
            train_interactions: Sparse CSR matrix of user listening history.
            lastfm_client: Optional LastFMClient for bio/image enrichment.
        """
        self.idx2artist = idx2artist
        self.tfidf_matrix = tfidf_matrix
        self.vectorizer = vectorizer
        self.train_interactions = train_interactions
        self.lastfm = lastfm_client or LastFMClient()

    def _get_shared_tags(self, item_id: int, user_artist_indices: Sequence[int]) -> list[str]:
        """Find overlapping tags between recommended artist and user's listened artists."""
        if self.tfidf_matrix is None or self.vectorizer is None:
            return []

        item_tags = set(t for t, _ in get_top_tags(self.vectorizer, self.tfidf_matrix, item_id, n=10))
        if not item_tags:
            return []

        user_tags = set()
        for aid in user_artist_indices:
            for tag, _ in get_top_tags(self.vectorizer, self.tfidf_matrix, aid, n=5):
                user_tags.add(tag)

        shared = sorted(list(item_tags & user_tags))
        return shared[:5]

    def _find_anchor_artists(
        self,
        user_id: int,
        item_id: int,
        top_k: int = 2,
    ) -> list[dict[str, Any]]:
        """Find the user's listened artists most similar to the recommended item."""
        if self.train_interactions is None or user_id >= self.train_interactions.shape[0]:
            return []

        user_row = self.train_interactions[user_id]
        if user_row.nnz == 0:
            return []

        listened_indices = user_row.indices
        play_counts = user_row.data

        anchors = []

        if self.tfidf_matrix is not None and item_id < self.tfidf_matrix.shape[0]:
            # Compute cosine similarity between candidate and user's listened items
            item_vec = self.tfidf_matrix[item_id]
            listened_vecs = self.tfidf_matrix[listened_indices]
            sims = (listened_vecs @ item_vec.T).toarray().ravel()

            # Weight similarity by log play count
            weights = np.log1p(play_counts)
            weighted_scores = sims * weights

            top_indices = np.argsort(weighted_scores)[::-1][:top_k]
            for idx in top_indices:
                anchor_idx = int(listened_indices[idx])
                name = self.idx2artist[anchor_idx] if self.idx2artist else f"artist_{anchor_idx}"
                shared = self._get_shared_tags(item_id, [anchor_idx])
                anchors.append({
                    "artist_id": anchor_idx,
                    "artist_name": name,
                    "play_count": int(play_counts[idx]),
                    "shared_tags": shared,
                })
        else:
            # Fallback: take user's most listened artists
            top_played = np.argsort(play_counts)[::-1][:top_k]
            for idx in top_played:
                anchor_idx = int(listened_indices[idx])
                name = self.idx2artist[anchor_idx] if self.idx2artist else f"artist_{anchor_idx}"
                anchors.append({
                    "artist_id": anchor_idx,
                    "artist_name": name,
                    "play_count": int(play_counts[idx]),
                    "shared_tags": [],
                })

        return anchors

    def explain(
        self,
        user_id: int,
        item_id: int,
        metadata: dict[str, Any] | None = None,
        fetch_online_metadata: bool = True,
    ) -> Explanation:
        """Generate a complete, structured explanation for a recommendation.

        Args:
            user_id: Internal user ID.
            item_id: Recommended item ID.
            metadata: Metadata dict attached to the Recommendation (containing component scores).
            fetch_online_metadata: Whether to fetch bio/image from Last.fm API.

        Returns:
            Explanation object.
        """
        artist_name = (
            self.idx2artist[item_id]
            if self.idx2artist and item_id < len(self.idx2artist)
            else f"artist_{item_id}"
        )

        # 1. Compute signal breakdown percentages
        signal_breakdown = {"collaborative": 40.0, "content": 30.0, "popularity": 30.0}
        if metadata:
            active_weights = metadata.get("active_weights", {})
            w_cf = active_weights.get("collaborative", 0.4)
            w_cnt = active_weights.get("content", 0.3)
            w_pop = active_weights.get("popularity", 0.3)

            s_cf = metadata.get("cf_score", 0.5)
            s_cnt = metadata.get("content_score", 0.5)
            s_pop = metadata.get("popularity_score", 0.5)

            contrib_cf = w_cf * s_cf
            contrib_cnt = w_cnt * s_cnt
            contrib_pop = w_pop * s_pop
            total_contrib = contrib_cf + contrib_cnt + contrib_pop

            if total_contrib > 0:
                signal_breakdown = {
                    "collaborative": round((contrib_cf / total_contrib) * 100, 1),
                    "content": round((contrib_cnt / total_contrib) * 100, 1),
                    "popularity": round((contrib_pop / total_contrib) * 100, 1),
                }

        # 2. Find anchor artists
        anchors = self._find_anchor_artists(user_id, item_id, top_k=2)

        # 3. Find shared tags
        shared_tags = []
        if self.train_interactions is not None and user_id < self.train_interactions.shape[0]:
            user_indices = self.train_interactions[user_id].indices
            shared_tags = self._get_shared_tags(item_id, user_indices)

        # 4. Construct natural language summary
        anchor_names = [a["artist_name"].title() for a in anchors if a.get("artist_name")]
        if anchor_names and shared_tags:
            tag_str = ", ".join(t.replace("_", " ") for t in shared_tags[:3])
            summary = (
                f"Recommended because you listen to {', '.join(anchor_names)}. "
                f"They share musical styles including {tag_str}."
            )
        elif anchor_names:
            summary = (
                f"Recommended based on your listening history with {', '.join(anchor_names)} "
                f"and listeners with similar taste."
            )
        elif shared_tags:
            tag_str = ", ".join(t.replace("_", " ") for t in shared_tags[:3])
            summary = f"Recommended based on matching genres: {tag_str}."
        else:
            summary = "Recommended based on community listening trends and taste match."

        # 5. Last.fm artist metadata
        bio_snippet = ""
        image_url = ""
        lastfm_url = f"https://www.last.fm/music/{artist_name.replace(' ', '+')}"

        if fetch_online_metadata and self.lastfm:
            try:
                info = self.lastfm.get_artist_info(artist_name)
                bio_snippet = info.get("bio", "")
                image_url = info.get("image_url", "")
                lastfm_url = info.get("url", lastfm_url)
            except Exception as e:
                logger.debug("Could not fetch Last.fm metadata for %s: %s", artist_name, e)

        return Explanation(
            item_id=item_id,
            item_name=artist_name,
            summary=summary,
            anchor_artists=anchors,
            shared_tags=shared_tags,
            signal_breakdown=signal_breakdown,
            bio_snippet=bio_snippet,
            image_url=image_url,
            lastfm_url=lastfm_url,
        )
