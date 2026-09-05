"""
Abstract base class for all recommenders.

Every recommender in this project (popularity, content-based, collaborative,
hybrid) inherits from BaseRecommender. This enforces a consistent interface
for training, predicting, and explaining recommendations.

Design rationale:
- Uniform API makes it straightforward to swap models in evaluation.
- The `explain` method ensures every recommender can produce structured
  explanations, which is a core project requirement.
- Type hints document expected input/output shapes without runtime cost.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Recommendation:
    """A single recommendation with metadata and explanation.

    Attributes:
        item_id: Internal integer ID of the recommended item.
        item_name: Human-readable name (artist name or track title).
        score: Recommendation score (higher = more relevant).
        reasons: List of human-readable explanation strings.
        metadata: Optional dictionary for additional info (genre, tags, etc.).
    """

    item_id: int
    item_name: str
    score: float
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for API responses."""
        return {
            "item_id": self.item_id,
            "item_name": self.item_name,
            "score": round(self.score, 4),
            "reasons": self.reasons,
            "metadata": self.metadata,
        }


class BaseRecommender(ABC):
    """Abstract base class for all recommendation models.

    Subclasses must implement:
        - fit(): Train or build the model.
        - recommend(): Generate Top-K recommendations for a user.
        - explain(): Provide explanation for why an item was recommended.

    Optional override:
        - get_similar_items(): Find items similar to a given item.
    """

    def __init__(self, name: str):
        """Initialize the recommender.

        Args:
            name: Human-readable name for this recommender (used in logs
                  and evaluation tables).
        """
        self.name = name
        self._is_fitted = False

    @abstractmethod
    def fit(self, **kwargs) -> "BaseRecommender":
        """Train or build the recommendation model.

        Args:
            **kwargs: Model-specific training data and parameters.

        Returns:
            self, for method chaining.
        """

    @abstractmethod
    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_known: bool = True,
        **kwargs,
    ) -> list[Recommendation]:
        """Generate Top-N recommendations for a user.

        Args:
            user_id: Internal user ID.
            n: Number of recommendations to return.
            exclude_known: If True, exclude items the user has already
                interacted with.
            **kwargs: Additional parameters (e.g., candidate pool).

        Returns:
            List of Recommendation objects, sorted by score descending.
        """

    @abstractmethod
    def explain(self, user_id: int, item_id: int) -> list[str]:
        """Explain why an item was recommended to a user.

        Args:
            user_id: Internal user ID.
            item_id: Internal item ID.

        Returns:
            List of human-readable explanation strings.
        """

    def get_similar_items(
        self, item_id: int, n: int = 10
    ) -> list[Recommendation]:
        """Find items similar to a given item.

        Default implementation raises NotImplementedError.
        Subclasses should override if they support item similarity.

        Args:
            item_id: Internal item ID.
            n: Number of similar items to return.

        Returns:
            List of Recommendation objects.
        """
        raise NotImplementedError(
            f"{self.name} does not support item similarity queries."
        )

    def get_scores(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        """Get raw scores for a set of candidate items.

        Used by the hybrid recommender to combine scores from multiple models.
        Default implementation calls recommend() and filters, but subclasses
        should override with a more efficient implementation.

        Args:
            user_id: Internal user ID.
            item_ids: Array of internal item IDs to score.

        Returns:
            Array of scores, same length as item_ids.
        """
        raise NotImplementedError(
            f"{self.name} does not support batch scoring. "
            "Override get_scores() for hybrid recommendation."
        )

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been trained."""
        return self._is_fitted

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self._is_fitted})"
