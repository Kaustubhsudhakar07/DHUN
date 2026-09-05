"""
Beyond-Accuracy Metrics — Phase 8

Metrics that evaluate recommendation quality beyond pure relevance:
- Catalog Coverage: what fraction of the catalog gets recommended
- Diversity: how different the recommendations are within each list
- Novelty: tendency to recommend less popular (long-tail) items
- Personalization: how different recommendations are across users

These metrics are crucial for a music recommender because:
1. Users want to discover NEW music, not just popular hits.
2. A system that only recommends the top 100 artists to everyone
   is useless even if those artists are technically "relevant."
3. Diversity within a single recommendation list prevents monotony.

Reference: Kaminskas & Bridge (2016), "Diversity, Serendipity, Novelty,
and Coverage: A Survey and Empirical Analysis of Beyond-Accuracy Objectives
in Recommender Systems."
"""

import logging
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from src.recommenders.base import BaseRecommender

logger = logging.getLogger(__name__)


def catalog_coverage(
    all_recommendations: list[list[int]],
    n_items: int,
) -> float:
    """Fraction of the catalog that appears in at least one recommendation list.

    Args:
        all_recommendations: List of recommendation lists (each a list of item IDs).
        n_items: Total number of items in the catalog.

    Returns:
        Coverage ratio in [0, 1].
    """
    if n_items == 0:
        return 0.0
    recommended_items = set()
    for rec_list in all_recommendations:
        recommended_items.update(rec_list)
    return len(recommended_items) / n_items


def diversity(
    recommendations: list[int],
    item_features: csr_matrix | np.ndarray | None = None,
) -> float:
    """Average pairwise distance among items in a single recommendation list.

    Uses 1 - cosine_similarity as distance. If no feature matrix is provided,
    returns the number of unique items / total items as a simple diversity proxy.

    Args:
        recommendations: List of recommended item IDs.
        item_features: Optional feature matrix [n_items × n_features].

    Returns:
        Diversity score in [0, 1].
    """
    if len(recommendations) <= 1:
        return 0.0

    if item_features is None:
        # Fallback: uniqueness ratio
        return len(set(recommendations)) / len(recommendations)

    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    rec_indices = np.array(recommendations)
    if hasattr(item_features, "toarray"):
        vecs = item_features[rec_indices].toarray()
    else:
        vecs = item_features[rec_indices]

    # Pairwise cosine similarity
    sim_matrix = cos_sim(vecs)

    # Average off-diagonal similarity → convert to distance
    n = len(recommendations)
    if n <= 1:
        return 0.0

    # Sum upper triangle (excluding diagonal)
    upper_sum = (sim_matrix.sum() - np.trace(sim_matrix)) / 2
    n_pairs = n * (n - 1) / 2
    avg_similarity = upper_sum / n_pairs

    return float(1.0 - avg_similarity)


def novelty(
    recommendations: list[int],
    item_popularity: np.ndarray,
    n_users: int,
) -> float:
    """Average self-information (surprise) of recommended items.

    Novelty = (1/|L|) * Σ_{i ∈ L} log2(n_users / popularity(i))

    Higher novelty means less popular items are being recommended.

    Args:
        recommendations: List of recommended item IDs.
        item_popularity: Array where item_popularity[i] = number of users
            who interacted with item i.
        n_users: Total number of users.

    Returns:
        Novelty score (higher = more novel). Unbounded above 0.
    """
    if len(recommendations) == 0 or n_users == 0:
        return 0.0

    scores = []
    for item in recommendations:
        pop = item_popularity[item] if item < len(item_popularity) else 0
        if pop > 0:
            scores.append(np.log2(n_users / pop))
        else:
            scores.append(np.log2(n_users))  # Max novelty for unknown items

    return float(np.mean(scores))


def personalization(
    all_recommendations: list[list[int]],
) -> float:
    """1 - average Jaccard similarity between all pairs of user recommendation lists.

    High personalization means users get different recommendations.

    Args:
        all_recommendations: List of recommendation lists.

    Returns:
        Personalization score in [0, 1].
    """
    n = len(all_recommendations)
    if n <= 1:
        return 0.0

    # Convert to sets for fast intersection
    rec_sets = [set(r) for r in all_recommendations]

    # Sample pairs if too many users (O(n^2) is expensive)
    max_pairs = 10000
    total_pairs = n * (n - 1) // 2

    if total_pairs <= max_pairs:
        # Compute all pairs
        total_jaccard = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                intersection = len(rec_sets[i] & rec_sets[j])
                union = len(rec_sets[i] | rec_sets[j])
                if union > 0:
                    total_jaccard += intersection / union
                count += 1
        avg_jaccard = total_jaccard / count if count > 0 else 0.0
    else:
        # Random sampling
        rng = np.random.default_rng(42)
        indices = rng.integers(0, n, size=(max_pairs, 2))
        # Ensure i != j
        mask = indices[:, 0] != indices[:, 1]
        indices = indices[mask][:max_pairs]

        total_jaccard = 0.0
        for i, j in indices:
            intersection = len(rec_sets[i] & rec_sets[j])
            union = len(rec_sets[i] | rec_sets[j])
            if union > 0:
                total_jaccard += intersection / union
        avg_jaccard = total_jaccard / len(indices) if len(indices) > 0 else 0.0

    return float(1.0 - avg_jaccard)


def evaluate_beyond_accuracy(
    recommender: BaseRecommender,
    train_matrix: csr_matrix,
    k: int = 10,
    n_users: int | None = None,
    item_features: csr_matrix | np.ndarray | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Compute all beyond-accuracy metrics for a recommender.

    Args:
        recommender: A fitted BaseRecommender.
        train_matrix: Training interaction matrix [n_users × n_items].
        k: Number of recommendations per user.
        n_users: Limit evaluation to this many random users.
        item_features: Optional item feature matrix for diversity computation.
        seed: Random seed for user sampling.

    Returns:
        Dictionary with keys: coverage, diversity, novelty, personalization.
    """
    total_users = train_matrix.shape[0]
    n_items = train_matrix.shape[1]

    # Determine which users to evaluate
    user_ids = list(range(total_users))
    if n_users is not None and n_users < total_users:
        rng = np.random.default_rng(seed)
        user_ids = rng.choice(user_ids, size=n_users, replace=False).tolist()

    logger.info(
        "Evaluating beyond-accuracy metrics for %s on %d users (K=%d)...",
        recommender.name,
        len(user_ids),
        k,
    )

    # Compute item popularity from training data
    binary = train_matrix.copy()
    binary.data = np.ones_like(binary.data)
    item_popularity = np.asarray(binary.sum(axis=0)).ravel()

    # Generate recommendations for all users
    all_recs = []
    diversity_scores = []

    for i, uid in enumerate(user_ids):
        try:
            recs = recommender.recommend(
                user_id=uid,
                n=k,
                exclude_known=True,
                train_interactions=train_matrix,
                known_items=train_matrix[uid].indices,
            )
            rec_ids = [r.item_id for r in recs]
        except Exception:
            rec_ids = []

        all_recs.append(rec_ids)

        if len(rec_ids) > 1:
            div = diversity(rec_ids, item_features)
            diversity_scores.append(div)

        if (i + 1) % 10000 == 0:
            logger.info("  Generated recommendations for %d/%d users...", i + 1, len(user_ids))

    # Compute aggregate metrics
    cov = catalog_coverage(all_recs, n_items)
    avg_div = float(np.mean(diversity_scores)) if diversity_scores else 0.0

    novelty_scores = [
        novelty(rec_list, item_popularity, total_users)
        for rec_list in all_recs
        if len(rec_list) > 0
    ]
    avg_novelty = float(np.mean(novelty_scores)) if novelty_scores else 0.0

    pers = personalization(all_recs)

    results = {
        "coverage": cov,
        "diversity": avg_div,
        "novelty": avg_novelty,
        "personalization": pers,
    }

    logger.info(
        "Beyond-accuracy results for %s:\n"
        "  Coverage:        %.4f (%.1f%% of catalog)\n"
        "  Diversity:       %.4f\n"
        "  Novelty:         %.4f\n"
        "  Personalization: %.4f",
        recommender.name,
        cov,
        cov * 100,
        avg_div,
        avg_novelty,
        pers,
    )

    return results
