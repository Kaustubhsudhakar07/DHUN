"""
Ranking Metrics for Recommendation Evaluation — Phase 8

Standard information retrieval and recommendation metrics for implicit
feedback evaluation. All metrics operate on ranked recommendation lists
evaluated against held-out ground truth items.

Metrics implemented:
- Precision@K: fraction of top-K items that are relevant
- Recall@K: fraction of relevant items captured in top-K
- Hit Rate@K: binary — did any relevant item appear in top-K?
- MAP@K: Mean Average Precision at K
- NDCG@K: Normalized Discounted Cumulative Gain at K

All per-user metric functions accept:
    recommended: list/array of recommended item IDs (ordered by rank)
    relevant: set/list of ground truth relevant item IDs
    k: cutoff
"""

import logging
import time
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from src.recommenders.base import BaseRecommender

logger = logging.getLogger(__name__)


# =========================================================================
# PER-USER METRICS
# =========================================================================

def precision_at_k(
    recommended: list[int] | np.ndarray,
    relevant: set[int],
    k: int,
) -> float:
    """Precision@K: fraction of top-K recommendations that are relevant.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of ground truth relevant item IDs.
        k: Cutoff position.

    Returns:
        Precision score in [0, 1].
    """
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(
    recommended: list[int] | np.ndarray,
    relevant: set[int],
    k: int,
) -> float:
    """Recall@K: fraction of relevant items captured in top-K.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of ground truth relevant item IDs.
        k: Cutoff position.

    Returns:
        Recall score in [0, 1].
    """
    if len(relevant) == 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def hit_rate_at_k(
    recommended: list[int] | np.ndarray,
    relevant: set[int],
    k: int,
) -> float:
    """Hit Rate@K: 1 if any relevant item in top-K, else 0.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of ground truth relevant item IDs.
        k: Cutoff position.

    Returns:
        1.0 or 0.0.
    """
    top_k = recommended[:k]
    return 1.0 if any(item in relevant for item in top_k) else 0.0


def average_precision_at_k(
    recommended: list[int] | np.ndarray,
    relevant: set[int],
    k: int,
) -> float:
    """Average Precision@K for a single user.

    AP@K = (1/min(|relevant|,K)) * Σ_{i=1}^{K} rel(i) * Precision@i

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of ground truth relevant item IDs.
        k: Cutoff position.

    Returns:
        AP score in [0, 1].
    """
    if len(relevant) == 0:
        return 0.0

    top_k = recommended[:k]
    score = 0.0
    hits = 0

    for i, item in enumerate(top_k):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)

    return score / min(len(relevant), k)


def ndcg_at_k(
    recommended: list[int] | np.ndarray,
    relevant: set[int],
    k: int,
) -> float:
    """NDCG@K: Normalized Discounted Cumulative Gain.

    For binary relevance (implicit feedback):
        DCG@K = Σ_{i=1}^{K} rel(i) / log2(i+1)
        IDCG@K = Σ_{i=1}^{min(|relevant|,K)} 1 / log2(i+1)

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of ground truth relevant item IDs.
        k: Cutoff position.

    Returns:
        NDCG score in [0, 1].
    """
    if len(relevant) == 0:
        return 0.0

    top_k = recommended[:k]

    # DCG
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # +2 because i is 0-indexed

    # IDCG (ideal: all relevant items at the top)
    n_relevant = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))

    if idcg == 0:
        return 0.0

    return dcg / idcg


# =========================================================================
# FULL MODEL EVALUATION
# =========================================================================

def evaluate_model(
    recommender: BaseRecommender,
    test_matrix: csr_matrix,
    train_matrix: csr_matrix,
    k_values: list[int] | None = None,
    n_users: int | None = None,
    seed: int = 42,
) -> dict[str, dict[int, float]]:
    """Evaluate a recommender across all test users.

    For each user with at least one held-out item in the test set:
    1. Generate top-max(K) recommendations (excluding training items).
    2. Compute all metrics at each K value.
    3. Average across users.

    Args:
        recommender: A fitted BaseRecommender subclass.
        test_matrix: Sparse CSR test matrix [n_users × n_items].
        train_matrix: Sparse CSR training matrix [n_users × n_items]
            (used for known-item exclusion).
        k_values: List of K cutoffs to evaluate. Defaults to [5, 10, 20, 50].
        n_users: If set, evaluate on a random sample of this many users
            (for speed during development).
        seed: Random seed for user sampling.

    Returns:
        Dictionary mapping metric_name → {k: avg_score}.
        Example: {"precision": {5: 0.02, 10: 0.015}, ...}
    """
    if k_values is None:
        from src.config import EVAL_K_VALUES
        k_values = EVAL_K_VALUES

    max_k = max(k_values)

    # Identify test users (those with at least one held-out item)
    nnz_per_user = np.diff(test_matrix.indptr)
    test_users = np.where(nnz_per_user > 0)[0].tolist()

    if n_users is not None and n_users < len(test_users):
        rng = np.random.default_rng(seed)
        test_users = rng.choice(test_users, size=n_users, replace=False).tolist()

    logger.info(
        "Evaluating %s on %d users at K=%s...",
        recommender.name,
        len(test_users),
        k_values,
    )

    # Initialize accumulators
    metrics = {
        "precision": {k: 0.0 for k in k_values},
        "recall": {k: 0.0 for k in k_values},
        "hit_rate": {k: 0.0 for k in k_values},
        "map": {k: 0.0 for k in k_values},
        "ndcg": {k: 0.0 for k in k_values},
    }

    start = time.time()
    evaluated = 0

    for i, user_id in enumerate(test_users):
        # Ground truth
        relevant = set(test_matrix[user_id].indices.tolist())
        if len(relevant) == 0:
            continue

        # Generate recommendations
        try:
            recs = recommender.recommend(
                user_id=user_id,
                n=max_k,
                exclude_known=True,
                train_interactions=train_matrix,
                known_items=train_matrix[user_id].indices,
            )
        except Exception as e:
            logger.warning("Error recommending for user %d: %s", user_id, e)
            continue

        recommended = [r.item_id for r in recs]

        # Compute metrics at each K
        for k in k_values:
            metrics["precision"][k] += precision_at_k(recommended, relevant, k)
            metrics["recall"][k] += recall_at_k(recommended, relevant, k)
            metrics["hit_rate"][k] += hit_rate_at_k(recommended, relevant, k)
            metrics["map"][k] += average_precision_at_k(recommended, relevant, k)
            metrics["ndcg"][k] += ndcg_at_k(recommended, relevant, k)

        evaluated += 1

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - start
            logger.info(
                "  Evaluated %d/%d users (%.1f users/sec)...",
                i + 1,
                len(test_users),
                (i + 1) / elapsed,
            )

    # Average across users
    if evaluated > 0:
        for metric in metrics:
            for k in k_values:
                metrics[metric][k] /= evaluated

    elapsed = time.time() - start
    logger.info(
        "Evaluation complete: %d users in %.1f seconds (%.1f users/sec).",
        evaluated,
        elapsed,
        evaluated / elapsed if elapsed > 0 else 0,
    )

    return metrics


def format_metrics_table(
    results: dict[str, dict[str, dict[int, float]]],
    k_values: list[int] | None = None,
) -> str:
    """Format evaluation results as a readable comparison table.

    Args:
        results: Dict mapping model_name → metrics dict from evaluate_model().
        k_values: K values to display. If None, uses all found.

    Returns:
        Formatted string table.
    """
    if not results:
        return "No results to display."

    # Determine K values from first model
    first_model = next(iter(results.values()))
    first_metric = next(iter(first_model.values()))
    if k_values is None:
        k_values = sorted(first_metric.keys())

    metric_names = ["precision", "recall", "hit_rate", "map", "ndcg"]
    model_names = list(results.keys())

    lines = []
    for metric in metric_names:
        lines.append(f"\n{'='*60}")
        lines.append(f"  {metric.upper()}")
        lines.append(f"{'='*60}")

        # Header
        header = f"{'K':>5}"
        for model in model_names:
            header += f"  {model:>15}"
        lines.append(header)
        lines.append("-" * len(header))

        for k in k_values:
            row = f"{k:>5}"
            for model in model_names:
                val = results[model].get(metric, {}).get(k, 0.0)
                row += f"  {val:>15.4f}"
            lines.append(row)

    return "\n".join(lines)
