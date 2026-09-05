"""Evaluation metrics for recommendation quality."""

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    average_precision_at_k,
    ndcg_at_k,
    evaluate_model,
    format_metrics_table,
)
from src.evaluation.beyond_accuracy import (
    catalog_coverage,
    diversity,
    novelty,
    personalization,
    evaluate_beyond_accuracy,
)

__all__ = [
    # Ranking metrics
    "precision_at_k",
    "recall_at_k",
    "hit_rate_at_k",
    "average_precision_at_k",
    "ndcg_at_k",
    "evaluate_model",
    "format_metrics_table",
    # Beyond-accuracy metrics
    "catalog_coverage",
    "diversity",
    "novelty",
    "personalization",
    "evaluate_beyond_accuracy",
]
