"""
Evaluate recommendation models — Phase 8.

Computes both ranking metrics (Precision@K, Recall@K, Hit Rate@K, MAP@K, NDCG@K)
and beyond-accuracy metrics (Coverage, Diversity, Novelty, Personalization)
across models, and outputs a formatted comparative table.

Usage:
    python scripts/evaluate_models.py --n-users 1000 --split val
    python scripts/evaluate_models.py --models popularity als --n-users 500
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Disable openblas threading warnings
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy.sparse import load_npz

from src.config import (
    ALS_MODEL,
    ARTIST_MAPPING,
    BPR_MODEL,
    EVAL_K_VALUES,
    MODELS_DIR,
    POPULARITY_SCORES,
    HYBRID_WEIGHTS,
    TEST_INTERACTIONS,
    TFIDF_MATRIX,
    TFIDF_VECTORIZER,
    TRAIN_INTERACTIONS,
    VAL_INTERACTIONS,
)
from src.evaluation.beyond_accuracy import evaluate_beyond_accuracy
from src.evaluation.metrics import evaluate_model, format_metrics_table
from src.features.tfidf import load_tfidf_features
from src.recommenders.als import ALSRecommender
from src.recommenders.bpr import BPRRecommender
from src.recommenders.content import ContentBasedRecommender
from src.recommenders.popularity import PopularityRecommender
from src.recommenders.hybrid import HybridRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_models")


def load_eval_data(split="val"):
    """Load train, eval (val or test) sparse matrices and artist mappings."""
    eval_path = VAL_INTERACTIONS if split == "val" else TEST_INTERACTIONS
    logger.info("Loading training matrix from %s...", TRAIN_INTERACTIONS)
    train_matrix = load_npz(TRAIN_INTERACTIONS)

    logger.info("Loading evaluation matrix (%s) from %s...", split, eval_path)
    eval_matrix = load_npz(eval_path)

    idx2artist = None
    if ARTIST_MAPPING.exists():
        with open(ARTIST_MAPPING, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        idx2artist = mapping.get("idx2artist", None)

    return train_matrix, eval_matrix, idx2artist


def init_models(model_names, train_matrix, idx2artist):
    """Instantiate and load models requested for evaluation."""
    models = {}

    # 1. Popularity
    if "popularity" in model_names:
        logger.info("Initializing PopularityRecommender...")
        pop = PopularityRecommender()
        if POPULARITY_SCORES.exists():
            pop.load(POPULARITY_SCORES, idx2artist=idx2artist)
        else:
            pop.fit(train_matrix, idx2artist=idx2artist)
        models["Popularity"] = pop

    # 2. Content-Based
    if "content" in model_names:
        logger.info("Initializing ContentBasedRecommender...")
        if TFIDF_MATRIX.exists() and TFIDF_VECTORIZER.exists():
            tfidf_matrix, vectorizer = load_tfidf_features()
            cb = ContentBasedRecommender()
            cb.fit(
                tfidf_matrix=tfidf_matrix,
                train_interactions=train_matrix,
                idx2artist=idx2artist,
                vectorizer=vectorizer,
            )
            models["ContentBased"] = cb
        else:
            logger.warning("TF-IDF features not found. Skipping ContentBased model.")

    # 3. ALS
    if "als" in model_names:
        logger.info("Initializing ALSRecommender...")
        if ALS_MODEL.exists():
            als = ALSRecommender(use_gpu=False)
            als.load(ALS_MODEL, train_interactions=train_matrix, idx2artist=idx2artist)
            models["ALS"] = als
        else:
            logger.warning("ALS model not found at %s. Skipping ALS.", ALS_MODEL)

    # 4. BPR
    if "bpr" in model_names:
        logger.info("Initializing BPRRecommender...")
        if BPR_MODEL.exists():
            bpr = BPRRecommender(use_gpu=False)
            bpr.load(BPR_MODEL, train_interactions=train_matrix, idx2artist=idx2artist)
            models["BPR"] = bpr
        else:
            logger.warning("BPR model not found at %s. Skipping BPR.", BPR_MODEL)

    # 5. Hybrid
    if "hybrid" in model_names:
        logger.info("Initializing HybridRecommender...")
        # Ensure sub-models are available
        pop_sub = models.get("Popularity")
        if pop_sub is None:
            pop_sub = PopularityRecommender()
            if POPULARITY_SCORES.exists():
                pop_sub.load(POPULARITY_SCORES, idx2artist=idx2artist)
            else:
                pop_sub.fit(train_matrix, idx2artist=idx2artist)

        cnt_sub = models.get("ContentBased")
        if cnt_sub is None and TFIDF_MATRIX.exists() and TFIDF_VECTORIZER.exists():
            tfidf_matrix, vectorizer = load_tfidf_features()
            cnt_sub = ContentBasedRecommender()
            cnt_sub.fit(
                tfidf_matrix=tfidf_matrix,
                train_interactions=train_matrix,
                idx2artist=idx2artist,
                vectorizer=vectorizer,
            )

        cf_sub = models.get("ALS")
        if cf_sub is None and ALS_MODEL.exists():
            cf_sub = ALSRecommender(use_gpu=False)
            cf_sub.load(ALS_MODEL, train_interactions=train_matrix, idx2artist=idx2artist)

        hybrid = HybridRecommender(
            cf_recommender=cf_sub,
            content_recommender=cnt_sub,
            popularity_recommender=pop_sub,
        )
        hybrid.fit(train_interactions=train_matrix, idx2artist=idx2artist)
        if HYBRID_WEIGHTS.exists():
            hybrid.load(HYBRID_WEIGHTS)
        models["Hybrid"] = hybrid

    return models


def generate_markdown_summary(ranking_results, beyond_results, k_values):
    """Generate a clean markdown table summarizing all metrics."""
    lines = [
        "# Model Evaluation Results",
        "",
        "## 1. Ranking Accuracy Metrics",
        "",
    ]

    models = list(ranking_results.keys())

    # Table for K=10 (standard primary benchmark)
    lines.append("### Primary Benchmark (K=10)")
    lines.append("")
    lines.append("| Model | Precision@10 | Recall@10 | Hit Rate@10 | MAP@10 | NDCG@10 |")
    lines.append("|---|---|---|---|---|---|")
    for m in models:
        p = ranking_results[m]["precision"].get(10, 0.0)
        r = ranking_results[m]["recall"].get(10, 0.0)
        hr = ranking_results[m]["hit_rate"].get(10, 0.0)
        map_ = ranking_results[m]["map"].get(10, 0.0)
        ndcg = ranking_results[m]["ndcg"].get(10, 0.0)
        lines.append(f"| **{m}** | {p:.4f} | {r:.4f} | {hr:.4f} | {map_:.4f} | {ndcg:.4f} |")

    lines.append("")
    lines.append("## 2. Beyond-Accuracy Metrics (K=10)")
    lines.append("")
    lines.append("| Model | Catalog Coverage | Diversity | Novelty | Personalization |")
    lines.append("|---|---|---|---|---|")
    for m in models:
        if m in beyond_results:
            cov = beyond_results[m]["coverage"] * 100
            div = beyond_results[m]["diversity"]
            nov = beyond_results[m]["novelty"]
            pers = beyond_results[m]["personalization"]
            lines.append(f"| **{m}** | {cov:.2f}% | {div:.4f} | {nov:.4f} | {pers:.4f} |")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate recommendation models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["all", "popularity", "content", "als", "bpr", "hybrid"],
        help="Models to evaluate (default: all)",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="Evaluation split (default: val)",
    )
    parser.add_argument(
        "--n-users",
        type=int,
        default=1000,
        help="Number of test users to sample (default: 1000; set 0 for all users)",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[5, 10, 20],
        help="List of K cutoffs (default: 5 10 20)",
    )
    parser.add_argument(
        "--output",
        default=str(MODELS_DIR / "evaluation_results.json"),
        help="Path to output JSON file",
    )

    args = parser.parse_args()
    selected_models = set(args.models)
    if "all" in selected_models:
        selected_models = {"popularity", "content", "als", "bpr", "hybrid"}

    n_users = args.n_users if args.n_users > 0 else None

    # Load data
    train_matrix, eval_matrix, idx2artist = load_eval_data(split=args.split)

    # Initialize models
    models = init_models(selected_models, train_matrix, idx2artist)
    if not models:
        logger.error("No models available for evaluation. Please train models first.")
        sys.exit(1)

    # Load TF-IDF for diversity feature distance if available
    item_features = None
    if TFIDF_MATRIX.exists():
        tfidf_matrix, _ = load_tfidf_features()
        item_features = tfidf_matrix

    ranking_results = {}
    beyond_results = {}

    for name, recommender in models.items():
        logger.info("\n========== Evaluating %s ==========", name)

        # 1. Ranking metrics
        t0 = time.time()
        ranking_res = evaluate_model(
            recommender=recommender,
            test_matrix=eval_matrix,
            train_matrix=train_matrix,
            k_values=args.k_values,
            n_users=n_users,
        )
        ranking_results[name] = ranking_res
        logger.info("Ranking evaluation completed in %.2fs.", time.time() - t0)

        # 2. Beyond-accuracy metrics (at K=10)
        t1 = time.time()
        beyond_res = evaluate_beyond_accuracy(
            recommender=recommender,
            train_matrix=train_matrix,
            k=10,
            n_users=n_users,
            item_features=item_features,
        )
        beyond_results[name] = beyond_res
        logger.info("Beyond-accuracy evaluation completed in %.2fs.", time.time() - t1)

    # Print summary table
    print("\n" + format_metrics_table(ranking_results, args.k_values))

    # Generate and print markdown report
    md_report = generate_markdown_summary(ranking_results, beyond_results, args.k_values)
    print("\n" + md_report)

    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_results = {
        "split": args.split,
        "n_users_evaluated": n_users,
        "k_values": args.k_values,
        "ranking": ranking_results,
        "beyond_accuracy": beyond_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved evaluation results to %s.", output_path)


if __name__ == "__main__":
    main()
