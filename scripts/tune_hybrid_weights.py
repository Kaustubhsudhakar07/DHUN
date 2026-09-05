"""
Grid Search Weight Tuning for Hybrid Recommender — Phase 9

Precomputes candidate pools and sub-model normalized scores for a sample of
validation users, then performs ultra-fast vectorized grid search over
(w_cf, w_content, w_pop) to find the Pareto-optimal configuration.

Usage:
    python scripts/tune_hybrid_weights.py --n-users 500 --step 0.05
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy.sparse import load_npz

from src.config import (
    ALS_MODEL,
    ARTIST_MAPPING,
    HYBRID_WEIGHTS,
    MODELS_DIR,
    POPULARITY_SCORES,
    TFIDF_MATRIX,
    TFIDF_VECTORIZER,
    TRAIN_INTERACTIONS,
    VAL_INTERACTIONS,
)
from src.evaluation.metrics import ndcg_at_k, recall_at_k, hit_rate_at_k, average_precision_at_k
from src.features.tfidf import load_tfidf_features
from src.ranking.ranker import CandidatePool, PostFilter, ScoreNormalizer
from src.recommenders.als import ALSRecommender
from src.recommenders.content import ContentBasedRecommender
from src.recommenders.hybrid import HybridRecommender
from src.recommenders.popularity import PopularityRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tune_hybrid_weights")


def generate_weight_grid(step: float = 0.05) -> list[tuple[float, float, float]]:
    """Generate (w_cf, w_content, w_pop) triplets that sum to 1.0."""
    grid = []
    steps = int(round(1.0 / step))

    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            k = steps - i - j
            w_cf = round(i * step, 3)
            w_cnt = round(j * step, 3)
            w_pop = round(k * step, 3)

            # Prioritize viable recommendation combinations:
            if w_cf >= 0.2 and w_cnt >= 0.05 and w_pop >= 0.02:
                grid.append((w_cf, w_cnt, w_pop))

    return grid


def precompute_user_candidates(
    users: list[int],
    train_matrix,
    val_matrix,
    als,
    cb,
    pop,
    n_per_source: int = 40,
    pool_size: int = 150,
):
    """Precompute candidate IDs and normalized scores per model for each user."""
    logger.info("Precomputing candidate pools and scores for %d validation users...", len(users))
    t0 = time.time()
    user_data = []

    for idx, u in enumerate(users):
        known_items = set(train_matrix[u].indices.tolist())
        relevant = set(val_matrix[u].indices.tolist())
        if not relevant:
            continue

        # 1. Retrieve candidates from each model
        cf_recs = als.recommend(u, n=n_per_source, exclude_known=True)
        cf_cands = [r.item_id for r in cf_recs]

        cnt_recs = cb.recommend(u, n=n_per_source, exclude_known=True)
        cnt_cands = [r.item_id for r in cnt_recs]

        pop_recs = pop.recommend(u, n=n_per_source, exclude_known=True, known_items=train_matrix[u].indices)
        pop_cands = [r.item_id for r in pop_recs]

        # Merge candidate pool
        merged = CandidatePool.merge(cf_cands, cnt_cands, pop_cands, max_candidates=pool_size)
        merged = PostFilter.filter_known(merged, known_items)

        if len(merged) == 0:
            continue

        # 2. Get normalized scores
        cf_raw = als.get_scores(u, merged)
        cnt_raw = cb.get_scores(u, merged)
        pop_raw = pop.get_scores(u, merged)

        norm_cf = ScoreNormalizer.min_max(cf_raw)
        norm_cnt = ScoreNormalizer.min_max(cnt_raw)
        norm_pop = ScoreNormalizer.min_max(pop_raw)

        # Has signals?
        has_cf = bool(cf_raw.max() > cf_raw.min()) if len(cf_raw) > 0 else False
        has_cnt = bool(cnt_raw.max() > 0.0) if len(cnt_raw) > 0 else False

        user_data.append({
            "user_id": u,
            "relevant": relevant,
            "candidates": merged,
            "norm_cf": norm_cf,
            "norm_cnt": norm_cnt,
            "norm_pop": norm_pop,
            "has_cf": has_cf,
            "has_cnt": has_cnt,
        })

        if (idx + 1) % 100 == 0:
            logger.info("  Precomputed %d/%d users...", idx + 1, len(users))

    elapsed = time.time() - t0
    logger.info("Precomputed %d valid users in %.2fs (%.1f users/sec).", len(user_data), elapsed, len(user_data)/elapsed)
    return user_data


def evaluate_weights_fast(user_data, w_cf, w_cnt, w_pop, k=10):
    """Fast vectorized evaluation of a single weight configuration."""
    ndcg_total = 0.0
    recall_total = 0.0
    hr_total = 0.0
    map_total = 0.0

    for u in user_data:
        # Dynamic rebalancing if user lacks signal
        cf_w = w_cf if u["has_cf"] else 0.0
        cnt_w = w_cnt if u["has_cnt"] else 0.0
        pop_w = w_pop
        tot = cf_w + cnt_w + pop_w
        if tot <= 0:
            pop_w = 1.0
            tot = 1.0

        cf_w /= tot
        cnt_w /= tot
        pop_w /= tot

        scores = (cf_w * u["norm_cf"]) + (cnt_w * u["norm_cnt"]) + (pop_w * u["norm_pop"])
        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_items = u["candidates"][top_k_indices].tolist()

        rel = u["relevant"]
        ndcg_total += ndcg_at_k(top_k_items, rel, k)
        recall_total += recall_at_k(top_k_items, rel, k)
        hr_total += hit_rate_at_k(top_k_items, rel, k)
        map_total += average_precision_at_k(top_k_items, rel, k)

    n = len(user_data)
    return {
        "ndcg@10": ndcg_total / n,
        "recall@10": recall_total / n,
        "hit_rate@10": hr_total / n,
        "map@10": map_total / n,
    }


def main():
    parser = argparse.ArgumentParser(description="Tune Hybrid Recommender weights on validation set.")
    parser.add_argument("--n-users", type=int, default=500, help="Number of validation users to sample (default: 500)")
    parser.add_argument("--step", type=float, default=0.05, help="Grid step size (default: 0.05)")
    parser.add_argument("--output", default=str(HYBRID_WEIGHTS), help="Path to save best weights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling users")

    args = parser.parse_args()

    # 1. Load Data
    logger.info("Loading train and validation matrices...")
    train_matrix = load_npz(TRAIN_INTERACTIONS)
    val_matrix = load_npz(VAL_INTERACTIONS)

    idx2artist = None
    if ARTIST_MAPPING.exists():
        with open(ARTIST_MAPPING, "r", encoding="utf-8") as f:
            idx2artist = json.load(f).get("idx2artist", None)

    # 2. Load Sub-models
    logger.info("Loading sub-models...")
    pop = PopularityRecommender().load(POPULARITY_SCORES, idx2artist=idx2artist)

    tfidf_matrix, vectorizer = load_tfidf_features()
    cb = ContentBasedRecommender()
    cb.fit(
        tfidf_matrix=tfidf_matrix,
        train_interactions=train_matrix,
        idx2artist=idx2artist,
        vectorizer=vectorizer,
    )

    als = ALSRecommender(use_gpu=False).load(
        ALS_MODEL, train_interactions=train_matrix, idx2artist=idx2artist
    )

    # 3. Sample Users
    nnz_per_user = np.diff(val_matrix.indptr)
    val_users = np.where(nnz_per_user > 0)[0].tolist()
    rng = np.random.default_rng(args.seed)
    sampled_users = rng.choice(val_users, size=min(args.n_users, len(val_users)), replace=False).tolist()

    # 4. Precompute Candidates & Scores
    user_data = precompute_user_candidates(
        sampled_users,
        train_matrix,
        val_matrix,
        als,
        cb,
        pop,
    )

    # 5. Fast Grid Search
    grid = generate_weight_grid(step=args.step)
    logger.info("Running fast grid search over %d weight configurations...", len(grid))

    results = []
    best_config = None
    best_score = -1.0

    t_grid_start = time.time()
    for w_cf, w_cnt, w_pop in grid:
        metrics = evaluate_weights_fast(user_data, w_cf, w_cnt, w_pop, k=10)
        # Objective: balance NDCG and Recall
        score = metrics["ndcg@10"] + 0.5 * metrics["recall@10"]

        record = {
            "weights": {
                "collaborative": w_cf,
                "content": w_cnt,
                "popularity": w_pop,
            },
            "ndcg@10": metrics["ndcg@10"],
            "recall@10": metrics["recall@10"],
            "hit_rate@10": metrics["hit_rate@10"],
            "map@10": metrics["map@10"],
            "score": score,
        }
        results.append(record)

        if score > best_score:
            best_score = score
            best_config = record

    t_grid_elapsed = time.time() - t_grid_start
    logger.info("Grid search evaluated %d configurations in %.2fs (%.1f configs/sec)!", len(grid), t_grid_elapsed, len(grid)/t_grid_elapsed)

    # Sort results by score descending
    results.sort(key=lambda r: r["score"], reverse=True)

    print("\n" + "=" * 80)
    print("TOP 10 HYBRID WEIGHT CONFIGURATIONS (by NDCG@10 + 0.5*Recall@10)")
    print("=" * 80)
    print(f"{'Rank':>4}  {'CF':>6}  {'Content':>8}  {'Pop':>6}  |  {'NDCG@10':>8}  {'Recall@10':>9}  {'HitRate@10':>10}  {'MAP@10':>8}")
    print("-" * 80)
    for rank, rec in enumerate(results[:10], 1):
        w = rec["weights"]
        print(
            f"{rank:4d}  {w['collaborative']:6.2f}  {w['content']:8.2f}  {w['popularity']:6.2f}  |  "
            f"{rec['ndcg@10']:8.4f}  {rec['recall@10']:9.4f}  {rec['hit_rate@10']:10.4f}  {rec['map@10']:8.4f}"
        )
    print("=" * 80)

    # 6. Save Best Weights
    hybrid = HybridRecommender(
        cf_recommender=als,
        content_recommender=cb,
        popularity_recommender=pop,
        weights=best_config["weights"],
    )
    hybrid.fit(train_interactions=train_matrix, idx2artist=idx2artist)
    output_path = Path(args.output)
    hybrid.save(output_path)

    # Save log
    log_path = MODELS_DIR / "hybrid_tuning_results.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_config": best_config,
                "top_10": results[:10],
                "total_configurations_tested": len(grid),
                "n_users_evaluated": len(user_data),
                "step": args.step,
            },
            f,
            indent=2,
        )
    logger.info("Saved tuning results to %s.", log_path)


if __name__ == "__main__":
    main()
