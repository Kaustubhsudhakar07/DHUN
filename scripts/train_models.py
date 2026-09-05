"""
Train and save recommendation models — Phase 7.

Usage:
    python scripts/train_models.py --models all
    python scripts/train_models.py --models popularity als
    python scripts/train_models.py --models bpr --bpr-iterations 50
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

# Set OPENBLAS threads before importing numpy/implicit
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy.sparse import load_npz

from src.config import (
    ALS_DEFAULTS,
    ALS_MODEL,
    ARTIST_MAPPING,
    BPR_DEFAULTS,
    BPR_MODEL,
    MODELS_DIR,
    POPULARITY_SCORES,
    TFIDF_MATRIX,
    TFIDF_VECTORIZER,
    TRAIN_INTERACTIONS,
)
from src.features.tfidf import build_tfidf_features
from src.recommenders.als import ALSRecommender
from src.recommenders.bpr import BPRRecommender
from src.recommenders.popularity import PopularityRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_models")


def load_train_data():
    """Load training interaction matrix and artist mappings."""
    logger.info("Loading training interaction matrix from %s...", TRAIN_INTERACTIONS)
    t0 = time.time()
    train_matrix = load_npz(TRAIN_INTERACTIONS)
    logger.info(
        "Loaded train matrix in %.2fs: shape=%s, nnz=%d, sparsity=%.4f%%.",
        time.time() - t0,
        train_matrix.shape,
        train_matrix.nnz,
        (1 - train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1])) * 100,
    )

    idx2artist = None
    if ARTIST_MAPPING.exists():
        with open(ARTIST_MAPPING, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        idx2artist = mapping.get("idx2artist", None)
        logger.info("Loaded artist mapping: %d artists.", len(idx2artist))

    return train_matrix, idx2artist


def train_popularity(train_matrix, idx2artist):
    """Train and save PopularityRecommender."""
    logger.info("--- Training Popularity Recommender ---")
    t0 = time.time()
    pop = PopularityRecommender()
    pop.fit(train_interactions=train_matrix, idx2artist=idx2artist)
    pop.save(path=POPULARITY_SCORES)
    elapsed = time.time() - t0
    logger.info("Popularity recommender trained and saved in %.2fs.", elapsed)
    return pop


def train_tfidf():
    """Build and save TF-IDF features if not already built."""
    logger.info("--- Checking / Building TF-IDF Features ---")
    if TFIDF_MATRIX.exists() and TFIDF_VECTORIZER.exists():
        logger.info("TF-IDF matrix and vectorizer already exist in %s.", MODELS_DIR)
        return
    t0 = time.time()
    build_tfidf_features()
    logger.info("TF-IDF features built and saved in %.2fs.", time.time() - t0)


def train_als(train_matrix, idx2artist, factors=None, iterations=None, reg=None, alpha=None):
    """Train and save ALSRecommender."""
    logger.info("--- Training ALS Recommender ---")
    factors = factors or ALS_DEFAULTS["factors"]
    iterations = iterations or ALS_DEFAULTS["iterations"]
    reg = reg or ALS_DEFAULTS["regularization"]
    alpha = alpha or 40.0

    t0 = time.time()
    als = ALSRecommender(
        factors=factors,
        regularization=reg,
        iterations=iterations,
        alpha=alpha,
        use_gpu=False,
    )
    als.fit(train_interactions=train_matrix, idx2artist=idx2artist)
    als.save(path=ALS_MODEL)
    elapsed = time.time() - t0
    logger.info("ALS recommender trained and saved in %.2fs.", elapsed)
    return als


def train_bpr(train_matrix, idx2artist, factors=None, iterations=None, lr=None, reg=None):
    """Train and save BPRRecommender."""
    logger.info("--- Training BPR Recommender ---")
    factors = factors or BPR_DEFAULTS["factors"]
    iterations = iterations or 50  # 50 iterations provides solid convergence in ~1 min
    lr = lr or BPR_DEFAULTS["learning_rate"]
    reg = reg or BPR_DEFAULTS["regularization"]

    t0 = time.time()
    bpr = BPRRecommender(
        factors=factors,
        learning_rate=lr,
        regularization=reg,
        iterations=iterations,
        use_gpu=False,
    )
    bpr.fit(train_interactions=train_matrix, idx2artist=idx2artist)
    bpr.save(path=BPR_MODEL)
    elapsed = time.time() - t0
    logger.info("BPR recommender trained and saved in %.2fs.", elapsed)
    return bpr


def main():
    parser = argparse.ArgumentParser(description="Train and save recommendation models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["all", "popularity", "tfidf", "als", "bpr"],
        help="Models to train (default: all)",
    )
    parser.add_argument("--als-factors", type=int, default=ALS_DEFAULTS["factors"])
    parser.add_argument("--als-iterations", type=int, default=ALS_DEFAULTS["iterations"])
    parser.add_argument("--bpr-factors", type=int, default=BPR_DEFAULTS["factors"])
    parser.add_argument("--bpr-iterations", type=int, default=50)

    args = parser.parse_args()
    selected_models = set(args.models)
    if "all" in selected_models:
        selected_models = {"popularity", "tfidf", "als", "bpr"}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. TF-IDF
    if "tfidf" in selected_models:
        train_tfidf()

    # Load matrix if needed for other models
    if any(m in selected_models for m in ["popularity", "als", "bpr"]):
        train_matrix, idx2artist = load_train_data()

        # 2. Popularity
        if "popularity" in selected_models:
            train_popularity(train_matrix, idx2artist)

        # 3. ALS
        if "als" in selected_models:
            train_als(
                train_matrix,
                idx2artist,
                factors=args.als_factors,
                iterations=args.als_iterations,
            )

        # 4. BPR
        if "bpr" in selected_models:
            train_bpr(
                train_matrix,
                idx2artist,
                factors=args.bpr_factors,
                iterations=args.bpr_iterations,
            )

    logger.info("All requested models trained and saved successfully.")


if __name__ == "__main__":
    main()
