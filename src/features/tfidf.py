"""
TF-IDF Feature Engineering — Phase 6

Transforms artist tag documents (produced by the preprocessing pipeline)
into a sparse TF-IDF matrix suitable for content-based recommendation.

Key design choices:
- sublinear_tf=True: log(1+tf) dampens the effect of very frequent tags
  within a single artist's document.
- max_features=5000: caps vocabulary to most discriminative tags.
- min_df=3, max_df=0.8: removes noise tags (too rare / too common).
- Cosine similarity is precomputed for top-K neighbors per artist to
  keep memory bounded (full pairwise is O(n_artists^2)).

Run standalone:
    python -m src.features.tfidf
"""

import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PROCESSED_ARTIST_FEATURES,
    TFIDF_VECTORIZER,
    TFIDF_MATRIX,
    MODELS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Path for precomputed similarity neighborhoods
TFIDF_SIMILARITY = MODELS_DIR / "tfidf_similarity.npz"


# =========================================================================
# BUILD
# =========================================================================

def build_tfidf_features(
    max_features: int = 5000,
    min_df: int = 3,
    max_df: float = 0.8,
    top_k_neighbors: int = 50,
) -> Tuple[csr_matrix, TfidfVectorizer, pd.DataFrame]:
    """Build TF-IDF feature matrix from artist tag documents.

    Args:
        max_features: Maximum vocabulary size.
        min_df: Minimum document frequency for a tag to be included.
        max_df: Maximum document frequency (fraction) for a tag.
        top_k_neighbors: Number of nearest neighbors to precompute per artist.

    Returns:
        Tuple of (tfidf_matrix, vectorizer, features_df).
    """
    start = time.time()
    logger.info("=" * 60)
    logger.info("BUILDING TF-IDF FEATURES — Phase 6")
    logger.info("=" * 60)

    # Load artist features
    if not PROCESSED_ARTIST_FEATURES.exists():
        raise FileNotFoundError(
            f"Artist features not found: {PROCESSED_ARTIST_FEATURES}\n"
            "Run: python -m src.preprocessing.pipeline"
        )

    features_df = pd.read_parquet(PROCESSED_ARTIST_FEATURES)
    logger.info(
        "Loaded artist features: %s artists, %s with tags.",
        f"{len(features_df):,}",
        f"{(features_df['tags'].str.len() > 0).sum():,}",
    )

    # Fill empty tag documents with empty string
    documents = features_df["tags"].fillna("").tolist()

    # Fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        norm="l2",
        token_pattern=r"(?u)\b\w[\w]*\b",  # match single char tokens too
    )

    tfidf_matrix = vectorizer.fit_transform(documents)
    vocab_size = len(vectorizer.vocabulary_)

    # Stats on coverage
    nonzero_rows = tfidf_matrix.getnnz(axis=1)
    artists_with_features = (nonzero_rows > 0).sum()
    artists_without = len(features_df) - artists_with_features

    logger.info(
        "TF-IDF matrix shape: %s × %s (vocab size: %d)",
        f"{tfidf_matrix.shape[0]:,}",
        f"{tfidf_matrix.shape[1]:,}",
        vocab_size,
    )
    logger.info(
        "Artists with TF-IDF features: %s (%.1f%%). Without: %s.",
        f"{artists_with_features:,}",
        (artists_with_features / len(features_df)) * 100,
        f"{artists_without:,}",
    )
    logger.info("TF-IDF matrix density: %.4f%%", tfidf_matrix.nnz / np.prod(tfidf_matrix.shape) * 100)

    # Save vectorizer and matrix
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(TFIDF_VECTORIZER, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info("Saved vectorizer to %s.", TFIDF_VECTORIZER.name)

    save_npz(TFIDF_MATRIX, csr_matrix(tfidf_matrix))
    logger.info("Saved TF-IDF matrix to %s.", TFIDF_MATRIX.name)

    # Precompute truncated cosine similarity neighborhoods
    logger.info(
        "Precomputing top-%d cosine similarity neighbors per artist...",
        top_k_neighbors,
    )
    sim_matrix = _build_truncated_similarity(
        tfidf_matrix, top_k=top_k_neighbors
    )
    save_npz(TFIDF_SIMILARITY, sim_matrix)
    logger.info("Saved similarity neighborhoods to %s.", TFIDF_SIMILARITY.name)

    elapsed = time.time() - start
    logger.info("TF-IDF feature engineering complete in %.1f seconds.", elapsed)
    logger.info("=" * 60)

    return tfidf_matrix, vectorizer, features_df


def _build_truncated_similarity(
    tfidf_matrix: csr_matrix,
    top_k: int = 50,
    batch_size: int = 1000,
) -> csr_matrix:
    """Compute truncated cosine similarity, keeping only top-K neighbors per row.

    Processes in batches to avoid OOM on large catalogs.

    Args:
        tfidf_matrix: Sparse TF-IDF matrix [n_artists × n_features].
        top_k: Number of neighbors to retain per artist.
        batch_size: Number of artists to process at a time.

    Returns:
        Sparse CSR matrix [n_artists × n_artists] with at most top_k
        non-zero entries per row.
    """
    n_artists = tfidf_matrix.shape[0]
    rows, cols, vals = [], [], []

    n_batches = (n_artists + batch_size - 1) // batch_size
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_artists)

        # Compute similarity for this batch against all artists
        batch_sim = cosine_similarity(
            tfidf_matrix[start_idx:end_idx], tfidf_matrix
        )

        for local_i in range(batch_sim.shape[0]):
            global_i = start_idx + local_i
            sim_row = batch_sim[local_i]

            # Zero out self-similarity
            sim_row[global_i] = 0.0

            # Get top-K indices
            if top_k < len(sim_row):
                top_indices = np.argpartition(sim_row, -top_k)[-top_k:]
            else:
                top_indices = np.arange(len(sim_row))

            # Filter out zero similarities
            mask = sim_row[top_indices] > 0
            top_indices = top_indices[mask]

            rows.extend([global_i] * len(top_indices))
            cols.extend(top_indices.tolist())
            vals.extend(sim_row[top_indices].tolist())

        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            logger.info(
                "  Similarity batch %d/%d (artists %d–%d)",
                batch_idx + 1,
                n_batches,
                start_idx,
                end_idx - 1,
            )

    sim_sparse = csr_matrix(
        (vals, (rows, cols)),
        shape=(n_artists, n_artists),
        dtype=np.float32,
    )
    return sim_sparse


# =========================================================================
# LOAD
# =========================================================================

def load_tfidf_features() -> Tuple[csr_matrix, TfidfVectorizer]:
    """Load pre-built TF-IDF matrix and vectorizer from disk.

    Returns:
        Tuple of (tfidf_matrix, vectorizer).

    Raises:
        FileNotFoundError: If artifacts don't exist (run build first).
    """
    if not TFIDF_MATRIX.exists():
        raise FileNotFoundError(
            f"TF-IDF matrix not found: {TFIDF_MATRIX}\n"
            "Run: python -m src.features.tfidf"
        )
    if not TFIDF_VECTORIZER.exists():
        raise FileNotFoundError(
            f"TF-IDF vectorizer not found: {TFIDF_VECTORIZER}\n"
            "Run: python -m src.features.tfidf"
        )

    tfidf_matrix = load_npz(TFIDF_MATRIX)
    with open(TFIDF_VECTORIZER, "rb") as f:
        vectorizer = pickle.load(f)

    return tfidf_matrix, vectorizer


def load_tfidf_similarity() -> csr_matrix:
    """Load precomputed truncated cosine similarity matrix.

    Returns:
        Sparse CSR similarity matrix [n_artists × n_artists].
    """
    if not TFIDF_SIMILARITY.exists():
        raise FileNotFoundError(
            f"Similarity matrix not found: {TFIDF_SIMILARITY}\n"
            "Run: python -m src.features.tfidf"
        )
    return load_npz(TFIDF_SIMILARITY)


def get_top_tags(
    vectorizer: TfidfVectorizer,
    tfidf_matrix: csr_matrix,
    artist_idx: int,
    n: int = 10,
) -> list[Tuple[str, float]]:
    """Get the top-N TF-IDF tags for an artist.

    Useful for explanations and debugging.

    Args:
        vectorizer: Fitted TF-IDF vectorizer.
        tfidf_matrix: TF-IDF matrix.
        artist_idx: Integer index of the artist.
        n: Number of top tags to return.

    Returns:
        List of (tag_name, tfidf_score) tuples, sorted by score descending.
    """
    feature_names = vectorizer.get_feature_names_out()
    row = tfidf_matrix[artist_idx].toarray().ravel()

    if row.sum() == 0:
        return []

    top_indices = np.argsort(row)[::-1][:n]
    return [
        (feature_names[i], float(row[i]))
        for i in top_indices
        if row[i] > 0
    ]


# =========================================================================
# CLI ENTRY POINT
# =========================================================================

if __name__ == "__main__":
    build_tfidf_features()
