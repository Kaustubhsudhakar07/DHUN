"""
Data Preprocessing Pipeline — Phase 5

Transforms raw interaction data and HetRec tags into production-ready
artifacts for recommendation modeling:
1. Data cleaning (remove invalid play counts, missing artists, whitespace).
2. Iterative k-core filtering (k_user >= 5, k_item >= 5) as established in Decision 006.
3. Canonical integer ID encoding for users and items.
4. Per-user Leave-1-Out train/validation/test split into sparse CSR matrices.
5. Artist content metadata extraction and tag document construction.

Run pipeline directly:
    python -m src.preprocessing.pipeline
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ARTIST_MAPPING,
    MIN_ARTIST_INTERACTIONS,
    MIN_USER_INTERACTIONS,
    PROCESSED_ARTIST_FEATURES,
    PROCESSED_DATA_DIR,
    PROCESSED_INTERACTIONS,
    TEST_INTERACTIONS,
    TRAIN_INTERACTIONS,
    USER_MAPPING,
    VAL_INTERACTIONS,
)
from src.data.loader import (
    get_artist_tags,
    load_hetrec_artists,
    load_hetrec_tags,
    load_hetrec_user_tagged_artists,
    load_lastfm_360k_interactions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =========================================================================
# 1. CLEANING
# =========================================================================

def clean_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw interactions.

    - Drops rows where play_count is NaN or <= 0
    - Drops rows where user_id or artist_name is missing
    - Strips whitespace and lowercases artist names
    - Aggregates duplicate (user_id, artist_name) pairs by summing play counts

    Args:
        df: Raw DataFrame containing at least ['user_id', 'artist_name', 'play_count'].

    Returns:
        Cleaned DataFrame with deduplicated user-artist pairs.
    """
    initial_len = len(df)
    logger.info("Cleaning raw interactions (initial rows: %s)...", f"{initial_len:,}")

    # Drop nulls
    cleaned = df.dropna(subset=["user_id", "artist_name", "play_count"]).copy()

    # Drop non-positive play counts
    cleaned = cleaned[cleaned["play_count"] > 0]

    # Normalize artist names
    cleaned["artist_name"] = (
        cleaned["artist_name"].astype(str).str.strip().str.lower()
    )
    cleaned = cleaned[cleaned["artist_name"].str.len() > 0]

    # Deduplicate user-artist interactions if any duplicate scrobbles exist
    cleaned = (
        cleaned.groupby(["user_id", "artist_name"], as_index=False)["play_count"]
        .sum()
    )

    dropped = initial_len - len(cleaned)
    logger.info(
        "Cleaning complete: retained %s rows (dropped %s, %.3f%%).",
        f"{len(cleaned):,}",
        f"{dropped:,}",
        (dropped / initial_len) * 100 if initial_len > 0 else 0,
    )
    return cleaned


# =========================================================================
# 2. ITERATIVE K-CORE FILTERING
# =========================================================================

def filter_k_core(
    df: pd.DataFrame,
    min_user_interactions: int = MIN_USER_INTERACTIONS,
    min_artist_interactions: int = MIN_ARTIST_INTERACTIONS,
    max_iter: int = 10,
) -> pd.DataFrame:
    """Apply iterative k-core filtering so all users and artists satisfy minimum degrees.

    Decision 006 justification:
    - Eliminates tail noise and disconnected vertices.
    - Guarantees every user has >= 5 interactions (needed for 1 val + 1 test + >= 3 train).
    - Shrinks artist catalog dimension while retaining > 95% interaction volume.

    Args:
        df: Cleaned interactions DataFrame.
        min_user_interactions: Minimum interactions required per user.
        min_artist_interactions: Minimum distinct listeners required per artist.
        max_iter: Maximum filtering iterations.

    Returns:
        k-core filtered DataFrame.
    """
    logger.info(
        "Applying iterative k-core filtering (k_user >= %d, k_artist >= %d)...",
        min_user_interactions,
        min_artist_interactions,
    )

    filtered = df[["user_id", "artist_name", "play_count"]].copy()

    for iteration in range(1, max_iter + 1):
        prev_len = len(filtered)

        # Filter artists
        artist_counts = filtered.groupby("artist_name")["user_id"].nunique()
        valid_artists = artist_counts[artist_counts >= min_artist_interactions].index
        filtered = filtered[filtered["artist_name"].isin(valid_artists)]

        # Filter users
        user_counts = filtered.groupby("user_id")["artist_name"].nunique()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        filtered = filtered[filtered["user_id"].isin(valid_users)]

        curr_len = len(filtered)
        logger.info(
            "Iteration %d: %s rows remaining (dropped %s in this pass)",
            iteration,
            f"{curr_len:,}",
            f"{prev_len - curr_len:,}",
        )

        if curr_len == prev_len:
            logger.info("k-core converged after %d iterations.", iteration)
            break

    n_users = filtered["user_id"].nunique()
    n_artists = filtered["artist_name"].nunique()
    logger.info(
        "Filtered dataset: %s interactions across %s users and %s artists.",
        f"{len(filtered):,}",
        f"{n_users:,}",
        f"{n_artists:,}",
    )
    return filtered


# =========================================================================
# 3. CANONICAL INTEGER ENCODING & MAPPINGS
# =========================================================================

def create_and_apply_mappings(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Assign contiguous integer indices [0 .. N-1] to users and artists.

    Args:
        df: Filtered interactions DataFrame with columns ['user_id', 'artist_name', 'play_count'].

    Returns:
        Tuple containing:
        - df_indexed: DataFrame with ['user_idx', 'artist_idx', 'play_count']
        - user_meta: Dict with 'user2idx' and 'idx2user'
        - artist_meta: Dict with 'artist2idx' and 'idx2artist'
    """
    logger.info("Encoding unique users and artists to contiguous integer indices...")

    # Sort lexicographically for deterministic ID assignment
    unique_users = np.sort(df["user_id"].unique())
    unique_artists = np.sort(df["artist_name"].unique())

    user2idx = {u: int(i) for i, u in enumerate(unique_users)}
    idx2user = list(unique_users)

    artist2idx = {a: int(i) for i, a in enumerate(unique_artists)}
    idx2artist = list(unique_artists)

    df_indexed = pd.DataFrame({
        "user_idx": df["user_id"].map(user2idx).astype(np.int32),
        "artist_idx": df["artist_name"].map(artist2idx).astype(np.int32),
        "play_count": df["play_count"].astype(np.float32),
    })

    user_meta = {"user2idx": user2idx, "idx2user": idx2user}
    artist_meta = {"artist2idx": artist2idx, "idx2artist": idx2artist}

    logger.info(
        "Encoding complete: %d users (0..%d), %d artists (0..%d).",
        len(idx2user),
        len(idx2user) - 1,
        len(idx2artist),
        len(idx2artist) - 1,
    )
    return df_indexed, user_meta, artist_meta


# =========================================================================
# 4. LEAVE-1-OUT TRAIN / VALIDATION / TEST SPLIT
# =========================================================================

def split_leave_one_out(
    df_indexed: pd.DataFrame, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute Leave-1-Out splitting per user.

    For every user:
    - Hold out exactly 1 interaction for the test set.
    - Hold out exactly 1 interaction for the validation set.
    - The remaining interactions (>= 3) form the training set.

    Vectorized implementation using deterministic random ranking.

    Args:
        df_indexed: DataFrame containing ['user_idx', 'artist_idx', 'play_count'].
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    logger.info("Performing deterministic Leave-1-Out per-user train/val/test split (seed=%d)...", seed)

    rng = np.random.default_rng(seed)
    # Assign uniform random float to each interaction row
    df_indexed = df_indexed.copy()
    df_indexed["_rand"] = rng.random(len(df_indexed))

    # Rank interactions within each user
    df_indexed["_rank"] = df_indexed.groupby("user_idx")["_rand"].rank(method="first", ascending=True)

    test_mask = df_indexed["_rank"] == 1
    val_mask = df_indexed["_rank"] == 2
    train_mask = df_indexed["_rank"] > 2

    test_df = df_indexed[test_mask][["user_idx", "artist_idx", "play_count"]].reset_index(drop=True)
    val_df = df_indexed[val_mask][["user_idx", "artist_idx", "play_count"]].reset_index(drop=True)
    train_df = df_indexed[train_mask][["user_idx", "artist_idx", "play_count"]].reset_index(drop=True)

    n_users = df_indexed["user_idx"].nunique()
    logger.info(
        "Split results:\n"
        "  - Train: %s interactions (%.2f%%)\n"
        "  - Val:   %s interactions (exactly 1 per user, %.2f%%)\n"
        "  - Test:  %s interactions (exactly 1 per user, %.2f%%)",
        f"{len(train_df):,}",
        (len(train_df) / len(df_indexed)) * 100,
        f"{len(val_df):,}",
        (len(val_df) / len(df_indexed)) * 100,
        f"{len(test_df):,}",
        (len(test_df) / len(df_indexed)) * 100,
    )

    assert len(val_df) == n_users, f"Validation set size {len(val_df)} != total users {n_users}"
    assert len(test_df) == n_users, f"Test set size {len(test_df)} != total users {n_users}"

    return train_df, val_df, test_df


def build_sparse_matrix(
    df: pd.DataFrame, n_users: int, n_artists: int
) -> csr_matrix:
    """Construct SciPy CSR sparse matrix from indexed interactions.

    Matrix dimensions: [n_users, n_artists]
    Value: raw play counts.

    Args:
        df: DataFrame with ['user_idx', 'artist_idx', 'play_count'].
        n_users: Total user count.
        n_artists: Total artist count.

    Returns:
        SciPy CSR sparse matrix.
    """
    row = df["user_idx"].values
    col = df["artist_idx"].values
    data = df["play_count"].values

    return csr_matrix(
        (data, (row, col)),
        shape=(n_users, n_artists),
        dtype=np.float32,
    )


# =========================================================================
# 5. HETREC TAG EXTRACTION & CONTENT FEATURE ALIGNMENT
# =========================================================================

def build_artist_content_features(artist2idx: dict[str, int]) -> pd.DataFrame:
    """Extract and align HetRec 2011 tag metadata for the filtered catalog.

    Constructs a textual tag document for each artist to enable TF-IDF
    feature extraction in Phase 7.

    Args:
        artist2idx: Mapping from normalized artist name to integer index.

    Returns:
        DataFrame with columns:
            [artist_idx, artist_name, tags, tag_count_dict_json]
    """
    logger.info("Extracting and aligning HetRec tag features for %d catalog artists...", len(artist2idx))

    hetrec_artists = load_hetrec_artists()
    hetrec_tags = load_hetrec_tags()
    hetrec_user_tagged = load_hetrec_user_tagged_artists()

    artist_tags = get_artist_tags(hetrec_user_tagged, hetrec_tags)

    # Merge artist names from HetRec
    hetrec_named = artist_tags.merge(
        hetrec_artists[["artist_id", "name"]], on="artist_id", how="inner"
    )
    hetrec_named["norm_name"] = (
        hetrec_named["name"].astype(str).str.strip().str.lower()
    )

    # Filter to artists present in our active catalog
    matched_tags = hetrec_named[hetrec_named["norm_name"].isin(artist2idx)].copy()
    matched_tags["artist_idx"] = matched_tags["norm_name"].map(artist2idx)

    # Aggregate tags per artist
    records = []
    for artist_name, artist_idx in artist2idx.items():
        sub = matched_tags[matched_tags["artist_idx"] == artist_idx]
        if len(sub) > 0:
            # Build tag string weighted by user assignment counts
            tag_tokens = []
            tag_dict = {}
            for _, row in sub.iterrows():
                t_val = str(row["tag_value"]).strip().lower().replace(" ", "_")
                t_count = int(row["tag_count"])
                tag_dict[t_val] = t_count
                # Repeat tag token based on log frequency for TF-IDF doc
                repeat = max(1, int(np.log2(t_count + 1)))
                tag_tokens.extend([t_val] * repeat)

            tag_doc = " ".join(tag_tokens)
            tag_json = json.dumps(tag_dict)
        else:
            tag_doc = ""
            tag_json = "{}"

        records.append({
            "artist_idx": artist_idx,
            "artist_name": artist_name,
            "tags": tag_doc,
            "tag_counts_json": tag_json,
        })

    features_df = pd.DataFrame(records).sort_values("artist_idx").reset_index(drop=True)
    tagged_count = (features_df["tags"].str.len() > 0).sum()
    logger.info(
        "Content features constructed: %s / %s artists (%.2f%%) have tags from HetRec.",
        f"{tagged_count:,}",
        f"{len(features_df):,}",
        (tagged_count / len(features_df)) * 100,
    )
    return features_df


# =========================================================================
# 6. PIPELINE ORCHESTRATOR
# =========================================================================

def run_preprocessing_pipeline():
    """Execute the full data preprocessing pipeline and save all artifacts."""
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("STARTING PREPROCESSING PIPELINE — Phase 5")
    logger.info("=" * 70)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    raw_df = load_lastfm_360k_interactions()

    # 2. Clean
    cleaned_df = clean_interactions(raw_df)

    # 3. Filter k-core
    filtered_df = filter_k_core(cleaned_df)

    # 4. Create and apply contiguous integer mappings
    indexed_df, user_meta, artist_meta = create_and_apply_mappings(filtered_df)
    n_users = len(user_meta["idx2user"])
    n_artists = len(artist_meta["idx2artist"])

    # 5. Split Leave-1-Out per user
    train_df, val_df, test_df = split_leave_one_out(indexed_df, seed=42)

    # 6. Build sparse matrices
    logger.info("Building CSR sparse matrices for train, val, and test...")
    train_csr = build_sparse_matrix(train_df, n_users, n_artists)
    val_csr = build_sparse_matrix(val_df, n_users, n_artists)
    test_csr = build_sparse_matrix(test_df, n_users, n_artists)

    # 7. Save interaction artifacts
    logger.info("Saving interaction matrices to %s...", PROCESSED_DATA_DIR)
    save_npz(TRAIN_INTERACTIONS, train_csr)
    save_npz(VAL_INTERACTIONS, val_csr)
    save_npz(TEST_INTERACTIONS, test_csr)
    logger.info("Saved train (%s), val (%s), test (%s).", TRAIN_INTERACTIONS.name, VAL_INTERACTIONS.name, TEST_INTERACTIONS.name)

    logger.info("Saving ID mappings to JSON...")
    with open(USER_MAPPING, "w", encoding="utf-8") as f:
        json.dump(user_meta, f)
    with open(ARTIST_MAPPING, "w", encoding="utf-8") as f:
        json.dump(artist_meta, f)
    logger.info("Saved user mapping (%s) and artist mapping (%s).", USER_MAPPING.name, ARTIST_MAPPING.name)

    # Save processed tabular interactions
    logger.info("Saving full processed interactions dataframe (parquet)...")
    indexed_df.to_parquet(PROCESSED_INTERACTIONS, index=False)
    logger.info("Saved %s.", PROCESSED_INTERACTIONS.name)

    # 8. Build and save artist content features
    features_df = build_artist_content_features(artist_meta["artist2idx"])
    logger.info("Saving artist content features to %s...", PROCESSED_ARTIST_FEATURES.name)
    features_df.to_parquet(PROCESSED_ARTIST_FEATURES, index=False)
    logger.info("Saved %s.", PROCESSED_ARTIST_FEATURES.name)

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("PREPROCESSING PIPELINE COMPLETE in %.1f seconds (%.2f minutes)", elapsed, elapsed / 60)
    logger.info("Artifacts produced:")
    logger.info("  - %s (shape: %s, nnz: %s)", TRAIN_INTERACTIONS.name, train_csr.shape, f"{train_csr.nnz:,}")
    logger.info("  - %s (shape: %s, nnz: %s)", VAL_INTERACTIONS.name, val_csr.shape, f"{val_csr.nnz:,}")
    logger.info("  - %s (shape: %s, nnz: %s)", TEST_INTERACTIONS.name, test_csr.shape, f"{test_csr.nnz:,}")
    logger.info("  - %s (%s users)", USER_MAPPING.name, f"{n_users:,}")
    logger.info("  - %s (%s artists)", ARTIST_MAPPING.name, f"{n_artists:,}")
    logger.info("  - %s (%s rows)", PROCESSED_ARTIST_FEATURES.name, f"{len(features_df):,}")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_preprocessing_pipeline()
