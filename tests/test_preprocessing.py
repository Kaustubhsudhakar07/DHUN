"""
Unit tests for data preprocessing pipeline (Phase 5).
"""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.preprocessing.pipeline import (
    build_sparse_matrix,
    clean_interactions,
    create_and_apply_mappings,
    filter_k_core,
    split_leave_one_out,
)


@pytest.fixture
def sample_raw_interactions():
    """Create sample dirty interaction data."""
    return pd.DataFrame({
        "user_id": ["u1", "u1", "u1", "u2", "u2", "u3", "u4", None, "u5"],
        "artist_name": [" Radiohead ", "Coldplay", "Coldplay", "Radiohead", "Muse", "Muse", "Coldplay", "Muse", "Ghost"],
        "play_count": [10, 5, 15, 20, -1, 30, 0, 50, np.nan],
    })


@pytest.fixture
def sample_k_core_interactions():
    """Create interaction data designed for k-core testing (k=2)."""
    # u1 listens to a1, a2, a3
    # u2 listens to a1, a2
    # u3 listens to a1, a3
    # u4 listens to a4 only (should be dropped)
    # a4 has only u4 (should be dropped)
    return pd.DataFrame({
        "user_id": ["u1", "u1", "u1", "u2", "u2", "u3", "u3", "u4"],
        "artist_name": ["a1", "a2", "a3", "a1", "a2", "a1", "a3", "a4"],
        "play_count": [10, 20, 30, 40, 50, 60, 70, 80],
    })


def test_clean_interactions(sample_raw_interactions):
    cleaned = clean_interactions(sample_raw_interactions)

    # Missing rows, negative/zero play counts, and NaN should be dropped
    assert None not in cleaned["user_id"].values
    assert (cleaned["play_count"] <= 0).sum() == 0
    assert cleaned["play_count"].isna().sum() == 0

    # Artist names stripped and lowercased
    assert "radiohead" in cleaned["artist_name"].values
    assert " Radiohead " not in cleaned["artist_name"].values

    # u1 + coldplay should be aggregated (5 + 15 = 20)
    u1_coldplay = cleaned[(cleaned["user_id"] == "u1") & (cleaned["artist_name"] == "coldplay")]
    assert len(u1_coldplay) == 1
    assert u1_coldplay["play_count"].iloc[0] == 20


def test_filter_k_core(sample_k_core_interactions):
    filtered = filter_k_core(sample_k_core_interactions, min_user_interactions=2, min_artist_interactions=2)

    # u4 and a4 should be filtered out
    assert "u4" not in filtered["user_id"].values
    assert "a4" not in filtered["artist_name"].values

    # All remaining users have >= 2 items
    user_counts = filtered.groupby("user_id")["artist_name"].nunique()
    assert (user_counts >= 2).all()

    # All remaining artists have >= 2 users
    artist_counts = filtered.groupby("artist_name")["user_id"].nunique()
    assert (artist_counts >= 2).all()


def test_create_and_apply_mappings():
    df = pd.DataFrame({
        "user_id": ["user_b", "user_a", "user_c"],
        "artist_name": ["artist_z", "artist_y", "artist_x"],
        "play_count": [10.0, 20.0, 30.0],
    })

    indexed_df, user_meta, artist_meta = create_and_apply_mappings(df)

    # Indices must be contiguous integers from 0 to N-1
    assert set(indexed_df["user_idx"].values) == {0, 1, 2}
    assert set(indexed_df["artist_idx"].values) == {0, 1, 2}

    # Lexicographical ordering checks
    assert user_meta["idx2user"] == ["user_a", "user_b", "user_c"]
    assert artist_meta["idx2artist"] == ["artist_x", "artist_y", "artist_z"]

    # Mapping round-trip
    for u, idx in user_meta["user2idx"].items():
        assert user_meta["idx2user"][idx] == u
    for a, idx in artist_meta["artist2idx"].items():
        assert artist_meta["idx2artist"][idx] == a


def test_split_leave_one_out():
    # 2 users, each with 5 interactions
    users = ["u0"] * 5 + ["u1"] * 6
    artists = [0, 1, 2, 3, 4] + [0, 1, 2, 3, 4, 5]
    df = pd.DataFrame({
        "user_idx": users,
        "artist_idx": artists,
        "play_count": [10.0] * 11,
    })

    train, val, test = split_leave_one_out(df, seed=42)

    # Every user must have exactly 1 in test and 1 in val
    assert len(test) == 2
    assert len(val) == 2
    assert len(train) == 11 - 2 - 2

    test_users = test["user_idx"].value_counts()
    val_users = val["user_idx"].value_counts()
    assert (test_users == 1).all()
    assert (val_users == 1).all()

    # Total rows preserved
    assert len(train) + len(val) + len(test) == len(df)


def test_build_sparse_matrix():
    df = pd.DataFrame({
        "user_idx": [0, 0, 1],
        "artist_idx": [1, 2, 0],
        "play_count": [15.0, 25.0, 50.0],
    })

    mat = build_sparse_matrix(df, n_users=2, n_artists=3)

    assert isinstance(mat, csr_matrix)
    assert mat.shape == (2, 3)
    assert mat[0, 1] == 15.0
    assert mat[0, 2] == 25.0
    assert mat[1, 0] == 50.0
    assert mat[0, 0] == 0.0
