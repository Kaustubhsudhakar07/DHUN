"""
Dataset loading utilities.

Provides functions to load and validate the raw datasets:
- Last.fm 360K (user-artist play counts)
- HetRec 2011 (artist tags, social relations)

All loaders return Pandas DataFrames with consistent column naming.
"""

from pathlib import Path

import pandas as pd

from src.config import (
    HETREC_ARTISTS,
    HETREC_TAGS,
    HETREC_USER_ARTISTS,
    HETREC_USER_TAGGED_ARTISTS,
    HETREC_USER_FRIENDS,
    LASTFM_360K_INTERACTIONS,
    LASTFM_360K_PROFILES,
)


# =========================================================================
# Last.fm 360K Dataset
# =========================================================================

def load_lastfm_360k_interactions(filepath: Path | None = None) -> pd.DataFrame:
    """Load Last.fm 360K user-artist play count data.

    File format: TSV with columns
        user_id (sha1) | artist_mbid | artist_name | play_count

    Args:
        filepath: Path to the TSV file. Defaults to config path.

    Returns:
        DataFrame with columns: [user_id, artist_mbid, artist_name, play_count]

    Raises:
        FileNotFoundError: If the data file doesn't exist.
    """
    filepath = filepath or LASTFM_360K_INTERACTIONS
    if not filepath.exists():
        raise FileNotFoundError(
            f"Last.fm 360K interactions file not found: {filepath}\n"
            "Run: python scripts/download_datasets.py"
        )

    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=["user_id", "artist_mbid", "artist_name", "play_count"],
        dtype={
            "user_id": str,
            "artist_mbid": str,
            "artist_name": str,
        },
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace",
    )

    # play_count should be numeric; coerce errors to NaN
    df["play_count"] = pd.to_numeric(df["play_count"], errors="coerce")

    return df


def load_lastfm_360k_profiles(filepath: Path | None = None) -> pd.DataFrame:
    """Load Last.fm 360K user profile data.

    File format: TSV with columns
        user_id (sha1) | gender | age | country | signup_date

    Args:
        filepath: Path to the TSV file. Defaults to config path.

    Returns:
        DataFrame with columns: [user_id, gender, age, country, signup_date]

    Raises:
        FileNotFoundError: If the data file doesn't exist.
    """
    filepath = filepath or LASTFM_360K_PROFILES
    if not filepath.exists():
        raise FileNotFoundError(
            f"Last.fm 360K profiles file not found: {filepath}\n"
            "Run: python scripts/download_datasets.py"
        )

    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=["user_id", "gender", "age", "country", "signup_date"],
        dtype={
            "user_id": str,
            "gender": str,
            "country": str,
            "signup_date": str,
        },
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace",
    )

    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    return df


# =========================================================================
# HetRec 2011 Dataset
# =========================================================================

def load_hetrec_artists(filepath: Path | None = None) -> pd.DataFrame:
    """Load HetRec 2011 artist data.

    File format: TAB-separated with header
        id | name | url | pictureURL

    Args:
        filepath: Path to artists.dat. Defaults to config path.

    Returns:
        DataFrame with columns: [artist_id, name, url, picture_url]
    """
    filepath = filepath or HETREC_ARTISTS
    if not filepath.exists():
        raise FileNotFoundError(
            f"HetRec artists file not found: {filepath}\n"
            "Run: python scripts/download_datasets.py"
        )

    df = pd.read_csv(
        filepath,
        sep="\t",
        encoding="utf-8",
        encoding_errors="replace",
    )

    # Normalize column names
    df.columns = ["artist_id", "name", "url", "picture_url"]

    return df


def load_hetrec_tags(filepath: Path | None = None) -> pd.DataFrame:
    """Load HetRec 2011 tag data.

    File format: TAB-separated with header
        tagID | tagValue

    Args:
        filepath: Path to tags.dat. Defaults to config path.

    Returns:
        DataFrame with columns: [tag_id, tag_value]
    """
    filepath = filepath or HETREC_TAGS
    if not filepath.exists():
        raise FileNotFoundError(
            f"HetRec tags file not found: {filepath}\n"
            "Run: python scripts/download_datasets.py"
        )

    df = pd.read_csv(
        filepath,
        sep="\t",
        encoding="utf-8",
        encoding_errors="replace",
    )

    df.columns = ["tag_id", "tag_value"]

    return df


def load_hetrec_user_artists(filepath: Path | None = None) -> pd.DataFrame:
    """Load HetRec 2011 user-artist listening data.

    File format: TAB-separated with header
        userID | artistID | weight

    Args:
        filepath: Path to user_artists.dat. Defaults to config path.

    Returns:
        DataFrame with columns: [user_id, artist_id, weight]
    """
    filepath = filepath or HETREC_USER_ARTISTS
    if not filepath.exists():
        raise FileNotFoundError(
            f"HetRec user_artists file not found: {filepath}\n"
            "Run: python scripts/download_datasets.py"
        )

    df = pd.read_csv(
        filepath,
        sep="\t",
        encoding="utf-8",
        encoding_errors="replace",
    )

    df.columns = ["user_id", "artist_id", "weight"]

    return df


def load_hetrec_user_tagged_artists(
    filepath: Path | None = None,
) -> pd.DataFrame:
    """Load HetRec 2011 user-tag-artist assignment data.

    File format: TAB-separated with header
        userID | artistID | tagID | day | month | year

    Args:
        filepath: Path to user_taggedartists.dat. Defaults to config path.

    Returns:
        DataFrame with columns:
            [user_id, artist_id, tag_id, day, month, year]
    """
    filepath = filepath or HETREC_USER_TAGGED_ARTISTS
    if not filepath.exists():
        raise FileNotFoundError(
            f"HetRec user_taggedartists file not found: {filepath}\n"
            "Run: python scripts/download_datasets.py"
        )

    df = pd.read_csv(
        filepath,
        sep="\t",
        encoding="utf-8",
        encoding_errors="replace",
    )

    df.columns = ["user_id", "artist_id", "tag_id", "day", "month", "year"]

    return df


def load_hetrec_user_friends(filepath: Path | None = None) -> pd.DataFrame:
    """Load HetRec 2011 social relations data.

    File format: TAB-separated with header
        userID | friendID

    Args:
        filepath: Path to user_friends.dat. Defaults to config path.

    Returns:
        DataFrame with columns: [user_id, friend_id]
    """
    filepath = filepath or HETREC_USER_FRIENDS
    if not filepath.exists():
        raise FileNotFoundError(
            f"HetRec user_friends file not found: {filepath}\n"
            "Run: python scripts/download_datasets.py"
        )

    df = pd.read_csv(
        filepath,
        sep="\t",
        encoding="utf-8",
        encoding_errors="replace",
    )

    df.columns = ["user_id", "friend_id"]

    return df


# =========================================================================
# Convenience functions
# =========================================================================

def get_artist_tags(
    tagged_artists_df: pd.DataFrame | None = None,
    tags_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join tagged artists with tag values to get artist-tag pairs.

    Args:
        tagged_artists_df: User-tagged-artist DataFrame.
            If None, loads from disk.
        tags_df: Tags DataFrame. If None, loads from disk.

    Returns:
        DataFrame with columns: [artist_id, tag_id, tag_value, tag_count]
        where tag_count is how many users assigned that tag to the artist.
    """
    if tagged_artists_df is None:
        tagged_artists_df = load_hetrec_user_tagged_artists()
    if tags_df is None:
        tags_df = load_hetrec_tags()

    # Count how many users assigned each tag to each artist
    tag_counts = (
        tagged_artists_df.groupby(["artist_id", "tag_id"])
        .size()
        .reset_index(name="tag_count")
    )

    # Join with tag names
    artist_tags = tag_counts.merge(tags_df, on="tag_id", how="left")

    return artist_tags.sort_values(
        ["artist_id", "tag_count"], ascending=[True, False]
    )


def print_dataset_summary():
    """Print a summary of all loaded datasets.

    Useful for quick validation after download.
    """
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    # Last.fm 360K
    try:
        interactions = load_lastfm_360k_interactions()
        print(f"\n--- Last.fm 360K Interactions ---")
        print(f"  Rows:            {len(interactions):,}")
        print(f"  Unique users:    {interactions['user_id'].nunique():,}")
        print(f"  Unique artists:  {interactions['artist_name'].nunique():,}")
        print(f"  Artists w/ MBID: {interactions['artist_mbid'].notna().sum():,}")
        print(f"  Play count range: {interactions['play_count'].min():.0f} – {interactions['play_count'].max():.0f}")
        print(f"  Play count mean:  {interactions['play_count'].mean():.1f}")
        print(f"  Memory usage:     {interactions.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    except FileNotFoundError as e:
        print(f"\n  [SKIP] {e}")

    # Last.fm 360K profiles
    try:
        profiles = load_lastfm_360k_profiles()
        print(f"\n--- Last.fm 360K Profiles ---")
        print(f"  Rows:        {len(profiles):,}")
        print(f"  Gender dist: {profiles['gender'].value_counts().to_dict()}")
        print(f"  Age range:   {profiles['age'].min():.0f} – {profiles['age'].max():.0f}")
        print(f"  Countries:   {profiles['country'].nunique()}")
    except FileNotFoundError as e:
        print(f"\n  [SKIP] {e}")

    # HetRec 2011
    try:
        artists = load_hetrec_artists()
        tags = load_hetrec_tags()
        user_artists = load_hetrec_user_artists()
        user_tagged = load_hetrec_user_tagged_artists()
        friends = load_hetrec_user_friends()

        print(f"\n--- HetRec 2011 ---")
        print(f"  Artists:          {len(artists):,}")
        print(f"  Tags:             {len(tags):,}")
        print(f"  User-artist rows: {len(user_artists):,}")
        print(f"  Tag assignments:  {len(user_tagged):,}")
        print(f"  Social relations: {len(friends):,}")
        print(f"  Unique users:     {user_artists['user_id'].nunique():,}")

        # Artist-tag coverage
        artist_tags = get_artist_tags(user_tagged, tags)
        tagged_artists = artist_tags["artist_id"].nunique()
        print(f"  Artists with tags: {tagged_artists:,} / {len(artists):,}")
    except FileNotFoundError as e:
        print(f"\n  [SKIP] {e}")

    print("\n" + "=" * 60)
