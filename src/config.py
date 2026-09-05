"""
Centralized configuration and path management.

All paths, constants, and environment variables are managed here.
Other modules import from this file instead of hardcoding paths.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from .env file
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# Root of the music-recommender project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Model artifacts
MODELS_DIR = PROJECT_ROOT / "models"

# Notebooks
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ---------------------------------------------------------------------------
# Dataset file paths (populated after download in Phase 3)
# ---------------------------------------------------------------------------
# Last.fm 360K
LASTFM_360K_INTERACTIONS = RAW_DATA_DIR / "lastfm-dataset-360K" / "usersha1-artmbid-artname-plays.tsv"
LASTFM_360K_PROFILES = RAW_DATA_DIR / "lastfm-dataset-360K" / "usersha1-profile.tsv"

# HetRec 2011
HETREC_ARTISTS = RAW_DATA_DIR / "hetrec2011-lastfm-2k" / "artists.dat"
HETREC_TAGS = RAW_DATA_DIR / "hetrec2011-lastfm-2k" / "tags.dat"
HETREC_USER_ARTISTS = RAW_DATA_DIR / "hetrec2011-lastfm-2k" / "user_artists.dat"
HETREC_USER_TAGGED_ARTISTS = RAW_DATA_DIR / "hetrec2011-lastfm-2k" / "user_taggedartists.dat"
HETREC_USER_FRIENDS = RAW_DATA_DIR / "hetrec2011-lastfm-2k" / "user_friends.dat"

# ---------------------------------------------------------------------------
# Processed data paths
# ---------------------------------------------------------------------------
PROCESSED_INTERACTIONS = PROCESSED_DATA_DIR / "interactions.parquet"
PROCESSED_ARTIST_FEATURES = PROCESSED_DATA_DIR / "artist_features.parquet"
TRAIN_INTERACTIONS = PROCESSED_DATA_DIR / "train_interactions.npz"
VAL_INTERACTIONS = PROCESSED_DATA_DIR / "val_interactions.npz"
TEST_INTERACTIONS = PROCESSED_DATA_DIR / "test_interactions.npz"
USER_MAPPING = PROCESSED_DATA_DIR / "user_mapping.json"
ARTIST_MAPPING = PROCESSED_DATA_DIR / "artist_mapping.json"
MASTER_CATALOG = PROCESSED_DATA_DIR / "master_catalog.parquet"
CATALOG_SUMMARY = PROCESSED_DATA_DIR / "catalog_summary.json"

# Configure PyArrow memory pool on Windows to prevent allocator errors
try:
    import pyarrow as pa
    pa.set_memory_pool(pa.system_memory_pool())
except Exception:
    pass


# External datasets (DhunDNA Multi-Source Catalog)
SPOTIFY_TRACKS_114K = EXTERNAL_DATA_DIR / "spotify_tracks_114k.csv"
BOLLYWOOD_SONGS_10K = EXTERNAL_DATA_DIR / "bollywood_songs_10k.csv"

# ---------------------------------------------------------------------------
# Model artifact paths
# ---------------------------------------------------------------------------
ALS_MODEL = MODELS_DIR / "als_model.npz"
BPR_MODEL = MODELS_DIR / "bpr_model.npz"
TFIDF_VECTORIZER = MODELS_DIR / "tfidf_vectorizer.pkl"
TFIDF_MATRIX = MODELS_DIR / "tfidf_matrix.npz"
POPULARITY_SCORES = MODELS_DIR / "popularity_scores.json"
HYBRID_WEIGHTS = MODELS_DIR / "hybrid_weights.json"
EVALUATION_RESULTS = MODELS_DIR / "evaluation_results.json"
HYBRID_TUNING_RESULTS = MODELS_DIR / "hybrid_tuning_results.json"

# ---------------------------------------------------------------------------
# Last.fm API configuration
# ---------------------------------------------------------------------------
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY", "")
LASTFM_API_SECRET = os.getenv("LASTFM_API_SECRET", "")
LASTFM_API_BASE_URL = "https://ws.audioscrobbler.com/2.0/"

# Rate limiting
LASTFM_RATE_LIMIT_CALLS = 5  # Max calls per second
LASTFM_REQUEST_TIMEOUT = 10  # Seconds
LASTFM_MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Model hyperparameter defaults
# ---------------------------------------------------------------------------
# These are starting values; actual values will be tuned during experiments.

ALS_DEFAULTS = {
    "factors": 64,
    "regularization": 0.01,
    "iterations": 15,
    "use_gpu": False,
}

BPR_DEFAULTS = {
    "factors": 64,
    "learning_rate": 0.01,
    "regularization": 0.01,
    "iterations": 100,
    "use_gpu": False,
}

# ---------------------------------------------------------------------------
# Recommendation defaults
# ---------------------------------------------------------------------------
DEFAULT_TOP_K = 10
CANDIDATE_POOL_SIZE = 200

# Hybrid weights (initial; tuned in Phase 9)
DEFAULT_HYBRID_WEIGHTS = {
    "collaborative": 0.6,
    "content": 0.3,
    "popularity": 0.1,
}

# ---------------------------------------------------------------------------
# Preprocessing thresholds (initial; justified in Phase 5)
# ---------------------------------------------------------------------------
MIN_USER_INTERACTIONS = 5    # Users with fewer interactions are excluded
MIN_ARTIST_INTERACTIONS = 5  # Artists with fewer listeners are excluded

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_K_VALUES = [5, 10, 20, 50]

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
FASTAPI_HOST = "127.0.0.1"
FASTAPI_PORT = 8000
STREAMLIT_PORT = 8501
