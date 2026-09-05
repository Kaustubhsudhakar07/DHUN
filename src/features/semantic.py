"""
Multilingual Semantic Music Engine for DhunDNA (Phase 3 & 4)

Provides:
1. Classical Baseline: Multilingual Word + Char N-Gram TF-IDF with Cosine Similarity.
2. Neural Semantic Model: Sentence-Transformers ('paraphrase-multilingual-MiniLM-L12-v2').
3. Natural language semantic query retrieval:
   e.g. '90s romantic Hindi songs about heartbreak and longing'
4. Song-to-song semantic similarity.

Strictly complies with lyrics copyright requirements by building semantic document
representations from legal metadata, thematic descriptors, mood keywords, and genres
without redistributing proprietary lyrics text.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("dhundna.semantic")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DEFAULT_MULTILINGUAL_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def build_track_semantic_text(row: Union[pd.Series, Dict[str, Any]]) -> str:
    """Construct a rich semantic descriptor from permissible metadata, language, and era.

    Complies with copyright rules: combines title, artist, genre, album, language, era,
    and thematic genre expansions into a clean semantic document.
    """
    title = str(row.get("track_name", "")).strip()
    artist = str(row.get("artist_name", "")).strip()
    all_artists = str(row.get("all_artists", "")).strip()
    album = str(row.get("album_name", "")).strip()
    genre = str(row.get("genre", "")).strip()
    language = str(row.get("language", "")).strip()
    era = str(row.get("era", "")).strip()

    # Thematic expansions based on genre & language
    thematic_cues = []
    if "bollywood" in genre.lower() or language.lower() == "hindi":
        thematic_cues.extend(["hindi film song", "bollywood music", "indian melody", "desi"])
    if "punjabi" in genre.lower() or language.lower() == "punjabi":
        thematic_cues.extend(["punjabi track", "bhangra", "desi beats"])
    if "romantic" in title.lower() or "dil" in title.lower() or "pyar" in title.lower() or "ishq" in title.lower():
        thematic_cues.extend(["love", "romance", "romantic", "affection", "heart"])
    if "sad" in genre.lower() or "judaai" in title.lower() or "dard" in title.lower() or "alvida" in title.lower():
        thematic_cues.extend(["sad", "breakup", "heartbreak", "longing", "separation"])

    doc_parts = [
        title,
        f"by {artist}" if artist else "",
        f"album {album}" if album and album != "Bollywood Classic" else "",
        f"genre {genre}" if genre else "",
        f"language {language}" if language else "",
        f"era {era}" if era and era != "Unknown" else "",
        " ".join(thematic_cues),
    ]
    return " • ".join(p for p in doc_parts if p).lower()


class SemanticMusicEngine:
    """Hybrid Semantic Search Engine supporting both Neural Embeddings and Classical TF-IDF."""

    def __init__(
        self,
        use_neural: bool = True,
        model_name: str = DEFAULT_MULTILINGUAL_MODEL,
        max_features: int = 15000,
    ):
        self.use_neural = use_neural
        self.model_name = model_name
        self.max_features = max_features

        self.neural_model = None
        self.embeddings: Optional[np.ndarray] = None

        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

        self.catalog_df: Optional[pd.DataFrame] = None
        self.track_id_to_idx: Dict[str, int] = {}
        self.is_fitted = False

    def _init_neural_model(self):
        """Lazy load sentence transformer."""
        if self.neural_model is None and self.use_neural:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading multilingual neural model '%s'...", self.model_name)
                self.neural_model = SentenceTransformer(self.model_name)
                logger.info("Neural model loaded successfully.")
            except Exception as e:
                logger.warning("Could not load SentenceTransformer (%s). Falling back to TF-IDF.", e)
                self.use_neural = False

    def fit(self, catalog_df: pd.DataFrame, max_tracks: int = 25000):
        """Fit semantic representations on the master catalog.

        Args:
            catalog_df: Master catalog DataFrame.
            max_tracks: Max tracks to index for high-speed retrieval.
        """
        logger.info("Fitting SemanticMusicEngine on %d tracks (sampled from %d)...", min(len(catalog_df), max_tracks), len(catalog_df))

        # Prioritize Indian tracks and popular global tracks
        if len(catalog_df) > max_tracks:
            indian_mask = catalog_df["language"].isin(["Hindi", "Punjabi", "Tamil", "Telugu"])
            indian_subset = catalog_df[indian_mask]
            global_subset = catalog_df[~indian_mask].sort_values("popularity", ascending=False)
            needed_global = max_tracks - len(indian_subset)
            sampled_df = pd.concat([indian_subset, global_subset.head(needed_global)]).reset_index(drop=True)
        else:
            sampled_df = catalog_df.reset_index(drop=True)

        self.catalog_df = sampled_df
        self.track_id_to_idx = {str(tid): idx for idx, tid in enumerate(sampled_df["track_id"])}

        # Build semantic document texts
        docs = [build_track_semantic_text(row) for _, row in sampled_df.iterrows()]

        # 1. Classical Multilingual TF-IDF Baseline (word + char ngrams)
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=self.max_features,
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(docs)
        logger.info("TF-IDF semantic matrix shape: %s.", self.tfidf_matrix.shape)

        # 2. Neural Embeddings (if enabled)
        if self.use_neural:
            self._init_neural_model()
            if self.neural_model is not None:
                logger.info("Encoding %d track descriptions with %s...", len(docs), self.model_name)
                self.embeddings = self.neural_model.encode(
                    docs,
                    batch_size=128,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                logger.info("Neural embeddings shape: %s.", self.embeddings.shape)

        self.is_fitted = True
        return self

    def search(
        self,
        query: str,
        top_k: int = 10,
        language_filter: Optional[str] = None,
        era_filter: Optional[str] = None,
        use_neural: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Search tracks by natural language semantic query."""
        if not self.is_fitted or self.catalog_df is None:
            raise RuntimeError("SemanticMusicEngine is not fitted.")

        use_nn = self.use_neural if use_neural is None else use_neural

        # Score computation
        if use_nn and self.neural_model is not None and self.embeddings is not None:
            q_emb = self.neural_model.encode([query], normalize_embeddings=True)
            sims = np.dot(self.embeddings, q_emb.T).ravel()
            model_used = "neural_multilingual"
        else:
            q_vec = self.tfidf_vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self.tfidf_matrix).ravel()
            model_used = "tfidf_baseline"

        # Apply optional filters
        filtered_indices = np.arange(len(self.catalog_df))
        if language_filter:
            mask = self.catalog_df["language"].str.lower() == language_filter.lower()
            sims[~mask] = -1.0
        if era_filter:
            mask = self.catalog_df["era"].str.lower() == era_filter.lower()
            sims[~mask] = -1.0

        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(sims[idx])
            if score <= 0.0:
                continue
            row = self.catalog_df.iloc[idx]
            results.append({
                "track_id": row["track_id"],
                "track_name": row["track_name"],
                "artist_name": row["artist_name"],
                "album_name": row["album_name"],
                "release_year": int(row["release_year"]),
                "era": row["era"],
                "language": row["language"],
                "genre": row["genre"],
                "popularity": float(row["popularity"]),
                "score": round(score, 4),
                "model_used": model_used,
            })

        return results

    def get_similar_tracks(
        self,
        track_id: str,
        top_k: int = 10,
        use_neural: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve most semantically similar tracks to a given track ID."""
        if str(track_id) not in self.track_id_to_idx:
            return []

        target_idx = self.track_id_to_idx[str(track_id)]
        use_nn = self.use_neural if use_neural is None else use_neural

        if use_nn and self.embeddings is not None:
            target_vec = self.embeddings[target_idx].reshape(1, -1)
            sims = np.dot(self.embeddings, target_vec.T).ravel()
        else:
            target_vec = self.tfidf_matrix[target_idx]
            sims = cosine_similarity(target_vec, self.tfidf_matrix).ravel()

        # Exclude self
        sims[target_idx] = -1.0
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(sims[idx])
            row = self.catalog_df.iloc[idx]
            results.append({
                "track_id": row["track_id"],
                "track_name": row["track_name"],
                "artist_name": row["artist_name"],
                "album_name": row["album_name"],
                "release_year": int(row["release_year"]),
                "era": row["era"],
                "language": row["language"],
                "genre": row["genre"],
                "popularity": float(row["popularity"]),
                "score": round(score, 4),
            })

        return results
