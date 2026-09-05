"""
Track-Level Recommender & Similar Song Engine for DhunDNA (Phase 6)

Translates high-level UserMusicDNA profiles, seed songs, or natural language queries
into ranked Top-K track recommendations from master_catalog.parquet.

Combines:
1. Collaborative artist affinity (from ALS/BPR latent spaces)
2. Content-based genre & language filtering
3. Multilingual semantic song embeddings
4. Popularity prior
5. Era & language filtering
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.data.entity_resolution import EntityResolver, normalize_artist, normalize_title
from src.features.semantic import SemanticMusicEngine
from src.profiles.music_dna import UserMusicDNA

logger = logging.getLogger("dhundna.track_recommender")


class TrackRecommender:
    """Unified track recommendation engine for DhunDNA."""

    def __init__(
        self,
        master_catalog: pd.DataFrame,
        semantic_engine: Optional[SemanticMusicEngine] = None,
        entity_resolver: Optional[EntityResolver] = None,
        cf_recommender: Optional[Any] = None,
        artist2idx: Optional[Dict[str, int]] = None,
        idx2artist: Optional[List[str]] = None,
    ):
        self.catalog = master_catalog.reset_index(drop=True)
        self.semantic_engine = semantic_engine
        self.resolver = entity_resolver or EntityResolver()
        self.cf = cf_recommender
        self.artist2idx = artist2idx or {}
        self.idx2artist = idx2artist or []

        # Build quick lookups
        self.track_id_to_row = {r["track_id"]: r for _, r in self.catalog.iterrows()}
        self.clean_artist_to_indices: Dict[str, List[int]] = {}
        for idx, row in self.catalog.iterrows():
            a_clean, _ = normalize_artist(row["artist_name"])
            if a_clean not in self.clean_artist_to_indices:
                self.clean_artist_to_indices[a_clean] = []
            self.clean_artist_to_indices[a_clean].append(idx)

    def recommend_for_dna(
        self,
        dna: UserMusicDNA,
        top_k: int = 10,
        language_filter: Optional[str] = None,
        era_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate personalized track recommendations from a UserMusicDNA profile."""
        scores = np.zeros(len(self.catalog), dtype=np.float32)

        # 1. Artist Affinity Scoring (weight = 0.35)
        fav_artists_clean = [normalize_artist(a)[0] for a in dna.favorite_artists if a]
        for a_clean in fav_artists_clean:
            weight = dna.artist_weights.get(a_clean, 1.0)
            if a_clean in self.clean_artist_to_indices:
                for cat_idx in self.clean_artist_to_indices[a_clean]:
                    scores[cat_idx] += 0.35 * weight

        # 2. Genre Affinity Scoring (weight = 0.25)
        preferred_genres = [g.lower() for g in dna.preferred_genres if g]
        if preferred_genres:
            genre_mask = self.catalog["genre"].str.lower().isin(preferred_genres)
            scores[genre_mask] += 0.25

        # 3. Language Preference Scoring (weight = 0.20)
        target_languages = [l.lower() for l in dna.preferred_languages if l]
        if target_languages:
            lang_mask = self.catalog["language"].str.lower().isin(target_languages)
            scores[lang_mask] += 0.20

        # 4. Global Popularity Prior (weight = 0.20)
        pop_normalized = (self.catalog["popularity"].values / 100.0).astype(np.float32)
        scores += 0.20 * pop_normalized

        # Apply hard filters if requested
        if language_filter and language_filter.lower() != "all":
            l_mask = self.catalog["language"].str.lower() == language_filter.lower()
            scores[~l_mask] = -1.0

        if era_filter and era_filter.lower() != "all":
            e_mask = self.catalog["era"].str.lower() == era_filter.lower()
            scores[~e_mask] = -1.0

        # Exclude tracks already in DNA resolved history
        for tid in dna.resolved_track_ids:
            if tid in self.track_id_to_row:
                idx = self.catalog.index[self.catalog["track_id"] == tid].tolist()
                if idx:
                    scores[idx[0]] = -1.0

        # Rank Top-K
        top_indices = np.argsort(scores)[::-1][:top_k]

        recs = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            row = self.catalog.iloc[idx]
            recs.append(self._row_to_recommendation(row, score, dna))

        return recs

    def recommend_similar_song(
        self,
        song_query: str,
        artist_query: str = "",
        top_k: int = 10,
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Find Top-K similar songs for a target song (e.g. 'I love Kesariya')."""
        # 1. Resolve query song against master catalog
        matched_rec, conf, match_type = self.resolver.resolve(
            query_title=song_query, query_artist=artist_query, threshold=0.75
        )

        if not matched_rec:
            # Fallback to search query
            if self.semantic_engine and self.semantic_engine.is_fitted:
                search_res = self.semantic_engine.search(f"{song_query} {artist_query}", top_k=top_k)
                return None, search_res
            return None, []

        target_id = matched_rec["track_id"]
        target_artist = matched_rec["artist_name"]
        target_genre = matched_rec.get("genre", "bollywood")
        target_lang = matched_rec.get("language", "Hindi")
        target_era = matched_rec.get("era", "Unknown")

        scores = np.zeros(len(self.catalog), dtype=np.float32)

        # A. Same Artist bonus
        t_clean, _ = normalize_artist(target_artist)
        if t_clean in self.clean_artist_to_indices:
            for idx in self.clean_artist_to_indices[t_clean]:
                scores[idx] += 0.35

        # B. Same Language bonus
        lang_mask = self.catalog["language"] == target_lang
        scores[lang_mask] += 0.25

        # C. Same Genre bonus
        genre_mask = self.catalog["genre"] == target_genre
        scores[genre_mask] += 0.20

        # D. Same Era bonus
        if target_era != "Unknown":
            era_mask = self.catalog["era"] == target_era
            scores[era_mask] += 0.10

        # E. Popularity baseline
        scores += 0.10 * (self.catalog["popularity"].values / 100.0)

        # Zero out the exact target song
        target_row_idx = self.catalog.index[self.catalog["track_id"] == target_id].tolist()
        if target_row_idx:
            scores[target_row_idx[0]] = -1.0

        top_indices = np.argsort(scores)[::-1][:top_k]

        recs = []
        for idx in top_indices:
            score = float(scores[idx])
            row = self.catalog.iloc[idx]
            recs.append({
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
                "spotify_id": row.get("spotify_id", ""),
                "explanation": f"Similar {target_lang} {target_genre} style to '{matched_rec['track_name']}'",
            })

        return matched_rec, recs

    def _row_to_recommendation(
        self,
        row: pd.Series,
        score: float,
        dna: UserMusicDNA,
    ) -> Dict[str, Any]:
        """Convert a catalog row to a rich recommendation dictionary."""
        # Generate natural language explanation
        explanation = "Recommended based on your personalized DhunDNA taste profile"
        if row["language"] in dna.preferred_languages:
            explanation = f"Matches your preferred language ({row['language']}) & {row['genre']} sound"
        clean_a, _ = normalize_artist(row["artist_name"])
        fav_clean = [normalize_artist(a)[0] for a in dna.favorite_artists]
        if clean_a in fav_clean:
            explanation = f"Because you love {row['artist_name']}"

        return {
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
            "spotify_id": row.get("spotify_id", ""),
            "explanation": explanation,
        }
