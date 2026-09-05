"""
Master Music Catalog Builder for DhunDNA (Phase 2)

Constructs the unified master track catalog by synthesizing:
1. Spotify Tracks 114k (114,000 global + Indian tracks across 114 genres)
2. Bollywood Cinema Songs 10k (10,827 classic Hindi cinema tracks, 1960s–2009)
3. Existing Last.fm 360K artist index cross-linkage (98,104 artists)

Applies entity resolution to merge duplicates, infers languages and release eras,
and outputs a query-optimized master_catalog.parquet.
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import pandas as pd

from src.config import (
    ARTIST_MAPPING,
    BOLLYWOOD_SONGS_10K,
    MASTER_CATALOG,
    PROCESSED_DATA_DIR,
    SPOTIFY_TRACKS_114K,
)
from src.data.entity_resolution import normalize_artist, normalize_title

logger = logging.getLogger("dhundna.catalog")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def infer_era(year: int) -> str:
    """Categorize integer release year into standard musical eras."""
    if year <= 0:
        return "Unknown"
    elif year < 1990:
        return "pre-1990"
    elif 1990 <= year <= 1999:
        return "1990s"
    elif 2000 <= year <= 2009:
        return "2000s"
    elif 2010 <= year <= 2019:
        return "2010s"
    elif year >= 2020:
        return "2020s"
    return "Unknown"


def infer_language_and_genre(row: pd.Series, source: str) -> tuple[str, str]:
    """Classify language and musical genre from artist names, film names, and tags."""
    if source == "bollywood":
        return "Hindi", "bollywood"

    genre = str(row.get("track_genre", "")).lower()
    artists_str = str(row.get("artists", "")).lower()
    track_str = str(row.get("track_name", "")).lower()

    # Regional Indian language detection patterns
    if "punjabi" in artists_str or "moose wala" in artists_str or "diljit" in artists_str or "ap dhillon" in artists_str:
        return "Punjabi", "punjabi"
    if "anirudh" in artists_str or "ilayaraja" in artists_str or "tamil" in genre:
        return "Tamil", "kollywood"
    if "telugu" in genre or "devi sri prasad" in artists_str or "keeravani" in artists_str:
        return "Telugu", "tollywood"

    # Bollywood / Hindi indicators
    indian_keywords = [
        "arijit singh", "pritam", "shreya ghoshal", "rahman", "lata mangeshkar",
        "kishore kumar", "kumar sanu", "alka yagnik", "udit narayan", "sonu nigam",
        "atif aslam", "sunidhi chauhan", "badshah", "neha kakkar", "vishal-shekhar",
        "shankar-ehsaan-loy", "amitabh bhattacharya", "sachin-jigar", "mohit chauhan",
        "kailash kher", "prateek kuhad", "ritviz", "the local train", "anuv jain",
    ]
    if genre == "indian" or any(k in artists_str for k in indian_keywords):
        return "Hindi", "bollywood"

    # Global genres
    return "English / Global", genre if genre else "pop"


def extract_year_from_album_or_str(val: str, default: int = 0) -> int:
    """Attempt to extract 4-digit release year from album string."""
    if not isinstance(val, str):
        return default
    matches = re.findall(r"\b(19\d\d|20[0-2]\d)\b", val)
    if matches:
        return int(matches[-1])
    return default


def build_master_catalog() -> pd.DataFrame:
    """Build, deduplicate, and persist the unified DhunDNA master catalog."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    # 1. Load artist mapping for catalog cross-linking
    artist2idx = {}
    if ARTIST_MAPPING.exists():
        with open(ARTIST_MAPPING, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
            artist2idx = mapping_data.get("artist2idx", {})
        logger.info("Loaded artist mapping with %d artists for cross-linking.", len(artist2idx))

    # 2. Ingest Bollywood Cinema Songs
    if BOLLYWOOD_SONGS_10K.exists():
        logger.info("Ingesting Bollywood dataset from %s...", BOLLYWOOD_SONGS_10K.name)
        df_bolly = pd.read_csv(
            BOLLYWOOD_SONGS_10K,
            usecols=["Title", "Film", "Year", "Singer", "Composer", "Lyricist"],
            encoding="utf-8",
        )
        df_bolly = df_bolly.dropna(subset=["Title"]).copy()

        for idx, row in df_bolly.iterrows():
            title = str(row["Title"]).strip()
            film = str(row["Film"]).strip() if pd.notna(row["Film"]) else ""
            singer = str(row["Singer"]).strip() if pd.notna(row["Singer"]) else ""
            composer = str(row["Composer"]).strip() if pd.notna(row["Composer"]) else ""
            lyricist = str(row["Lyricist"]).strip() if pd.notna(row["Lyricist"]) else ""

            # Year cleaning
            year_val = int(row["Year"]) if pd.notna(row["Year"]) and str(row["Year"]).replace("-", "").isdigit() else 0
            if year_val < 1900 or year_val > 2025:
                year_val = 0

            # Construct artist credits
            artists_list = [p for p in [singer, composer, lyricist] if p]
            all_artists = ", ".join(artists_list) if artists_list else singer
            primary_a, _ = normalize_artist(singer if singer else composer)

            # Match against Last.fm artist index
            cat_idx = artist2idx.get(primary_a.lower(), -1)

            records.append({
                "track_id": f"bolly_{idx}",
                "track_name": title.title(),
                "artist_name": (singer if singer else composer).title(),
                "all_artists": all_artists.title(),
                "album_name": film.title() if film else "Bollywood Classic",
                "release_year": year_val,
                "era": infer_era(year_val),
                "language": "Hindi",
                "genre": "bollywood",
                "popularity": 65 if year_val >= 1990 else 50,  # Baseline cinema weight
                "spotify_id": "",
                "catalog_artist_idx": cat_idx,
                "source": "bollywood_cinema",
                "clean_title": normalize_title(title),
                "clean_artist": primary_a,
            })
        logger.info("Ingested %d Bollywood cinema tracks.", len(df_bolly))

    # 3. Ingest Spotify 114k Dataset
    if SPOTIFY_TRACKS_114K.exists():
        logger.info("Ingesting Spotify tracks dataset from %s...", SPOTIFY_TRACKS_114K.name)
        df_sp = pd.read_csv(SPOTIFY_TRACKS_114K)
        # Drop duplicates on track_id
        df_sp = df_sp.drop_duplicates(subset=["track_id"]).copy()

        # Build lookup of existing clean titles to deduplicate
        existing_lookup = {(r["clean_title"], r["clean_artist"]): idx for idx, r in enumerate(records)}

        for _, row in df_sp.iterrows():
            sp_id = str(row["track_id"])
            t_name = str(row["track_name"]).strip()
            a_name = str(row["artists"]).strip()
            alb_name = str(row["album_name"]).strip()
            pop = float(row["popularity"]) if pd.notna(row["popularity"]) else 0.0

            clean_t = normalize_title(t_name)
            primary_a, _ = normalize_artist(a_name)

            # Deduplication: if exact match already exists from Bollywood, merge Spotify ID & popularity
            match_key = (clean_t, primary_a)
            if match_key in existing_lookup:
                existing_idx = existing_lookup[match_key]
                records[existing_idx]["spotify_id"] = sp_id
                records[existing_idx]["popularity"] = max(records[existing_idx]["popularity"], pop)
                continue

            # Infer year from album
            year_val = extract_year_from_album_or_str(alb_name, default=0)
            lang, genre = infer_language_and_genre(row, source="spotify")
            cat_idx = artist2idx.get(primary_a.lower(), -1)

            records.append({
                "track_id": f"sp_{sp_id}",
                "track_name": t_name,
                "artist_name": primary_a.title() if primary_a else a_name,
                "all_artists": a_name,
                "album_name": alb_name,
                "release_year": year_val,
                "era": infer_era(year_val),
                "language": lang,
                "genre": genre,
                "popularity": pop,
                "spotify_id": sp_id,
                "catalog_artist_idx": cat_idx,
                "source": "spotify_tracks",
                "clean_title": clean_t,
                "clean_artist": primary_a,
            })

        logger.info("Total master records compiled: %d.", len(records))

    df_master = pd.DataFrame(records)
    # Reorder columns
    cols = [
        "track_id",
        "track_name",
        "artist_name",
        "all_artists",
        "album_name",
        "release_year",
        "era",
        "language",
        "genre",
        "popularity",
        "spotify_id",
        "catalog_artist_idx",
        "source",
    ]
    df_master = df_master[cols]

    # Save to parquet
    df_master.to_parquet(MASTER_CATALOG, index=False)
    logger.info("Master catalog successfully saved to %s (%d rows).", MASTER_CATALOG, len(df_master))
    return df_master


if __name__ == "__main__":
    df = build_master_catalog()
    print("Catalog Shape:", df.shape)
    print("\nLanguage Distribution:\n", df["language"].value_counts())
    print("\nEra Distribution:\n", df["era"].value_counts())
    print("\nSample Rows:\n", df[["track_name", "artist_name", "language", "release_year", "popularity"]].head(8))
