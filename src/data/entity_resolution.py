"""
Entity Resolution & Track Deduplication Engine for DhunDNA (Phase 2)

Provides robust normalization, string canonicalization, and fuzzy matching
to resolve identical tracks across disparate sources (Spotify, Bollywood, Last.fm):
E.g.:
- 'Kesariya (From "Brahmastra")' <-> 'Kesariya' <-> 'Kesariya - Brahmastra'
- 'Tum Se Hi (From "Jab We Met")' <-> 'Tum Se Hi'
"""

import logging
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("dhundna.entity_resolution")


def strip_accents_and_diacritics(text: str) -> str:
    """Normalize unicode characters and strip accents."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_title(title: str) -> str:
    """Canonicalize a song title by stripping extraneous noise, soundtrack tags, and punctuation.

    Examples:
        'Kesariya (From "Brahmastra")' -> 'kesariya'
        'Tum Se Hi - Jab We Met' -> 'tum se hi'
        'Bohemian Rhapsody - Remastered 2011' -> 'bohemian rhapsody'
    """
    if not title or not isinstance(title, str):
        return ""

    t = strip_accents_and_diacritics(title).lower().strip()

    # 1. Remove bracketed / parenthetical qualifiers: (from ...), [feat. ...], (deluxe version), etc.
    t = re.sub(
        r"[\(\[\{](?:from|feat\.?|featuring|with|remix|version|ost|original|soundtrack|sound\s*track|lyrics|remastered|deluxe|live|mono|stereo|reprise|edit|radio).*?[\)\]\}]",
        "",
        t,
        flags=re.IGNORECASE,
    )
    # Also strip any lingering parenthesized expressions at the end
    t = re.sub(r"[\(\[\{].*?[\)\]\}]", "", t)

    # 2. Remove dash-separated suffixes e.g. " - From Brahmastra", " - Remastered 2011"
    t = re.sub(
        r"\s*-\s*(?:from|soundtrack|remix|remastered|ost|live|radio|deluxe|version).*$",
        "",
        t,
        flags=re.IGNORECASE,
    )

    # 3. Strip quotes and extraneous punctuation
    t = re.sub(r"['\"\`]", "", t)
    t = re.sub(r"[^\w\s]", " ", t)

    # 4. Collapse whitespace
    return " ".join(t.split())


def normalize_artist(artist: str) -> Tuple[str, List[str]]:
    """Parse and canonicalize artist strings, extracting primary and contributing artists.

    Examples:
        'Pritam;Arijit Singh;Amitabh Bhattacharya' -> ('pritam', ['pritam', 'arijit singh', 'amitabh bhattacharya'])
        'A.R. Rahman feat. Alka Yagnik' -> ('ar rahman', ['ar rahman', 'alka yagnik'])
    """
    if not artist or not isinstance(artist, str):
        return "", []

    raw = strip_accents_and_diacritics(artist).lower().strip()

    # Split by standard collaboration tokens: ';', ',', '/', '&', 'feat.', 'ft.', 'featuring'
    split_pattern = r"\s*(?:;|,|\/|&|\bfeat\.?\b|\bft\.?\b|\bfeaturing\b|\bwith\b)\s*"
    parts = [p.strip() for p in re.split(split_pattern, raw) if p.strip()]

    # Clean punctuation inside each artist name
    cleaned_artists = []
    for p in parts:
        clean_p = re.sub(r"[^\w\s]", "", p).strip()
        clean_p = " ".join(clean_p.split())
        if clean_p and clean_p not in cleaned_artists:
            cleaned_artists.append(clean_p)

    primary = cleaned_artists[0] if cleaned_artists else ""
    return primary, cleaned_artists


def token_sort_similarity(str1: str, str2: str) -> float:
    """Compute token-sort ratio similarity between two strings."""
    if not str1 or not str2:
        return 0.0
    if str1 == str2:
        return 1.0

    tokens1 = " ".join(sorted(str1.split()))
    tokens2 = " ".join(sorted(str2.split()))
    return SequenceMatcher(None, tokens1, tokens2).ratio()


class EntityResolver:
    """In-memory entity resolver that maps heterogeneous track queries to canonical records."""

    def __init__(self, canonical_catalog: Optional[List[Dict[str, Any]]] = None):
        self.exact_index: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.title_index: Dict[str, List[Dict[str, Any]]] = {}
        self.total_indexed = 0

        if canonical_catalog:
            self.index_catalog(canonical_catalog)

    def index_catalog(self, catalog_records: List[Dict[str, Any]]):
        """Index a list of catalog dictionaries containing 'track_name' and 'artist_name'."""
        for rec in catalog_records:
            t_name = rec.get("track_name", "")
            a_name = rec.get("artist_name", "")
            c_title = normalize_title(t_name)
            primary_a, _ = normalize_artist(a_name)

            if not c_title:
                continue

            # Store in exact tuple index
            key = (c_title, primary_a)
            if key not in self.exact_index:
                self.exact_index[key] = rec

            # Store in title bucket for fuzzy matching
            if c_title not in self.title_index:
                self.title_index[c_title] = []
            self.title_index[c_title].append(rec)
            self.total_indexed += 1

        logger.info("Indexed %d catalog items for entity resolution.", self.total_indexed)

    def resolve(
        self,
        query_title: str,
        query_artist: str = "",
        threshold: float = 0.82,
    ) -> Tuple[Optional[Dict[str, Any]], float, str]:
        """Resolve a track query to a canonical catalog record.

        Returns:
            Tuple of (matched_record_dict, confidence_score, match_type)
        """
        c_title = normalize_title(query_title)
        primary_a, all_query_artists = normalize_artist(query_artist)

        if not c_title:
            return None, 0.0, "unmatched_empty_title"

        # 1. Exact Match: (Normalized Title, Primary Artist)
        if primary_a and (c_title, primary_a) in self.exact_index:
            return self.exact_index[(c_title, primary_a)], 1.0, "exact_title_artist"

        # 2. Check if any contributing artist matches exactly
        for a in all_query_artists:
            if (c_title, a) in self.exact_index:
                return self.exact_index[(c_title, a)], 0.98, "exact_title_collaborator"

        # 3. Exact Title match with artist similarity check
        if c_title in self.title_index:
            candidates = self.title_index[c_title]
            best_match = None
            best_score = 0.0
            for cand in candidates:
                cand_a = cand.get("artist_name", "")
                cand_primary, cand_all = normalize_artist(cand_a)
                if not primary_a:
                    best_match = cand
                    best_score = 0.85
                    break
                sims = [token_sort_similarity(primary_a, cand_primary)]
                for ca in cand_all:
                    sims.append(token_sort_similarity(primary_a, ca))
                    if primary_a in ca or ca in primary_a:
                        sims.append(0.90)
                max_sim = max(sims)
                if max_sim > best_score:
                    best_score = max_sim
                    best_match = cand

            if best_match and best_score >= 0.60:
                return best_match, round(best_score, 2), "exact_title_fuzzy_artist"

        # 4. Fuzzy Title Matching within same artist candidates (if primary artist exists)
        if primary_a:
            best_fuzzy = None
            best_fuzzy_score = 0.0
            # Scan title index keys with prefix or token similarity
            for indexed_title, cands in self.title_index.items():
                t_sim = token_sort_similarity(c_title, indexed_title)
                if t_sim >= threshold and t_sim > best_fuzzy_score:
                    for cand in cands:
                        cand_primary, _ = normalize_artist(cand.get("artist_name", ""))
                        if token_sort_similarity(primary_a, cand_primary) >= 0.70:
                            best_fuzzy_score = t_sim
                            best_fuzzy = cand
                            break

            if best_fuzzy and best_fuzzy_score >= threshold:
                return best_fuzzy, round(best_fuzzy_score, 2), "fuzzy_title_artist"

        return None, 0.0, "unmatched"
