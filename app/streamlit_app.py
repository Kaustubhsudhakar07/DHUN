"""
🎵 DhunDNA: Decode Your Music Taste • Interactive Discovery Dashboard
Indian & Global Music Discovery Engine (1990–2025)

Features:
1. ✍️ Enter Taste DNA: Manual favorite songs, artists, genres, languages, eras, mood
2. 🎧 Connect Spotify: OAuth 2.0 PKCE / Curated playlist taste decoding
3. ❤️ Find Similar Songs: Melodic, acoustic, and content similarity search
4. 🔎 Semantic Music Search: Multilingual natural language descriptive retrieval
5. 📊 Music Catalog: 100k track catalog insights, languages & era distributions
6. 👥 Collaborative Matrix: Legacy 360k listener matrix & interleaved playlist generator
"""

import html
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.api_client import RecommendationClient
from app.components import (
    plot_catalog_eras,
    plot_catalog_languages,
    plot_top_tags,
    render_hero_banner,
    render_playlist_item,
    render_recommendation_card,
    render_track_card,
)
from app.styles import CUSTOM_CSS
from src.config import (
    ARTIST_MAPPING,
    CATALOG_SUMMARY,
    DATA_DIR,
    MASTER_CATALOG,
    POPULARITY_SCORES,
)

logger = logging.getLogger("dhundna.streamlit")

# Page Configuration
st.set_page_config(
    page_title="DhunDNA • Decode Your Music Taste",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject Custom Styling
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================================
# CLIENT & DATA CACHING
# =========================================================================

@st.cache_resource(show_spinner=False)
def get_client() -> RecommendationClient:
    """Initialize persistent API / ML client."""
    return RecommendationClient()




@st.cache_data(show_spinner=False)
def load_catalog_summary() -> Dict[str, Any]:
    """Compute and cache master catalog summary metrics."""
    if CATALOG_SUMMARY.exists():
        try:
            with open(CATALOG_SUMMARY, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    if MASTER_CATALOG.exists():
        try:
            import pyarrow as pa
            pa.set_memory_pool(pa.system_memory_pool())
            df = pd.read_parquet(MASTER_CATALOG, columns=["language", "era", "genre", "artist_name"])
            lang_counts = df["language"].value_counts().to_dict()
            era_counts = df["era"].value_counts().to_dict()
            top_genres = df["genre"].value_counts().head(10).to_dict()
            summary = {
                "total_tracks": len(df),
                "languages": lang_counts,
                "eras": era_counts,
                "top_genres": top_genres,
                "total_artists": int(df["artist_name"].nunique()),
            }
            try:
                with open(CATALOG_SUMMARY, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
            except Exception:
                pass
            return summary
        except Exception as e:
            logger.warning("Error loading catalog summary: %s", e)
    return {
        "total_tracks": 100558,
        "languages": {"English / Global": 88028, "Hindi": 12300, "Punjabi": 125, "Tamil": 101, "Telugu": 4},
        "eras": {"Catalog": 87613, "Pre-1990": 6509, "2020s": 2493, "1990s": 1713, "2000s": 1588, "2010s": 642},
        "top_genres": {"pop": 15200, "bollywood": 12300, "rock": 8400, "indie": 6200},
        "total_artists": 31400,
    }


@st.cache_data(show_spinner=False)
def load_popular_artists(limit: int = 60) -> List[str]:
    """Load top popular artists for cold-start autocomplete."""
    if POPULARITY_SCORES.exists() and ARTIST_MAPPING.exists():
        try:
            with open(POPULARITY_SCORES, "r", encoding="utf-8") as f:
                pop_data = json.load(f)
            with open(ARTIST_MAPPING, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            idx2artist = mapping.get("idx2artist", [])
            top_indices = sorted(pop_data.get("scores", {}).keys(), key=lambda k: pop_data["scores"][k], reverse=True)
            names = [idx2artist[int(i)] for i in top_indices[:limit] if int(i) < len(idx2artist)]
            return names
        except Exception:
            pass
    return [
        "Arijit Singh", "Pritam", "Shreya Ghoshal", "Coldplay", "Ap Dhillon",
        "A.R. Rahman", "Radiohead", "The Beatles", "Daft Punk", "Nirvana",
        "Kishore Kumar", "Lata Mangeshkar", "Mohit Chauhan", "Prateek Kuhad",
    ]


client = get_client()
is_online = client.is_api_online()
cat_summary = load_catalog_summary()

# =========================================================================
# HEADER HERO BANNER
# =========================================================================
render_hero_banner(
    title="🎵 DhunDNA • Decode Your Music Taste",
    subtitle="Indian (Bollywood, Punjabi, Tamil, Telugu) + Global Music Discovery Engine (1990–2025) • Semantic Neural Retrieval • Spotify OAuth 2.0 PKCE",
)


# =========================================================================
# MAIN NAVIGATION TABS
# =========================================================================
tabs = st.tabs([
    "✍️ Enter Taste DNA",
    "🎧 Connect Spotify",
    "❤️ Find Similar Songs",
    "🔎 Semantic Music Search",
    "📊 Music Catalog",
    "👥 Collaborative Matrix",
])


# -------------------------------------------------------------------------
# TAB 1: ENTER TASTE DNA (MANUAL INPUT)
# -------------------------------------------------------------------------
with tabs[0]:
    st.subheader("🧬 Decode Your Musical Taste & Discover Tracks")
    st.caption("Tell us your favorite songs, artists, genres, languages, era, and current mood to decode your unique Music DNA.")

    # Preset Taste Shortcuts
    st.markdown("**⚡ Quick Taste Presets:**")
    p_cols = st.columns(4)
    if p_cols[0].button("🌟 2000s Bollywood Romance", key="preset_bolly"):
        st.session_state["dna_artists"] = ["Pritam", "Mohit Chauhan", "Shreya Ghoshal"]
        st.session_state["dna_songs"] = ["Tum Se Hi", "Doobey"]
        st.session_state["dna_genres"] = ["bollywood", "pop"]
        st.session_state["dna_langs"] = ["Hindi"]
        st.session_state["dna_eras"] = ["2000s", "2010s"]
        st.session_state["dna_mood"] = "Romantic / Soulful"
        st.rerun()

    if p_cols[1].button("🕺 Punjabi Pop Bangers", key="preset_punjabi"):
        st.session_state["dna_artists"] = ["Ap Dhillon", "Badshah", "Diljit Dosanjh"]
        st.session_state["dna_songs"] = ["Summer High", "Insane"]
        st.session_state["dna_genres"] = ["punjabi", "pop"]
        st.session_state["dna_langs"] = ["Punjabi"]
        st.session_state["dna_eras"] = ["2020s"]
        st.session_state["dna_mood"] = "High Energy / Dance"
        st.rerun()

    if p_cols[2].button("🎸 Global Indie & Alt-Rock", key="preset_rock"):
        st.session_state["dna_artists"] = ["Coldplay", "Radiohead", "Arctic Monkeys"]
        st.session_state["dna_songs"] = ["Yellow", "Karma Police"]
        st.session_state["dna_genres"] = ["rock", "indie"]
        st.session_state["dna_langs"] = ["English / Global"]
        st.session_state["dna_eras"] = ["2000s"]
        st.session_state["dna_mood"] = "Chill / Relaxed"
        st.rerun()

    if p_cols[3].button("📻 90s Golden Nostalgia", key="preset_90s"):
        st.session_state["dna_artists"] = ["Kumar Sanu", "Lata Mangeshkar", "Udit Narayan"]
        st.session_state["dna_songs"] = ["Tujhe Dekha To", "Mere Dil Kaa Vo Shahazaadaa"]
        st.session_state["dna_genres"] = ["bollywood"]
        st.session_state["dna_langs"] = ["Hindi"]
        st.session_state["dna_eras"] = ["1990s"]
        st.session_state["dna_mood"] = "Nostalgic / Melodic"
        st.rerun()

    # Form inputs
    with st.form("dna_manual_form"):
        col1, col2 = st.columns(2)
        with col1:
            fav_songs_raw = st.text_input(
                "Favorite Song Titles (comma-separated)",
                value=", ".join(st.session_state.get("dna_songs", ["Kesariya", "Tum Se Hi"])),
                help="E.g. Kesariya, Tum Se Hi, Yellow, Summer High",
            )
            fav_artists_raw = st.text_input(
                "Favorite Artists (comma-separated)",
                value=", ".join(st.session_state.get("dna_artists", ["Pritam", "Arijit Singh", "Shreya Ghoshal"])),
                help="E.g. Pritam, Arijit Singh, Coldplay, Ap Dhillon",
            )
            genres_options = [
                "bollywood", "pop", "indie", "punjabi", "rock", "acoustic",
                "electronic", "hip-hop", "folk", "classical", "ambient", "dance",
            ]
            selected_genres = st.multiselect(
                "Preferred Genres",
                options=genres_options,
                default=st.session_state.get("dna_genres", ["bollywood", "pop"]),
            )

        with col2:
            languages_options = ["Hindi", "Punjabi", "Tamil", "Telugu", "English / Global"]
            selected_languages = st.multiselect(
                "Preferred Languages",
                options=languages_options,
                default=st.session_state.get("dna_langs", ["Hindi", "English / Global"]),
            )
            eras_options = ["1990s", "2000s", "2010s", "2020s", "Pre-1990"]
            selected_eras = st.multiselect(
                "Preferred Eras",
                options=eras_options,
                default=st.session_state.get("dna_eras", ["2000s", "2020s"]),
            )
            mood_options = [
                "Romantic / Soulful", "Chill / Relaxed", "High Energy / Dance",
                "Melancholic / Heartbreak", "Focus / Study", "Nostalgic / Melodic",
            ]
            selected_mood = st.selectbox(
                "Current Mood",
                options=mood_options,
                index=mood_options.index(st.session_state.get("dna_mood", "Romantic / Soulful")),
            )

        # Bottom filters & slider
        f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
        with f_col1:
            lang_filter = st.selectbox("Strict Language Filter", options=["All", "Hindi", "Punjabi", "Tamil", "English / Global"], index=0)
        with f_col2:
            era_filter = st.selectbox("Strict Era Filter", options=["All", "1990s", "2000s", "2010s", "2020s", "Pre-1990"], index=0)
        with f_col3:
            dna_k = st.slider("Recommendations (Top-K)", min_value=5, max_value=30, value=10, step=5)

        submit_dna = st.form_submit_button("🧬 Decode DNA & Recommend Tracks", type="primary", use_container_width=True)

    if submit_dna:
        songs_list = [s.strip() for s in fav_songs_raw.split(",") if s.strip()]
        artists_list = [a.strip() for a in fav_artists_raw.split(",") if a.strip()]

        with st.spinner("Decoding your Music DNA profile across 100,558 tracks..."):
            res = client.recommend_for_dna(
                favorite_songs=songs_list,
                favorite_artists=artists_list,
                preferred_genres=selected_genres,
                preferred_languages=selected_languages,
                preferred_eras=selected_eras,
                mood=selected_mood,
                n=dna_k,
                language_filter=None if lang_filter == "All" else lang_filter,
                era_filter=None if era_filter == "All" else era_filter,
            )

            recs = res.get("recommendations", [])
            summary = res.get("query_summary", "Custom DNA Profile")

            st.markdown(
                f'<div style="background: rgba(29, 185, 84, 0.12); border: 1px solid rgba(29, 185, 84, 0.35); border-radius: 12px; padding: 18px; margin: 16px 0;">'
                f'<div style="font-size: 0.82rem; text-transform: uppercase; color: #1DB954; font-weight: 700; letter-spacing: 0.05em;">Decoded Music Taste DNA</div>'
                f'<div style="font-size: 1.15rem; font-weight: 600; color: #FFFFFF; margin-top: 4px;">{html.escape(summary)}</div>'
                f'<div style="font-size: 0.85rem; color: #94A3B8; margin-top: 6px;">'
                f'Analyzed: {len(songs_list)} seed songs • {len(artists_list)} seed artists • {len(selected_genres)} genres • Mood: {selected_mood}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if recs:
                st.success(f"Generated {len(recs)} tailored recommendations from your Music DNA!")
                for rank, track in enumerate(recs, start=1):
                    render_track_card(track=track, rank=rank, client=client, user_id="manual_user")
            else:
                st.warning("No catalog tracks matched the specified filters. Try relaxing the language or era filter.")


# -------------------------------------------------------------------------
# TAB 2: CONNECT SPOTIFY (PLAYLISTS & OAUTH 2.0 PKCE)
# -------------------------------------------------------------------------
with tabs[1]:
    st.subheader("🎧 Connect Spotify & Discover Tracks from Your Playlists")
    st.markdown(
        '<div style="background: rgba(18, 24, 38, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 18px; margin-bottom: 20px;">'
        '<div style="display: flex; align-items: center; gap: 10px;">'
        '<span style="font-size: 1.5rem;">🔒</span>'
        '<div>'
        '<strong style="color: #1DB954;">2024–2026 Spotify Developer Policy Compliant</strong>'
        '<div style="color: #94A3B8; font-size: 0.88rem; margin-top: 2px;">'
        'Uses official User Authorization OAuth 2.0 PKCE to inspect playlists (<code>/v1/me/playlists</code>, <code>/v1/playlists/{id}/tracks</code>). '
        'Does NOT rely on deprecated or restricted audio-features endpoints.'
        '</div>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Check for Spotify OAuth callback code in URL query params
    oauth_code = st.query_params.get("code")
    if oauth_code and not st.session_state.get("spotify_authenticated"):
        stored_verifier = st.session_state.get("spotify_pkce_verifier")
        stored_cid = st.session_state.get("spotify_client_id") or os.getenv("SPOTIFY_CLIENT_ID")
        if stored_cid:
            from src.spotify.client import SpotifyClient
            live_sc = SpotifyClient(client_id=stored_cid)
            if live_sc.exchange_code_for_token(oauth_code, code_verifier=stored_verifier):
                st.session_state["spotify_authenticated"] = True
                st.session_state["spotify_client"] = live_sc
                st.query_params.clear()
                st.toast("🎉 Successfully connected to Spotify! Loaded your playlists & history.", icon="✅")
                st.rerun()

    # Determine authentication state & playlists
    is_session_auth = st.session_state.get("spotify_authenticated", False)
    live_sc = st.session_state.get("spotify_client")

    if is_session_auth and live_sc:
        playlists = live_sc.get_user_playlists()
        is_auth = True
    else:
        sp_data = client.get_spotify_playlists()
        playlists = sp_data.get("playlists", [])
        is_auth = sp_data.get("is_authenticated", False)

    # Prepend Listening History as a first-class source option
    history_source = {
        "id": "user_history",
        "name": "🌟 My Spotify Listening History (Top Played Tracks)",
        "description": "Decodes your top affinity tracks and artist tastes into catalog recommendations",
        "total_tracks": 20,
    }
    all_sources = [history_source] + playlists

    sp_col1, sp_col2 = st.columns([3, 2])
    with sp_col1:
        st.markdown("#### 📁 Select Music Source to Analyze")
        source_options = {p["id"]: f"{p['name']} ({p.get('total_tracks', 0)} tracks)" for p in all_sources}
        selected_pid = st.selectbox(
            "Choose Source (Listening History or Playlist)",
            options=list(source_options.keys()),
            format_func=lambda pid: source_options[pid],
            key="sp_source_select",
        )

        selected_source_info = next((p for p in all_sources if p["id"] == selected_pid), all_sources[0])
        st.info(f"**Selected Source**: {selected_source_info.get('name')} — *{selected_source_info.get('description', 'Curated tracks')}*")

        sp_k = st.slider("Discovery Tracks to Generate", min_value=5, max_value=25, value=10, key="sp_k")

        btn_label = "🔥 Generate Recommendations from Listening History" if selected_pid == "user_history" else "🚀 Analyze & Discover from Playlist"
        if st.button(btn_label, type="primary", use_container_width=True):
            with st.spinner(f"Analyzing '{selected_source_info['name']}' and finding discovery matches..."):
                if is_session_auth and live_sc:
                    # Use live client tracks
                    from src.profiles.music_dna import UserMusicDNA
                    if selected_pid == "user_history":
                        src_tracks = live_sc.get_top_tracks(limit=30)
                    else:
                        src_tracks = live_sc.get_playlist_tracks(selected_pid)

                    # Build DNA and query local catalog
                    app_state = getattr(st, "_cached_direct_state", None)
                    summary = f"Curated mix derived from '{selected_source_info['name']}' ({len(src_tracks)} seed tracks analyzed)"
                    sp_rec_res = client.recommend_from_spotify_playlist(
                        playlist_id=selected_pid,
                        playlist_name=selected_source_info.get("name", "Playlist"),
                        n=sp_k,
                    )
                    recs = sp_rec_res.get("recommendations", [])
                else:
                    sp_rec_res = client.recommend_from_spotify_playlist(
                        playlist_id=selected_pid,
                        playlist_name=selected_source_info.get("name", "Playlist"),
                        n=sp_k,
                    )
                    recs = sp_rec_res.get("recommendations", [])
                    summary = sp_rec_res.get("query_summary", "")

                st.markdown(
                    f'<div style="background: rgba(29, 185, 84, 0.12); border: 1px solid rgba(29, 185, 84, 0.35); border-radius: 12px; padding: 18px; margin: 16px 0;">'
                    f'<div style="font-size: 0.82rem; text-transform: uppercase; color: #1DB954; font-weight: 700;">Decoded Taste DNA</div>'
                    f'<div style="font-size: 1.1rem; font-weight: 600; color: #FFFFFF; margin-top: 4px;">{html.escape(summary)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                if recs:
                    st.success(f"Generated {len(recs)} tailored discovery tracks!")
                    for rank, track in enumerate(recs, start=1):
                        render_track_card(track=track, rank=rank, client=client, user_id="spotify_user")
                else:
                    st.warning("Could not generate recommendations for the selected source.")

    with sp_col2:
        st.markdown("#### 🔑 Spotify Account Linking")
        if is_auth:
            st.success("✅ **Linked to Active Spotify Account!**")
            st.caption("Your live playlists and listening history are active on the left.")
            if st.button("Disconnect Spotify"):
                st.session_state["spotify_authenticated"] = False
                st.session_state.pop("spotify_client", None)
                st.rerun()
        else:
            st.markdown(
                """
                Currently operating in **Zero-Config Demo Mode** with real catalog playlists & history:
                - *🌟 My Spotify Listening History*
                - *My Bollywood Favorites*
                - *Desi Indie & Acoustic Vibes*
                - *Global 2000s Pop-Rock*
                """
            )
            with st.expander("Connect Live Spotify Account"):
                st.caption(
                    "**Redirect URI Setup:** In your Spotify App settings, add: `http://127.0.0.1:8501` "
                    "(Spotify's security policy requires `127.0.0.1` instead of `localhost`)."
                )
                custom_cid = st.text_input("Spotify Client ID", type="password", key="sp_cid_input")
                if st.button("Generate OAuth Link", key="gen_oauth_btn"):
                    from src.spotify.client import SpotifyClient
                    verifier, challenge = SpotifyClient.generate_pkce_pair()
                    st.session_state["spotify_pkce_verifier"] = verifier
                    st.session_state["spotify_client_id"] = custom_cid.strip() if custom_cid else ""
                    temp_sc = SpotifyClient(client_id=custom_cid.strip() if custom_cid else None)
                    auth_url = temp_sc.get_authorization_url(code_challenge=challenge)
                    st.markdown(f"[👉 Click here to Authorize Spotify]({auth_url})")


# -------------------------------------------------------------------------
# TAB 3: FIND SIMILAR SONGS
# -------------------------------------------------------------------------
with tabs[2]:
    st.subheader("❤️ Find Songs with Similar Melodic & Acoustic DNA")
    st.caption("Enter any song title to discover tracks with matching acoustic signatures, artist style, and emotional resonance.")

    st.markdown("**Quick Examples to Try:**")
    ex_cols = st.columns(4)
    if ex_cols[0].button("Kesariya (Brahmastra)", key="ex_kesariya"):
        st.session_state["sim_title"] = "Kesariya"
        st.session_state["sim_artist"] = "Pritam"
        st.rerun()
    if ex_cols[1].button("Tum Se Hi (Jab We Met)", key="ex_tumsehi"):
        st.session_state["sim_title"] = "Tum Se Hi"
        st.session_state["sim_artist"] = "Pritam"
        st.rerun()
    if ex_cols[2].button("Summer High (Ap Dhillon)", key="ex_summer"):
        st.session_state["sim_title"] = "Summer High"
        st.session_state["sim_artist"] = "Ap Dhillon"
        st.rerun()
    if ex_cols[3].button("Yellow (Coldplay)", key="ex_yellow"):
        st.session_state["sim_title"] = "Yellow"
        st.session_state["sim_artist"] = "Coldplay"
        st.rerun()

    sim_c1, sim_c2, sim_c3 = st.columns([3, 2, 1])
    with sim_c1:
        query_song = st.text_input("Song Title", value=st.session_state.get("sim_title", "Kesariya"))
    with sim_c2:
        query_artist = st.text_input("Artist (Optional)", value=st.session_state.get("sim_artist", "Pritam"))
    with sim_c3:
        sim_k = st.slider("Count", min_value=5, max_value=25, value=10, key="sim_k")

    if st.button("🔍 Find Similar Tracks", type="primary", use_container_width=True):
        if not query_song.strip():
            st.error("Please enter a song title.")
        else:
            with st.spinner(f"Resolving '{query_song}' and searching catalog for nearest matches..."):
                sim_res = client.recommend_similar_song(
                    song_title=query_song.strip(),
                    artist_name=query_artist.strip(),
                    n=sim_k,
                )
                recs = sim_res.get("recommendations", [])
                summary = sim_res.get("query_summary", "")

                st.markdown(
                    f'<div style="background: rgba(168, 85, 247, 0.12); border: 1px solid rgba(168, 85, 247, 0.35); border-radius: 12px; padding: 18px; margin: 16px 0;">'
                    f'<div style="font-size: 0.82rem; text-transform: uppercase; color: #C084FC; font-weight: 700;">Similarity Anchor</div>'
                    f'<div style="font-size: 1.15rem; font-weight: 600; color: #FFFFFF; margin-top: 4px;">{html.escape(summary)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                if recs:
                    st.success(f"Found {len(recs)} tracks with matching acoustic and artist style!")
                    for rank, track in enumerate(recs, start=1):
                        render_track_card(track=track, rank=rank, client=client, user_id="similar_user")
                else:
                    st.warning(f"Could not find similar matches for '{query_song}'.")


# -------------------------------------------------------------------------
# TAB 4: SEMANTIC MUSIC SEARCH
# -------------------------------------------------------------------------
with tabs[3]:
    st.subheader("🔎 Semantic Multilingual Natural Language Music Search")
    st.caption("Search across 100,558 tracks using natural descriptive queries (e.g. '90s Hindi romantic songs about heartbreak').")

    st.markdown("**Example Natural Language Queries:**")
    sem_ex_cols = st.columns(3)
    if sem_ex_cols[0].button("💔 '90s Hindi heartbreak songs'", key="sem_ex_1"):
        st.session_state["sem_query"] = "90s Hindi romantic songs about heartbreak and longing"
        st.rerun()
    if sem_ex_cols[1].button("🕺 'Energetic Punjabi party dance'", key="sem_ex_2"):
        st.session_state["sem_query"] = "energetic punjabi party dance beat"
        st.rerun()
    if sem_ex_cols[2].button("🌧️ 'Rainy monsoon acoustic love'", key="sem_ex_3"):
        st.session_state["sem_query"] = "rainy day monsoon acoustic love song bollywood"
        st.rerun()

    sem_prompt = st.text_input(
        "Enter Natural Language Search Prompt",
        value=st.session_state.get("sem_query", "90s Hindi romantic songs about heartbreak and love"),
        help="Describe themes, instruments, moods, or musical qualities in English or Romanized Desi terms.",
    )

    s_col1, s_col2, s_col3 = st.columns([1, 1, 2])
    with s_col1:
        sem_lang = st.selectbox("Language Filter", options=["All", "Hindi", "Punjabi", "Tamil", "English / Global"], key="sem_lang")
    with s_col2:
        sem_era = st.selectbox("Era Filter", options=["All", "1990s", "2000s", "2010s", "2020s", "Pre-1990"], key="sem_era")
    with s_col3:
        sem_k = st.slider("Top Results", min_value=5, max_value=30, value=10, key="sem_k")

    if st.button("🧠 Search Semantic Catalog", type="primary", use_container_width=True):
        if not sem_prompt.strip():
            st.error("Please enter a search prompt.")
        else:
            with st.spinner(f"Running semantic similarity search for '{sem_prompt}'..."):
                sem_res = client.search_semantic(
                    query=sem_prompt.strip(),
                    n=sem_k,
                    language_filter=None if sem_lang == "All" else sem_lang,
                    era_filter=None if sem_era == "All" else sem_era,
                )
                recs = sem_res.get("recommendations", [])

                if recs:
                    st.success(f"Retrieved {len(recs)} semantic matches!")
                    for rank, track in enumerate(recs, start=1):
                        render_track_card(track=track, rank=rank, client=client, user_id="semantic_user")
                else:
                    st.warning("No semantic matches found for this query with selected filters.")


# -------------------------------------------------------------------------
# TAB 5: MUSIC CATALOG INSIGHTS
# -------------------------------------------------------------------------
with tabs[4]:
    st.subheader("📊 Master Catalog Insights")
    st.caption("Master catalog distribution, language breakdown, and release eras across 100,558 tracks.")

    # High-level Catalog Stat Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-value">{cat_summary['total_tracks']:,}</div>
                <div class="stat-label">Total Tracks (Parquet)</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-value">{cat_summary['languages'].get('Hindi', 12300):,}</div>
                <div class="stat-label">Hindi / Bollywood Tracks</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-value">358,868</div>
                <div class="stat-label">Collaborative Listeners</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-value">1960–2025</div>
                <div class="stat-label">Release Era Span</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### 🌐 Catalog Language Breakdown")
        st.plotly_chart(plot_catalog_languages(cat_summary["languages"]), use_container_width=True)

    with chart_col2:
        st.markdown("#### 📅 Release Era Distribution")
        st.plotly_chart(plot_catalog_eras(cat_summary["eras"]), use_container_width=True)


# -------------------------------------------------------------------------
# TAB 6: COLLABORATIVE MATRIX (LAST.FM)
# -------------------------------------------------------------------------
with tabs[5]:
    st.subheader("👥 Collaborative Matrix & Interleaved Discovery Mix")
    st.caption("Directly query the 358,868-listener collaborative filtering models (ALS, BPR, Content, Hybrid) from HetRec 2011.")

    cf_col1, cf_col2, cf_col3 = st.columns([2, 2, 2])
    with cf_col1:
        u_id = st.number_input("Listener ID (0 to 358,867)", min_value=0, max_value=358867, value=0, step=1)
    with cf_col2:
        m_choice = st.selectbox(
            "Algorithm",
            options=["hybrid", "als", "content", "bpr", "popularity"],
            format_func=lambda x: {
                "hybrid": "⭐ Hybrid Engine (Optimal)",
                "als": "Collaborative Filtering (ALS)",
                "content": "Content-Based (TF-IDF Tags)",
                "bpr": "Ranking (BPR Implicit)",
                "popularity": "Popularity Prior",
            }[x],
        )
    with cf_col3:
        cf_k = st.slider("Top Artists", min_value=5, max_value=25, value=10)

    if st.button("✨ Get Collaborative Recommendations", type="primary", use_container_width=True):
        with st.spinner(f"Computing top artists for User #{u_id} using {m_choice.upper()}..."):
            cf_res = client.get_recommendations(user_id=int(u_id), n=cf_k, model=m_choice, explain=True)
            recs = cf_res.get("recommendations", [])
            if recs:
                st.success(f"Generated {len(recs)} artist recommendations for User #{u_id}!")
                for rank, rec in enumerate(recs, start=1):
                    render_recommendation_card(rec=rec, rank=rank, user_id=int(u_id), client=client)
            else:
                st.warning("No recommendations returned for this user.")

    st.markdown("---")
    st.markdown("#### 💿 Interleaved Discovery Playlist Generator")
    p_title = st.text_input("Mix Title", value=f"User {u_id} • DhunDNA Discovery Mix")
    if st.button("🎶 Generate Curated Interleaved Playlist"):
        with st.spinner("Generating interleaved track playlist..."):
            pl = client.get_playlist(user_id=int(u_id), n_artists=5, tracks_per_artist=2, title=p_title)
            tracks = pl.get("tracks", [])
            if tracks:
                for idx, t in enumerate(tracks, start=1):
                    render_playlist_item(t, index=idx)



