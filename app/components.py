"""
Reusable Streamlit UI Components & Plotly Visualizations (Phase 12)

Provides visual components for recommendation cards, explainability callouts,
playlist players, interactive Plotly charts (radar, donut, bar), and feedback widgets.
"""

from typing import Any, Dict, List, Optional
import html
import plotly.graph_objects as go
import streamlit as st


def render_hero_banner(
    title: str = "🎵 Antigravity Audio Intelligence",
    subtitle: str = "Hybrid Music Recommendation Engine • Matrix Factorization • Tag Content Filtering • Explainable AI",
    status_label: Optional[str] = None,
    is_online: bool = True,
):
    """Render a premium glassmorphic hero header."""
    status_pill_html = ""
    if status_label:
        status_color = "#1DB954" if is_online else "#F59E0B"
        status_pill_html = (
            f'<div style="background: rgba(0,0,0,0.3); padding: 6px 14px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.1); font-size: 0.8rem; display: flex; align-items: center; gap: 8px;">'
            f'<span style="width: 8px; height: 8px; border-radius: 50%; background: {status_color}; display: inline-block;"></span>'
            f'<span style="color: #CBD5E1; font-weight: 500;">{html.escape(status_label)}</span>'
            f'</div>'
        )

    banner_html = (
        f'<div class="hero-container">'
        f'<div style="display: flex; justify-content: space-between; align-items: flex-start;">'
        f'<div>'
        f'<h1 class="hero-title">{html.escape(title)}</h1>'
        f'<p class="hero-subtitle">{html.escape(subtitle)}</p>'
        f'</div>'
        f'{status_pill_html}'
        f'</div>'
        f'</div>'
    )
    st.markdown(banner_html, unsafe_allow_html=True)


def render_recommendation_card(
    rec: Dict[str, Any],
    rank: int,
    user_id: int,
    client: Any,
    show_feedback: bool = True,
):
    """Render a single recommendation card with rich explainability and feedback controls."""
    artist_name = rec.get("artist_name", "Unknown Artist")
    artist_id = rec.get("artist_id", 0)
    score = rec.get("score", 0.0)
    explanation = rec.get("explanation", "Personalized match based on your listening history.")
    shared_tags = rec.get("shared_tags", [])
    anchors = rec.get("anchor_artists", [])
    signals = rec.get("signal_breakdown", {})
    lastfm_url = rec.get("lastfm_url", f"https://www.last.fm/music/{artist_name.replace(' ', '+')}")

    # Build badge HTML without newlines or markdown indentation
    badges_list = []
    for a in anchors[:3]:
        a_name = a if isinstance(a, str) else a.get("artist_name", "")
        if a_name:
            badges_list.append(f'<span class="badge-anchor">🔗 {html.escape(a_name)}</span>')

    for t in shared_tags[:5]:
        tag_name = t if isinstance(t, str) else t.get("tag", "")
        if tag_name:
            badges_list.append(f'<span class="badge-tag">#{html.escape(tag_name)}</span>')

    badges_html = f'<div style="margin-top: 8px;">{" ".join(badges_list)}</div>' if badges_list else ""

    # Signal breakdown summary
    sig_parts = []
    for k, v in signals.items():
        sig_parts.append(f"{k.capitalize()}: {v:.0f}%")
    sig_summary = " • ".join(sig_parts) if sig_parts else "Balanced Hybrid"

    score_pct = int(min(100, max(0, score * 100))) if score <= 1.0 else int(min(100, score))

    card_html = (
        f'<div class="rec-card">'
        f'<div class="rec-header">'
        f'<div style="display: flex; align-items: center; gap: 12px;">'
        f'<span class="rec-rank">#{rank}</span>'
        f'<a href="{lastfm_url}" target="_blank" class="rec-artist-name">{html.escape(artist_name)} ↗</a>'
        f'</div>'
        f'<div class="rec-score-badge">Match: {score_pct}%</div>'
        f'</div>'
        f'<div class="rec-explanation">💡 <strong>Why recommended:</strong> {html.escape(explanation)}</div>'
        f'{badges_html}'
        f'<div style="margin-top: 8px; font-size: 0.78rem; color: #64748B;">Signal contribution: <span style="color: #94A3B8;">{sig_summary}</span></div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    if show_feedback:
        cols = st.columns([1, 1, 1, 7])
        with cols[0]:
            if st.button("👍", key=f"like_{user_id}_{artist_id}_{rank}", help="Recommend more like this"):
                client.post_feedback(user_id=user_id, artist_id=artist_id, artist_name=artist_name, feedback_type="thumbs_up")
                st.toast(f"Liked {artist_name}! We'll tune future recommendations.")
        with cols[1]:
            if st.button("👎", key=f"dislike_{user_id}_{artist_id}_{rank}", help="Don't recommend this"):
                client.post_feedback(user_id=user_id, artist_id=artist_id, artist_name=artist_name, feedback_type="thumbs_down")
                st.toast(f"Marked {artist_name} as not interested.")
        with cols[2]:
            if st.button("❤️", key=f"fav_{user_id}_{artist_id}_{rank}", help="Add to favorites"):
                client.post_feedback(user_id=user_id, artist_id=artist_id, artist_name=artist_name, feedback_type="favorite")
                st.toast(f"Added {artist_name} to favorites!")


def render_track_card(
    track: Dict[str, Any],
    rank: int,
    client: Any,
    user_id: str = "guest",
    show_feedback: bool = True,
):
    """Render a rich song / track card for DhunDNA."""
    title = track.get("track_name", "Unknown Track")
    artist = track.get("artist_name", "Unknown Artist")
    album = track.get("album_name", "")
    year = track.get("release_year", 0)
    era = track.get("era", "Unknown")
    language = track.get("language", "Global")
    genre = track.get("genre", "pop")
    score = track.get("score", 0.0)
    popularity = track.get("popularity", 0.0)
    explanation = track.get("explanation", "Personalized DhunDNA recommendation")
    sp_id = track.get("spotify_id", "")

    sp_url = (
        f"https://open.spotify.com/track/{sp_id}"
        if sp_id
        else f"https://www.google.com/search?q={html.escape(title)}+{html.escape(artist)}+song"
    )
    score_pct = int(min(100, max(0, score * 100))) if score <= 1.0 else int(min(100, score))

    badges = [
        f'<span class="badge-tag" style="background: rgba(34, 197, 94, 0.15); color: #4ADE80; border-color: rgba(34, 197, 94, 0.3);">{html.escape(language)}</span>',
        f'<span class="badge-tag" style="background: rgba(168, 85, 247, 0.15); color: #C084FC; border-color: rgba(168, 85, 247, 0.3);">{html.escape(genre)}</span>',
    ]
    if era != "Unknown":
        badges.append(f'<span class="badge-tag" style="background: rgba(245, 158, 11, 0.15); color: #FBBF24; border-color: rgba(245, 158, 11, 0.3);">{html.escape(era)}</span>')
    if year > 0:
        badges.append(f'<span class="badge-tag">📅 {year}</span>')

    badges_html = " ".join(badges)
    album_html = f" • 💿 {html.escape(album)}" if album else ""

    card_html = (
        f'<div class="rec-card">'
        f'<div class="rec-header">'
        f'<div style="display: flex; align-items: center; gap: 12px;">'
        f'<span class="rec-rank">#{rank}</span>'
        f'<div>'
        f'<a href="{sp_url}" target="_blank" class="rec-artist-name" style="font-size: 1.15rem;">{html.escape(title)} ↗</a>'
        f'<div style="color: #94A3B8; font-size: 0.85rem; margin-top: 2px;">🎤 <strong>{html.escape(artist)}</strong>{album_html}</div>'
        f'</div>'
        f'</div>'
        f'<div class="rec-score-badge">Match: {score_pct}%</div>'
        f'</div>'
        f'<div class="rec-explanation">💡 <strong>Why this track:</strong> {html.escape(explanation)}</div>'
        f'<div style="margin-top: 8px;">{badges_html}</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    if show_feedback:
        cols = st.columns([1, 1, 1, 7])
        with cols[0]:
            if st.button("👍", key=f"track_like_{user_id}_{rank}", help="Like track"):
                client.post_feedback(user_id=0, artist_id=0, artist_name=f"{title} - {artist}", feedback_type="thumbs_up")
                st.toast(f"Liked '{title}'!")
        with cols[1]:
            if st.button("👎", key=f"track_dislike_{user_id}_{rank}", help="Not interested"):
                client.post_feedback(user_id=0, artist_id=0, artist_name=f"{title} - {artist}", feedback_type="thumbs_down")
                st.toast(f"Marked '{title}' as not interested.")
        with cols[2]:
            if st.button("❤️", key=f"track_fav_{user_id}_{rank}", help="Save to favorites"):
                client.post_feedback(user_id=0, artist_id=0, artist_name=f"{title} - {artist}", feedback_type="favorite")
                st.toast(f"Saved '{title}' to favorites!")


def render_playlist_item(track: Dict[str, Any], index: int):
    """Render a clean playlist row item."""
    title = track.get("track_title", "Track")
    artist = track.get("artist_name", "Artist")
    explanation = track.get("explanation", "Curated for you")
    url = track.get("url", "#")

    row_html = (
        f'<div class="playlist-track-row">'
        f'<div style="display: flex; align-items: center;">'
        f'<div class="track-number">{index:02d}</div>'
        f'<div class="track-info">'
        f'<div class="track-title"><a href="{url}" target="_blank" style="color: #F8FAFC; text-decoration: none;">{html.escape(title)} ↗</a></div>'
        f'<div class="track-artist">{html.escape(artist)}</div>'
        f'</div>'
        f'</div>'
        f'<div><span class="track-badge">{html.escape(explanation)}</span></div>'
        f'</div>'
    )
    st.markdown(row_html, unsafe_allow_html=True)


# =========================================================================
# Plotly Chart Generators
# =========================================================================

def plot_ndcg_comparison(eval_data: Dict[str, Any]) -> go.Figure:
    """Create a grouped bar chart comparing NDCG across models."""
    ranking = eval_data.get("ranking", {})
    models = list(ranking.keys())
    k_vals = ["5", "10", "20"]

    colors = {
        "Popularity": "#64748B",
        "ContentBased": "#0EA5E9",
        "ALS": "#8B5CF6",
        "BPR": "#F43F5E",
        "Hybrid": "#1DB954",
    }

    fig = go.Figure()
    all_ndcg = []
    for m in models:
        m_ndcg = [ranking[m]["ndcg"].get(k, 0.0) for k in k_vals]
        all_ndcg.extend(m_ndcg)
        line_dict = dict(color="#22C55E", width=1.5) if m == "Hybrid" else dict(width=0)
        fig.add_trace(go.Bar(
            name=m,
            x=[f"NDCG@{k}" for k in k_vals],
            y=m_ndcg,
            marker=dict(
                color=colors.get(m, "#A855F7"),
                line=line_dict,
            ),
            text=[f"{v:.3f}" for v in m_ndcg],
            textposition="outside",
            textfont=dict(size=11, family="Inter, sans-serif", color="#F1F5F9"),
        ))

    max_y = (max(all_ndcg) * 1.25) if all_ndcg else 0.12

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.3)",
        barmode="group",
        bargap=0.22,
        bargroupgap=0.08,
        font=dict(family="Inter, sans-serif", color="#E2E8F0"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        yaxis=dict(
            title="NDCG Accuracy",
            range=[0, max_y],
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.1)",
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.0)",
            tickfont=dict(size=12, family="Inter, sans-serif", color="#CBD5E1"),
        ),
        margin=dict(l=40, r=25, t=15, b=55),
        height=370,
    )
    return fig


def plot_beyond_accuracy_radar(eval_data: Dict[str, Any]) -> go.Figure:
    """Create a Radar / Spider chart comparing beyond-accuracy dimensions."""
    ba = eval_data.get("beyond_accuracy", {})
    categories = ["Catalog Coverage", "Intra-List Diversity", "Novelty", "Personalization"]

    fig = go.Figure()

    line_colors = {
        "Popularity": "#64748B",
        "ContentBased": "#0EA5E9",
        "ALS": "#8B5CF6",
        "BPR": "#F43F5E",
        "Hybrid": "#1DB954",
    }

    fill_colors = {
        "Popularity": "rgba(100, 116, 139, 0.04)",
        "ContentBased": "rgba(14, 165, 233, 0.06)",
        "ALS": "rgba(139, 92, 246, 0.06)",
        "BPR": "rgba(244, 63, 94, 0.06)",
        "Hybrid": "rgba(29, 185, 84, 0.30)",
    }

    line_widths = {
        "Popularity": 1.5,
        "ContentBased": 2.0,
        "ALS": 2.0,
        "BPR": 2.0,
        "Hybrid": 3.5,
    }

    for m, vals in ba.items():
        cov = min(1.0, vals.get("coverage", 0.0) * 25.0)
        div = vals.get("diversity", 0.0)
        nov = min(1.0, vals.get("novelty", 0.0) / 10.0)
        pers = vals.get("personalization", 0.0)

        r_vals = [cov, div, nov, pers, cov]
        theta_vals = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=r_vals,
            theta=theta_vals,
            fill="toself",
            name=m,
            fillcolor=fill_colors.get(m, "rgba(255,255,255,0.04)"),
            line=dict(
                color=line_colors.get(m, "#A855F7"),
                width=line_widths.get(m, 2.0),
            ),
            marker=dict(size=4 if m != "Hybrid" else 7),
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor="rgba(255,255,255,0.08)",
                linecolor="rgba(255,255,255,0.08)",
                tickfont=dict(size=9, color="#94A3B8"),
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.08)",
                linecolor="rgba(255,255,255,0.12)",
                tickfont=dict(size=11, family="Inter, sans-serif", color="#E2E8F0"),
            ),
            bgcolor="rgba(15, 23, 42, 0.35)",
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#E2E8F0"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=45, r=45, t=15, b=55),
        height=370,
    )
    return fig


def plot_signal_donut(signals: Dict[str, float]) -> go.Figure:
    """Create a mini donut chart of hybrid signal percentages."""
    labels = [k.capitalize() for k in signals.keys()]
    values = list(signals.values())
    color_map = {
        "Collaborative": "#8B5CF6",
        "Content": "#1DB954",
        "Popularity": "#0EA5E9",
    }
    colors = [color_map.get(lbl, "#64748B") for lbl in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors),
        textinfo="label+percent",
        textfont=dict(size=12),
    )])

    fig.update_layout(
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=180,
    )
    return fig


def plot_top_tags(tags: List[str], weights: List[float], artist_name: str) -> go.Figure:
    """Create horizontal bar chart of top TF-IDF tags for an artist."""
    fig = go.Figure(go.Bar(
        x=weights[::-1],
        y=tags[::-1],
        orientation="h",
        marker=dict(
            color=weights[::-1],
            colorscale=[[0, "#0EA5E9"], [1, "#1DB954"]],
        ),
        text=[f"{w:.3f}" for w in weights[::-1]],
        textposition="outside",
    ))

    fig.update_layout(
        title=f"Top TF-IDF Tag Affinity Profile: {artist_name}",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#E2E8F0"),
        xaxis=dict(title="TF-IDF Weight", gridcolor="rgba(255,255,255,0.08)"),
        margin=dict(l=80, r=40, t=50, b=40),
        height=320,
    )
    return fig


def plot_catalog_languages(languages: Dict[str, int]) -> go.Figure:
    """Lightweight bar chart for catalog languages."""
    langs = list(languages.keys())
    counts = list(languages.values())
    colors = ["#1DB954", "#8B5CF6", "#0EA5E9", "#EC4899", "#F59E0B"]
    fig = go.Figure(go.Bar(
        x=langs,
        y=counts,
        marker_color=colors[:len(langs)],
        text=[f"{c:,}" for c in counts],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=30, b=30),
        height=320,
    )
    return fig


def plot_catalog_eras(eras: Dict[str, int]) -> go.Figure:
    """Lightweight bar chart for catalog era breakdown."""
    normalized: Dict[str, int] = {}
    for k, v in eras.items():
        k_lower = k.strip().lower()
        if "pre-1990" in k_lower or "pre 1990" in k_lower:
            norm_key = "Pre-1990"
        elif "unknown" in k_lower or "catalog" in k_lower:
            norm_key = "Global / Catalog"
        else:
            norm_key = k.strip()
        normalized[norm_key] = normalized.get(norm_key, 0) + v

    era_order = ["Pre-1990", "1990s", "2000s", "2010s", "2020s", "Global / Catalog"]
    ordered = [e for e in era_order if e in normalized]
    for k in normalized:
        if k not in ordered:
            ordered.append(k)

    counts = [normalized[e] for e in ordered]
    colors = ["#A78BFA", "#38BDF8", "#34D399", "#FBBF24", "#F43F5E", "#64748B", "#06B6D4"]
    fig = go.Figure(go.Bar(
        x=ordered,
        y=counts,
        marker_color=colors[:len(ordered)],
        text=[f"{c:,}" for c in counts],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=30, b=30),
        height=320,
    )
    return fig


