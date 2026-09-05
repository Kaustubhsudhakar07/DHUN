"""
Custom CSS & Styling Tokens for Streamlit Music Recommender (Phase 12)

Provides modern dark-mode aesthetics, glassmorphic card styling, vibrant gradient
accents (Spotify green, neon violet, electric cyan), and responsive layout components.
"""

CUSTOM_CSS = """
<style>
/* =========================================================================
   Global Theme & Typography
   ========================================================================= */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
    letter-spacing: -0.02em;
    font-weight: 600;
}

/* App Background */
.stApp {
    background: radial-gradient(circle at 15% 15%, rgba(29, 185, 84, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 25%, rgba(138, 43, 226, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 50% 80%, rgba(0, 240, 255, 0.04) 0%, transparent 50%),
                #0B0E14;
    color: #E2E8F0;
}

/* Hide empty sidebar completely */
[data-testid="stSidebar"], [data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

/* =========================================================================
   Hero Header Banner
   ========================================================================= */
.hero-container {
    background: linear-gradient(135deg, rgba(29, 185, 84, 0.15) 0%, rgba(138, 43, 226, 0.18) 50%, rgba(14, 17, 23, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(12px);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    background: linear-gradient(90deg, #1DB954, #1ed760, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #94A3B8;
    font-size: 0.98rem;
    margin: 0;
    line-height: 1.5;
}

/* =========================================================================
   Glassmorphic Recommendation Card
   ========================================================================= */
.rec-card {
    background: rgba(18, 24, 38, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
    transition: all 0.25s ease;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}

.rec-card:hover {
    border-color: rgba(29, 185, 84, 0.4);
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(29, 185, 84, 0.12);
}

.rec-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.rec-rank {
    background: linear-gradient(135deg, #1DB954, #15883e);
    color: #000000;
    font-weight: 700;
    font-size: 0.85rem;
    padding: 3px 10px;
    border-radius: 20px;
}

.rec-artist-name {
    font-size: 1.25rem;
    font-weight: 600;
    color: #FFFFFF;
    text-decoration: none;
}

.rec-score-badge {
    background: rgba(138, 43, 226, 0.2);
    color: #C084FC;
    border: 1px solid rgba(138, 43, 226, 0.4);
    font-weight: 600;
    font-size: 0.82rem;
    padding: 4px 10px;
    border-radius: 12px;
}

.rec-explanation {
    background: rgba(255, 255, 255, 0.03);
    border-left: 3px solid #1DB954;
    padding: 10px 14px;
    border-radius: 0 8px 8px 0;
    margin: 10px 0;
    font-size: 0.9rem;
    color: #CBD5E1;
    line-height: 1.45;
}

/* =========================================================================
   Badges & Pills
   ========================================================================= */
.badge-tag {
    display: inline-block;
    background: rgba(56, 189, 248, 0.12);
    color: #38BDF8;
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 12px;
    padding: 2px 9px;
    font-size: 0.78rem;
    margin: 2px 4px 2px 0;
    font-weight: 500;
}

.badge-anchor {
    display: inline-block;
    background: rgba(34, 197, 94, 0.12);
    color: #4ADE80;
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 12px;
    padding: 2px 9px;
    font-size: 0.78rem;
    margin: 2px 4px 2px 0;
    font-weight: 500;
}

.badge-model {
    display: inline-block;
    background: rgba(168, 85, 247, 0.15);
    color: #C084FC;
    border: 1px solid rgba(168, 85, 247, 0.3);
    border-radius: 8px;
    padding: 3px 8px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* =========================================================================
   Stat Metric Callout Card
   ========================================================================= */
.stat-card {
    background: rgba(18, 24, 38, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    backdrop-filter: blur(8px);
}

.stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1DB954;
    margin-bottom: 4px;
}

.stat-label {
    font-size: 0.8rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* =========================================================================
   Playlist Item
   ========================================================================= */
.playlist-track-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(22, 28, 44, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 8px;
    transition: background 0.2s ease;
}

.playlist-track-row:hover {
    background: rgba(29, 185, 84, 0.08);
    border-color: rgba(29, 185, 84, 0.2);
}

.track-number {
    font-weight: 700;
    color: #64748B;
    width: 28px;
}

.track-info {
    flex-grow: 1;
    margin-left: 12px;
}

.track-title {
    font-weight: 600;
    color: #F8FAFC;
    font-size: 0.95rem;
}

.track-artist {
    color: #94A3B8;
    font-size: 0.85rem;
}

.track-badge {
    background: rgba(255, 255, 255, 0.05);
    color: #A78BFA;
    font-size: 0.75rem;
    padding: 3px 8px;
    border-radius: 6px;
    border: 1px solid rgba(167, 139, 250, 0.2);
}

/* Tab Styling Enhancements */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: rgba(15, 20, 30, 0.5);
    padding: 6px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 500;
    color: #94A3B8;
}

.stTabs [aria-selected="true"] {
    background-color: rgba(29, 185, 84, 0.15) !important;
    color: #1DB954 !important;
    font-weight: 600;
}
</style>
"""
