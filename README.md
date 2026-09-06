# 🎵 DhunDNA • Decode Your Music Taste
### Indian (Bollywood, Punjabi, Tamil, Telugu) + Global Music Discovery & Recommendation Engine (1990–2025)

A production-grade hybrid music recommendation and semantic discovery engine combining **collaborative filtering (ALS, BPR)**, **content-based tag profiling (TF-IDF)**, **multilingual neural semantic retrieval (`paraphrase-multilingual-MiniLM-L12-v2`)**, and **2024–2026 compliant Spotify OAuth 2.0 PKCE playlist integration** across a **100,558 real track catalog**.

---

## 🚀 Key Highlights & Capabilities

- **🎵 Master Music Catalog (100,558 Tracks)**:
  - **12,300+ Bollywood / Hindi Cinema Tracks** spanning 1960–2025 (classic Kishore Kumar, Lata Mangeshkar, R.D. Burman up to modern Pritam, Arijit Singh, Shreya Ghoshal, Amit Trivedi).
  - **Punjabi, Tamil, and Telugu Tracks** (Ap Dhillon, Diljit Dosanjh, Badshah, Anirudh, Sid Sriram).
  - **88,000+ Global Tracks** across 114 genres.
  - Zero synthetic / fabricated data — all real tracks, verified release years, and Spotify IDs.
- **🧬 User Music DNA Profiling**:
  - Source-agnostic taste representation (`UserMusicDNA`) capturing artist affinities, genre weights, language priors, era distributions, and mood acoustic signatures.
  - Seamlessly generated via manual input, connected Spotify playlists, or historical Last.fm scrobbles.
- **🎧 2024–2026 Spotify Developer Policy Compliant**:
  - Strictly uses permitted User Authorization endpoints (`GET /v1/me/playlists`, `GET /v1/playlists/{id}/tracks`) via OAuth 2.0 PKCE.
  - Zero reliance on deprecated/restricted endpoints (`/v1/audio-features`, `/v1/recommendations`).
  - Includes realistic demo mixes (*"My Bollywood Favorites"*, *"Desi Indie & Acoustic Vibes"*, *"Global 2000s Pop-Rock"*) for zero-config testing.
- **❤️ Similar Song Discovery**:
  - Entity-resolved seed track search (e.g. *Kesariya*, *Tum Se Hi*, *Summer High*, *Yellow*) returning acoustic, melodic, and collaborative similar tracks.
- **🔎 Multilingual Semantic Music Search**:
  - Natural language descriptive retrieval powered by `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` and N-gram TF-IDF.
  - Supports queries like *"90s Hindi romantic songs about heartbreak"*, *"energetic punjabi party dance beat"*, or *"rainy monsoon acoustic love"*.
- **🌐 Live Cloud Deployment**:
  - Hosted and accessible 24/7 on Streamlit Community Cloud: **[https://dhundna.streamlit.app](https://dhundna.streamlit.app)**.
- **100% Test Coverage**: **146 passed automated unit and integration tests** (135 legacy tests preserved + 11 new DhunDNA tests).

---

## 🖥️ Streamlined 6-Tab Interactive Dashboard

The dashboard provides a focused, dark-mode glassmorphic user experience across 6 core discovery interfaces:

1. **✍️ Enter Taste DNA**: Enter favorite songs, artists, genres, languages, eras, and moods to decode your unique taste profile and receive hybrid recommendations with explanation callouts.
2. **🎧 Connect Spotify**: Seamless OAuth 2.0 PKCE connection or instant demo mixes (*"My Bollywood Favorites"*, *"Desi Indie & Acoustic"*, *"Global Pop"*) to decode playlist DNA.
3. **❤️ Find Similar Songs**: Entity-resolved similarity search (e.g. *Kesariya*, *Tum Se Hi*, *Summer High*, *Yellow*) retrieving acoustically, melodically, and collaboratively related tracks.
4. **🔎 Semantic Music Search**: Natural language descriptive retrieval across 100,558 tracks using conversational prompts (e.g. *"rainy monsoon acoustic love song"*).
5. **📊 Music Catalog Insights**: Explore the 100,558-track catalog with interactive Plotly charts showing language breakdown (Hindi, Punjabi, Tamil, Telugu, Global) and release era distribution (1960–2025).
6. **👥 Collaborative Matrix**: Direct exploration of the 358,868-listener Last.fm collaborative matrix (ALS, BPR, Content, Popularity) and customizable interleaved discovery playlists.

---


## 🏛️ System Architecture

```
                                  ┌───────────────────────────┐
                                  │   User Listening Events   │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
              ┌──────────────────────────────────────────────────────────────────┐
              │                Candidate Generation & Retrieval                  │
              │                                                                  │
              │   ┌────────────────┐   ┌─────────────────┐   ┌───────────────┐   │
              │   │   ALS Matrix   │   │  TF-IDF Centroid│   │  Global Log   │   │
              │   │ Factorization  │   │  Content Match  │   │  Popularity   │   │
              │   └───────┬────────┘   └────────┬────────┘   └───────┬───────┘   │
              └───────────┼─────────────────────┼────────────────────┼───────────┘
                          │                     │                    │
                          ▼                     ▼                    ▼
              ┌──────────────────────────────────────────────────────────────────┐
              │            Dynamic Hybrid Re-Ranker & Normalizer                 │
              │       Score = 0.40·S_CF + 0.30·S_Content + 0.30·S_Popularity     │
              └─────────────────────────────────┬────────────────────────────────┘
                                                │
                                                ▼
              ┌──────────────────────────────────────────────────────────────────┐
              │           Explainability & Playlist Synthesis Engine             │
              │   • Anchor Artists ("Because you listen to X")                   │
              │   • Shared Genre/Mood Badges                                     │
              │   • Interleaved Track Playlist Sequencing                        │
              └────────────────────────┬─────────────────┬───────────────────────┘
                                       │                 │
                                       ▼                 ▼
                        ┌────────────────────┐     ┌─────────────────────┐
                        │ FastAPI REST Server│     │ Streamlit Web App   │
                        │ (Port 8000)        │     │ (Port 8501)         │
                        └────────────────────┘     └─────────────────────┘
```

---

## 📂 Project Structure

```
music-recommender/
├── api/                       # FastAPI REST API
│   ├── main.py                # Server app, lifespan state, and route handlers
│   └── schemas.py             # Pydantic request/response validation models
├── app/                       # Streamlit Interactive Dashboard
│   ├── streamlit_app.py       # Multi-tab modern dark-mode application
│   ├── api_client.py          # Dual-mode API client (REST + direct fallback)
│   ├── components.py          # Plotly radar/donut/bar charts & UI cards
│   └── styles.py              # Custom CSS styling tokens & glassmorphic classes
├── data/
│   ├── raw/                   # Raw Last.fm 360k & HetRec datasets
│   └── processed/             # Parquet files & sparse interaction matrices (.npz)
├── models/                    # Trained model artifacts & evaluation metrics
│   ├── als_model.npz          # 64-factor Implicit ALS model weights
│   ├── bpr_model.npz          # 65-factor Implicit BPR model weights
│   ├── tfidf_matrix.npz       # 98,104 x 2,889 artist tag TF-IDF feature matrix
│   ├── tfidf_vectorizer.pkl   # Fitted tag vectorizer
│   ├── popularity_scores.json # Global smoothed log-play popularity priors
│   ├── hybrid_weights.json    # Optimal grid-search weights (0.4, 0.3, 0.3)
│   └── evaluation_results.json# Complete offline benchmark metrics
├── scripts/                   # CLI pipelines & training utilities
│   ├── train_models.py        # Model training entrypoint
│   ├── evaluate_models.py     # Evaluation & benchmarking suite
│   └── tune_hybrid_weights.py # Grid search optimizer for hybrid weights
├── src/                       # Core Recommender System Library
│   ├── config.py              # Central paths & hyperparameter defaults
│   ├── recommenders/          # ALS, BPR, Content, Popularity, Hybrid, TrackRecommender
│   ├── ranking/               # CandidatePool, ScoreNormalizer, PostFilter
│   ├── evaluation/            # Precision, Recall, NDCG, MAP, Diversity, Novelty
│   ├── explainability/        # ExplanationEngine & PlaylistGenerator
│   ├── data/                  # EntityResolver & MasterCatalog builder
│   ├── features/              # TF-IDF Tag features & Multilingual Semantic Engine
│   ├── profiles/              # UserMusicDNA taste profiling
│   ├── spotify/               # OAuth 2.0 PKCE Spotify API client
│   └── lastfm/                # Rate-limited Last.fm API client with fallback
└── tests/                     # 146 automated unit & integration tests
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11)
- Windows / macOS / Linux

### 2. Environment Setup
```bash
# Clone repository and navigate to project folder
git clone <repo-url>
cd music-recommender

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate       # Windows PowerShell
# source venv/bin/activate  # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables (Optional)
If you have a Last.fm API key, add it to `.env`:
```env
LASTFM_API_KEY=your_lastfm_api_key_here
LASTFM_API_SECRET=your_lastfm_api_secret_here
```
*(Note: If no API key is provided, the system automatically runs in offline mock mode with zero degradation in functionality).*

---

## 🏃 Running the Application

### Option A: Complete System (FastAPI + Streamlit UI)

In Terminal 1 (Start REST API Server):
```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```
- API Docs (Swagger UI): `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/health`

In Terminal 2 (Start Streamlit UI):
```bash
python -m streamlit run app/streamlit_app.py --server.port 8501
```
- Open your browser to `http://localhost:8501`.

### Option B: Standalone Streamlit UI (Direct In-Memory Mode)
If you prefer to run only the Streamlit application without running FastAPI:
```bash
python -m streamlit run app/streamlit_app.py
```
*The app automatically detects that the REST API is offline and falls back to running the ML models in-memory.*

### Option C: Live Cloud Deployment (Streamlit Community Cloud)
Access the live deployed production version directly in your browser with zero local setup:
👉 **[https://dhundna.streamlit.app](https://dhundna.streamlit.app)**

*(Models and master catalog are automatically provisioned on the cloud from GitHub Release assets via `src/data/download_bundle.py`).*

---

## 🧪 Testing & Verification

Run the comprehensive test suite across all 9 modules:
```bash
# Run all tests quietly
pytest tests/ -q

# Run specific suite with verbose output
pytest tests/test_hybrid.py -v
pytest tests/test_api.py -v
pytest tests/test_app.py -v
```

---

## 📖 CLI Training & Tuning Pipelines

To retrain or re-tune components from scratch:

```bash
# Train all models (ALS, BPR, Content-Based, Popularity)
python scripts/train_models.py

# Evaluate all models on validation split
python scripts/evaluate_models.py --split val --k 5 10 20

# Run hybrid weight grid search optimization
python scripts/tune_hybrid_weights.py --users 500 --step 0.1
```

---

