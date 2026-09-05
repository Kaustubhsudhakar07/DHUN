"""
Exploratory Data Analysis — Phase 4

Comprehensive analysis of both datasets to drive modeling decisions.
Generates statistics and visualizations saved to notebooks/figures/.

Run from project root:
    python notebooks/01_eda.py

Sections:
    1. Last.fm 360K Interactions Overview
    2. User Activity Analysis
    3. Artist Popularity (Long-Tail)
    4. Play Count Distribution
    5. Interaction Sparsity
    6. User Demographics (Profiles)
    7. HetRec 2011 Tags Analysis
    8. Cross-Dataset Overlap
    9. Modeling Implications Summary
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import (
    load_lastfm_360k_interactions,
    load_lastfm_360k_profiles,
    load_hetrec_artists,
    load_hetrec_tags,
    load_hetrec_user_artists,
    load_hetrec_user_tagged_artists,
    get_artist_tags,
)

# Output directory for figures
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Plotting style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def save_fig(fig, name: str):
    """Save figure to the figures directory."""
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] {path}")


# =========================================================================
# 1. LAST.FM 360K — LOAD AND OVERVIEW
# =========================================================================

def section_1_overview(interactions: pd.DataFrame):
    """Basic dataset statistics."""
    print("\n" + "=" * 70)
    print("1. LAST.FM 360K — DATASET OVERVIEW")
    print("=" * 70)

    print(f"\n  Total interaction rows:     {len(interactions):>12,}")
    print(f"  Unique users:               {interactions['user_id'].nunique():>12,}")
    print(f"  Unique artists:             {interactions['artist_name'].nunique():>12,}")

    # Missing values
    print(f"\n  --- Missing Values ---")
    for col in interactions.columns:
        n_miss = interactions[col].isna().sum()
        pct = n_miss / len(interactions) * 100
        print(f"  {col:25s}  {n_miss:>10,}  ({pct:.2f}%)")

    # Play count stats
    pc = interactions["play_count"].dropna()
    print(f"\n  --- Play Count Statistics ---")
    print(f"  Min:      {pc.min():>12,.0f}")
    print(f"  Max:      {pc.max():>12,.0f}")
    print(f"  Mean:     {pc.mean():>12,.1f}")
    print(f"  Median:   {pc.median():>12,.0f}")
    print(f"  Std:      {pc.std():>12,.1f}")
    print(f"  Q25:      {pc.quantile(0.25):>12,.0f}")
    print(f"  Q75:      {pc.quantile(0.75):>12,.0f}")
    print(f"  Q95:      {pc.quantile(0.95):>12,.0f}")
    print(f"  Q99:      {pc.quantile(0.99):>12,.0f}")

    # Zero play counts
    n_zero = (pc == 0).sum()
    print(f"\n  Rows with play_count = 0:  {n_zero:>10,} ({n_zero/len(interactions)*100:.2f}%)")

    return pc


# =========================================================================
# 2. USER ACTIVITY ANALYSIS
# =========================================================================

def section_2_user_activity(interactions: pd.DataFrame):
    """Analyze per-user interaction counts."""
    print("\n" + "=" * 70)
    print("2. USER ACTIVITY ANALYSIS")
    print("=" * 70)

    user_stats = interactions.groupby("user_id").agg(
        n_artists=("artist_name", "nunique"),
        total_plays=("play_count", "sum"),
        mean_plays=("play_count", "mean"),
    ).reset_index()

    print(f"\n  --- Artists per User ---")
    print(f"  Min:      {user_stats['n_artists'].min():>10,}")
    print(f"  Max:      {user_stats['n_artists'].max():>10,}")
    print(f"  Mean:     {user_stats['n_artists'].mean():>10,.1f}")
    print(f"  Median:   {user_stats['n_artists'].median():>10,.0f}")
    print(f"  Q25:      {user_stats['n_artists'].quantile(0.25):>10,.0f}")
    print(f"  Q75:      {user_stats['n_artists'].quantile(0.75):>10,.0f}")

    print(f"\n  --- Total Plays per User ---")
    print(f"  Min:      {user_stats['total_plays'].min():>10,}")
    print(f"  Max:      {user_stats['total_plays'].max():>10,}")
    print(f"  Mean:     {user_stats['total_plays'].mean():>10,.1f}")
    print(f"  Median:   {user_stats['total_plays'].median():>10,.0f}")

    # Users with very few interactions (cold-start analysis)
    thresholds = [1, 2, 3, 5, 10, 20, 50]
    print(f"\n  --- Users by Minimum Artists Listened ---")
    for t in thresholds:
        n = (user_stats["n_artists"] >= t).sum()
        pct = n / len(user_stats) * 100
        print(f"  >= {t:3d} artists:  {n:>10,} users  ({pct:.1f}%)")

    # Figure: Distribution of artists per user
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(user_stats["n_artists"].clip(upper=200), bins=100, color="#2196F3",
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Number of Artists")
    ax.set_ylabel("Number of Users")
    ax.set_title("Distribution of Artists per User (clipped at 200)")

    ax.axvline(user_stats["n_artists"].median(), color="red", linestyle="--",
               label=f'Median: {user_stats["n_artists"].median():.0f}')
    ax.legend()

    ax = axes[1]
    ax.hist(np.log10(user_stats["total_plays"].clip(lower=1)), bins=100,
            color="#FF9800", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("log10(Total Plays)")
    ax.set_ylabel("Number of Users")
    ax.set_title("Distribution of Total Plays per User (log scale)")
    ax.axvline(np.log10(user_stats["total_plays"].median()), color="red",
               linestyle="--",
               label=f'Median: {user_stats["total_plays"].median():,.0f}')
    ax.legend()

    fig.suptitle("User Activity Distribution", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "02_user_activity_distribution")

    return user_stats


# =========================================================================
# 3. ARTIST POPULARITY (LONG-TAIL ANALYSIS)
# =========================================================================

def section_3_artist_popularity(interactions: pd.DataFrame):
    """Analyze artist popularity distribution — the long tail."""
    print("\n" + "=" * 70)
    print("3. ARTIST POPULARITY — LONG-TAIL ANALYSIS")
    print("=" * 70)

    artist_stats = interactions.groupby("artist_name").agg(
        n_listeners=("user_id", "nunique"),
        total_plays=("play_count", "sum"),
    ).reset_index().sort_values("n_listeners", ascending=False).reset_index(drop=True)

    print(f"\n  --- Top 20 Artists by Listeners ---")
    top20 = artist_stats.head(20)
    for i, row in top20.iterrows():
        print(f"  {i+1:3d}. {row['artist_name'][:40]:40s}  "
              f"{row['n_listeners']:>8,} listeners  "
              f"{row['total_plays']:>12,} plays")

    # Long-tail statistics
    total_artists = len(artist_stats)
    top_1pct = int(total_artists * 0.01)
    top_5pct = int(total_artists * 0.05)
    top_10pct = int(total_artists * 0.10)

    total_listeners = artist_stats["n_listeners"].sum()
    top_1pct_listeners = artist_stats.head(top_1pct)["n_listeners"].sum()
    top_5pct_listeners = artist_stats.head(top_5pct)["n_listeners"].sum()
    top_10pct_listeners = artist_stats.head(top_10pct)["n_listeners"].sum()

    print(f"\n  --- Long-Tail Distribution ---")
    print(f"  Total artists:         {total_artists:>10,}")
    print(f"  Top 1% ({top_1pct:,}) cover:   {top_1pct_listeners/total_listeners*100:.1f}% of all listener-events")
    print(f"  Top 5% ({top_5pct:,}) cover:   {top_5pct_listeners/total_listeners*100:.1f}% of all listener-events")
    print(f"  Top 10% ({top_10pct:,}) cover:  {top_10pct_listeners/total_listeners*100:.1f}% of all listener-events")

    # Artists with very few listeners
    print(f"\n  --- Artists by Listener Count ---")
    for t in [1, 2, 5, 10, 50, 100]:
        n = (artist_stats["n_listeners"] <= t).sum()
        pct = n / total_artists * 100
        print(f"  <= {t:4d} listeners:  {n:>10,} artists  ({pct:.1f}%)")

    # Figure: Long-tail plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ranks = np.arange(1, len(artist_stats) + 1)
    ax.plot(ranks, artist_stats["n_listeners"].values, color="#E91E63", linewidth=0.5)
    ax.set_xlabel("Artist Rank")
    ax.set_ylabel("Number of Listeners")
    ax.set_title("Artist Popularity — Long Tail")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(top_1pct, color="orange", linestyle="--", alpha=0.7, label="Top 1%")
    ax.axvline(top_10pct, color="green", linestyle="--", alpha=0.7, label="Top 10%")
    ax.legend()

    ax = axes[1]
    cumulative = np.cumsum(artist_stats["n_listeners"].values) / total_listeners * 100
    ax.plot(ranks / total_artists * 100, cumulative, color="#9C27B0", linewidth=1.5)
    ax.set_xlabel("% of Artists (ranked by popularity)")
    ax.set_ylabel("Cumulative % of Listener-Events")
    ax.set_title("Cumulative Popularity Distribution")
    ax.axhline(80, color="red", linestyle=":", alpha=0.5, label="80% of interactions")
    ax.axhline(50, color="orange", linestyle=":", alpha=0.5, label="50% of interactions")
    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    fig.suptitle("Artist Popularity — Long Tail Analysis", fontsize=15,
                 fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "03_artist_long_tail")

    return artist_stats


# =========================================================================
# 4. PLAY COUNT DISTRIBUTION
# =========================================================================

def section_4_play_counts(interactions: pd.DataFrame, play_counts: pd.Series):
    """Analyze play count distribution."""
    print("\n" + "=" * 70)
    print("4. PLAY COUNT DISTRIBUTION")
    print("=" * 70)

    pc = play_counts.dropna()
    pc_positive = pc[pc > 0]

    print(f"\n  Total interactions:    {len(pc):>12,}")
    print(f"  With play_count > 0:   {len(pc_positive):>12,}")
    print(f"  With play_count = 0:   {(pc == 0).sum():>12,}")

    # Distribution in buckets
    buckets = [(1, 10), (11, 50), (51, 100), (101, 500),
               (501, 1000), (1001, 5000), (5001, 50000), (50001, int(pc.max()))]
    print(f"\n  --- Play Count Buckets ---")
    for lo, hi in buckets:
        n = ((pc >= lo) & (pc <= hi)).sum()
        pct = n / len(pc) * 100
        print(f"  {lo:>6,} – {hi:>6,}:  {n:>10,}  ({pct:.1f}%)")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(np.log10(pc_positive), bins=100, color="#4CAF50",
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel("log10(Play Count)")
    ax.set_ylabel("Frequency")
    ax.set_title("Play Count Distribution (log scale)")
    med_val = pc_positive.median()
    mean_val = pc_positive.mean()
    ax.axvline(np.log10(med_val), color="red", linestyle="--",
               label=f"Median: {med_val:.0f}")
    ax.axvline(np.log10(mean_val), color="orange", linestyle="--",
               label=f"Mean: {mean_val:.0f}")
    ax.legend()

    ax = axes[1]
    ax.boxplot([pc_positive.clip(upper=pc_positive.quantile(0.99))],
               vert=True, widths=0.6,
               patch_artist=True,
               boxprops=dict(facecolor="#03A9F4", alpha=0.7))
    ax.set_ylabel("Play Count")
    ax.set_title(f"Play Count Box Plot (clipped at 99th percentile = {pc_positive.quantile(0.99):,.0f})")
    ax.set_xticklabels(["Play Count"])

    fig.suptitle("Play Count Distribution Analysis", fontsize=15,
                 fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "04_play_count_distribution")


# =========================================================================
# 5. INTERACTION MATRIX SPARSITY
# =========================================================================

def section_5_sparsity(interactions: pd.DataFrame):
    """Calculate and visualize interaction matrix sparsity."""
    print("\n" + "=" * 70)
    print("5. INTERACTION MATRIX SPARSITY")
    print("=" * 70)

    n_users = interactions["user_id"].nunique()
    n_artists = interactions["artist_name"].nunique()
    n_interactions = len(interactions)
    total_cells = n_users * n_artists
    sparsity = 1 - (n_interactions / total_cells)
    density = n_interactions / total_cells

    print(f"\n  Users:             {n_users:>12,}")
    print(f"  Artists:           {n_artists:>12,}")
    print(f"  Total cells:       {total_cells:>12,}")
    print(f"  Non-zero cells:    {n_interactions:>12,}")
    print(f"  Sparsity:          {sparsity*100:>11.6f}%")
    print(f"  Density:           {density*100:>11.6f}%")
    print(f"  Avg interactions per user:    {n_interactions/n_users:>8.1f}")
    print(f"  Avg listeners per artist:    {n_interactions/n_artists:>8.1f}")

    print(f"\n  [INSIGHT] The interaction matrix is {sparsity*100:.4f}% sparse.")
    print(f"  This means each user has interacted with only "
          f"{n_interactions/n_users:.1f} out of {n_artists:,} artists on average.")
    print(f"  This extreme sparsity is WHY collaborative filtering with")
    print(f"  matrix factorization (ALS/BPR) is appropriate — these methods")
    print(f"  learn dense latent factors that generalize across the sparse matrix.")


# =========================================================================
# 6. USER DEMOGRAPHICS
# =========================================================================

def section_6_demographics(profiles: pd.DataFrame):
    """Analyze user demographic data from profiles."""
    print("\n" + "=" * 70)
    print("6. USER DEMOGRAPHICS (PROFILES)")
    print("=" * 70)

    print(f"\n  Total profiles:  {len(profiles):,}")

    # Gender
    print(f"\n  --- Gender ---")
    gender_counts = profiles["gender"].value_counts()
    for g, c in gender_counts.items():
        label = {"m": "Male", "f": "Female", "": "Not specified"}.get(g, g)
        print(f"  {label:20s}  {c:>10,}  ({c/len(profiles)*100:.1f}%)")

    # Age (after cleaning)
    age = profiles["age"].dropna()
    valid_age = age[(age >= 5) & (age <= 100)]  # Reasonable range
    invalid_age = len(age) - len(valid_age)

    print(f"\n  --- Age ---")
    print(f"  Raw range:       {age.min():.0f} – {age.max():.0f}")
    print(f"  Valid (5–100):   {len(valid_age):,} ({len(valid_age)/len(age)*100:.1f}%)")
    print(f"  Invalid:         {invalid_age:,} ({invalid_age/len(age)*100:.1f}%)")
    if len(valid_age) > 0:
        print(f"  Valid mean:      {valid_age.mean():.1f}")
        print(f"  Valid median:    {valid_age.median():.0f}")

    # Country
    country_counts = profiles["country"].value_counts()
    print(f"\n  --- Top 15 Countries ---")
    for country, count in country_counts.head(15).items():
        print(f"  {str(country)[:25]:25s}  {count:>8,}  ({count/len(profiles)*100:.1f}%)")

    # Figures
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Gender pie
    ax = axes[0]
    labels = [{"m": "Male", "f": "Female"}.get(g, "Not specified")
              for g in gender_counts.index[:3]]
    colors = ["#2196F3", "#E91E63", "#9E9E9E"]
    ax.pie(gender_counts.values[:3], labels=labels, autopct="%1.1f%%",
           colors=colors[:len(labels)], startangle=90)
    ax.set_title("Gender Distribution")

    # Age histogram
    ax = axes[1]
    if len(valid_age) > 0:
        ax.hist(valid_age, bins=50, color="#4CAF50", edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        ax.set_title(f"Age Distribution (valid: {len(valid_age):,})")
        ax.axvline(valid_age.median(), color="red", linestyle="--",
                   label=f"Median: {valid_age.median():.0f}")
        ax.legend()

    # Top countries bar
    ax = axes[2]
    top10_countries = country_counts.head(10)
    ax.barh(range(len(top10_countries)), top10_countries.values, color="#FF9800")
    ax.set_yticks(range(len(top10_countries)))
    ax.set_yticklabels([str(c)[:15] for c in top10_countries.index])
    ax.set_xlabel("Number of Users")
    ax.set_title("Top 10 Countries")
    ax.invert_yaxis()

    fig.suptitle("User Demographics", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "06_user_demographics")


# =========================================================================
# 7. HETREC 2011 TAGS ANALYSIS
# =========================================================================

def section_7_tags():
    """Analyze the HetRec 2011 tag data."""
    print("\n" + "=" * 70)
    print("7. HETREC 2011 — TAGS ANALYSIS")
    print("=" * 70)

    artists = load_hetrec_artists()
    tags = load_hetrec_tags()
    user_tagged = load_hetrec_user_tagged_artists()
    artist_tags = get_artist_tags(user_tagged, tags)

    print(f"\n  Total artists in HetRec:    {len(artists):,}")
    print(f"  Total unique tags:          {len(tags):,}")
    print(f"  Total tag assignments:      {len(user_tagged):,}")
    print(f"  Artist-tag pairs:           {len(artist_tags):,}")
    print(f"  Artists with >=1 tag:        {artist_tags['artist_id'].nunique():,}")

    # Tags per artist
    tags_per_artist = artist_tags.groupby("artist_id").size()
    print(f"\n  --- Tags per Artist ---")
    print(f"  Min:      {tags_per_artist.min():>8,}")
    print(f"  Max:      {tags_per_artist.max():>8,}")
    print(f"  Mean:     {tags_per_artist.mean():>8.1f}")
    print(f"  Median:   {tags_per_artist.median():>8.0f}")

    # Most popular tags (by number of artists they are applied to)
    tag_popularity = artist_tags.groupby("tag_value")["artist_id"].nunique().sort_values(ascending=False)
    print("\n  --- Top 30 Tags (by artists tagged) ---")
    for tag, count in tag_popularity.head(30).items():
        tag_str = str(tag).encode("ascii", "replace").decode("ascii")
        print(f"  {tag_str[:35]:35s}  {count:>6,} artists")

    # Tag frequency distribution
    tag_artist_counts = artist_tags.groupby("tag_value")["artist_id"].nunique()
    print(f"\n  --- Tag Frequency Distribution ---")
    for t in [1, 2, 5, 10, 50, 100]:
        n = (tag_artist_counts <= t).sum()
        pct = n / len(tag_artist_counts) * 100
        print(f"  Applied to <= {t:4d} artists:  {n:>6,} tags  ({pct:.1f}%)")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    top_tags = tag_popularity.head(20)
    ax.barh(range(len(top_tags)), top_tags.values, color="#9C27B0")
    ax.set_yticks(range(len(top_tags)))
    ax.set_yticklabels(top_tags.index)
    ax.set_xlabel("Number of Artists")
    ax.set_title("Top 20 Tags by Artist Coverage")
    ax.invert_yaxis()

    ax = axes[1]
    ax.hist(tags_per_artist.clip(upper=50), bins=50, color="#00BCD4",
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Number of Unique Tags")
    ax.set_ylabel("Number of Artists")
    ax.set_title("Tags per Artist Distribution (clipped at 50)")
    ax.axvline(tags_per_artist.median(), color="red", linestyle="--",
               label=f"Median: {tags_per_artist.median():.0f}")
    ax.legend()

    fig.suptitle("HetRec 2011 — Tag Analysis", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "07_hetrec_tags")

    return artist_tags, artists


# =========================================================================
# 8. CROSS-DATASET OVERLAP
# =========================================================================

def section_8_overlap(interactions: pd.DataFrame, hetrec_artists: pd.DataFrame,
                      artist_tags: pd.DataFrame):
    """Analyze overlap between Last.fm 360K and HetRec 2011."""
    print("\n" + "=" * 70)
    print("8. CROSS-DATASET OVERLAP ANALYSIS")
    print("=" * 70)

    # Normalize artist names for matching
    artists_360k = set(interactions["artist_name"].dropna().str.lower().str.strip().unique())
    artists_hetrec = set(hetrec_artists["name"].dropna().str.lower().str.strip().unique())
    tagged_artist_ids = set(artist_tags["artist_id"].unique())
    hetrec_tagged_names = set(
        hetrec_artists[hetrec_artists["artist_id"].isin(tagged_artist_ids)]["name"]
        .str.lower().str.strip().unique()
    )

    overlap = artists_360k & artists_hetrec
    overlap_tagged = artists_360k & hetrec_tagged_names

    print(f"\n  Artists in Last.fm 360K:       {len(artists_360k):>10,}")
    print(f"  Artists in HetRec 2011:        {len(artists_hetrec):>10,}")
    print(f"  Artists with tags in HetRec:   {len(hetrec_tagged_names):>10,}")
    print(f"  Name overlap (all):            {len(overlap):>10,}")
    print(f"  Name overlap (tagged only):    {len(overlap_tagged):>10,}")
    print(f"  360K artists with tag data:    {len(overlap_tagged)/len(artists_360k)*100:.2f}%")

    # How many 360K interactions are covered by tagged artists?
    interactions_lower = interactions["artist_name"].str.lower().str.strip()
    covered_interactions = interactions_lower.isin(hetrec_tagged_names).sum()
    print(f"\n  360K interactions with tag data:  {covered_interactions:>10,} "
          f"({covered_interactions/len(interactions)*100:.1f}%)")

    print(f"\n  [INSIGHT] Only {len(overlap_tagged)/len(artists_360k)*100:.1f}% of 360K artists "
          f"have tag data from HetRec.")
    print(f"  However, these tagged artists may cover a disproportionate share of")
    print(f"  interactions (popular artists are more likely to be tagged).")
    print(f"  The Last.fm API can supplement tags for untagged artists at runtime.")

    # Figure: Bar chart for dataset overlap
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["360K Only", "Overlap", "HetRec Only"]
    values = [
        len(artists_360k - artists_hetrec),
        len(overlap),
        len(artists_hetrec - artists_360k),
    ]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    bars = ax.bar(categories, values, color=colors)
    ax.set_ylabel("Number of Artists")
    ax.set_title("Artist Overlap Between Datasets")
    for bar, v in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 2000, f"{v:,}", ha="center", fontsize=11)
    save_fig(fig, "08_dataset_overlap_bar")

    return overlap_tagged


# =========================================================================
# 9. MODELING IMPLICATIONS SUMMARY
# =========================================================================

def section_9_summary(interactions, user_stats, artist_stats):
    """Summarize key findings and their modeling implications."""
    print("\n" + "=" * 70)
    print("9. MODELING IMPLICATIONS SUMMARY")
    print("=" * 70)

    n_users = interactions["user_id"].nunique()
    n_artists = interactions["artist_name"].nunique()
    n_interactions = len(interactions)
    sparsity = 1 - (n_interactions / (n_users * n_artists))

    print(f"""
  +------------------------------+-------------------------------------+
  | FINDING                      | IMPLICATION                         |
  +------------------------------+-------------------------------------+
  | Sparsity: {sparsity*100:.4f}%         | Matrix factorization (ALS/BPR)      |
  |                              | over memory-based CF                |
  +------------------------------+-------------------------------------+
  | Implicit feedback only       | Use ALS/BPR, NOT SVD/NMF on         |
  | (play counts, no ratings)    | fabricated ratings                  |
  +------------------------------+-------------------------------------+
  | Strong long-tail             | Popularity baseline will be strong  |
  | (top 1% artists dominate)    | but biased toward head items        |
  +------------------------------+-------------------------------------+
  | ~{user_stats[user_stats['n_artists']<5].shape[0]:,} users with <5 artists  | Min interaction threshold needed    |
  |                              | for train/test split                |
  +------------------------------+-------------------------------------+
  | HetRec tag coverage limited  | Content-based needs Last.fm API     |
  | (~{n_artists:,} vs 12,523 tagged)  | supplementation for full coverage   |
  +------------------------------+-------------------------------------+
  | Age data has invalid values  | Profile features unreliable for     |
  | (range: -1337 to 1002)       | demographic-based recommendation    |
  +------------------------------+-------------------------------------+
  | Play count heavily skewed    | Log-transform or confidence         |
  | (median << mean)             | weighting for ALS                   |
  +------------------------------+-------------------------------------+
    """)

    # Recommended preprocessing steps based on EDA
    print("  RECOMMENDED PREPROCESSING (based on EDA findings):")
    print("  1. Remove rows with play_count = 0 or NaN")
    print("  2. Remove users with < 5 artist interactions")
    print("  3. Remove artists with < 5 unique listeners")
    print("  4. Clean age: set invalid values (< 5 or > 100) to NaN")
    print("  5. Apply log1p transform to play counts for confidence weighting")
    print("  6. Use artist name (lowered, stripped) as matching key across datasets")
    print("  7. Supplement HetRec tags with Last.fm API tags where missing")


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 70)
    print("  EXPLORATORY DATA ANALYSIS — Music Recommendation System")
    print("  Phase 4")
    print("=" * 70)

    # Load data
    print("\n[LOADING] Last.fm 360K interactions...")
    interactions = load_lastfm_360k_interactions()
    print(f"  Loaded {len(interactions):,} rows")

    print("[LOADING] Last.fm 360K profiles...")
    profiles = load_lastfm_360k_profiles()
    print(f"  Loaded {len(profiles):,} rows")

    # Run all sections
    play_counts = section_1_overview(interactions)
    user_stats = section_2_user_activity(interactions)
    artist_stats = section_3_artist_popularity(interactions)
    section_4_play_counts(interactions, play_counts)
    section_5_sparsity(interactions)
    section_6_demographics(profiles)
    artist_tags, hetrec_artists = section_7_tags()
    section_8_overlap(interactions, hetrec_artists, artist_tags)
    section_9_summary(interactions, user_stats, artist_stats)

    print(f"\n{'=' * 70}")
    print(f"  EDA COMPLETE — Figures saved to {FIGURES_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
