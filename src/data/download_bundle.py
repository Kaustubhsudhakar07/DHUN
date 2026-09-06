"""
Automatic Data & Model Artifact Provisioner for Streamlit Cloud Deployment.

Ensures all required pre-trained models and processed catalog files are present.
Downloads the release bundle directly into the project directory if running in the cloud.
"""

import logging
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("dhundna.bundle")

DEFAULT_GITHUB_RELEASE_URL = (
    "https://github.com/Kaustubhsudhakar07/DHUN/releases/download/v1.0.0/dhun_data_bundle.zip"
)

REQUIRED_RELATIVE_FILES = [
    Path("models") / "als_model.npz",
    Path("models") / "popularity_scores.json",
    Path("models") / "tfidf_matrix.npz",
    Path("models") / "tfidf_vectorizer.pkl",
    Path("data") / "processed" / "master_catalog.parquet",
    Path("data") / "processed" / "artist_mapping.json",
]


def is_bundle_present(project_root: Path) -> bool:
    """Check if all essential model and catalog files exist and are non-empty."""
    return all(
        (project_root / rel_path).exists()
        and (project_root / rel_path).stat().st_size > 500
        for rel_path in REQUIRED_RELATIVE_FILES
    )


def ensure_bundle_downloaded(
    project_root: Path, progress_callback: Optional[Callable[[str], None]] = None
) -> bool:
    """Download and extract the model & catalog bundle if missing.

    Uses an atomic .tmp file to prevent corrupt or partial extractions.
    """
    if is_bundle_present(project_root):
        return True

    bundle_url = None
    gdrive_id = None

    # Check Streamlit secrets
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            bundle_url = st.secrets.get("DATA_BUNDLE_URL", None)
            gdrive_id = st.secrets.get("GDRIVE_FILE_ID", None)
    except Exception:
        pass

    if not bundle_url:
        bundle_url = os.environ.get("DATA_BUNDLE_URL", None)
    if not gdrive_id:
        gdrive_id = os.environ.get("GDRIVE_FILE_ID", None)

    zip_tmp = project_root / "dhun_data_bundle.zip.tmp"

    # Strategy 1: Google Drive via gdown if configured
    if gdrive_id and gdrive_id != "YOUR_FILE_ID":
        try:
            import gdown

            if progress_callback:
                progress_callback("Downloading data bundle from Google Drive...")
            logger.info("Downloading bundle from Google Drive file ID: %s", gdrive_id)
            gdown.download(id=gdrive_id, output=str(zip_tmp), quiet=False)
            if zip_tmp.exists() and zipfile.is_zipfile(zip_tmp):
                if progress_callback:
                    progress_callback("Extracting models and music catalog...")
                with zipfile.ZipFile(zip_tmp, "r") as z:
                    z.extractall(project_root)
                zip_tmp.unlink(missing_ok=True)
                return is_bundle_present(project_root)
        except Exception as e:
            logger.warning("Google Drive download failed: %s", e)
            if zip_tmp.exists():
                zip_tmp.unlink(missing_ok=True)

    # Strategy 2: Direct URL (GitHub Releases by default)
    target_url = bundle_url or DEFAULT_GITHUB_RELEASE_URL
    try:
        if progress_callback:
            progress_callback("Connecting to release asset download...")
        logger.info("Downloading bundle from URL: %s", target_url)

        req = urllib.request.Request(
            target_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req) as resp, open(zip_tmp, "wb") as out_f:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 256  # 256 KB blocks
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total_size > 0:
                    pct = int(min(100, downloaded * 100 / total_size))
                    progress_callback(f"Downloading models & catalog ({downloaded / (1024*1024):.0f} / {total_size / (1024*1024):.0f} MB - {pct}%)...")

        if zip_tmp.exists() and zipfile.is_zipfile(zip_tmp):
            if progress_callback:
                progress_callback("Extracting models and catalog (unpacking 100k tracks)...")
            with zipfile.ZipFile(zip_tmp, "r") as z:
                z.extractall(project_root)
            zip_tmp.unlink(missing_ok=True)
            return is_bundle_present(project_root)
        else:
            logger.error("Downloaded file is not a valid zip archive.")
            if zip_tmp.exists():
                zip_tmp.unlink(missing_ok=True)
    except Exception as e:
        logger.error("Bundle download failed: %s", e)
        if zip_tmp.exists():
            zip_tmp.unlink(missing_ok=True)

    return is_bundle_present(project_root)
