"""
Automatic Data & Model Artifact Provisioner for Streamlit Cloud Deployment.

Ensures required pre-trained models and processed catalog files are available.
If running on Streamlit Cloud (where models are not committed to git due to GitHub's
100MB file limit), this module downloads and extracts `dhun_data_bundle.zip` automatically.
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


def is_bundle_present(project_root: Path) -> bool:
    """Check if critical data and model artifacts are already extracted."""
    required_files = [
        project_root / "models" / "als_model.npz",
        project_root / "data" / "processed" / "master_catalog.parquet",
    ]
    return all(p.exists() and p.stat().st_size > 1000 for p in required_files)


def ensure_bundle_downloaded(
    project_root: Path, progress_callback: Optional[Callable[[str], None]] = None
) -> bool:
    """Download and extract model & data bundle if missing.

    Supports:
    1. Direct URL from Streamlit Secrets `DATA_BUNDLE_URL` (or GitHub Release).
    2. Google Drive File ID from Streamlit Secrets `GDRIVE_FILE_ID` via gdown.
    3. Default GitHub Release URL as fallback.
    """
    if is_bundle_present(project_root):
        return True

    bundle_url = None
    gdrive_id = None

    # Check Streamlit secrets if available
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            bundle_url = st.secrets.get("DATA_BUNDLE_URL", None)
            gdrive_id = st.secrets.get("GDRIVE_FILE_ID", None)
    except Exception:
        pass

    # Check environment variables
    if not bundle_url:
        bundle_url = os.environ.get("DATA_BUNDLE_URL", None)
    if not gdrive_id:
        gdrive_id = os.environ.get("GDRIVE_FILE_ID", None)

    zip_path = project_root / "dhun_data_bundle.zip"

    # Strategy 1: Google Drive via gdown if GDRIVE_FILE_ID is configured
    if gdrive_id and gdrive_id != "YOUR_FILE_ID":
        try:
            import gdown

            if progress_callback:
                progress_callback("Downloading data bundle from Google Drive...")
            logger.info("Downloading bundle from Google Drive file ID: %s", gdrive_id)
            gdown.download(id=gdrive_id, output=str(zip_path), quiet=False)
            if zip_path.exists() and zip_path.stat().st_size > 10000:
                if progress_callback:
                    progress_callback("Extracting models and catalog...")
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(project_root)
                zip_path.unlink(missing_ok=True)
                return True
        except Exception as e:
            logger.warning("Google Drive download failed: %s", e)
            if zip_path.exists():
                zip_path.unlink(missing_ok=True)

    # Strategy 2: Direct HTTP URL (GitHub Releases or custom host)
    target_url = bundle_url or DEFAULT_GITHUB_RELEASE_URL
    try:
        if progress_callback:
            progress_callback(f"Downloading artifacts from cloud storage...")
        logger.info("Downloading bundle from URL: %s", target_url)

        def _reporthook(count, block_size, total_size):
            if progress_callback and total_size > 0:
                pct = int(min(100, count * block_size * 100 / total_size))
                progress_callback(f"Downloading models & catalog... {pct}%")

        req = urllib.request.Request(
            target_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as out_f:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 128
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total_size > 0:
                    pct = int(min(100, downloaded * 100 / total_size))
                    progress_callback(f"Downloading models & catalog... {pct}%")

        if zip_path.exists() and zip_path.stat().st_size > 10000:
            if progress_callback:
                progress_callback("Extracting models and catalog...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(project_root)
            zip_path.unlink(missing_ok=True)
            return True
    except Exception as e:
        logger.warning("Direct URL download failed: %s", e)
        if zip_path.exists():
            zip_path.unlink(missing_ok=True)

    return is_bundle_present(project_root)
