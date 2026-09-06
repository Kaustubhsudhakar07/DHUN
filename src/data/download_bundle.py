"""
Automatic Data & Model Artifact Provisioner for Streamlit Cloud Deployment.

Ensures all required pre-trained models and processed catalog files are present.
Downloads the release bundle directly into the project directory if running in the cloud.
"""

import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple
import requests

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


def download_with_requests(
    url: str, dest_path: Path, progress_callback: Optional[Callable[[str], None]] = None
) -> None:
    """Download a file via streaming requests with progress updates."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; DhunDNABot/1.0)"}
    resp = requests.get(url, stream=True, timeout=(15, 600), headers=headers)
    resp.raise_for_status()

    total_size = int(resp.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 1024 * 512  # 512 KB chunks

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            if progress_callback and total_size > 0:
                pct = int(min(100, downloaded * 100 / total_size))
                mb_curr = downloaded / (1024 * 1024)
                mb_tot = total_size / (1024 * 1024)
                progress_callback(
                    f"Downloading models & catalog: {mb_curr:.1f} / {mb_tot:.1f} MB ({pct}%)..."
                )


def ensure_bundle_downloaded(
    project_root: Path, progress_callback: Optional[Callable[[str], None]] = None
) -> Tuple[bool, str]:
    """Download and extract the model & catalog bundle if missing.

    Returns:
        Tuple of (success: bool, message: str)
    """
    if is_bundle_present(project_root):
        return True, "Bundle already present"

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
    errors = []

    # Strategy 1: Direct HTTP URL (GitHub Releases or custom host)
    target_url = bundle_url or DEFAULT_GITHUB_RELEASE_URL
    try:
        if progress_callback:
            progress_callback("Connecting to release asset download...")
        logger.info("Downloading bundle from URL: %s", target_url)

        download_with_requests(target_url, zip_tmp, progress_callback=progress_callback)

        if zip_tmp.exists() and zipfile.is_zipfile(zip_tmp):
            if progress_callback:
                progress_callback("Extracting models and catalog (unpacking 100k tracks)...")
            with zipfile.ZipFile(zip_tmp, "r") as z:
                z.extractall(project_root)
            zip_tmp.unlink(missing_ok=True)
            if is_bundle_present(project_root):
                return True, "Extracted successfully"
            else:
                missing = [
                    str(r)
                    for r in REQUIRED_RELATIVE_FILES
                    if not (project_root / r).exists()
                ]
                return False, f"Zip extracted, but missing files: {', '.join(missing)}"
        else:
            errors.append(
                f"Downloaded file is not a valid zip archive (size: {zip_tmp.stat().st_size if zip_tmp.exists() else 0} bytes)."
            )
            if zip_tmp.exists():
                zip_tmp.unlink(missing_ok=True)
    except Exception as e:
        logger.error("Direct URL download failed: %s", e)
        errors.append(f"Direct download error: {str(e)}")
        if zip_tmp.exists():
            zip_tmp.unlink(missing_ok=True)

    # Strategy 2: Google Drive via gdown if configured
    if gdrive_id and len(str(gdrive_id).strip()) > 5 and gdrive_id != "YOUR_FILE_ID":
        try:
            import gdown

            if progress_callback:
                progress_callback("Downloading data bundle from Google Drive...")
            logger.info("Downloading bundle from Google Drive file ID: %s", gdrive_id)
            gdown.download(id=str(gdrive_id).strip(), output=str(zip_tmp), quiet=False)
            if zip_tmp.exists() and zipfile.is_zipfile(zip_tmp):
                if progress_callback:
                    progress_callback("Extracting models and music catalog...")
                with zipfile.ZipFile(zip_tmp, "r") as z:
                    z.extractall(project_root)
                zip_tmp.unlink(missing_ok=True)
                if is_bundle_present(project_root):
                    return True, "Extracted successfully from Google Drive"
                else:
                    return False, "Google Drive zip extracted, but required files missing."
        except Exception as e:
            logger.warning("Google Drive download failed: %s", e)
            errors.append(f"Google Drive error: {str(e)}")
            if zip_tmp.exists():
                zip_tmp.unlink(missing_ok=True)

    final_msg = " | ".join(errors) if errors else "Download failed"
    return is_bundle_present(project_root), final_msg
