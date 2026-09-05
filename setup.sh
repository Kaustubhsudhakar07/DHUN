#!/bin/bash
# -----------------------------------------------------------
# setup.sh — Download model & data artifacts from Google Drive
# Runs once during Streamlit Cloud build step.
# -----------------------------------------------------------
set -e

pip install -q gdown

# Replace YOUR_FILE_ID with the actual Google Drive file ID
# after uploading dhun_data_bundle.zip to Google Drive.
FILE_ID="${GDRIVE_FILE_ID:-YOUR_FILE_ID}"

echo "📦 Downloading data bundle from Google Drive..."
gdown --id "$FILE_ID" -O dhun_data_bundle.zip

echo "📂 Extracting data bundle..."
unzip -o dhun_data_bundle.zip -d .
rm dhun_data_bundle.zip

echo "✅ Data & models ready!"
ls -la models/
ls -la data/processed/
