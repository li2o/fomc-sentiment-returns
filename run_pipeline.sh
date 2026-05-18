#!/usr/bin/env bash
# run_pipeline.sh
# ---------------
# Reproducible end-to-end data pipeline: fetch → clean → manifest.
#
# Run from the repo root:
#     bash run_pipeline.sh
#
# Optional environment overrides:
#     START_YEAR=2015 END_YEAR=2020 bash run_pipeline.sh
#
# Steps
# -----
#   1. Fetch Fed speeches          (src/scraping/fetch_fed_speeches.py)
#   2. Clean Fed speeches          (data/processing/run_cleaning_fed_speeches.py)
#   3. Fetch FOMC materials        (src/scraping/fetch_fomc_minutes_statements.py)
#      (minutes, policy statements, SEP)
#   4. Clean FOMC materials        (data/processing/run_cleaning_fomc_materials.py)
#   5. Generate raw data manifests (tools/generate_manifest.py)
#
# Notes
# -----
#   - Steps 1 and 3 make HTTP requests; expect them to take several minutes.
#   - Steps 2 and 4 are local-only and are fast.
#   - Step 5 writes SHA-256 checksums to data/metadata/manifest_*.json.
#     Run `python tools/generate_manifest.py --verify` at any later point to
#     confirm raw files have not changed.

set -euo pipefail

PYTHON="${PYTHON:-.venv/Scripts/python.exe}"
START_YEAR="${START_YEAR:-2012}"
END_YEAR="${END_YEAR:-$(date +%Y)}"

echo "========================================================"
echo " FED Data Pipeline"
echo " Range  : ${START_YEAR}–${END_YEAR}"
echo " Python : $PYTHON"
echo "========================================================"

# ------------------------------------------------------------------
# Step 1: Fetch Fed speeches
# ------------------------------------------------------------------
echo ""
echo "--- Step 1/5: Fetching Fed speeches (${START_YEAR}–${END_YEAR}) ---"
"$PYTHON" src/scraping/fetch_fed_speeches.py \
    --start-year "$START_YEAR" \
    --end-year   "$END_YEAR" \
    --out        data/metadata/fed_speeches.csv \
    --texts-dir  data/raw/fed_texts/fed_speeches

# ------------------------------------------------------------------
# Step 2: Clean Fed speeches
# ------------------------------------------------------------------
echo ""
echo "--- Step 2/5: Cleaning Fed speeches ---"
"$PYTHON" data/processing/run_cleaning_fed_speeches.py

# ------------------------------------------------------------------
# Step 3: Fetch FOMC materials (minutes, policy statements, SEP)
# ------------------------------------------------------------------
echo ""
echo "--- Step 3/5: Fetching FOMC materials (${START_YEAR}–${END_YEAR}) ---"
"$PYTHON" src/scraping/fetch_fomc_minutes_statements.py \
    --start-year "$START_YEAR" \
    --end-year   "$END_YEAR" \
    --out        data/metadata/fomc_minutes_statements.csv \
    --texts-dir  data/raw/fed_texts/fomc_minutes_statements

# ------------------------------------------------------------------
# Step 4: Clean FOMC materials
# ------------------------------------------------------------------
echo ""
echo "--- Step 4/5: Cleaning FOMC materials ---"
"$PYTHON" data/processing/run_cleaning_fomc_materials.py

# ------------------------------------------------------------------
# Step 5: Generate raw data manifests
# ------------------------------------------------------------------
echo ""
echo "--- Step 5/5: Generating raw data manifests ---"
"$PYTHON" tools/generate_manifest.py

echo ""
echo "========================================================"
echo " Pipeline complete."
echo " To verify raw files have not changed in a future run:"
echo "     $PYTHON tools/generate_manifest.py --verify"
echo "========================================================"
