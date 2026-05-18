#!/usr/bin/env bash
# run_pipeline.sh
# ---------------
# Reproducible end-to-end pipeline for the thesis:
#   "FOMC Communication Sentiment and Financial Market Returns"
#
# Run from the repo root:
#     bash run_pipeline.sh
#
# Optional environment overrides:
#     START_YEAR=2015 END_YEAR=2020 bash run_pipeline.sh
#
# Steps
# -----
#   1. Fetch FOMC minutes & policy statements  (src/scraping/fetch_fomc_minutes_statements.py)
#   2. Fetch FOMC press conference transcripts  (src/scraping/fetch_fomc_press_conferences.py)
#   3. Fetch Bitcoin hourly OHLCV data          (src/market/bitcoin_data_api.py)
#   4. Clean FOMC minutes & statements          (data/processing/run_cleaning_fomc_materials.py)
#   5. Clean press conference transcripts       (data/processing/run_cleaning_fomc_press_conferences.py)
#   6. Run CentralBankRoBERTa classification    (llm_analysis/scripts/run_centralbankroberta_analysis.py)
#   7. Build minutes master dataset             (analysis/build_minutes_master_dataset.py)
#   8. Build press conferences master dataset   (analysis/build_press_conferences_master_dataset.py)
#   9. Run R analysis                           (analysis/analysis.r)
#
# Notes
# -----
#   - Steps 1-3 make HTTP requests; expect several minutes each.
#   - Step 6 requires a GPU for reasonable speed (falls back to CPU).
#   - Step 9 requires R with packages: tidyverse, sandwich, lmtest, patchwork, broom, systemfonts.
#   - Market data for SPX and ZT (data/market/spx_yahoo_1d.csv, zt_yahoo_1d.csv) must be
#     present; fetch via Yahoo Finance (e.g. yfinance) if not available.

set -euo pipefail

PYTHON="${PYTHON:-.venv/Scripts/python.exe}"
RSCRIPT="${RSCRIPT:-Rscript}"
START_YEAR="${START_YEAR:-2012}"
END_YEAR="${END_YEAR:-$(date +%Y)}"

echo "========================================================"
echo " FOMC Sentiment Pipeline"
echo " Range  : ${START_YEAR}–${END_YEAR}"
echo " Python : $PYTHON"
echo " Rscript: $RSCRIPT"
echo "========================================================"

# ------------------------------------------------------------------
# Step 1: Fetch FOMC minutes & policy statements
# ------------------------------------------------------------------
echo ""
echo "--- Step 1/9: Fetching FOMC minutes & policy statements (${START_YEAR}–${END_YEAR}) ---"
"$PYTHON" src/scraping/fetch_fomc_minutes_statements.py \
    --start-year "$START_YEAR" \
    --end-year   "$END_YEAR" \
    --out        data/metadata/fomc_minutes_statements.csv \
    --texts-dir  data/raw/fed_texts/fomc_minutes_statements

# ------------------------------------------------------------------
# Step 2: Fetch FOMC press conference transcripts
# ------------------------------------------------------------------
echo ""
echo "--- Step 2/9: Fetching FOMC press conference transcripts (${START_YEAR}–${END_YEAR}) ---"
"$PYTHON" src/scraping/fetch_fomc_press_conferences.py \
    --start-year "$START_YEAR" \
    --end-year   "$END_YEAR" \
    --out        data/metadata/fomc_press_conferences.csv \
    --texts-dir  data/raw/fed_texts/fomc_press_conferences

# ------------------------------------------------------------------
# Step 3: Fetch Bitcoin hourly OHLCV data
# ------------------------------------------------------------------
echo ""
echo "--- Step 3/9: Fetching Bitcoin hourly data (Bitstamp) ---"
"$PYTHON" src/market/bitcoin_data_api.py

# ------------------------------------------------------------------
# Step 4: Clean FOMC minutes & statements
# ------------------------------------------------------------------
echo ""
echo "--- Step 4/9: Cleaning FOMC minutes & statements ---"
"$PYTHON" data/processing/run_cleaning_fomc_materials.py

# ------------------------------------------------------------------
# Step 5: Clean press conference transcripts
# ------------------------------------------------------------------
echo ""
echo "--- Step 5/9: Cleaning press conference transcripts ---"
"$PYTHON" data/processing/run_cleaning_fomc_press_conferences.py

# ------------------------------------------------------------------
# Step 6: Run CentralBankRoBERTa (agent + sentiment classification)
# ------------------------------------------------------------------
echo ""
echo "--- Step 6/9: Running CentralBankRoBERTa classification ---"
"$PYTHON" llm_analysis/scripts/run_centralbankroberta_analysis.py \
    --fomc-text-dir        data/processed/fed_texts/fomc_minutes_statements \
    --press-conf-text-dir  data/processed/fed_texts/fomc_press_conferences

# ------------------------------------------------------------------
# Step 7: Build minutes master dataset
# ------------------------------------------------------------------
echo ""
echo "--- Step 7/9: Building minutes master dataset ---"
"$PYTHON" analysis/build_minutes_master_dataset.py

# ------------------------------------------------------------------
# Step 8: Build press conferences master dataset
# ------------------------------------------------------------------
echo ""
echo "--- Step 8/9: Building press conferences master dataset ---"
"$PYTHON" analysis/build_press_conferences_master_dataset.py

# ------------------------------------------------------------------
# Step 9: Run R analysis (produces all figures and tables)
# ------------------------------------------------------------------
echo ""
echo "--- Step 9/9: Running R analysis ---"
"$RSCRIPT" analysis/analysis.r

echo ""
echo "========================================================"
echo " Pipeline complete."
echo " Outputs: analysis/outputs/"
echo "========================================================"
