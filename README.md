# FOMC Communication Sentiment and Financial Market Returns

This repository contains the code and data for the bachelor's thesis:

> **"FOMC Communication Sentiment and Financial Market Returns"**  
> Lionel Zaugg, University of St. Gallen (HSG), 2025

The thesis measures whether the tone of Federal Reserve communication — as classified sentence-by-sentence by [CentralBankRoBERTa](https://huggingface.co/Moritz-Pfeifer) — predicts returns on the S&P 500, two-year Treasury futures, and Bitcoin across horizons of 0–9 calendar days.

## Repository Structure

```
├── src/
│   ├── scraping/
│   │   ├── fetch_fomc_minutes_statements.py   # Scrape FOMC minutes & policy statements
│   │   └── fetch_fomc_press_conferences.py    # Scrape press conference PDF transcripts
│   └── market/
│       └── bitcoin_data_api.py                # Fetch BTC/USD hourly OHLCV from Bitstamp
├── data/
│   ├── metadata/
│   │   ├── fomc_minutes_statements.csv        # FOMC document metadata
│   │   └── fomc_press_conferences.csv         # Press conference metadata
│   ├── raw/fed_texts/
│   │   ├── fomc_minutes_statements/           # Raw minutes & statement text files
│   │   └── fomc_press_conferences/            # Raw press conference text files
│   ├── processed/fed_texts/
│   │   ├── fomc_minutes_statements/           # Cleaned minutes & statement text files
│   │   └── fomc_press_conferences/            # Cleaned press conference text files
│   ├── market/
│   │   ├── bitcoin_bitstamp_1h.csv            # BTC/USD hourly OHLCV (Bitstamp)
│   │   ├── spx_yahoo_1d.csv                   # S&P 500 daily OHLCV (Yahoo Finance)
│   │   └── zt_yahoo_1d.csv                    # 2-year Treasury futures daily (Yahoo Finance)
│   ├── control variables/
│   │   ├── monetary policy surprises data.xlsx # Bauer & Swanson (2023) MPS series
│   │   ├── shocks_fed_jk_t.csv                # Jarociński & Karadi (2020) shock series
│   │   └── financial condition.csv            # NFCI (Chicago Fed)
│   ├── master_dataset_minutes_reduced.csv     # Regression-ready minutes dataset
│   └── master_dataset_press_conferences_reduced.csv  # Regression-ready press conf. dataset
├── data/processing/
│   ├── cleaners.py                            # Shared text cleaning functions
│   ├── run_cleaning_fomc_materials.py         # Clean minutes & policy statements
│   └── run_cleaning_fomc_press_conferences.py # Clean press conference transcripts
├── llm_analysis/
│   ├── scripts/
│   │   ├── run_centralbankroberta_analysis.py # CentralBankRoBERTa inference
│   │   └── visualize_document.py              # HTML colour-coded document visualisation
│   └── outputs/
│       ├── sentence_level/                    # Per-sentence agent & sentiment labels
│       └── document_level/                    # Aggregated per-document scores
├── analysis/
│   ├── build_minutes_master_dataset.py        # Build minutes regression dataset
│   ├── build_press_conferences_master_dataset.py  # Build press conf. regression dataset
│   ├── analysis.r                             # All regression analysis & figures (R)
│   └── outputs/                              # Generated PDF/PNG figures and CSV results
├── requirements.txt
└── run_pipeline.sh
```

## Data Coverage

| Corpus | Documents | Sentences | Period |
|---|---|---|---|
| FOMC Minutes | 107 | ~27,000 | 2012–2025 |
| Policy Statements | 110 | ~3,400 | 2012–2025 |
| Press Conference Transcripts | 86 | ~19,000 | 2012–2025 |
| Bitcoin (Bitstamp 1h) | ~122,000 candles | — | 2012–2025 |

## Pipeline Overview

### 1. Data Collection

**FOMC minutes & policy statements:**
```bash
python src/scraping/fetch_fomc_minutes_statements.py \
  --start-year 2012 --end-year 2025 \
  --out data/metadata/fomc_minutes_statements.csv \
  --texts-dir data/raw/fed_texts/fomc_minutes_statements
```

**FOMC press conference transcripts (PDF):**
```bash
python src/scraping/fetch_fomc_press_conferences.py \
  --start-year 2012 --end-year 2025 \
  --out data/metadata/fomc_press_conferences.csv \
  --texts-dir data/raw/fed_texts/fomc_press_conferences
```

**Bitcoin hourly price data:**
```bash
python src/market/bitcoin_data_api.py
# Output: data/market/bitcoin_bitstamp_1h.csv
```

S&P 500 (`spx_yahoo_1d.csv`) and two-year Treasury futures (`zt_yahoo_1d.csv`) are fetched from Yahoo Finance via `yfinance` and stored in `data/market/`.

### 2. Text Cleaning

```bash
python data/processing/run_cleaning_fomc_materials.py
python data/processing/run_cleaning_fomc_press_conferences.py
```

The press conference cleaner retains only Chair/Chairman speaking turns and removes reporter interventions.

### 3. LLM Analysis — CentralBankRoBERTa

Runs two fine-tuned RoBERTa classifiers from [Moritz-Pfeifer/CentralBankRoBERTa](https://huggingface.co/Moritz-Pfeifer) on each sentence:

- **Agent classifier**: which economic agent is the sentence about? (Households, Firms, Financial Sector, Government, Central Bank)
- **Sentiment classifier**: Positive or Negative?

```bash
python llm_analysis/scripts/run_centralbankroberta_analysis.py \
  --fomc-text-dir       data/processed/fed_texts/fomc_minutes_statements \
  --press-conf-text-dir data/processed/fed_texts/fomc_press_conferences
```

Outputs:
- `llm_analysis/outputs/sentence_level/` — agent & sentiment label per sentence
- `llm_analysis/outputs/document_level/fomc_document_level.csv` — aggregated per-document scores
- `llm_analysis/outputs/document_level/press_conferences_document_level.csv`

**Key document-level variable:** `net_sentiment_{agent}` = positive share − negative share for that agent's sentences, range [−1, +1].

**Visualise a single document** (produces a colour-coded HTML file as in Figure 3 of the thesis):
```bash
python llm_analysis/scripts/visualize_document.py --document-id fomc_00134
```

### 4. Build Master Datasets

Merges LLM sentiment scores with asset returns and monetary-policy-surprise controls into two regression-ready CSVs.

```bash
python analysis/build_minutes_master_dataset.py
python analysis/build_press_conferences_master_dataset.py
```

**Return construction:** `r_h = log(P_close_{t+h} / P_open_t)` where `t` is the event day and `h ∈ {0,1,2,3,5,7,9}` calendar days. For Bitcoin the event-day open is the Bitstamp candle opening nearest to the release time; for SPX and ZT it is the opening price of the US trading session on the event day.

**Monetary-policy-surprise controls** from Bauer & Swanson (2023) are merged on meeting date.

Outputs:
- `data/master_dataset_minutes_reduced.csv`
- `data/master_dataset_press_conferences_reduced.csv`

### 5. Statistical Analysis (R)

```bash
Rscript analysis/analysis.r
# Outputs: analysis/outputs/  (PDF and PNG figures, CSV coefficient tables)
```

The R script (`analysis/analysis.r`) reproduces all figures and regression results in the thesis:

- **Descriptive:** sentence-count distributions, agent-share pie charts, agent net-sentiment box plots, rolling 4-document mean sentiment time series
- **Event-level local projections:** OLS with HC1 heteroskedasticity-robust standard errors, estimated separately for each horizon `h`
- **Hypotheses:**
  - H1: Household + Firm sentiment → S&P 500 returns (minutes & press conferences)
  - H2: Household + Firm sentiment → Two-year Treasury futures returns
  - H3: Financial-sector sentiment → SPX, BTC, and ZT returns; BTC robustness on post-2020 sub-sample

**Required R packages:** `tidyverse`, `lubridate`, `broom`, `sandwich`, `lmtest`, `modelsummary`, `ragg`, `systemfonts`, `patchwork`  
The script calls `install.packages()` for any missing packages automatically.

## Installation

```bash
pip install -r requirements.txt
python -m playwright install chromium   # only needed for scraping
```

Key Python dependencies: `transformers`, `torch`, `pandas`, `numpy`, `playwright`, `beautifulsoup4`, `yfinance`, `requests`, `openpyxl`

## Notes

- The `data/master_dataset_*_reduced.csv` files are pre-built and included in the repository; steps 1–4 only need to be re-run to update the data.
- 11 documents lack a precise `release_time` and are excluded from return calculations.
- Monetary-policy-surprise controls are available only through 2023; observations after 2023 enter regressions without this control.
- Plots use Computer Modern / CMU Serif if installed as a system font; they fall back to the default serif font otherwise.
