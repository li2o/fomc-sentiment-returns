# FOMC Communication Sentiment and Financial Market Returns

This repository contains the code and data for the bachelor's thesis:

> **"Federal Reserve Communication and Financial Market
Reactions:
Evidence from Agent-Specific Sentiment in FOMC Documents"**  
> Lionel Zaugg, University of St. Gallen (HSG), 2026

The thesis examines whether agent-specific sentiment in Federal Reserve communication is associated with subsequent returns on the S&P 500, two-year Treasury futures, and Bitcoin across horizons of 0–9 calendar days. Sentiment is measured sentence-by-sentence using [CentralBankRoBERTa](https://huggingface.co/Moritz-Pfeifer), a RoBERTa model fine-tuned on central bank texts.

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
│   │   └── bitcoin_bitstamp_1h.csv            # BTC/USD hourly OHLCV (Bitstamp, 2012–2025)
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
├── LICENSE
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

Note: Bitstamp hourly Bitcoin data is included from January 2012. Bitcoin trading was very illiquid before 2014; the thesis discusses this data limitation. S&P 500 and two-year Treasury futures data must be fetched by the user (see [Data availability](#data-availability-and-third-party-data)).

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

**S&P 500 and two-year Treasury futures** must be fetched separately (see [Data availability](#data-availability-and-third-party-data)) and saved as `data/market/spx_yahoo_1d.csv` and `data/market/zt_yahoo_1d.csv`.

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

**Visualise a single document** (colour-coded HTML as in Figure 3 of the thesis):
```bash
python llm_analysis/scripts/visualize_document.py --document-id fomc_00134
```

### 4. Build Master Datasets

Merges LLM sentiment scores with asset returns and monetary-policy-surprise controls.

```bash
python analysis/build_minutes_master_dataset.py
python analysis/build_press_conferences_master_dataset.py
```

**Return construction:** `r_h = log(P_close_{t+h} / P_open_t)` where `t` is the event day and `h ∈ {0,1,2,3,5,7,9}` calendar days. For Bitcoin the event-day open is the Bitstamp candle nearest to the document release time; for SPX and ZT it is the US session opening price on the event day.

Outputs: `data/master_dataset_minutes_reduced.csv` and `data/master_dataset_press_conferences_reduced.csv`.

### 5. Statistical Analysis (R)

```bash
Rscript analysis/analysis.r
# Outputs: analysis/outputs/  (PDF and PNG figures, CSV coefficient tables)
```

Reproduces all figures and regression results in the thesis:

- **Descriptive:** sentence-count distributions, agent-share pie charts, agent net-sentiment box plots, rolling 4-document mean sentiment time series
- **Event-level local projections:** OLS with HC1 heteroskedasticity-robust standard errors per horizon `h`
- **Hypotheses:**
  - H1: Household + Firm sentiment → S&P 500 returns (minutes & press conferences)
  - H2: Household + Firm sentiment → Two-year Treasury futures returns
  - H3: Financial-sector sentiment → SPX, BTC, and ZT returns; BTC robustness on post-2020 sub-sample

**Required R packages:** `tidyverse`, `lubridate`, `broom`, `sandwich`, `lmtest`, `modelsummary`, `ragg`, `systemfonts`, `patchwork`  
The script installs any missing packages automatically on first run.

## Installation

```bash
pip install -r requirements.txt
python -m playwright install chromium   # only needed for scraping
```

## Data Availability and Third-Party Data

**Included in this repository:**

| Data | Source | License |
|---|---|---|
| FOMC minutes & policy statements (text) | Federal Reserve Board | Public domain ([Fed copyright policy](https://www.federalreserve.gov/accessibility.htm)) |
| Press conference transcripts (text) | Federal Reserve Board | Public domain |
| FOMC document metadata | Federal Reserve Board | Public domain |
| Bitcoin OHLCV 1h (Bitstamp) | Bitstamp via public API | Included for research reproducibility |
| MPS controls | Bauer & Swanson (2023), FRBSF | See original paper |
| JK shock series | Jarociński & Karadi (2020) | See original paper |
| NFCI | Federal Reserve Bank of Chicago | Public domain |

**Not included — must be fetched by the user:**

| Data | Source | Why excluded |
|---|---|---|
| `data/market/spx_yahoo_1d.csv` | Yahoo Finance (ticker: `^GSPC`) | Yahoo Finance [terms of service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html) prohibit redistribution |
| `data/market/zt_yahoo_1d.csv` | Yahoo Finance (ticker: `ZT=F`) | Same |

To obtain these files, fetch daily OHLCV data from Yahoo Finance (e.g. via `yfinance`) and save them with columns `Date, Open, High, Low, Close, Volume` to the paths above. The `build_minutes_master_dataset.py` script documents the expected format.

**Attribution:** Federal Reserve texts are cited as sourced from the Board of Governors of the Federal Reserve System. Use of this repository does not imply endorsement by the Federal Reserve.

## License

The code in this repository is released under the [MIT License](LICENSE). The license applies to the scripts and analysis code only. Data files are subject to their respective upstream terms as described above.

## Notes

- The `data/master_dataset_*_reduced.csv` files are pre-built and included; steps 1–4 only need re-running to update the data.
- 11 documents lack a precise `release_time` and are excluded from return calculations.
- Monetary-policy-surprise controls (Bauer & Swanson 2023) are available through 2023; observations after 2023 enter regressions without this control.
- Plots use Computer Modern / CMU Serif if installed as a system font; they fall back to the default serif font otherwise.
