# FOMC Sentiment & Bitcoin Returns

This project scrapes, cleans, and analyses Federal Reserve text data (FOMC Minutes, Policy Statements, Fed Speeches) with the goal of measuring the effect of central bank communication sentiment on Bitcoin returns.

## Project Structure

```
Code/
├── src/
│   ├── scraping/
│   │   ├── fetch_fed_speeches.py              # Scrape Fed speeches & testimony
│   │   └── fetch_fomc_minutes_statements.py   # Scrape FOMC minutes & statements
│   └── market/
│       ├── bitcoin_data_api.py                # Fetch BTC/USD hourly OHLCV from Bitstamp
│       └── build_event_impact_dataset.py      # Combine LLM signals with BTC return windows
├── data/
│   ├── metadata/
│   │   ├── fomc_minutes_statements.csv        # FOMC document metadata (canonical)
│   │   ├── fed_speeches.csv                   # Fed speeches metadata (canonical)
│   ├── raw/fed_texts/
│   │   ├── fed_speeches/                      # Raw speech text files
│   │   └── fomc_minutes_statements/           # Raw FOMC text files
│   ├── processed/fed_texts/
│   │   ├── fed_speeches/                      # Cleaned speech text files
│   │   └── fomc_minutes_statements/           # Cleaned FOMC text files
│   └── bitcoin/
│       ├── bitcoin_bitstamp_1h.csv
│       └── bitcoin_bitstamp_1h.json
├── data/processing/
│   ├── cleaners.py                            # Text cleaning functions
│   ├── run_cleaning_fed_speeches.py
│   └── run_cleaning_fomc_materials.py
├── llm_analysis/
│   ├── scripts/
│   │   ├── run_centralbankroberta_analysis.py # Run CentralBankRoBERTa on text corpora
│   │   ├── check_pipeline_inputs.py
│   │   └── visualize_document.py              # HTML visualisation of a single document
│   ├── outputs/
│   │   ├── sentence_level/                    # Per-sentence agent & sentiment labels
│   │   └── document_level/                    # Aggregated per-document results + BTC returns
│   └── logs/
├── analysis/
│   ├── _common.py                             # Shared constants and helpers
│   ├── build_event_returns.py                 # Append BTC return windows to document CSV
│   ├── summary.py                             # Corpus overview & sentiment summary
│   ├── correlation_analysis.py                # Spearman correlation: sentiment vs. BTC returns
│   ├── ols_regression.py                      # OLS regression with HAC standard errors
│   ├── quantile_regression.py                 # Quantile regression (q10–q90)
│   ├── sign_test.py                           # Directional hit-rate / binomial test
│   ├── agent_sentiment_correlation.py         # Cross-agent sentiment correlation
│   ├── run_all.py                             # Master runner → combined HTML report
│   └── outputs/                              # Generated HTML reports
├── tools/                                     # Utility scripts (path fixing, manifests)
├── tests/
├── requirements.txt
└── run_pipeline.sh
```

## Data Availability

Bloomberg Terminal exports are not included in the public repository. The public analysis uses the included reduced master datasets and public/source-recreatable inputs. Scripts that previously consumed Bloomberg event-calendar exports now fall back to public/default timing assumptions.

## Pipeline Overview

### 1. Data Collection

**Scrape FOMC materials (minutes & statements):**
```bash
python src/scraping/fetch_fomc_minutes_statements.py \
  --start-year 2012 --end-year 2025 \
  --out data/metadata/fomc_minutes_statements.csv \
  --texts-dir data/raw/fed_texts/fomc_minutes_statements
```

**Scrape Fed speeches:**
```bash
python src/scraping/fetch_fed_speeches.py \
  --start-year 2012 --end-year 2025 \
  --out data/metadata/fed_speeches.csv \
  --texts-dir data/raw/fed_texts/fed_speeches
```

**Fetch Bitcoin hourly price data (Bitstamp):**
```bash
python src/market/bitcoin_data_api.py
# Output: data/bitcoin/bitcoin_bitstamp_1h.csv
```

### 2. Text Cleaning

```bash
python data/processing/run_cleaning_fomc_materials.py
python data/processing/run_cleaning_fed_speeches.py
# Output: data/processed/fed_texts/
```

### 3. LLM Analysis — CentralBankRoBERTa

Runs two fine-tuned RoBERTa classifiers from [Moritz-Pfeifer/CentralBankRoBERTa](https://huggingface.co/Moritz-Pfeifer) on each sentence:
- **Agent classifier**: which economic agent is the sentence about? (households, firms, financial sector, government, central bank)
- **Sentiment classifier**: positive or negative?

```bash
# FOMC minutes & statements only
python llm_analysis/scripts/run_centralbankroberta_analysis.py \
  --skip-speeches \
  --fomc-text-dir data/processed/fed_texts/fomc_minutes_statements

# Both corpora
python llm_analysis/scripts/run_centralbankroberta_analysis.py \
  --fomc-text-dir data/processed/fed_texts/fomc_minutes_statements \
  --speeches-text-dir data/processed/fed_texts/fed_speeches
```

Outputs:
- `llm_analysis/outputs/sentence_level/fomc_sentence_level.csv` — agent & sentiment label per sentence
- `llm_analysis/outputs/document_level/fomc_document_level.csv` — aggregated per-document scores

**Key document-level variables:**
- `net_sentiment_{agent}` — positive share minus negative share for that agent's sentences
- `agent_share_{agent}` — share of sentences attributed to that agent
- `dominant_agent`, `dominant_sentiment`

**Visualise a single document:**
```bash
python llm_analysis/scripts/visualize_document.py --document-id fomc_00134
# Output: llm_analysis/outputs/fomc_00134_visualisation.html
```

### 4. Bitcoin Return Windows

Appends Bitcoin log returns to the document-level CSV. Baseline price = close of the candle ending 1 hour before document release.

```bash
python analysis/build_event_returns.py
```

Return windows added: **1h, 3h, 9h, 24h, 72h, 144h, 216h, 288h, 360h** (simple + log returns).

### 5. Statistical Analysis

```bash
python analysis/run_all.py
# Output: analysis/outputs/full_analysis_3periods_extended.html
```

The analysis covers **FOMC Minutes and Policy Statements**, split into three sub-periods:
- **2012–2019** (pre-COVID, n≈60 per type)
- **2020–Jul 2023** (COVID + tightening cycle, n≈48)
- **Aug 2023–2025** (post-tightening, n≈30)

Four methods are applied per period, document type, agent, and return window:

| Script | Method | Key output |
|---|---|---|
| `correlation_analysis.py` | Spearman ρ | Heatmap + p-values |
| `ols_regression.py` | OLS with HAC SE (Newey-West) | β coefficients, coefficient plot |
| `quantile_regression.py` | Quantile regression q10–q90 | Tail asymmetry |
| `sign_test.py` | Binomial directional hit-rate | % correct sign prediction |

**Cross-agent sentiment correlation (Minutes only):**
```bash
python analysis/agent_sentiment_correlation.py
# Output: analysis/outputs/agent_sentiment_correlation.html
```

**Corpus summary with visualisations:**
```bash
python analysis/summary.py
# Output: analysis/outputs/summary.html
```

## Data Coverage

| Corpus | Documents | Sentences | Period |
|---|---|---|---|
| FOMC Minutes | 107 | ~27,000 | 2012–2025 |
| Policy Statements | 110 | ~3,400 | 2012–2025 |
| Summary of Econ. Projections | 36 | — | 2012–2025 |
| Bitcoin (Bitstamp 1h) | 122,736 candles | — | 2012–2025 |

## Installation

```bash
pip install -r requirements.txt
python -m playwright install chromium   # only needed for scraping
```

Key dependencies: `transformers`, `torch`, `statsmodels`, `seaborn`, `pandas`, `playwright`, `beautifulsoup4`

## Notes

- The `analysis/` scripts filter to `["Minutes", "Policy Statement"]` by default. Summary of Economic Projections documents are present in the document-level CSV but excluded from the regression analyses.
- HAC (Newey-West) standard errors are used for return windows ≥ 9h to account for potential autocorrelation.
- 11 documents have missing `release_time` and are excluded from Bitcoin return calculations.
- All analysis outputs are HTML files that can be opened directly in a browser.
