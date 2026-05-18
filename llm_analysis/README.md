# LLM Analysis

This folder contains model-based analysis workflows for the FED text corpora in this workspace.

## Structure

- `scripts/` - runnable Python scripts for inference and aggregation
- `outputs/sentence_level/` - sentence-by-sentence model predictions
- `outputs/document_level/` - aggregated document-level outputs
- `notebooks/` - optional exploratory notebooks
- `logs/` - run summaries and diagnostics

## First script

The first analysis script is:

- `scripts/run_centralbankroberta_analysis.py`

It uses the public Hugging Face models from the `CentralBankRoBERTa` project for:

- economic agent classification: `households`, `firms`, `financial sector`, `government`, `central bank`
- binary sentiment classification: `positive`, `negative`

## Expected inputs

By default, the script reads:

- `data/metadata/fed_speeches.csv`
- `data/metadata/fomc_minutes_statements.csv`
- text files from `data/raw/fed_texts/fed_speeches/`
- text files from `data/raw/fed_texts/fomc_minutes_statements/`

The metadata files are standardized so `text_path` values point to the local text corpora.

## Quick input check

Before running model inference, validate required files/columns:

```bash
python llm_analysis/scripts/check_pipeline_inputs.py
```

## Install dependencies

```bash
pip install pandas torch transformers tqdm
```

## Run

```bash
python llm_analysis/scripts/run_centralbankroberta_analysis.py
```

## Outputs

The script writes:

- sentence-level CSVs per corpus
- document-level CSVs per corpus
- combined sentence-level and document-level CSVs
- a JSON run summary in `logs/`

## Build event-impact table

After document-level outputs exist, build a modeling-ready table with BTC post-event windows:

```bash
python src/market/build_event_impact_dataset.py
```

This writes:

- `data/processed/event_impact_dataset.csv`

## Notes

- The model is best used at sentence level; document-level metrics are aggregated from sentence predictions.
- Sentiment is binary only; there is no built-in neutral class.
- The first run downloads model weights from Hugging Face.
