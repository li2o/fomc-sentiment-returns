# LLM Analysis

Sentence-level classification of FOMC texts using [CentralBankRoBERTa](https://huggingface.co/Moritz-Pfeifer).

## Structure

- `scripts/run_centralbankroberta_analysis.py` — main inference script
- `scripts/visualize_document.py` — HTML colour-coded visualisation for a single document
- `outputs/sentence_level/` — per-sentence agent & sentiment labels
- `outputs/document_level/` — aggregated per-document scores

## Models

Two fine-tuned RoBERTa classifiers:

- **Agent classifier** — which economic agent is the sentence about? (Households, Firms, Financial Sector, Government, Central Bank)
- **Sentiment classifier** — Positive or Negative?

## Run

```bash
python llm_analysis/scripts/run_centralbankroberta_analysis.py \
  --fomc-text-dir       data/processed/fed_texts/fomc_minutes_statements \
  --press-conf-text-dir data/processed/fed_texts/fomc_press_conferences
```

The first run downloads model weights from Hugging Face (~500 MB). A GPU is recommended.

## Outputs

| File | Description |
|---|---|
| `outputs/sentence_level/fomc_sentence_level.csv` | Agent & sentiment label per sentence (minutes & statements) |
| `outputs/sentence_level/press_conferences_sentence_level.csv` | Agent & sentiment label per sentence (press conferences) |
| `outputs/document_level/fomc_document_level.csv` | Aggregated per-document scores (minutes & statements) |
| `outputs/document_level/press_conferences_document_level.csv` | Aggregated per-document scores (press conferences) |

**Key document-level variables:**
- `net_sentiment_{agent}` — positive share minus negative share for that agent's sentences, range [−1, +1]
- `agent_share_{agent}` — share of all sentences attributed to that agent
- `dominant_agent`, `dominant_sentiment`

## Visualise a single document

```bash
python llm_analysis/scripts/visualize_document.py --document-id fomc_00134
# Output: llm_analysis/outputs/fomc_00134_visualisation.html
```

## Notes

- Sentiment is binary (Positive / Negative); there is no neutral class.
- Document-level scores are aggregated from sentence-level predictions.
