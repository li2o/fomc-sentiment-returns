from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


AGENT_MODEL_NAME = "Moritz-Pfeifer/CentralBankRoBERTa-agent-classifier"
SENTIMENT_MODEL_NAME = "Moritz-Pfeifer/CentralBankRoBERTa-sentiment-classifier"

AGENT_FALLBACK_LABELS = {
    0: "households",
    1: "firms",
    2: "financial sector",
    3: "government",
    4: "central bank",
}

SENTIMENT_FALLBACK_LABELS = {
    0: "negative",
    1: "positive",
}

SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘(])")
WHITESPACE_REGEX = re.compile(r"\s+")


@dataclass
class CorpusConfig:
    name: str
    metadata_path: Path
    text_dir: Path


class SequenceClassifier:
    def __init__(
        self,
        model_name: str,
        fallback_labels: dict[int, str],
        batch_size: int = 16,
        max_length: int = 256,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self._build_label_map(fallback_labels)
        self.labels = [self.id2label[idx] for idx in sorted(self.id2label)]

    def _build_label_map(self, fallback_labels: dict[int, str]) -> dict[int, str]:
        config_map = getattr(self.model.config, "id2label", {}) or {}
        normalized = {}

        for idx, label in config_map.items():
            try:
                idx_int = int(idx)
            except (TypeError, ValueError):
                idx_int = idx

            label_str = str(label).strip()
            if not label_str or label_str.upper().startswith("LABEL_"):
                label_str = fallback_labels.get(idx_int, label_str)

            normalized[idx_int] = normalize_label(label_str)

        if not normalized:
            normalized = {idx: normalize_label(label) for idx, label in fallback_labels.items()}

        return normalized

    def predict(self, texts: list[str]) -> tuple[list[str], list[dict[str, float]]]:
        labels: list[str] = []
        probabilities: list[dict[str, float]] = []

        if not texts:
            return labels, probabilities

        with torch.inference_mode():
            for start in tqdm(
                range(0, len(texts), self.batch_size),
                desc=f"Running {self.model_name}",
                leave=False,
            ):
                batch = texts[start : start + self.batch_size]
                encodings = self.tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encodings = {key: value.to(self.device) for key, value in encodings.items()}
                logits = self.model(**encodings).logits
                probs = torch.softmax(logits, dim=-1).detach().cpu()
                pred_ids = probs.argmax(dim=-1).tolist()

                for pred_id, row_probs in zip(pred_ids, probs.tolist()):
                    label = self.id2label[pred_id]
                    prob_dict = {
                        self.id2label[idx]: float(prob)
                        for idx, prob in enumerate(row_probs)
                    }
                    labels.append(label)
                    probabilities.append(prob_dict)

        return labels, probabilities


def normalize_label(label: str) -> str:
    normalized = label.strip().lower().replace("_", " ")
    normalized = WHITESPACE_REGEX.sub(" ", normalized)

    alias_map = {
        "banks": "financial sector",
        "bank": "financial sector",
        "financial institutes": "financial sector",
        "financial institutions": "financial sector",
        "financial institute": "financial sector",
        "audience": normalized,
    }
    return alias_map.get(normalized, normalized)


def split_into_sentences(text: str, min_chars: int = 20) -> list[str]:
    cleaned = text.replace("\ufeff", " ").replace("\xa0", " ")
    cleaned = cleaned.replace("\r", "\n")
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    cleaned = WHITESPACE_REGEX.sub(" ", cleaned).strip()

    if not cleaned:
        return []

    parts = SENTENCE_SPLIT_REGEX.split(cleaned)
    sentences = []
    for part in parts:
        sentence = part.strip()
        if len(sentence) >= min_chars:
            sentences.append(sentence)
    return sentences


def resolve_text_path(raw_text_path: str, workspace_root: Path, preferred_text_dir: Path) -> Path | None:
    candidates: list[Path] = []

    if raw_text_path:
        normalized = raw_text_path.replace("\\", "/")
        candidates.append(workspace_root / normalized)
        candidates.append(preferred_text_dir / Path(normalized).name)
    else:
        return None

    filename = Path(raw_text_path).name
    candidates.append(preferred_text_dir / filename)
    candidates.append(workspace_root / filename)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def aggregate_document_rows(df: pd.DataFrame, agent_labels: list[str], sentiment_labels: list[str]) -> pd.DataFrame:
    document_rows: list[dict[str, object]] = []

    for doc_id, group in df.groupby("document_id", sort=False):
        first = group.iloc[0]
        row = {
            "document_id": doc_id,
            "source_corpus": first["source_corpus"],
            "resolved_text_path": first["resolved_text_path"],
            "sentence_count": int(len(group)),
        }

        metadata_columns = [
            column
            for column in group.columns
            if column.startswith("meta_")
        ]
        for column in metadata_columns:
            row[column[5:]] = first[column]

        agent_counts = group["agent_label"].value_counts()
        sentiment_counts = group["sentiment_label"].value_counts()

        for label in agent_labels:
            row[f"agent_count_{label}"] = int(agent_counts.get(label, 0))
            row[f"agent_share_{label}"] = float((group["agent_label"] == label).mean())
            row[f"agent_prob_mean_{label}"] = float(group[f"agent_prob_{label}"].mean())

        for label in sentiment_labels:
            row[f"sentiment_count_{label}"] = int(sentiment_counts.get(label, 0))
            row[f"sentiment_share_{label}"] = float((group["sentiment_label"] == label).mean())
            row[f"sentiment_prob_mean_{label}"] = float(group[f"sentiment_prob_{label}"].mean())

        row["dominant_agent"] = max(
            agent_labels,
            key=lambda label: (row[f"agent_share_{label}"], row[f"agent_prob_mean_{label}"]),
        )
        row["dominant_sentiment"] = max(
            sentiment_labels,
            key=lambda label: (row[f"sentiment_share_{label}"], row[f"sentiment_prob_mean_{label}"]),
        )
        row["net_sentiment"] = float(
            row.get("sentiment_share_positive", 0.0) - row.get("sentiment_share_negative", 0.0)
        )

        # Per-agent sentiment breakdown
        for agent in agent_labels:
            agent_sentences = group[group["agent_label"] == agent]
            n = len(agent_sentences)
            for sentiment in sentiment_labels:
                share = float((agent_sentences["sentiment_label"] == sentiment).mean()) if n > 0 else float("nan")
                row[f"sentiment_{sentiment}_share_{agent}"] = share
            pos = row.get(f"sentiment_positive_share_{agent}", float("nan"))
            neg = row.get(f"sentiment_negative_share_{agent}", float("nan"))
            row[f"net_sentiment_{agent}"] = (pos - neg) if (pos == pos and neg == neg) else float("nan")

        document_rows.append(row)

    return pd.DataFrame(document_rows)


def process_corpus(
    corpus: CorpusConfig,
    workspace_root: Path,
    output_root: Path,
    agent_model: SequenceClassifier,
    sentiment_model: SequenceClassifier,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    metadata = pd.read_csv(corpus.metadata_path, dtype=str).fillna("")

    sentence_records: list[dict[str, object]] = []
    missing_files: list[dict[str, str]] = []
    document_counter = 0

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"Preparing {corpus.name}"):
        raw_text_path = row.get("text_path", "")
        resolved = resolve_text_path(raw_text_path, workspace_root, corpus.text_dir)

        if resolved is None:
            missing_files.append(
                {
                    "source_corpus": corpus.name,
                    "text_path": raw_text_path,
                    "title": row.get("title", ""),
                    "date": row.get("date", ""),
                }
            )
            continue

        text = load_text(resolved)
        sentences = split_into_sentences(text)
        if not sentences:
            continue

        document_id = f"{corpus.name}_{document_counter:05d}"
        document_counter += 1

        base_record: dict[str, object] = {
            "document_id": document_id,
            "source_corpus": corpus.name,
            "resolved_text_path": str(resolved.relative_to(workspace_root)).replace("\\", "/"),
        }
        for column, value in row.items():
            base_record[f"meta_{column}"] = value

        for sentence_index, sentence in enumerate(sentences, start=1):
            record: dict[str, object] = dict(base_record)
            record["sentence_index"] = sentence_index
            record["sentence_text"] = sentence
            sentence_records.append(record)

    sentence_df = pd.DataFrame(sentence_records)
    if sentence_df.empty:
        return sentence_df, pd.DataFrame(), {
            "corpus": corpus.name,
            "documents_processed": 0,
            "sentences_processed": 0,
            "missing_files": missing_files,
        }

    texts = sentence_df["sentence_text"].tolist()
    agent_labels, agent_probabilities = agent_model.predict(texts)
    sentiment_labels, sentiment_probabilities = sentiment_model.predict(texts)

    sentence_df["agent_label"] = agent_labels
    sentence_df["sentiment_label"] = sentiment_labels

    for label in agent_model.labels:
        sentence_df[f"agent_prob_{label}"] = [probs.get(label, 0.0) for probs in agent_probabilities]

    for label in sentiment_model.labels:
        sentence_df[f"sentiment_prob_{label}"] = [probs.get(label, 0.0) for probs in sentiment_probabilities]

    document_df = aggregate_document_rows(sentence_df, agent_model.labels, sentiment_model.labels)

    sentence_output = output_root / "sentence_level" / f"{corpus.name}_sentence_level.csv"
    document_output = output_root / "document_level" / f"{corpus.name}_document_level.csv"
    sentence_df.to_csv(sentence_output, index=False)
    document_df.to_csv(document_output, index=False)

    summary = {
        "corpus": corpus.name,
        "documents_processed": int(document_df.shape[0]),
        "sentences_processed": int(sentence_df.shape[0]),
        "missing_files": missing_files,
        "sentence_output": str(sentence_output.relative_to(workspace_root)).replace("\\", "/"),
        "document_output": str(document_output.relative_to(workspace_root)).replace("\\", "/"),
    }
    return sentence_df, document_df, summary


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Run CentralBankRoBERTa agent and sentiment analysis on FED text corpora."
    )
    parser.add_argument("--workspace-root", type=Path, default=workspace_default)
    parser.add_argument("--speeches-metadata", type=Path, default=Path("data/metadata/fed_speeches.csv"))
    parser.add_argument("--speeches-text-dir", type=Path, default=Path("data/raw/fed_texts/fed_speeches"))
    parser.add_argument("--fomc-metadata", type=Path, default=Path("data/metadata/fomc_minutes_statements.csv"))
    parser.add_argument("--fomc-text-dir", type=Path, default=Path("data/raw/fed_texts/fomc_minutes_statements"))
    parser.add_argument("--output-dir", type=Path, default=Path("llm_analysis/outputs"))
    parser.add_argument("--log-dir", type=Path, default=Path("llm_analysis/logs"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--skip-speeches", action="store_true")
    parser.add_argument("--skip-fomc", action="store_true")
    return parser.parse_args()


def make_absolute(root: Path, maybe_relative: Path) -> Path:
    return maybe_relative if maybe_relative.is_absolute() else root / maybe_relative


def main() -> None:
    args = parse_args()
    workspace_root = args.workspace_root.resolve()
    output_root = make_absolute(workspace_root, args.output_dir)
    log_dir = make_absolute(workspace_root, args.log_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "sentence_level").mkdir(parents=True, exist_ok=True)
    (output_root / "document_level").mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    corpora: list[CorpusConfig] = []
    if not args.skip_speeches:
        corpora.append(
            CorpusConfig(
                name="speeches",
                metadata_path=make_absolute(workspace_root, args.speeches_metadata),
                text_dir=make_absolute(workspace_root, args.speeches_text_dir),
            )
        )
    if not args.skip_fomc:
        corpora.append(
            CorpusConfig(
                name="fomc",
                metadata_path=make_absolute(workspace_root, args.fomc_metadata),
                text_dir=make_absolute(workspace_root, args.fomc_text_dir),
            )
        )

    if not corpora:
        raise SystemExit("No corpora selected. Remove --skip-* flags or choose a corpus.")

    print("Loading models...")
    agent_model = SequenceClassifier(
        model_name=AGENT_MODEL_NAME,
        fallback_labels=AGENT_FALLBACK_LABELS,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    sentiment_model = SequenceClassifier(
        model_name=SENTIMENT_MODEL_NAME,
        fallback_labels=SENTIMENT_FALLBACK_LABELS,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    all_sentence_frames: list[pd.DataFrame] = []
    all_document_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    for corpus in corpora:
        print(f"Processing corpus: {corpus.name}")
        sentence_df, document_df, summary = process_corpus(
            corpus=corpus,
            workspace_root=workspace_root,
            output_root=output_root,
            agent_model=agent_model,
            sentiment_model=sentiment_model,
        )
        summaries.append(summary)
        if not sentence_df.empty:
            all_sentence_frames.append(sentence_df)
        if not document_df.empty:
            all_document_frames.append(document_df)

    if all_sentence_frames:
        combined_sentence_df = pd.concat(all_sentence_frames, ignore_index=True)
        combined_sentence_path = output_root / "sentence_level" / "all_corpora_sentence_level.csv"
        combined_sentence_df.to_csv(combined_sentence_path, index=False)
    else:
        combined_sentence_path = None

    if all_document_frames:
        combined_document_df = pd.concat(all_document_frames, ignore_index=True)
        combined_document_path = output_root / "document_level" / "all_corpora_document_level.csv"
        combined_document_df.to_csv(combined_document_path, index=False)
    else:
        combined_document_path = None

    run_summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(workspace_root),
        "models": {
            "agent": AGENT_MODEL_NAME,
            "sentiment": SENTIMENT_MODEL_NAME,
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "corpora": summaries,
        "combined_sentence_output": (
            str(combined_sentence_path.relative_to(workspace_root)).replace("\\", "/")
            if combined_sentence_path is not None
            else None
        ),
        "combined_document_output": (
            str(combined_document_path.relative_to(workspace_root)).replace("\\", "/")
            if combined_document_path is not None
            else None
        ),
    }

    summary_path = log_dir / f"run_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("Done.")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
