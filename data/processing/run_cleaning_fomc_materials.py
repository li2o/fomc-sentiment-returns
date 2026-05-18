"""
data/processing/run_cleaning_fomc_materials.py
-----------------------------------------------
Clean texts from the fomc_minutes_statements corpus and write
processed files to data/processed/fed_texts/fomc_minutes_statements/.

Usage (from repo root):
    .venv\\Scripts\\python.exe data\\processing\\run_cleaning_fomc_materials.py

What it does:
1. Reads raw files from:
       data/raw/fed_texts/fomc_minutes_statements/
2. Looks up each file's document_type in:
       data/metadata/fomc_minutes_statements.csv
3. Routes to the appropriate pipeline:
       Minutes              -> PIPELINE_FOMC_STRUCTURED_MINUTES
       Policy Statement     -> PIPELINE_FOMC_STRUCTURED_STATEMENTS
       Summary of Economic Projections -> PIPELINE_FOMC_STRUCTURED_SEP
4. Writes cleaned text files (same filename) to:
       data/processed/fed_texts/fomc_minutes_statements/
"""

from pathlib import Path
import sys
import csv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.processing.cleaners import (
    PIPELINE_FOMC_STRUCTURED_MINUTES,
    PIPELINE_FOMC_STRUCTURED_STATEMENTS,
    PIPELINE_FOMC_STRUCTURED_SEP,
    apply_pipeline,
)

RAW_DIR = ROOT / "data/raw/fed_texts/fomc_minutes_statements"
OUT_DIR = ROOT / "data/processed/fed_texts/fomc_minutes_statements"
META_CSV = ROOT / "data/metadata/fomc_minutes_statements.csv"

PIPELINE_MAP = {
    "minutes":                          PIPELINE_FOMC_STRUCTURED_MINUTES,
    "policy statement":                 PIPELINE_FOMC_STRUCTURED_STATEMENTS,
    "summary of economic projections":  PIPELINE_FOMC_STRUCTURED_SEP,
}


def load_doctype_map(csv_path: Path) -> dict[str, str]:
    """Return {filename: document_type} from the metadata CSV."""
    mapping = {}
    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            text_path = row.get("text_path", "")
            filename = Path(text_path).name
            mapping[filename] = row.get("document_type", "").strip()
    return mapping


def pick_pipeline(document_type: str) -> list:
    key = document_type.lower()
    pipeline = PIPELINE_MAP.get(key)
    if pipeline is None:
        # Fallback: use Minutes pipeline for unknown types
        print(f"  [WARN] Unknown document_type '{document_type}', using Minutes pipeline.")
        pipeline = PIPELINE_FOMC_STRUCTURED_MINUTES
    return pipeline


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    doctype_map = load_doctype_map(META_CSV)
    raw_files = sorted(RAW_DIR.glob("*.txt"))

    counts = {"minutes": 0, "policy statement": 0, "summary of economic projections": 0, "unknown": 0}
    skipped = 0

    print(f"Processing {len(raw_files)} files -> {OUT_DIR}")

    for raw_path in raw_files:
        doc_type = doctype_map.get(raw_path.name, "")
        if not doc_type:
            print(f"  [SKIP] {raw_path.name} not found in metadata CSV.")
            skipped += 1
            continue

        pipeline = pick_pipeline(doc_type)
        raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
        cleaned_text = apply_pipeline(raw_text, pipeline)

        out_path = OUT_DIR / raw_path.name
        out_path.write_text(cleaned_text, encoding="utf-8")

        key = doc_type.lower()
        if key in counts:
            counts[key] += 1
        else:
            counts["unknown"] += 1

    print("\nDone.")
    print(f"  Minutes:            {counts['minutes']}")
    print(f"  Policy Statements:  {counts['policy statement']}")
    print(f"  SEP:                {counts['summary of economic projections']}")
    if counts["unknown"]:
        print(f"  Unknown type:       {counts['unknown']}")
    if skipped:
        print(f"  Skipped (no meta):  {skipped}")
    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    run()
