"""
data/processing/run_cleaning_fed_speeches.py
--------------------------------------------
Clean texts from data/raw/fed_texts/fed_speeches and write processed files to
 data/processed/fed_texts/fed_speeches.

Usage (from repo root):
    .venv\Scripts\python.exe data\processing\run_cleaning_fed_speeches.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.processing.cleaners import PIPELINE_SPEECHES, apply_pipeline

RAW_DIR = ROOT / "data/raw/fed_texts/fed_speeches"
OUT_DIR = ROOT / "data/processed/fed_texts/fed_speeches"


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(RAW_DIR.glob("*.txt"))

    print(f"Processing {len(raw_files)} files -> {OUT_DIR}")

    for raw_path in raw_files:
        raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
        cleaned_text = apply_pipeline(raw_text, PIPELINE_SPEECHES)
        (OUT_DIR / raw_path.name).write_text(cleaned_text, encoding="utf-8")

    print("Done.")


if __name__ == "__main__":
    run()
