"""
Clean texts from data/raw/fed_texts/fomc_press_conferences and write
processed files to data/processed/fed_texts/fomc_press_conferences.

Usage (from repo root):
    .venv\Scripts\python.exe data\processing\run_cleaning_fomc_press_conferences.py
"""

from pathlib import Path
import csv
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.processing.cleaners import PIPELINE_FOMC_PRESS_CONFERENCES, apply_pipeline_stepwise

RAW_DIR = ROOT / "data/raw/fed_texts/fomc_press_conferences"
OUT_DIR = ROOT / "data/processed/fed_texts/fomc_press_conferences"
REPORT_DIR = ROOT / "data/processed/reports"
REPORT_CSV = REPORT_DIR / "fomc_press_conferences_cleaning_report.csv"
QA_STEP_NAME = "remove_fomc_press_conference_reporter_questions"


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(RAW_DIR.glob("*.txt"))
    report_rows: list[dict[str, str | int | float]] = []

    print(f"Processing {len(raw_files)} files -> {OUT_DIR}")

    for raw_path in raw_files:
        raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
        step_results = apply_pipeline_stepwise(raw_text, PIPELINE_FOMC_PRESS_CONFERENCES)
        cleaned_text = step_results[-1]["text_after"] if step_results else raw_text
        (OUT_DIR / raw_path.name).write_text(cleaned_text, encoding="utf-8")

        qa_step = next((row for row in step_results if row["step"] == QA_STEP_NAME), None)
        qa_before = len(qa_step["text_before"]) if qa_step else len(cleaned_text)
        qa_after = len(qa_step["text_after"]) if qa_step else len(cleaned_text)
        qa_removed = qa_step["chars_removed"] if qa_step else 0
        qa_lines_before = qa_step["text_before"].count("\n") + 1 if qa_step and qa_step["text_before"] else 0
        qa_lines_after = qa_step["text_after"].count("\n") + 1 if qa_step and qa_step["text_after"] else 0
        qa_pct_removed = (qa_removed / qa_before * 100.0) if qa_before else 0.0

        report_rows.append(
            {
                "filename": raw_path.name,
                "raw_chars": len(raw_text),
                "cleaned_chars": len(cleaned_text),
                "qa_chars_before": qa_before,
                "qa_chars_after": qa_after,
                "qa_chars_removed": qa_removed,
                "qa_lines_before": qa_lines_before,
                "qa_lines_after": qa_lines_after,
                "qa_pct_removed": round(qa_pct_removed, 2),
            }
        )

    with open(REPORT_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "filename",
                "raw_chars",
                "cleaned_chars",
                "qa_chars_before",
                "qa_chars_after",
                "qa_chars_removed",
                "qa_lines_before",
                "qa_lines_after",
                "qa_pct_removed",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print("\nQuestion-removal control table (% removed at the reporter-question step):")
    print(f"{'file':<34} {'before':>10} {'after':>10} {'removed':>10} {'% removed':>10}")
    print("-" * 80)

    total_before = 0
    total_after = 0
    total_removed = 0

    for row in report_rows:
        qa_chars_before = int(row["qa_chars_before"])
        qa_chars_after = int(row["qa_chars_after"])
        qa_chars_removed = int(row["qa_chars_removed"])
        qa_pct_removed = float(row["qa_pct_removed"])
        total_before += qa_chars_before
        total_after += qa_chars_after
        total_removed += qa_chars_removed
        short_name = str(row["filename"])[:34]
        print(
            f"{short_name:<34}"
            f" {qa_chars_before:>10}"
            f" {qa_chars_after:>10}"
            f" {qa_chars_removed:>10}"
            f" {qa_pct_removed:>9.2f}%"
        )

    total_pct = (total_removed / total_before * 100.0) if total_before else 0.0
    print("-" * 80)
    print(f"{'TOTAL':<34} {total_before:>10} {total_after:>10} {total_removed:>10} {total_pct:>9.2f}%")
    print(f"\nReport: {REPORT_CSV}")
    print("Done.")


if __name__ == "__main__":
    run()
