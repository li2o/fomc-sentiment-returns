from __future__ import annotations

from pathlib import Path

import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_FILES = {
    "fed_speeches": {
        "path": WORKSPACE_ROOT / "data" / "metadata" / "fed_speeches.csv",
        "columns": {
            "date",
            "type",
            "content_type",
            "title",
            "speech_url",
            "text_path",
        },
    },
    "fomc_minutes_statements": {
        "path": WORKSPACE_ROOT / "data" / "metadata" / "fomc_minutes_statements.csv",
        "columns": {
            "meeting_date",
            "document_date",
            "document_type",
            "title",
            "source_url",
            "text_path",
            "release_time",
        },
    },
    "bitcoin_hourly": {
        "path": WORKSPACE_ROOT / "data" / "market" / "bitcoin_bitstamp_1h.csv",
        "columns": {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        },
    },
}


def check_file(label: str, config: dict[str, object]) -> tuple[bool, str]:
    path = config["path"]
    assert isinstance(path, Path)

    if not path.exists():
        return False, f"MISSING file: {path.relative_to(WORKSPACE_ROOT)}"

    try:
        df = pd.read_csv(path, dtype=str, nrows=5)
    except Exception as exc:
        return False, f"READ ERROR in {path.relative_to(WORKSPACE_ROOT)}: {type(exc).__name__}: {exc}"

    required_columns = config["columns"]
    assert isinstance(required_columns, set)
    present_columns = set(df.columns)
    missing_columns = sorted(required_columns - present_columns)

    if missing_columns:
        return (
            False,
            f"MISSING columns in {path.relative_to(WORKSPACE_ROOT)}: {', '.join(missing_columns)}",
        )

    return True, f"OK {label}: {path.relative_to(WORKSPACE_ROOT)}"


def main() -> None:
    print("Validating canonical pipeline inputs...\n")
    failures: list[str] = []

    for label, config in REQUIRED_FILES.items():
        ok, message = check_file(label, config)
        print(message)
        if not ok:
            failures.append(message)

    if failures:
        print(f"\nValidation failed with {len(failures)} issue(s).")
        raise SystemExit(1)

    print("\nValidation succeeded. Public inputs are ready for NLP + BTC impact analysis.")


if __name__ == "__main__":
    main()
