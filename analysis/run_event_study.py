"""
Standalone runner for the event study only.

Usage:
    python analysis/run_event_study.py
Output:
    analysis/outputs/event_study.html
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common

_common.PERIODS = ["2012–2023", "2024–2025"]
_common.PERIOD_COLORS = {
    "2012–2023": "#76b7b2",
    "2024–2025": "#e15759",
}

def _assign_period(date: pd.Timestamp) -> str:
    if pd.isna(date):
        return "unknown"
    return "2012–2023" if date.year <= 2023 else "2024–2025"

_common._assign_period = _assign_period

def _patched_load():
    df = pd.read_csv(_common.DOC_CSV)
    df["document_date"] = pd.to_datetime(df["document_date"], format="%m/%d/%Y", errors="coerce")
    df = df[df["document_type"].isin(_common.DOC_TYPES)].copy()
    df["period"] = df["document_date"].apply(_assign_period)
    return df[df["period"] != "unknown"].copy()

_common.load_data = _patched_load

from _common import CSS
import event_study

OUT = Path(__file__).resolve().parent / "outputs" / "event_study.html"


def main() -> None:
    print("Running event study...")
    section = event_study.run()

    date_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FOMC Event Study — Abnormal Bitcoin Returns</title>
<style>{CSS}</style>
</head>
<body>
<h1>FOMC Event Study — Abnormal Bitcoin Returns</h1>
<p class="note">
  Source: CentralBankRoBERTa sentiment scores applied to FOMC Minutes and Policy Statements (2012–2025).<br>
  Bitcoin prices: Bitstamp 1h OHLCV. Abnormal return = log return minus mean hourly return
  over 30-day estimation window &times; window length.<br>
  Periods: 2012–2023 | 2024–2025. Generated: {date_str}
</p>
{section}
</body>
</html>"""

    OUT.write_text(html, encoding="utf-8")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
