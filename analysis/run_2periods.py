"""
Master runner with two periods: 2012–2023 and 2024–2025.
Patches _common constants before importing analysis modules.

Usage:
    python analysis/run_2periods.py
Output:
    analysis/outputs/full_analysis_2periods.html
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── patch _common before any analysis module imports it ──────────────────────
import _common

_common.PERIODS = ["2012–2023", "2024–2025"]
_common.PERIOD_COLORS = {
    "2012–2023": "#76b7b2",
    "2024–2025": "#e15759",
}

def _assign_period(date: pd.Timestamp) -> str:
    if pd.isna(date):
        return "unknown"
    if date.year <= 2023:
        return "2012–2023"
    return "2024–2025"

_common._assign_period = _assign_period

_original_load = _common.load_data
def _patched_load():
    df = pd.read_csv(_common.DOC_CSV)
    df["document_date"] = pd.to_datetime(df["document_date"], format="%m/%d/%Y", errors="coerce")
    df = df[df["document_type"].isin(_common.DOC_TYPES)].copy()
    df["period"] = df["document_date"].apply(_assign_period)
    df = df[df["period"] != "unknown"].copy()
    return df

_common.load_data = _patched_load

# ── now import analysis modules (they will use the patched _common) ───────────
import correlation_analysis
import ols_regression
import quantile_regression
import sign_test

OUT = Path(__file__).resolve().parent / "outputs" / "full_analysis_2periods.html"

CSS = _common.CSS


def main() -> None:
    # verify sample sizes
    df = _common.load_data()
    print("Sample sizes:")
    print(df.groupby(["document_type", "period"]).size().to_string())
    print()

    print("Running correlation analysis...")
    s1 = correlation_analysis.run()

    print("Running OLS regression...")
    s2 = ols_regression.run()

    print("Running quantile regression...")
    s3 = quantile_regression.run()

    print("Running sign test...")
    s4 = sign_test.run()

    date_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FOMC Sentiment → Bitcoin Return Analysis (2 Periods)</title>
<style>
{CSS}
</style>
</head>
<body>
<h1>FOMC Sentiment Effect on Bitcoin Returns</h1>
<p class="note">
  Source: CentralBankRoBERTa agent &amp; sentiment scores applied to FOMC Minutes and Policy Statements (2012–2025).<br>
  Bitcoin prices: Bitstamp 1h OHLCV. Baseline price = close of candle ending 1h before document release.<br>
  Periods: 2012–2023 | 2024–2025.<br>
  Generated: {date_str}
</p>
{s1}
{s2}
{s3}
{s4}
</body>
</html>"""

    OUT.write_text(html, encoding="utf-8")
    print(f"\nFull report saved to: {OUT}")


if __name__ == "__main__":
    main()
