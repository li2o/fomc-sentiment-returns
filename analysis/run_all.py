"""
Master runner: executes all analyses and combines output into a single HTML report.

Usage:
    python analysis/run_all.py
Output:
    analysis/outputs/full_analysis.html
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# make sure scripts can import _common
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import CSS
import correlation_analysis
import ols_regression
import quantile_regression
import sign_test
import event_study

OUT = Path(__file__).resolve().parent / "outputs" / "full_analysis_with_event_study.html"


def main() -> None:
    print("Running correlation analysis...")
    s1 = correlation_analysis.run()

    print("Running OLS regression...")
    s2 = ols_regression.run()

    print("Running quantile regression...")
    s3 = quantile_regression.run()

    print("Running sign test...")
    s4 = sign_test.run()

    print("Running event study...")
    s5 = event_study.run()

    date_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FOMC Sentiment → Bitcoin Return Analysis</title>
<style>
{CSS}
</style>
</head>
<body>
<h1>FOMC Sentiment Effect on Bitcoin Returns</h1>
<p class="note">
  Source: CentralBankRoBERTa agent &amp; sentiment scores applied to FOMC Minutes and Policy Statements (2012–2025).<br>
  Bitcoin prices: Bitstamp 1h OHLCV. Baseline price = close of candle ending 1h before document release.<br>
  Periods: 2012–2019 | 2020–Jul2023 | Aug2023–2025.<br>
  Generated: {date_str}
</p>
{s1}
{s2}
{s3}
{s4}
{s5}
</body>
</html>"""

    OUT.write_text(html, encoding="utf-8")
    print(f"\nFull report saved to: {OUT}")


if __name__ == "__main__":
    main()
