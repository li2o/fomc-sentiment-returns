"""
Compute mean-adjusted abnormal Bitcoin returns for each FOMC document.

Method:
  - Estimation window: 30 calendar days (720 hourly candles) ending 1h before release
  - Expected return: mean hourly log return over the estimation window
  - Abnormal return for window Xh:
      AR_Xh = btc_log_return_Xh - X * mean_hourly_return_estimation

  This converts raw log returns into abnormal returns by removing the baseline
  Bitcoin drift over the same horizon.

Appends columns to fomc_document_level.csv:
  btc_ar_{w}h   for each window w in WINDOWS

Usage:
    python analysis/build_abnormal_returns.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

WORKSPACE        = Path(__file__).resolve().parent.parent
BTC_CSV          = WORKSPACE / "data/bitcoin/bitcoin_bitstamp_1h.csv"
DOC_CSV          = WORKSPACE / "llm_analysis/outputs/document_level/fomc_document_level.csv"
ESTIMATION_HOURS = 720    # 30 days
WINDOWS          = [1, 3, 9, 24, 72, 144, 216, 288, 360]


def load_btc(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df["close"]


def mean_hourly_log_return(prices: pd.Series, baseline_ts: pd.Timestamp) -> float | None:
    """Mean hourly log return over the 720h estimation window ending at baseline_ts."""
    end   = baseline_ts
    start = baseline_ts - pd.Timedelta(hours=ESTIMATION_HOURS)
    window = prices.loc[start:end].dropna()
    if len(window) < 100:   # require at least 100 observations
        return None
    log_rets = np.log(window / window.shift(1)).dropna()
    return float(log_rets.mean())


def compute_abnormal_returns(doc: pd.Series, prices: pd.Series) -> dict[str, float | None]:
    result = {f"btc_ar_{w}h": None for w in WINDOWS}

    date_str = doc.get("document_date", "")
    time_str = doc.get("release_time", "")
    if not date_str or not time_str or pd.isna(date_str) or pd.isna(time_str):
        return result

    try:
        release_dt  = pd.Timestamp(f"{date_str} {time_str}")
    except Exception:
        return result

    baseline_ts = release_dt - pd.Timedelta(hours=1)
    mu = mean_hourly_log_return(prices, baseline_ts)
    if mu is None:
        return result

    for w in WINDOWS:
        raw_col = f"btc_log_return_{w}h"
        raw = doc.get(raw_col)
        if pd.isna(raw):
            continue
        # scale expected return to window length
        expected = mu * w
        result[f"btc_ar_{w}h"] = round(float(raw) - expected, 8)

    return result


def main() -> None:
    prices = load_btc(BTC_CSV)
    df     = pd.read_csv(DOC_CSV)

    # drop existing AR columns before recomputing
    existing = [c for c in df.columns if c.startswith("btc_ar_")]
    if existing:
        df = df.drop(columns=existing)

    records   = [compute_abnormal_returns(row, prices) for _, row in df.iterrows()]
    ar_df     = pd.DataFrame(records, index=df.index)
    df        = pd.concat([df, ar_df], axis=1)

    df.to_csv(DOC_CSV, index=False)

    subset = df[df["document_type"].isin(["Minutes", "Policy Statement"])]
    print(f"Documents processed : {len(df)}")
    print(f"Minutes + Statements: {len(subset)}")
    print(f"Missing AR (1h)     : {subset['btc_ar_1h'].isna().sum()}")
    print()
    ar_cols = [f"btc_ar_{w}h" for w in WINDOWS]
    print(subset.groupby("document_type")[ar_cols].apply(lambda g: g.notna().sum()).T.to_string())
    print(f"\nSaved to: {DOC_CSV}")


if __name__ == "__main__":
    main()
