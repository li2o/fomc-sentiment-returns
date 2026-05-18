"""
Compute Bitcoin returns around FOMC document releases and append them
to the document-level CSV.

Baseline : close of the candle ending at release_time - 1h
           (last fully completed candle before the release)
Forward  : close of the candle ending at release_time + Xh

Returns added (simple + log) for windows: 1h, 3h, 9h, 24h, 72h, 144h

Usage:
    python analysis/build_event_returns.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

WORKSPACE  = Path(__file__).resolve().parent.parent
BTC_CSV    = WORKSPACE / "data/bitcoin/bitcoin_bitstamp_1h.csv"
DOC_CSV    = WORKSPACE / "llm_analysis/outputs/document_level/fomc_document_level.csv"
OUT_CSV    = DOC_CSV          # update in-place
DOC_TYPES  = ["Minutes", "Policy Statement"]
WINDOWS    = [1, 3, 9, 24, 72, 144, 216, 288, 360]   # hours


def load_btc(path: Path) -> pd.Series:
    """Return a Series of close prices indexed by UTC timestamp (hourly)."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df["close"]


def get_close(prices: pd.Series, ts: pd.Timestamp) -> float | None:
    """Return close price at exactly ts, or None if not found."""
    try:
        return float(prices.loc[ts])
    except KeyError:
        return None


def compute_returns(doc: pd.Series, prices: pd.Series) -> dict[str, float | None]:
    result: dict[str, float | None] = {"btc_price_baseline": None}
    for w in WINDOWS:
        result[f"btc_return_{w}h"]     = None
        result[f"btc_log_return_{w}h"] = None

    # Parse release datetime
    date_str = doc.get("document_date", "")
    time_str = doc.get("release_time", "")
    if not date_str or not time_str or pd.isna(date_str) or pd.isna(time_str):
        return result

    try:
        release_dt = pd.Timestamp(f"{date_str} {time_str}")
    except Exception:
        return result

    # Baseline: candle closing 1h before release
    baseline_ts = release_dt - pd.Timedelta(hours=1)
    baseline_price = get_close(prices, baseline_ts)
    if baseline_price is None or baseline_price <= 0:
        return result

    result["btc_price_baseline"] = baseline_price

    for w in WINDOWS:
        forward_ts = release_dt + pd.Timedelta(hours=w)
        forward_price = get_close(prices, forward_ts)
        if forward_price is None or forward_price <= 0:
            continue
        simple = forward_price / baseline_price - 1
        log    = np.log(forward_price / baseline_price)
        result[f"btc_return_{w}h"]     = round(simple, 8)
        result[f"btc_log_return_{w}h"] = round(log,    8)

    return result


def main() -> None:
    prices = load_btc(BTC_CSV)
    df = pd.read_csv(DOC_CSV)

    # Drop any existing return columns before recomputing
    existing = [c for c in df.columns if c.startswith("btc_")]
    if existing:
        df = df.drop(columns=existing)

    records = [compute_returns(row, prices) for _, row in df.iterrows()]
    returns_df = pd.DataFrame(records, index=df.index)
    df = pd.concat([df, returns_df], axis=1)

    df.to_csv(OUT_CSV, index=False)

    # Summary
    subset = df[df["document_type"].isin(DOC_TYPES)]
    print(f"Documents processed : {len(df)}")
    print(f"Minutes + Statements: {len(subset)}")
    print(f"Missing baseline    : {df['btc_price_baseline'].isna().sum()}")
    print()
    cols = [f"btc_return_{w}h" for w in WINDOWS]
    print(subset.groupby("document_type")[cols].apply(lambda g: g.notna().sum()).T.to_string())
    print(f"\nSaved to: {OUT_CSV}")


if __name__ == "__main__":
    main()
