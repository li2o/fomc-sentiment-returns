"""
Build a minutes-only master dataset.

Output schema matches the historical master_dataset.csv columns so existing
downstream analysis scripts can keep running unchanged.

Data sources:
- llm_analysis/outputs/document_level/fomc_document_level.csv
- data/market/bitcoin_bitstamp_1h.csv
- data/market/spx_yahoo_1d.csv
- data/market/zt_yahoo_1d.csv
- data/control variables/monetary policy surprises data.xlsx
- data/control variables/shocks_fed_jk_t.csv
- data/control variables/financial condition.csv

Usage:
    python analysis/build_minutes_master_dataset.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parent.parent

DOC_CSV = WORKSPACE / "llm_analysis/outputs/document_level/fomc_document_level.csv"
BTC_CSV = WORKSPACE / "data/market/bitcoin_bitstamp_1h.csv"
SPX_CSV = WORKSPACE / "data/market/spx_yahoo_1d.csv"
ZT_CSV = WORKSPACE / "data/market/zt_yahoo_1d.csv"
JK_CSV = WORKSPACE / "data/control variables/shocks_fed_jk_t.csv"
MPS_XLSX = WORKSPACE / "data/control variables/monetary policy surprises data.xlsx"
NFCI_CSV = WORKSPACE / "data/control variables/financial condition.csv"

OUT_FULL = WORKSPACE / "data/master_dataset_minutes.csv"
OUT_REDUCED = WORKSPACE / "data/master_dataset_minutes_reduced.csv"


OUT_COLS = [
    "Date",
    "Time UTC",
    "Event",
    "BTC Close",
    "BTC Log Return Intraday",
    "BTC Log Return 1d",
    "BTC Log Return 2d",
    "BTC Log Return 3d",
    "BTC Log Return 5d",
    "BTC Log Return 7d",
    "BTC Log Return 9d",
    "SPX Close",
    "SPX Log Return Intraday",
    "SPX Log Return 1d",
    "SPX Log Return 2d",
    "SPX Log Return 3d",
    "SPX Log Return 5d",
    "SPX Log Return 7d",
    "SPX Log Return 9d",
    "ZT Close",
    "ZT Log Return Intraday",
    "ZT Log Return 1d",
    "ZT Log Return 2d",
    "ZT Log Return 3d",
    "ZT Log Return 5d",
    "ZT Log Return 7d",
    "ZT Log Return 9d",
    "NFCI",
    "mps",
    "MPS_ORTH",
    "MP_median",
    "CBI_median",
    "net_sentiment",
    "net_sentiment_households",
    "net_sentiment_firms",
    "net_sentiment_financial_sector",
    "net_sentiment_government",
    "net_sentiment_central_bank",
]

REDUCED_RENAMES = {
    "BTC Close": "btc_close",
    "BTC Log Return Intraday": "btc_r0",
    "BTC Log Return 1d": "btc_r1",
    "BTC Log Return 2d": "btc_r2",
    "BTC Log Return 3d": "btc_r3",
    "BTC Log Return 5d": "btc_r5",
    "BTC Log Return 7d": "btc_r7",
    "BTC Log Return 9d": "btc_r9",
    "SPX Close": "spx_close",
    "SPX Log Return Intraday": "spx_r0",
    "SPX Log Return 1d": "spx_r1",
    "SPX Log Return 2d": "spx_r2",
    "SPX Log Return 3d": "spx_r3",
    "SPX Log Return 5d": "spx_r5",
    "SPX Log Return 7d": "spx_r7",
    "SPX Log Return 9d": "spx_r9",
    "ZT Close": "zt_close",
    "ZT Log Return Intraday": "zt_r0",
    "ZT Log Return 1d": "zt_r1",
    "ZT Log Return 2d": "zt_r2",
    "ZT Log Return 3d": "zt_r3",
    "ZT Log Return 5d": "zt_r5",
    "ZT Log Return 7d": "zt_r7",
    "ZT Log Return 9d": "zt_r9",
    "net_sentiment": "s_Overall",
    "net_sentiment_households": "s_Households",
    "net_sentiment_firms": "s_Firms",
    "net_sentiment_financial_sector": "s_Financial_Sector",
    "net_sentiment_government": "s_Government",
    "net_sentiment_central_bank": "s_Central_Bank",
}

REDUCED_COLS = [
    "Date",
    "btc_close",
    "btc_r0",
    "btc_r1",
    "btc_r2",
    "btc_r3",
    "btc_r5",
    "btc_r7",
    "btc_r9",
    "spx_close",
    "spx_r0",
    "spx_r1",
    "spx_r2",
    "spx_r3",
    "spx_r5",
    "spx_r7",
    "spx_r9",
    "zt_close",
    "zt_r0",
    "zt_r1",
    "zt_r2",
    "zt_r3",
    "zt_r5",
    "zt_r7",
    "zt_r9",
    "NFCI",
    "mps",
    "MPS_ORTH",
    "MP_median",
    "CBI_median",
    "s_Overall",
    "s_Households",
    "s_Firms",
    "s_Financial_Sector",
    "s_Government",
    "s_Central_Bank",
]


def to_datetime_col(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def load_hourly_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    for col in ["open", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open", "close"]]


def load_daily_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.sort_values("date").set_index("date")
    for col in ["open", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open", "close"]]


def get_first_open_on_or_after(prices: pd.DataFrame, date: pd.Timestamp) -> float | None:
    subset = prices[prices.index >= date]
    if subset.empty:
        return None
    val = subset.iloc[0]["open"]
    if pd.isna(val) or val <= 0:
        return None
    return float(val)


def get_last_close_on_or_before(prices: pd.DataFrame, date: pd.Timestamp) -> float | None:
    subset = prices[prices.index <= date]
    if subset.empty:
        return None
    val = subset.iloc[-1]["close"]
    if pd.isna(val) or val <= 0:
        return None
    return float(val)


def safe_log_ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    if a <= 0 or b <= 0:
        return None
    return float(np.log(a / b))


def btc_returns_for_release(
    event_date: pd.Timestamp,
    btc_prices: pd.DataFrame,
) -> dict[str, float | None]:
    open_t = get_first_open_on_or_after(btc_prices, event_date)
    close_t = get_last_close_on_or_before(
        btc_prices,
        event_date + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1),
    )
    out = {
        "BTC Close": close_t,
        "BTC Log Return Intraday": None,
        "BTC Log Return 1d": None,
        "BTC Log Return 2d": None,
        "BTC Log Return 3d": None,
        "BTC Log Return 5d": None,
        "BTC Log Return 7d": None,
        "BTC Log Return 9d": None,
    }
    if open_t is None:
        return out

    out["BTC Log Return Intraday"] = safe_log_ratio(close_t, open_t)
    for days in [1, 2, 3, 5, 7, 9]:
        target_close = get_last_close_on_or_before(
            btc_prices,
            event_date + pd.Timedelta(days=days + 1) - pd.Timedelta(nanoseconds=1),
        )
        out[f"BTC Log Return {days}d"] = safe_log_ratio(target_close, open_t)
    return out


def daily_asset_returns_for_release(
    release_date: pd.Timestamp,
    prices: pd.DataFrame,
    asset_prefix: str,
) -> dict[str, float | None]:
    open_t = get_first_open_on_or_after(prices, release_date)
    close_t = get_last_close_on_or_before(prices, release_date)

    out = {
        f"{asset_prefix} Close": close_t,
        f"{asset_prefix} Log Return Intraday": safe_log_ratio(close_t, open_t),
        f"{asset_prefix} Log Return 1d": None,
        f"{asset_prefix} Log Return 2d": None,
        f"{asset_prefix} Log Return 3d": None,
        f"{asset_prefix} Log Return 5d": None,
        f"{asset_prefix} Log Return 7d": None,
        f"{asset_prefix} Log Return 9d": None,
    }

    if open_t is None:
        return out

    for n in [1, 2, 3, 5, 7, 9]:
        target = get_last_close_on_or_before(prices, release_date + pd.Timedelta(days=n))
        out[f"{asset_prefix} Log Return {n}d"] = safe_log_ratio(target, open_t)
    return out


def load_mps_controls() -> pd.DataFrame:
    try:
        mps = pd.read_excel(MPS_XLSX, sheet_name="FOMC (update 2023)")
        date_col = "Date" if "Date" in mps.columns else "date"
        mps["date"] = pd.to_datetime(mps[date_col], errors="coerce").dt.normalize()
        if "MPS" not in mps.columns:
            raise KeyError("No MPS column found")
        out = pd.DataFrame(
            {
                "date": mps["date"],
                "mps": pd.to_numeric(mps["MPS"], errors="coerce"),
                "MPS_ORTH": pd.to_numeric(mps.get("MPS_ORTH"), errors="coerce"),
            }
        )
        jk = pd.read_csv(JK_CSV)
        jk["date"] = pd.to_datetime(jk["date"], errors="coerce").dt.normalize()
        for col in ["MP_median", "CBI_median"]:
            jk[col] = pd.to_numeric(jk[col], errors="coerce")
        out = out.merge(jk[["date", "MP_median", "CBI_median"]], how="left", on="date")
        return out
    except Exception as exc:
        print(f"Warning: could not read {MPS_XLSX.name}; using {JK_CSV.name} fallback ({exc})")

    jk = pd.read_csv(JK_CSV)
    jk["date"] = pd.to_datetime(jk["date"], errors="coerce").dt.normalize()
    for col in ["MP_pm", "MP_median", "CBI_median"]:
        jk[col] = pd.to_numeric(jk[col], errors="coerce")
    jk = jk[["date", "MP_pm", "MP_median", "CBI_median"]].rename(
        columns={"MP_pm": "mps"}
    )
    jk["MPS_ORTH"] = jk["mps"]
    return jk[["date", "mps", "MPS_ORTH", "MP_median", "CBI_median"]]


def load_controls() -> tuple[pd.DataFrame, pd.DataFrame]:
    mps = load_mps_controls()
    nfci = pd.read_csv(NFCI_CSV)
    nfci["date"] = pd.to_datetime(nfci["date"], errors="coerce").dt.normalize()
    nfci["NFCI"] = pd.to_numeric(nfci["NFCI"], errors="coerce")
    nfci = nfci[["date", "NFCI"]].dropna(subset=["date"]).sort_values("date")
    return mps, nfci


def match_nfci_for_date(event_date: pd.Timestamp, nfci_df: pd.DataFrame) -> float | None:
    subset = nfci_df[nfci_df["date"] <= event_date]
    if subset.empty:
        return None
    val = subset.iloc[-1]["NFCI"]
    return None if pd.isna(val) else float(val)


def merge_minutes_controls(docs: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    control_cols = ["mps", "MPS_ORTH", "MP_median", "CBI_median"]
    out = docs.merge(controls, how="left", left_on="meeting_date", right_on="date")
    out = out.drop(columns=["date"])

    missing = out["mps"].isna()
    if missing.any():
        fallback = controls.rename(
            columns={col: f"{col}_next_day" for col in control_cols}
        ).copy()
        fallback["meeting_date_next_day"] = fallback["date"] - pd.Timedelta(days=1)
        fallback = fallback.drop(columns=["date"])
        out = out.merge(
            fallback,
            how="left",
            left_on="meeting_date",
            right_on="meeting_date_next_day",
        )
        for col in control_cols:
            out[col] = out[col].combine_first(out[f"{col}_next_day"])
            out = out.drop(columns=[f"{col}_next_day"])
        out = out.drop(columns=["meeting_date_next_day"])
    return out


def main() -> None:
    docs = pd.read_csv(DOC_CSV)
    docs["meeting_date"] = to_datetime_col(docs["meeting_date"]).dt.normalize()
    docs["document_date"] = to_datetime_col(docs["document_date"]).dt.normalize()
    docs = docs[docs["document_type"] == "Minutes"].copy()
    docs = docs.dropna(subset=["meeting_date", "document_date", "release_time"])

    btc_prices = load_hourly_prices(BTC_CSV)
    spx_prices = load_daily_prices(SPX_CSV)
    zt_prices = load_daily_prices(ZT_CSV)
    mps, nfci = load_controls()

    docs = merge_minutes_controls(docs, mps)

    rows: list[dict[str, object]] = []
    for _, r in docs.sort_values(["document_date", "meeting_date"]).iterrows():
        release_dt = pd.to_datetime(
            f"{r['document_date'].date()} {str(r['release_time'])}",
            errors="coerce",
        )
        if pd.isna(release_dt):
            continue

        event_date = pd.Timestamp(r["document_date"]).normalize()

        btc_ret = btc_returns_for_release(event_date, btc_prices)
        spx_ret = daily_asset_returns_for_release(event_date, spx_prices, "SPX")
        zt_ret = daily_asset_returns_for_release(event_date, zt_prices, "ZT")

        row: dict[str, object] = {
            "Date": event_date.date().isoformat(),
            "Time UTC": str(r["release_time"]),
            "Event": "FOMC Minutes Release",
            **btc_ret,
            **spx_ret,
            **zt_ret,
            "NFCI": match_nfci_for_date(event_date, nfci),
            "mps": r.get("mps"),
            "MPS_ORTH": r.get("MPS_ORTH"),
            "MP_median": r.get("MP_median"),
            "CBI_median": r.get("CBI_median"),
            "net_sentiment": r.get("net_sentiment"),
            "net_sentiment_households": r.get("net_sentiment_households"),
            "net_sentiment_firms": r.get("net_sentiment_firms"),
            "net_sentiment_financial_sector": r.get("net_sentiment_financial sector"),
            "net_sentiment_government": r.get("net_sentiment_government"),
            "net_sentiment_central_bank": r.get("net_sentiment_central bank"),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    for c in OUT_COLS:
        if c not in out.columns:
            out[c] = np.nan
    out = out[OUT_COLS].sort_values(["Date", "Time UTC"]).reset_index(drop=True)

    out.to_csv(OUT_FULL, index=False)

    reduced = out.rename(columns=REDUCED_RENAMES)
    reduced = reduced[REDUCED_COLS]
    numeric_cols = [c for c in reduced.columns if c != "Date"]
    reduced[numeric_cols] = reduced[numeric_cols].round(3)
    reduced.to_csv(OUT_REDUCED, index=False)

    print(f"Rows written: {len(out)}")
    print(f"Minutes rows : {(out['Event'] == 'FOMC Minutes Release').sum()}")
    print(f"Missing BTC baseline: {out['BTC Close'].isna().sum()}")
    print(f"Missing SPX close   : {out['SPX Close'].isna().sum()}")
    print(f"Missing ZT close    : {out['ZT Close'].isna().sum()}")
    print(f"Saved: {OUT_FULL}")
    print(f"Saved: {OUT_REDUCED}")


if __name__ == "__main__":
    main()
