"""
Build press-conference master datasets for 2012-2025.

Data sources:
- llm_analysis/outputs/document_level/press_conferences_document_level.csv
- data/metadata/fomc_press_conferences.csv
- data/market/bitcoin_bitstamp_1h.csv
- data/market/spx_yahoo_1d.csv
- data/market/zt_yahoo_1d.csv
- data/control variables/monetary policy surprises data.xlsx
- data/control variables/shocks_fed_jk_t.csv
- data/control variables/financial condition.csv

Outputs:
- data/master_dataset_press_conferences.csv
- data/master_dataset_press_conferences_reduced.csv

Usage:
    python analysis/build_press_conferences_master_dataset.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parent.parent

PC_CSV = WORKSPACE / "llm_analysis/outputs/document_level/press_conferences_document_level.csv"
PC_META_CSV = WORKSPACE / "data/metadata/fomc_press_conferences.csv"
BTC_CSV = WORKSPACE / "data/market/bitcoin_bitstamp_1h.csv"
SPX_CSV = WORKSPACE / "data/market/spx_yahoo_1d.csv"
ZT_CSV = WORKSPACE / "data/market/zt_yahoo_1d.csv"
JK_CSV = WORKSPACE / "data/control variables/shocks_fed_jk_t.csv"
MPS_XLSX = WORKSPACE / "data/control variables/monetary policy surprises data.xlsx"
NFCI_CSV = WORKSPACE / "data/control variables/financial condition.csv"

OUT_FULL = WORKSPACE / "data/master_dataset_press_conferences.csv"
OUT_REDUCED = WORKSPACE / "data/master_dataset_press_conferences_reduced.csv"

START_DATE = pd.Timestamp("2012-01-01")
END_DATE = pd.Timestamp("2025-12-31")

FULL_COLS = [
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


def btc_returns_for_release(event_date: pd.Timestamp, btc_prices: pd.DataFrame) -> dict[str, float | None]:
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

    for days in [1, 2, 3, 5, 7, 9]:
        target_close = get_last_close_on_or_before(prices, release_date + pd.Timedelta(days=days))
        out[f"{asset_prefix} Log Return {days}d"] = safe_log_ratio(target_close, open_t)
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


def match_nfci(event_date: pd.Timestamp, nfci: pd.DataFrame) -> float | None:
    subset = nfci[nfci["date"] <= event_date]
    if subset.empty:
        return None
    val = subset.iloc[-1]["NFCI"]
    return None if pd.isna(val) else float(val)


def load_press_conference_times() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "time_utc", "event"])


def default_press_conference_time(date: pd.Timestamp) -> str:
    # Public fallback: post-meeting press conferences are normally 30 minutes
    # after the FOMC statement: 18:30 UTC during US daylight time, otherwise
    # 19:30 UTC.
    return "18:30:00" if date.month in [3, 4, 5, 6, 7, 9, 10] else "19:30:00"


def main() -> None:
    pc = pd.read_csv(PC_CSV)
    pc["meeting_date"] = pd.to_datetime(pc["meeting_date"], errors="coerce").dt.normalize()
    pc = pc[(pc["meeting_date"] >= START_DATE) & (pc["meeting_date"] <= END_DATE)].copy()

    meta = pd.read_csv(PC_META_CSV)
    meta["meeting_date"] = pd.to_datetime(meta["meeting_date"], errors="coerce").dt.normalize()
    pc = pc.merge(
        meta[["meeting_date", "title"]].rename(columns={"title": "meta_title"}),
        how="left",
        on="meeting_date",
    )

    times = load_press_conference_times()
    pc = pc.merge(times, how="left", left_on="meeting_date", right_on="date").drop(columns=["date"])
    pc["time_utc"] = pc.apply(
        lambda row: row["time_utc"]
        if isinstance(row["time_utc"], str) and row["time_utc"] and row["time_utc"] != "nan"
        else default_press_conference_time(row["meeting_date"]),
        axis=1,
    )
    pc["event"] = pc["event"].where(pc["event"].notna(), pc["meta_title"])
    pc["event"] = pc["event"].fillna("FOMC Press Conference")

    btc_prices = load_hourly_prices(BTC_CSV)
    spx_prices = load_daily_prices(SPX_CSV)
    zt_prices = load_daily_prices(ZT_CSV)
    mps, nfci = load_controls()
    pc = pc.merge(mps, how="left", left_on="meeting_date", right_on="date").drop(columns=["date"])

    rows: list[dict[str, object]] = []
    for _, r in pc.sort_values("meeting_date").iterrows():
        event_date = pd.Timestamp(r["meeting_date"]).normalize()
        release_dt = pd.to_datetime(f"{event_date.date()} {r['time_utc']}", errors="coerce")
        if pd.isna(release_dt):
            continue

        row: dict[str, object] = {
            "Date": event_date.date().isoformat(),
            "Time UTC": str(r["time_utc"])[:5],
            "Event": r["event"],
            **btc_returns_for_release(event_date, btc_prices),
            **daily_asset_returns_for_release(event_date, spx_prices, "SPX"),
            **daily_asset_returns_for_release(event_date, zt_prices, "ZT"),
            "NFCI": match_nfci(event_date, nfci),
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

    full = pd.DataFrame(rows)
    for col in FULL_COLS:
        if col not in full.columns:
            full[col] = np.nan
    full = full[FULL_COLS].sort_values(["Date", "Time UTC"]).reset_index(drop=True)
    full.to_csv(OUT_FULL, index=False)

    reduced = full.rename(columns=REDUCED_RENAMES)
    reduced_cols = [
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
    reduced = reduced[reduced_cols]
    numeric_cols = [c for c in reduced.columns if c != "Date"]
    reduced[numeric_cols] = reduced[numeric_cols].round(3)
    reduced.to_csv(OUT_REDUCED, index=False)

    print(f"Rows written: {len(full)}")
    print(f"Date range   : {full['Date'].min()} to {full['Date'].max()}")
    print(f"Missing times: {pc['time_utc'].isna().sum()}")
    print(f"Missing BTC  : {full['BTC Close'].isna().sum()}")
    print(f"Missing SPX  : {full['SPX Close'].isna().sum()}")
    print(f"Missing ZT   : {full['ZT Close'].isna().sum()}")
    print(f"Saved: {OUT_FULL}")
    print(f"Saved: {OUT_REDUCED}")


if __name__ == "__main__":
    main()
