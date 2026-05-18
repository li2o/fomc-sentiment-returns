from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Paths:
    workspace_root: Path
    llm_document_level_dir: Path
    btc_hourly_csv: Path
    output_csv: Path


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Build a modeling-ready event impact table from LLM document-level outputs and BTC hourly data."
    )
    parser.add_argument("--workspace-root", type=Path, default=workspace_default)
    parser.add_argument(
        "--llm-document-level-dir",
        type=Path,
        default=Path("llm_analysis/outputs/document_level"),
    )
    parser.add_argument(
        "--btc-hourly-csv",
        type=Path,
        default=Path("data/bitcoin/bitcoin_bitstamp_1h.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/event_impact_dataset.csv"),
    )
    parser.add_argument("--windows-hours", nargs="+", type=int, default=[1, 6, 24])
    return parser.parse_args()


def to_absolute(root: Path, maybe_relative: Path) -> Path:
    return maybe_relative if maybe_relative.is_absolute() else root / maybe_relative


def resolve_document_level_input(document_level_dir: Path) -> Path:
    preferred = document_level_dir / "all_corpora_document_level.csv"
    if preferred.exists():
        return preferred

    parts = []
    for file_name in ["speeches_document_level.csv", "fomc_document_level.csv"]:
        candidate = document_level_dir / file_name
        if candidate.exists():
            parts.append(candidate)

    if len(parts) == 2:
        return preferred

    missing_hint = (
        "Expected one of:"
        "\n- llm_analysis/outputs/document_level/all_corpora_document_level.csv"
        "\n- both speeches_document_level.csv and fomc_document_level.csv"
        "\nRun: python llm_analysis/scripts/run_centralbankroberta_analysis.py"
    )
    raise FileNotFoundError(missing_hint)


def load_document_level(path: Path) -> pd.DataFrame:
    if path.name == "all_corpora_document_level.csv":
        return pd.read_csv(path, dtype=str).fillna("")

    document_level_dir = path.parent
    speeches = pd.read_csv(document_level_dir / "speeches_document_level.csv", dtype=str).fillna("")
    fomc = pd.read_csv(document_level_dir / "fomc_document_level.csv", dtype=str).fillna("")
    merged = pd.concat([speeches, fomc], ignore_index=True)
    merged.to_csv(path, index=False)
    return merged


def load_btc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"BTC file missing columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
    df["log_return_1h"] = np.log(df["close"]).diff()
    return df


def build_event_timestamp(df: pd.DataFrame) -> pd.Series:
    if "date" in df.columns:
        date_input: pd.Series = df["date"]
    else:
        date_input = pd.Series("", index=df.index, dtype=str)

    if "release_time" in df.columns:
        time_input: pd.Series = df["release_time"]
    elif "calendar_time" in df.columns:
        time_input = df["calendar_time"]
    else:
        time_input = pd.Series("", index=df.index, dtype=str)

    date_values = pd.to_datetime(date_input, errors="coerce")
    date_text = date_values.map(lambda value: value.strftime("%Y-%m-%d") if pd.notna(value) else "")

    time_values = time_input.astype(str).str.strip()
    hhmmss = time_values.where(time_values.str.match(r"^\d{1,2}:\d{2}:\d{2}$"), "00:00:00")

    combined = (date_text + " " + hhmmss).str.strip()
    return pd.to_datetime(combined, errors="coerce", utc=True)


def first_row_at_or_after(df: pd.DataFrame, target_ts: pd.Timestamp) -> pd.Series | None:
    match = df[df["timestamp"] >= target_ts]
    if match.empty:
        return None
    return match.iloc[0]


def compute_window_metrics(
    btc_df: pd.DataFrame,
    event_ts: pd.Timestamp,
    windows_hours: list[int],
) -> dict[str, float | str | None]:
    metrics: dict[str, float | str | None] = {}

    t0_row = first_row_at_or_after(btc_df, event_ts)
    if t0_row is None:
        metrics["event_btc_timestamp"] = None
        for window in windows_hours:
            metrics[f"ret_{window}h"] = np.nan
            metrics[f"vol_{window}h"] = np.nan
            metrics[f"volume_{window}h"] = np.nan
        return metrics

    t0_ts = pd.Timestamp(t0_row["timestamp"])
    t0_close = float(t0_row["close"])
    metrics["event_btc_timestamp"] = t0_ts.isoformat()

    for window in windows_hours:
        end_target = event_ts + pd.Timedelta(hours=window)
        end_row = first_row_at_or_after(btc_df, end_target)

        if end_row is None:
            metrics[f"ret_{window}h"] = np.nan
            metrics[f"vol_{window}h"] = np.nan
            metrics[f"volume_{window}h"] = np.nan
            continue

        end_ts = pd.Timestamp(end_row["timestamp"])
        end_close = float(end_row["close"])

        if t0_close > 0:
            metrics[f"ret_{window}h"] = (end_close / t0_close) - 1.0
        else:
            metrics[f"ret_{window}h"] = np.nan

        window_slice = btc_df[(btc_df["timestamp"] > t0_ts) & (btc_df["timestamp"] <= end_ts)].copy()
        metrics[f"vol_{window}h"] = float(window_slice["log_return_1h"].std(ddof=0)) if not window_slice.empty else np.nan
        metrics[f"volume_{window}h"] = float(window_slice["volume"].sum()) if not window_slice.empty else np.nan

    return metrics


def build_event_impact_table(
    document_df: pd.DataFrame,
    btc_df: pd.DataFrame,
    windows_hours: list[int],
) -> pd.DataFrame:
    required_meta_cols = {"date", "source_corpus", "document_id"}
    missing = required_meta_cols - set(document_df.columns)
    if missing:
        raise ValueError(f"Document-level LLM file missing required columns: {sorted(missing)}")

    working = document_df.copy()
    working["event_timestamp_utc"] = build_event_timestamp(working)

    metrics_rows: list[dict[str, float | str | None]] = []
    for _, row in working.iterrows():
        event_ts = row["event_timestamp_utc"]
        if pd.isna(event_ts):
            result: dict[str, float | str | None] = {"event_btc_timestamp": None}
            for window in windows_hours:
                result[f"ret_{window}h"] = np.nan
                result[f"vol_{window}h"] = np.nan
                result[f"volume_{window}h"] = np.nan
        else:
            result = compute_window_metrics(btc_df, pd.Timestamp(event_ts), windows_hours)
        metrics_rows.append(result)

    metrics_df = pd.DataFrame(metrics_rows)
    out = pd.concat([working.reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1)

    column_order_prefix = [
        "document_id",
        "source_corpus",
        "date",
        "release_time",
        "event_timestamp_utc",
        "event_btc_timestamp",
        "dominant_agent",
        "dominant_sentiment",
        "net_sentiment",
    ]

    present_prefix = [col for col in column_order_prefix if col in out.columns]
    other_cols = [col for col in out.columns if col not in present_prefix]
    return out[present_prefix + other_cols]


def main() -> None:
    args = parse_args()

    paths = Paths(
        workspace_root=args.workspace_root.resolve(),
        llm_document_level_dir=to_absolute(args.workspace_root.resolve(), args.llm_document_level_dir),
        btc_hourly_csv=to_absolute(args.workspace_root.resolve(), args.btc_hourly_csv),
        output_csv=to_absolute(args.workspace_root.resolve(), args.output_csv),
    )

    input_path = resolve_document_level_input(paths.llm_document_level_dir)
    document_df = load_document_level(input_path)
    btc_df = load_btc(paths.btc_hourly_csv)

    windows_hours = sorted(set(args.windows_hours))
    out_df = build_event_impact_table(document_df, btc_df, windows_hours)

    paths.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(paths.output_csv, index=False)

    print(f"Document-level input: {input_path.relative_to(paths.workspace_root)}")
    print(f"BTC hourly input: {paths.btc_hourly_csv.relative_to(paths.workspace_root)}")
    print(f"Output written: {paths.output_csv.relative_to(paths.workspace_root)}")
    print(f"Rows: {len(out_df)}")

    matched = out_df["event_btc_timestamp"].notna().sum() if "event_btc_timestamp" in out_df.columns else 0
    print(f"Rows with BTC mapping: {matched}")


if __name__ == "__main__":
    main()
