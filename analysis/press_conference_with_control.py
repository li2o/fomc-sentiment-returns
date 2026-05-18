"""
OLS regression with control variable MPS_ORTH:
  btc_log_return ~ net_sentiment_agent + MPS_ORTH
  - Document level: Press Conferences only
  - Period: 2012–2023 (limited by MPS data availability)
  - Control variable: MPS_ORTH from monetary-policy-surprises-data.xlsx
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    fig_to_b64, img_tag, html_table, sig_stars, AGENTS, WINDOWS, CSS
)


WORKSPACE = Path(__file__).resolve().parent.parent
PC_CSV = WORKSPACE / "llm_analysis/outputs/document_level/press_conferences_document_level.csv"
BTC_CSV = WORKSPACE / "data/bitcoin/bitcoin_bitstamp_1h.csv"
MPS_FILE = WORKSPACE / "data/metadata/monetary-policy-surprises-data.xlsx"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

HAC_WINDOWS = {1: 0, 3: 0, 9: 1, 24: 2, 72: 4, 144: 6, 216: 8, 288: 10, 360: 12}


def load_btc_data() -> pd.DataFrame:
    """Load Bitcoin hourly data and compute log returns."""
    df = pd.read_csv(BTC_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
    
    # Compute log returns for each window
    for window_hours in WINDOWS:
        window_rows = window_hours
        df[f"btc_log_return_{window_hours}h"] = (
            np.log(df["close"]).diff(window_rows)
        )
    
    return df


def get_mps_orth_data() -> pd.DataFrame:
    """Load MPS_ORTH control variable from xlsx file."""
    mps = pd.read_excel(MPS_FILE, sheet_name="FOMC (update 2023)")
    mps["Date"] = pd.to_datetime(mps["Date"], errors="coerce")
    mps = mps[["Date", "MPS_ORTH"]].copy().rename(columns={"Date": "meeting_date"})
    mps = mps[mps["MPS_ORTH"].notna()].copy()
    return mps


def add_bitcoin_returns(pc_df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    """Match press conferences with Bitcoin returns."""
    # Parse meeting_date
    pc_df["meeting_date"] = pd.to_datetime(pc_df["meeting_date"], errors="coerce")
    
    results = []
    for _, pc_row in pc_df.iterrows():
        event_date = pc_row["meeting_date"]
        
        # Find Bitcoin data on or after this date (assuming events happen at market open)
        # Use 14:00 UTC as approximation for Federal Reserve announcement time
        event_ts = pd.Timestamp(
            year=event_date.year,
            month=event_date.month,
            day=event_date.day,
            hour=14,
            minute=0,
            second=0,
            tz="UTC"
        )
        
        # Find first BTC candle at or after event time
        btc_after = btc_df[btc_df["timestamp"] >= event_ts]
        if btc_after.empty:
            # No Bitcoin data after this event
            for w in WINDOWS:
                pc_row[f"btc_log_return_{w}h"] = np.nan
                pc_row[f"btc_price_baseline"] = np.nan
        else:
            baseline_row = btc_after.iloc[0]
            pc_row["btc_price_baseline"] = baseline_row["close"]
            
            for w in WINDOWS:
                pc_row[f"btc_log_return_{w}h"] = baseline_row[f"btc_log_return_{w}h"]
        
        results.append(pc_row)
    
    return pd.DataFrame(results)


def load_data() -> pd.DataFrame:
    """Load and prepare all data."""
    # Load press conferences
    pc_df = pd.read_csv(PC_CSV)
    
    # Load Bitcoin data
    btc_df = load_btc_data()
    
    # Add Bitcoin returns to press conferences
    df = add_bitcoin_returns(pc_df, btc_df)
    
    # Load MPS_ORTH and merge
    mps_df = get_mps_orth_data()
    df = df.merge(mps_df, on="meeting_date", how="left")
    
    # Parse date and filter to 2023 and earlier
    df["meeting_date"] = pd.to_datetime(df["meeting_date"], errors="coerce")
    df = df[df["meeting_date"] <= pd.Timestamp("2023-12-31")].copy()
    
    # Create agent-specific sentiment columns if not present
    if "net_sentiment_households" not in df.columns and "net_sentiment_financial sector" not in df.columns:
        print("Note: Agent-specific sentiment columns not found in press conference data")
    
    return df


def run_univariate(df: pd.DataFrame) -> list[dict]:
    """Run univariate regressions with control variable."""
    results = []
    
    for agent in AGENTS:
        sentiment_col = f"net_sentiment_{agent}"
        
        # Skip if column doesn't exist
        if sentiment_col not in df.columns:
            continue
        
        for w in WINDOWS:
            y_col = f"btc_log_return_{w}h"
            
            # Prepare data: sentiment, control variable, and returns
            data = df[[sentiment_col, "MPS_ORTH", y_col]].copy()
            data = data.dropna()
            
            if len(data) < 10:
                continue
            
            # OLS with control variable: y ~ const + sentiment + MPS_ORTH
            X = sm.add_constant(data[[sentiment_col, "MPS_ORTH"]])
            model = sm.OLS(data[y_col], X).fit()
            
            # Apply HAC for longer windows
            lags = HAC_WINDOWS[w]
            if lags > 0:
                r = model.get_robustcov_results(cov_type="HAC", maxlags=lags)
                params = pd.Series(r.params, index=model.params.index)
                bse = pd.Series(r.bse, index=model.params.index)
                tvals = pd.Series(r.tvalues, index=model.params.index)
                pvals = pd.Series(r.pvalues, index=model.params.index)
            else:
                params = model.params
                bse = model.bse
                tvals = model.tvalues
                pvals = model.pvalues
            
            results.append({
                "Agent": agent.title(),
                "Window": f"{w}h",
                "N": len(data),
                "β_sentiment": f"{float(params[sentiment_col]):.4f}",
                "SE_sentiment": f"{float(bse[sentiment_col]):.4f}",
                "t_sentiment": f"{float(tvals[sentiment_col]):.2f}",
                "p_sentiment": f"{float(pvals[sentiment_col]):.3f}",
                "Sig_sentiment": sig_stars(float(pvals[sentiment_col])),
                "β_control": f"{float(params['MPS_ORTH']):.4f}",
                "SE_control": f"{float(bse['MPS_ORTH']):.4f}",
                "t_control": f"{float(tvals['MPS_ORTH']):.2f}",
                "p_control": f"{float(pvals['MPS_ORTH']):.3f}",
                "Sig_control": sig_stars(float(pvals['MPS_ORTH'])),
                "R²": f"{model.rsquared:.3f}",
                "SE type": "HAC" if lags > 0 else "OLS",
            })
    
    return results


def run_joint(df: pd.DataFrame, windows: list[int] = [1, 24]) -> dict[int, dict]:
    """Run joint model with all agents and control variable."""
    results = {}
    agent_cols = [f"net_sentiment_{a}" for a in AGENTS]
    
    # Filter available agent columns
    agent_cols = [c for c in agent_cols if c in df.columns]
    
    for w in windows:
        y_col = f"btc_log_return_{w}h"
        data = df[agent_cols + ["MPS_ORTH", y_col]].copy()
        data = data.dropna()
        
        if len(data) < 12:
            results[w] = {"error": "Insufficient data"}
            continue
        
        # Joint model: y ~ const + all_agents + MPS_ORTH
        X = sm.add_constant(data[agent_cols + ["MPS_ORTH"]])
        fitted = sm.OLS(data[y_col], X).fit()
        
        lags = HAC_WINDOWS[w]
        if lags > 0:
            r = fitted.get_robustcov_results(cov_type="HAC", maxlags=lags)
            params = pd.Series(r.params, index=fitted.params.index)
            bse = pd.Series(r.bse, index=fitted.params.index)
            tvals = pd.Series(r.tvalues, index=fitted.params.index)
            pvals = pd.Series(r.pvalues, index=fitted.params.index)
        else:
            params = fitted.params
            bse = fitted.bse
            tvals = fitted.tvalues
            pvals = fitted.pvalues
        
        rows = []
        for col, agent in zip(agent_cols, [a for a in AGENTS if f"net_sentiment_{a}" in agent_cols]):
            rows.append({
                "Variable": agent.title(),
                "β": f"{float(params[col]):.4f}",
                "SE": f"{float(bse[col]):.4f}",
                "t": f"{float(tvals[col]):.2f}",
                "p": f"{float(pvals[col]):.3f}",
                "Sig": sig_stars(float(pvals[col])),
            })
        
        # Add control variable
        rows.append({
            "Variable": "MPS_ORTH (control)",
            "β": f"{float(params['MPS_ORTH']):.4f}",
            "SE": f"{float(bse['MPS_ORTH']):.4f}",
            "t": f"{float(tvals['MPS_ORTH']):.2f}",
            "p": f"{float(pvals['MPS_ORTH']):.3f}",
            "Sig": sig_stars(float(pvals['MPS_ORTH'])),
        })
        
        results[w] = {
            "table": pd.DataFrame(rows),
            "R²": round(fitted.rsquared, 3),
            "N": len(data),
            "SE type": "HAC" if lags > 0 else "OLS",
        }
    
    return results


def build_html(uni_results: list[dict], joint_results: dict) -> str:
    """Build HTML report."""
    html = "<h3>Univariate regressions with MPS_ORTH control</h3>\n"
    html += '<p class="note">btc_log_return ~ sentiment + MPS_ORTH. HAC SE for ≥24h windows.</p>\n'
    
    # Create table from univariate results
    if uni_results:
        df_uni = pd.DataFrame(uni_results)
        html += html_table(df_uni, sig_col="Sig_sentiment")
    else:
        html += "<p>No univariate results available.</p>\n"
    
    # Joint models
    html += "<h3>Joint model (all agents + control) — 1h and 24h windows</h3>\n"
    for w in [1, 24]:
        if w in joint_results:
            res = joint_results[w]
            if "error" in res:
                html += f"<p><strong>{w}h window:</strong> {res['error']}</p>\n"
            else:
                html += f"<p><strong>{w}h window</strong> — N={res['N']}, R²={res['R²']}, SE: {res['SE type']}</p>\n"
                html += html_table(res["table"], sig_col="Sig")
    
    return html


def run() -> str:
    """Main execution function."""
    print("Loading data...")
    df = load_data()
    
    print(f"Loaded {len(df)} press conferences")
    print(f"Date range: {df['meeting_date'].min()} to {df['meeting_date'].max()}")
    
    # Check available agent columns
    available_agents = [a for a in AGENTS if f"net_sentiment_{a}" in df.columns]
    print(f"Available agents: {available_agents}")
    
    # Check MPS data
    with_mps = df["MPS_ORTH"].notna().sum()
    print(f"Rows with MPS_ORTH data: {with_mps}/{len(df)}")
    
    print("\nRunning univariate regressions...")
    uni = run_univariate(df)
    
    print("Running joint model...")
    joint = run_joint(df)
    
    date_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    html = f"""
<h2>Press Conference Sentiment → Bitcoin Return (with MPS_ORTH control)</h2>
<p class="note">
  Source: CentralBankRoBERTa agent & sentiment scores on FOMC Press Conferences (2012–2023).<br>
  Control variable: MPS_ORTH (Monetary Policy Surprise, orthogonalized).<br>
  Data period: 2012–2023 (limited by MPS data availability).<br>
  Generated: {date_str}
</p>
{build_html(uni, joint)}
"""
    
    return html


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Press Conferences + MPS_ORTH Control</title>
<style>
{CSS}
</style>
</head>
<body>
{run()}
</body>
</html>"""
    
    out_file = OUT_DIR / "press_conference_regression_with_control.html"
    out_file.write_text(html, encoding="utf-8")
    print(f"\nReport saved to: {out_file}")
