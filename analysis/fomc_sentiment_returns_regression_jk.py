"""
Robustness check: OLS regressions using Jarocinski-Karadi (2020) shocks as controls.

Model:
  CumulativeReturn(t+h) = α + β·NetSentiment_[agent] + γ·MP_median + δ·CBI_median + ζ·NFCI + ε

Controls:
  - MP_median   : Monetary Policy shock (median rotation, Jarocinski & Karadi 2020, updated Jan 2024)
  - CBI_median  : Central Bank Information shock (median rotation, same source)
  - NFCI        : Chicago Fed National Financial Conditions Index (weekly, last Friday ≤ meeting date)

Dependent variables:
  - BTC cumulative log return at horizons 1, 2, 3, 5, 7, 10, 14 days after FOMC
  - SPX cumulative log return at same horizons
  - ZT=F (2-Year Treasury Futures) cumulative log return at same horizons

Sample: 2015-01-01 to 2025-12-31  (JK data available through 2024-01-31, so effective end ~2024)
Source: FOMC Press Conferences document-level sentiment · Yahoo Finance · Jarocinski-Karadi (2024) · Chicago Fed NFCI
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import fig_to_b64, img_tag, html_table, sig_stars, AGENTS, AGENT_COLORS, CSS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE   = Path(__file__).resolve().parent.parent
PC_CSV      = WORKSPACE / "llm_analysis/outputs/document_level/press_conferences_document_level.csv"
RETURNS_CSV = WORKSPACE / "data/market/returns_14d.csv"
ZT_CSV      = WORKSPACE / "data/market/zt_yahoo_1d.csv"
JK_FILE     = WORKSPACE / "data/control variables/shocks_fed_jk_t.csv"
NFCI_FILE   = WORKSPACE / "data/control variables/financial condition.csv"
OUT_DIR     = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

HORIZONS    = [1, 2, 3, 5, 7, 10, 14]
DATE_MIN    = pd.Timestamp("2015-01-01")
DATE_MAX    = pd.Timestamp("2023-12-31")

ALL_SENTIMENTS = AGENTS + ["overall"]
SENT_COLORS    = {**AGENT_COLORS, "overall": "#333333"}

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_sentiments() -> pd.DataFrame:
    df = pd.read_csv(PC_CSV)
    df["meeting_date"] = pd.to_datetime(df["meeting_date"], errors="coerce")
    agent_cols = [f"net_sentiment_{a}" for a in AGENTS]
    df["net_sentiment_overall"] = pd.to_numeric(df["net_sentiment"], errors="coerce")
    keep = ["meeting_date"] + agent_cols + ["net_sentiment_overall"]
    return df[keep].dropna(subset=["meeting_date"]).copy()


def load_jk() -> pd.DataFrame:
    """Load Jarocinski-Karadi MP and CBI shocks."""
    jk = pd.read_csv(JK_FILE, parse_dates=["date"])
    jk["MP_median"]  = pd.to_numeric(jk["MP_median"],  errors="coerce")
    jk["CBI_median"] = pd.to_numeric(jk["CBI_median"], errors="coerce")
    return jk[["date", "MP_median", "CBI_median"]].dropna().sort_values("date").reset_index(drop=True)


def load_nfci() -> pd.DataFrame:
    nfci = pd.read_csv(NFCI_FILE, parse_dates=["date"])
    nfci["NFCI"] = pd.to_numeric(nfci["NFCI"], errors="coerce")
    return nfci[["date", "NFCI"]].dropna().sort_values("date").reset_index(drop=True)


def match_nfci(meeting_dates: pd.Series, nfci_df: pd.DataFrame) -> pd.Series:
    """Last available Friday on or before each meeting_date."""
    nfci_sorted = nfci_df.sort_values("date")
    result = []
    for d in meeting_dates:
        past = nfci_sorted[nfci_sorted["date"] <= d]
        result.append(float(past["NFCI"].iloc[-1]) if len(past) else float("nan"))
    return pd.Series(result, index=meeting_dates.index)


def load_returns() -> dict[int, pd.DataFrame]:
    """Return dict horizon -> DataFrame with meeting_date, btc_cum, spx_cum, zt_cum."""
    df = pd.read_csv(RETURNS_CSV, parse_dates=["fomc_date"])

    zt = pd.read_csv(ZT_CSV, parse_dates=["date"]).set_index("date")["close"]
    zt.index = zt.index.normalize()

    fomc_dates = df["fomc_date"].drop_duplicates().sort_values()
    zt_rows: dict[tuple, float] = {}
    for fomc_date in fomc_dates:
        past = zt[zt.index <= fomc_date]
        zt0 = float(past.iloc[-1]) if len(past) else np.nan
        for d in HORIZONS:
            target = fomc_date + pd.Timedelta(days=d)
            match = zt[zt.index == target]
            zt_c = float(match.iloc[0]) if len(match) else np.nan
            cum = (np.log(zt_c / zt0)
                   if not (np.isnan(zt_c) or np.isnan(zt0) or zt0 <= 0)
                   else np.nan)
            zt_rows[(fomc_date.date(), d)] = cum

    result = {}
    for h in HORIZONS:
        sub = df[df["days_after"] == h][
            ["fomc_date", "btc_cumulative_return", "spx_cumulative_return"]
        ].copy()
        sub = sub.rename(columns={"fomc_date": "meeting_date"})
        sub["zt_cumulative_return"] = sub["meeting_date"].apply(
            lambda d: zt_rows.get((d.date() if hasattr(d, "date") else d, h), np.nan)
        )
        result[h] = sub
    return result


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def run_ols(y: pd.Series, X_df: pd.DataFrame) -> dict | None:
    data = pd.concat([y, X_df], axis=1).dropna()
    if len(data) < 10:
        return None
    y_fit = data.iloc[:, 0]
    X_fit = sm.add_constant(data.iloc[:, 1:])
    model = sm.OLS(y_fit, X_fit).fit(cov_type="HC3")
    return {
        "model":  model,
        "n":      int(model.nobs),
        "r2":     model.rsquared,
        "r2_adj": model.rsquared_adj,
        "params": model.params,
        "bse":    model.bse,
        "tvals":  model.tvalues,
        "pvals":  model.pvalues,
    }


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def result_table_html(rows: list[dict]) -> str:
    records = []
    for r in rows:
        records.append({
            "Agent":        r["agent"],
            "Horizon":      f"Day {r['horizon']}",
            "Asset":        r["asset"],
            "N":            r["n"],
            "β_sentiment":  r["b_sent"],
            "SE":           r["se_sent"],
            "t":            r["t_sent"],
            "p":            r["p_sent"],
            "Sig":          r["sig_sent"],
            "β_MP_median":  r["b_mp"],
            "SE_MP":        r["se_mp"],
            "t_MP":         r["t_mp"],
            "p_MP":         r["p_mp"],
            "Sig_MP":       r["sig_mp"],
            "β_CBI_median": r["b_cbi"],
            "SE_CBI":       r["se_cbi"],
            "t_CBI":        r["t_cbi"],
            "p_CBI":        r["p_cbi"],
            "Sig_CBI":      r["sig_cbi"],
            "β_NFCI":       r["b_nfci"],
            "SE_NFCI":      r["se_nfci"],
            "t_NFCI":       r["t_nfci"],
            "p_NFCI":       r["p_nfci"],
            "Sig_NFCI":     r["sig_nfci"],
            "R²":           r["r2"],
            "R²_adj":       r["r2_adj"],
        })
    return html_table(pd.DataFrame(records), sig_col="Sig")


def make_coef_path_chart(all_rows: list[dict], asset: str) -> str:
    """β path across horizons with 95% CI band, ★ at p<0.10."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, color="#aaa", linewidth=0.8, linestyle="--")

    for agent in ALL_SENTIMENTS:
        rows = sorted(
            [r for r in all_rows if r["asset"] == asset and r["agent"] == agent],
            key=lambda x: x["horizon"]
        )
        if not rows:
            continue
        xs  = [r["horizon"] for r in rows]
        bs  = [r["b_sent_n"] for r in rows]
        ses = [r["se_sent_n"] for r in rows]
        ci  = [1.96 * s for s in ses]
        color = SENT_COLORS[agent]
        lw = 2.5 if agent == "overall" else 1.8
        label = "Overall" if agent == "overall" else agent.title()
        ax.plot(xs, bs, marker="o", color=color, label=label, linewidth=lw,
                markersize=6 if agent == "overall" else 5, zorder=4 if agent == "overall" else 3)
        ax.fill_between(xs,
                        [b - c for b, c in zip(bs, ci)],
                        [b + c for b, c in zip(bs, ci)],
                        color=color, alpha=0.12)
        for x, b, p in zip(xs, bs, [r["p_sent_n"] for r in rows]):
            if p < 0.10:
                ax.plot(x, b, marker="*", color=color, markersize=11, zorder=5)

    ax.set_xlabel("Calendar days after FOMC rate decision", fontsize=11)
    ax.set_ylabel("β (net sentiment → cumulative return)", fontsize=11)
    ax.set_title(f"{asset} – Sentiment Coefficient Path (JK controls)", fontsize=13)
    ax.set_xticks(HORIZONS)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.7)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return img_tag(b64)


def make_scatter_grid(all_rows: list[dict], asset: str, horizon: int = 7) -> str:
    """2×3 grid of scatter plots (one per sentiment) at a chosen horizon."""
    rows_h = [r for r in all_rows if r["asset"] == asset and r["horizon"] == horizon]
    if not rows_h:
        return ""

    ret_col_label = {
        "BTC": "BTC cumulative log return",
        "SPX": "SPX cumulative log return",
        "ZT":  "ZT=F cumulative log return",
    }.get(asset, "cumulative log return")

    ncols = 3
    nrows = (len(ALL_SENTIMENTS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 4))
    axes_flat = axes.flatten()

    for i, agent in enumerate(ALL_SENTIMENTS):
        ax = axes_flat[i]
        row = next((r for r in rows_h if r["agent"] == agent), None)
        if row is None:
            ax.set_visible(False)
            continue

        df = row["_df_h"][[row["_sent_col"], row["_ret_col"]]].dropna()
        x = df[row["_sent_col"]]
        y = df[row["_ret_col"]]
        color = SENT_COLORS[agent]

        ax.scatter(x, y, color=color, alpha=0.65, s=30, zorder=3)
        if len(x) > 2:
            m, b = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, m * x_line + b, color=color, linewidth=1.8)

        label = "Overall" if agent == "overall" else agent.title()
        ax.set_title(f"{label}  β={row['b_sent']} {row['sig_sent']}", fontsize=10)
        ax.set_xlabel(f"Net sentiment ({label.lower()})", fontsize=8)
        ax.set_ylabel(ret_col_label, fontsize=8)
        ax.axhline(0, color="#bbb", linewidth=0.6)
        ax.axvline(0, color="#bbb", linewidth=0.6)
        ax.grid(linewidth=0.3, alpha=0.5)

    for j in range(len(ALL_SENTIMENTS), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"{asset} Cumulative Return vs Sentiment — Day {horizon} after FOMC (JK controls)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return img_tag(b64)


def build_html(all_rows: list[dict], n_total: int, n_jk: int) -> str:
    sections = []

    for asset in ["BTC", "SPX", "ZT"]:
        asset_rows = [r for r in all_rows if r["asset"] == asset]

        sections.append(f"<h2>{asset} – Coefficient Path Across Horizons</h2>")
        sections.append("<p class='note'>Bands = 95% CI (HC3). ★ = significant at 10% level.</p>")
        sections.append(make_coef_path_chart(all_rows, asset))

        sections.append(f"<h2>{asset} – Sentiment vs Return Scatter (Day 7)</h2>")
        sections.append(make_scatter_grid(all_rows, asset, horizon=7))

        sections.append(f"<h2>{asset} – Full Regression Table</h2>")
        sections.append(result_table_html(asset_rows))

    horizons_str = ", ".join(str(h) for h in HORIZONS)
    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8">
<title>FOMC Sentiment → Returns (Jarocinski-Karadi Robustness)</title>
<style>{CSS}</style>
</head>
<body>
<h1>Robustness Check: FOMC Sentiment → Market Returns (Jarocinski-Karadi Controls)</h1>
<p class="note">
  Model: CumulativeReturn<sub>t,h</sub> = α + β·NetSentiment<sub>agent,t</sub>
  + γ·MP_median<sub>t</sub> + δ·CBI_median<sub>t</sub> + ζ·NFCI<sub>t</sub> + ε<br>
  Controls: <b>MP_median</b> and <b>CBI_median</b> are Monetary Policy and Central Bank Information shocks
  from Jarocinski &amp; Karadi (2020), updated through January 2024
  (<a href="https://github.com/marekjarocinski/jkshocks_update_fed_202401">jkshocks_update_fed_202401</a>).
  <b>NFCI</b> = Chicago Fed National Financial Conditions Index (last Friday ≤ meeting date).<br>
  Dependent variable: cumulative log return h calendar days after FOMC rate decision (h = {horizons_str}).<br>
  Sample: 2015–2023.
  Press conferences (2015–2025): <b>{n_total}</b> total; <b>{n_jk}</b> matched with JK shocks.<br>
  Standard errors: HC3 heteroskedasticity-robust.
</p>
{"".join(sections)}
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sentiment_df = load_sentiments()
    jk_df        = load_jk()
    nfci_df      = load_nfci()
    returns_dict = load_returns()

    # Restrict to 2015–2025
    sentiment_df = sentiment_df[
        (sentiment_df["meeting_date"] >= DATE_MIN) &
        (sentiment_df["meeting_date"] <= DATE_MAX)
    ].copy()
    n_total = len(sentiment_df)

    # Merge with JK shocks (left join on exact date)
    jk_df = jk_df.rename(columns={"date": "meeting_date"})
    base = sentiment_df.merge(jk_df, on="meeting_date", how="left")
    n_jk = int(base["MP_median"].notna().sum())

    # Add NFCI: last Friday on or before meeting_date
    base["NFCI"] = match_nfci(base["meeting_date"], nfci_df)

    print(
        f"Press conferences (2015-2025): {n_total}"
        f"  |  with JK shocks: {n_jk}"
        f"  |  with NFCI: {int(base['NFCI'].notna().sum())}"
    )

    all_rows: list[dict] = []

    for h in HORIZONS:
        ret = returns_dict[h]
        df_h = base.merge(ret, on="meeting_date", how="inner")

        for agent in ALL_SENTIMENTS:
            sent_col = f"net_sentiment_{agent}"

            for asset, ret_col in [
                ("BTC", "btc_cumulative_return"),
                ("SPX", "spx_cumulative_return"),
                ("ZT",  "zt_cumulative_return"),
            ]:
                X = df_h[[sent_col, "MP_median", "CBI_median", "NFCI"]]
                y = df_h[ret_col]

                res = run_ols(y, X)
                if res is None:
                    continue

                params = res["params"]
                bse    = res["bse"]
                tvals  = res["tvals"]
                pvals  = res["pvals"]

                all_rows.append({
                    "agent":    agent,
                    "horizon":  h,
                    "asset":    asset,
                    "n":        res["n"],
                    # sentiment
                    "b_sent":   f"{params[sent_col]:.4f}",
                    "se_sent":  f"{bse[sent_col]:.4f}",
                    "t_sent":   f"{tvals[sent_col]:.2f}",
                    "p_sent":   f"{pvals[sent_col]:.3f}",
                    "sig_sent": sig_stars(pvals[sent_col]),
                    # MP_median
                    "b_mp":     f"{params.get('MP_median', float('nan')):.4f}",
                    "se_mp":    f"{bse.get('MP_median', float('nan')):.4f}",
                    "t_mp":     f"{tvals.get('MP_median', float('nan')):.2f}",
                    "p_mp":     f"{pvals.get('MP_median', float('nan')):.3f}",
                    "sig_mp":   sig_stars(pvals.get("MP_median", 1.0)),
                    # CBI_median
                    "b_cbi":    f"{params.get('CBI_median', float('nan')):.4f}",
                    "se_cbi":   f"{bse.get('CBI_median', float('nan')):.4f}",
                    "t_cbi":    f"{tvals.get('CBI_median', float('nan')):.2f}",
                    "p_cbi":    f"{pvals.get('CBI_median', float('nan')):.3f}",
                    "sig_cbi":  sig_stars(pvals.get("CBI_median", 1.0)),
                    # NFCI
                    "b_nfci":   f"{params.get('NFCI', float('nan')):.4f}",
                    "se_nfci":  f"{bse.get('NFCI', float('nan')):.4f}",
                    "t_nfci":   f"{tvals.get('NFCI', float('nan')):.2f}",
                    "p_nfci":   f"{pvals.get('NFCI', float('nan')):.3f}",
                    "sig_nfci": sig_stars(pvals.get("NFCI", 1.0)),
                    # summary
                    "r2":       f"{res['r2']:.4f}",
                    "r2_adj":   f"{res['r2_adj']:.4f}",
                    # numeric for charts
                    "b_sent_n":  float(params[sent_col]),
                    "se_sent_n": float(bse[sent_col]),
                    "p_sent_n":  float(pvals[sent_col]),
                    "r2_n":      float(res["r2"]),
                    # raw data for scatter
                    "_df_h":    df_h,
                    "_sent_col": sent_col,
                    "_ret_col":  ret_col,
                })

    html = build_html(all_rows, n_total, n_jk)
    out_path = OUT_DIR / "fomc_sentiment_returns_regression_jk.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\nSaved → {out_path}")
    print(f"Total regressions run: {len(all_rows)}")


if __name__ == "__main__":
    main()
