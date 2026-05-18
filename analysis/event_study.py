"""
Event study: OLS and Spearman correlation using mean-adjusted abnormal returns (AR)
as the dependent variable instead of raw log returns.

Compares results to raw-return regressions to assess robustness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

from _common import (
    load_data, fig_to_b64, img_tag, html_table, sig_stars,
    AGENTS, WINDOWS, DOC_TYPES, PERIODS, PERIOD_COLORS, AGENT_COLORS, CSS
)

HAC_LAGS      = {1: 0, 3: 0, 9: 1, 24: 2, 72: 4, 144: 6, 216: 8, 288: 10, 360: 12}
EVENT_WINDOWS = [1, 3, 9, 24]   # short windows for event study
GROUPS        = [(dt, p) for dt in DOC_TYPES for p in PERIODS]


# ── helpers ───────────────────────────────────────────────────────────────────

def _ols_row(sub: pd.DataFrame, agent: str, w: int, y_col: str) -> dict | None:
    x_col = f"net_sentiment_{agent}"
    data  = sub[[x_col, y_col]].dropna()
    if len(data) < 10:
        return None
    X     = sm.add_constant(data[x_col])
    model = sm.OLS(data[y_col], X).fit()
    lags  = HAC_LAGS[w]
    if lags > 0:
        r    = model.get_robustcov_results(cov_type="HAC", maxlags=lags)
        idx  = list(model.params.index).index(x_col)
        beta, se, t, p = float(r.params[idx]), float(r.bse[idx]), float(r.tvalues[idx]), float(r.pvalues[idx])
    else:
        beta = float(model.params[x_col])
        se   = float(model.bse[x_col])
        t    = float(model.tvalues[x_col])
        p    = float(model.pvalues[x_col])
    return {
        "Agent":  agent.title(),
        "Window": f"{w}h",
        "N":      len(data),
        "β":      f"{beta:.4f}",
        "SE":     f"{se:.4f}",
        "t":      f"{t:.2f}",
        "p":      f"{p:.3f}",
        "Sig":    sig_stars(p),
        "R²":     f"{model.rsquared:.3f}",
    }


# ── plot: CAR (cumulative abnormal return) profile ────────────────────────────

def plot_car_profile(df: pd.DataFrame) -> str:
    """
    Average cumulative abnormal return split by sentiment tercile per agent.
    One row per agent, one column per document type.
    """
    fig, axes = plt.subplots(len(AGENTS), len(DOC_TYPES),
                             figsize=(12, 3.2 * len(AGENTS)), sharey=False)
    group_colors = {"Net Positive": "#2ca02c", "Net Negative": "#d62728"}

    for row_idx, agent in enumerate(AGENTS):
        for col_idx, doc_type in enumerate(DOC_TYPES):
            ax = axes[row_idx][col_idx]
            sub = df[df["document_type"] == doc_type].copy()
            sentiment_col = f"net_sentiment_{agent}"
            sub = sub.dropna(subset=[sentiment_col])
            if len(sub) < 9:
                ax.set_visible(False)
                continue
            sub = sub.copy()
            sub["tercile"] = sub[sentiment_col].apply(
                lambda v: "Net Positive" if v > 0 else "Net Negative"
            )

            for tercile in ["Net Negative", "Net Positive"]:
                group = sub[sub["tercile"] == tercile]
                means, lo, hi = [], [], []
                for w in EVENT_WINDOWS:
                    vals = group[f"btc_ar_{w}h"].dropna() * 100
                    n = len(vals)
                    if n > 1:
                        m  = vals.mean()
                        se = vals.std() / np.sqrt(n)
                        z  = stats.t.ppf(0.99, df=n - 1)   # 98% CI (two-tailed)
                        means.append(m)
                        lo.append(m - z * se)
                        hi.append(m + z * se)
                    else:
                        means.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                color = group_colors[tercile]
                ax.plot(EVENT_WINDOWS, means, marker="o", markersize=4,
                        color=color, label=f"{tercile} (n={len(group)})", linewidth=1.8)
                ax.fill_between(EVENT_WINDOWS, lo, hi, color=color, alpha=0.15)

            ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
            ax.set_xticks(EVENT_WINDOWS)
            ax.set_xticklabels([f"{w}h" for w in EVENT_WINDOWS], fontsize=8)
            ax.set_xlabel("Hours after release")
            ax.set_ylabel("Avg. CAR (%)")
            ax.set_title(f"{agent.title()} — {doc_type}", fontsize=9)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7, loc="best")

    fig.suptitle("Cumulative abnormal return by sentiment group per agent (shaded = 98% CI)", fontsize=11)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── plot: AR distribution at key windows ──────────────────────────────────────

def plot_ar_distribution(df: pd.DataFrame) -> str:
    focus = EVENT_WINDOWS
    fig, axes = plt.subplots(len(DOC_TYPES), len(focus), figsize=(13, 6), sharey=False)

    for row_idx, doc_type in enumerate(DOC_TYPES):
        sub = df[df["document_type"] == doc_type]
        for col_idx, w in enumerate(focus):
            ax = axes[row_idx][col_idx]
            col = f"btc_ar_{w}h"
            vals = sub[col].dropna() * 100
            ax.hist(vals, bins=25, color="#4E79A7", alpha=0.75, edgecolor="white")
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.axvline(vals.mean(), color="red", linewidth=1.2, linestyle="-", label=f"mean={vals.mean():.1f}%")
            ax.set_title(f"{doc_type} — {w}h", fontsize=9)
            ax.set_xlabel("AR (%)")
            ax.legend(fontsize=7)

    fig.suptitle("Distribution of abnormal Bitcoin returns around FOMC releases", fontsize=11)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── plot: AR heatmap (Spearman sentiment vs AR) ───────────────────────────────

def plot_ar_heatmap(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, len(PERIODS), figsize=(13, 8))
    for row_idx, doc_type in enumerate(DOC_TYPES):
        for col_idx, period in enumerate(PERIODS):
            ax = axes[row_idx][col_idx]
            sub = df[(df["document_type"] == doc_type) & (df["period"] == period)]
            rho_data, p_data = [], []
            for agent in AGENTS:
                x = sub[f"net_sentiment_{agent}"]
                row_r, row_p = [], []
                for w in EVENT_WINDOWS:
                    y    = sub[f"btc_ar_{w}h"]
                    mask = x.notna() & y.notna()
                    if mask.sum() < 8:
                        row_r.append(np.nan); row_p.append(np.nan)
                    else:
                        r, p = stats.spearmanr(x[mask], y[mask])
                        row_r.append(round(r, 3)); row_p.append(p)
                rho_data.append(row_r); p_data.append(row_p)

            rho_df = pd.DataFrame(rho_data, index=[a.title() for a in AGENTS],
                                  columns=[f"{w}h" for w in EVENT_WINDOWS])
            p_df   = pd.DataFrame(p_data,   index=[a.title() for a in AGENTS],
                                  columns=[f"{w}h" for w in EVENT_WINDOWS])
            sns.heatmap(rho_df, ax=ax, annot=True, fmt=".2f",
                        center=0, vmin=-0.5, vmax=0.5, cmap="RdYlGn",
                        linewidths=0.5, linecolor="#ddd", cbar_kws={"shrink": 0.8})
            for i, agent in enumerate(rho_df.index):
                for j, w_col in enumerate(rho_df.columns):
                    p = p_df.loc[agent, w_col]
                    if not np.isnan(p):
                        stars = sig_stars(p)
                        if stars:
                            ax.text(j + 0.75, i + 0.25, stars, ha="center", va="center",
                                    fontsize=7, color="black", fontweight="bold")
            ax.set_title(f"{doc_type} — {period}", fontsize=10)

    fig.suptitle("Spearman rho: net sentiment vs. abnormal Bitcoin return", fontsize=12)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── OLS on abnormal returns ───────────────────────────────────────────────────

def run_ols_ar(df: pd.DataFrame) -> str:
    html = ""
    for doc_type in DOC_TYPES:
        html += f"<h3>{doc_type}</h3>\n"
        for period in PERIODS:
            sub  = df[(df["document_type"] == doc_type) & (df["period"] == period)]
            rows = []
            for agent in AGENTS:
                for w in EVENT_WINDOWS:
                    row = _ols_row(sub, agent, w, f"btc_ar_{w}h")
                    if row:
                        rows.append(row)
            if rows:
                html += f"<p><strong>{period}</strong></p>\n"
                html += html_table(pd.DataFrame(rows), sig_col="Sig")
    return html


# ── comparison: raw vs AR ─────────────────────────────────────────────────────

def plot_raw_vs_ar(df: pd.DataFrame) -> str:
    """Scatter: OLS beta on raw log return vs OLS beta on AR, per agent × window × group."""
    raw_betas, ar_betas, labels = [], [], []

    for doc_type in DOC_TYPES:
        for period in PERIODS:
            sub = df[(df["document_type"] == doc_type) & (df["period"] == period)]
            for agent in AGENTS:
                x_col = f"net_sentiment_{agent}"
                for w in EVENT_WINDOWS:
                    for suffix, y_prefix in [("raw", "btc_log_return"), ("ar", "btc_ar")]:
                        y_col = f"{y_prefix}_{w}h"
                        data  = sub[[x_col, y_col]].dropna()
                        if len(data) < 10:
                            break
                    else:
                        continue
                    data_raw = sub[[x_col, f"btc_log_return_{w}h"]].dropna()
                    data_ar  = sub[[x_col, f"btc_ar_{w}h"]].dropna()
                    if len(data_raw) < 10 or len(data_ar) < 10:
                        continue
                    def beta(d, xc, yc):
                        X = sm.add_constant(d[xc])
                        return sm.OLS(d[yc], X).fit().params[xc]
                    raw_betas.append(beta(data_raw, x_col, f"btc_log_return_{w}h"))
                    ar_betas.append(beta(data_ar,  x_col, f"btc_ar_{w}h"))
                    labels.append(f"{agent[:3].title()} {w}h")

    if not raw_betas:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        return fig_to_b64(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(raw_betas, ar_betas, alpha=0.5, s=25, color="#4E79A7")
    lim = max(abs(min(raw_betas + ar_betas)), abs(max(raw_betas + ar_betas))) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, label="45° line (raw = AR)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("OLS β — raw log return")
    ax.set_ylabel("OLS β — abnormal return")
    ax.set_title("Raw return β vs. abnormal return β\n(each point = one agent × window × group)", fontsize=10)
    ax.legend(fontsize=8)
    r, p = stats.pearsonr(raw_betas, ar_betas)
    ax.text(0.05, 0.95, f"Pearson r = {r:.3f} (p={p:.3f})",
            transform=ax.transAxes, fontsize=8, va="top")
    fig.tight_layout()
    return fig_to_b64(fig)


# ── public run function ───────────────────────────────────────────────────────

def run() -> str:
    df = load_data()

    return f"""
<h2>5. Event Study — Mean-Adjusted Abnormal Returns</h2>
<p class="note">
  Abnormal return AR<sub>i,w</sub> = log return over window w &minus; (mean hourly log return
  over 30-day estimation window &times; w). The estimation window ends 1 hour before each
  document release. This removes baseline Bitcoin drift from the dependent variable.
</p>

<h3>Abnormal Return Distribution</h3>
<p class="note">Distribution of abnormal returns at key windows. Mean shown in red.</p>
{img_tag(plot_ar_distribution(df))}

<h3>Cumulative Abnormal Return Profile by Sentiment Tercile</h3>
<p class="note">
  Documents split into terciles by central bank net sentiment.
  Shows whether more positive/negative central bank language is associated with
  systematically higher/lower abnormal returns over the post-release horizon.
</p>
{img_tag(plot_car_profile(df))}

<h3>Spearman Correlation: Sentiment vs. Abnormal Return</h3>
<p class="note">Stars: * p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01 (uncorrected).</p>
{img_tag(plot_ar_heatmap(df))}

<h3>OLS on Abnormal Returns</h3>
<p class="note">
  btc_ar_Xh ~ net_sentiment_agent. HAC (Newey-West) SE for windows &ge; 9h.
  Stars: * p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01.
</p>
{run_ols_ar(df)}

<h3>Robustness: Raw Return β vs. Abnormal Return β</h3>
<p class="note">
  Each point is one agent &times; window &times; period &times; document-type combination.
  If raw and abnormal return estimates agree, points lie near the 45° line.
</p>
{img_tag(plot_raw_vs_ar(df))}
"""


if __name__ == "__main__":
    from pathlib import Path
    html = f"<!DOCTYPE html><html><head><meta charset='UTF-8'><style>{CSS}</style></head><body>{run()}</body></html>"
    out  = Path(__file__).parent / "outputs" / "event_study.html"
    out.write_text(html, encoding="utf-8")
    print(f"Saved: {out}")
