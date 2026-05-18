"""
OLS regression: btc_log_return_Xh ~ net_sentiment_agent
  - Run separately per document type × period × agent × window
  - HAC (Newey-West) standard errors for windows >= 24h
  - Joint model with all agents for 1h and 24h windows
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from _common import (
    load_data, fig_to_b64, img_tag, html_table, sig_stars,
    AGENTS, WINDOWS, DOC_TYPES, PERIODS, PERIOD_COLORS, CSS
)

HAC_WINDOWS = {1: 0, 3: 0, 9: 1, 24: 2, 72: 4, 144: 6, 216: 8, 288: 10, 360: 12}
GROUPS = [(dt, p) for dt in DOC_TYPES for p in PERIODS]


def _hac_series(model, lags: int) -> tuple:
    """Return (params, bse, tvalues, pvalues) as named Series with HAC SEs."""
    r = model.get_robustcov_results(cov_type="HAC", maxlags=lags)
    idx = model.params.index
    return (
        pd.Series(r.params,  index=idx),
        pd.Series(r.bse,     index=idx),
        pd.Series(r.tvalues, index=idx),
        pd.Series(r.pvalues, index=idx),
    )


def run_univariate(df: pd.DataFrame) -> dict[tuple, list[dict]]:
    results: dict[tuple, list[dict]] = {g: [] for g in GROUPS}
    for doc_type, period in GROUPS:
        sub = df[(df["document_type"] == doc_type) & (df["period"] == period)].copy()
        for agent in AGENTS:
            for w in WINDOWS:
                x_col = f"net_sentiment_{agent}"
                y_col = f"btc_log_return_{w}h"
                data = sub[[x_col, y_col]].dropna()
                if len(data) < 10:
                    continue
                X = sm.add_constant(data[x_col])
                model = sm.OLS(data[y_col], X).fit()
                lags = HAC_WINDOWS[w]
                if lags > 0:
                    params, bse, tvals, pvals = _hac_series(model, lags)
                else:
                    params, bse, tvals, pvals = model.params, model.bse, model.tvalues, model.pvalues
                results[(doc_type, period)].append({
                    "Agent":   agent.title(),
                    "Window":  f"{w}h",
                    "N":       len(data),
                    "β":       f"{float(params[x_col]):.4f}",
                    "SE":      f"{float(bse[x_col]):.4f}",
                    "t":       f"{float(tvals[x_col]):.2f}",
                    "p":       f"{float(pvals[x_col]):.3f}",
                    "Sig":     sig_stars(float(pvals[x_col])),
                    "R²":      f"{model.rsquared:.3f}",
                    "SE type": "HAC" if lags > 0 else "OLS",
                })
    return results


def run_joint(df: pd.DataFrame, windows: list[int] = [1, 24]) -> dict[tuple, dict]:
    results: dict[tuple, dict] = {}
    agent_cols = [f"net_sentiment_{a}" for a in AGENTS]
    for doc_type, period in GROUPS:
        sub = df[(df["document_type"] == doc_type) & (df["period"] == period)].copy()
        results[(doc_type, period)] = {}
        for w in windows:
            y_col = f"btc_log_return_{w}h"
            data = sub[agent_cols + [y_col]].dropna()
            if len(data) < 12:
                continue
            X = sm.add_constant(data[agent_cols])
            fitted = sm.OLS(data[y_col], X).fit()
            lags = HAC_WINDOWS[w]
            if lags > 0:
                params, bse, tvals, pvals = _hac_series(fitted, lags)
            else:
                params, bse, tvals, pvals = fitted.params, fitted.bse, fitted.tvalues, fitted.pvalues
            rows = []
            for col, agent in zip(agent_cols, AGENTS):
                rows.append({
                    "Agent": agent.title(),
                    "β":     f"{float(params[col]):.4f}",
                    "SE":    f"{float(bse[col]):.4f}",
                    "t":     f"{float(tvals[col]):.2f}",
                    "p":     f"{float(pvals[col]):.3f}",
                    "Sig":   sig_stars(float(pvals[col])),
                })
            results[(doc_type, period)][w] = {
                "table":   pd.DataFrame(rows),
                "R²":      round(fitted.rsquared, 3),
                "N":       len(data),
                "SE type": "HAC" if lags > 0 else "OLS",
            }
    return results


def plot_coef(results: dict[tuple, list[dict]]) -> str:
    """Coefficient plot per agent: across periods, per doc type."""
    offset_step = 0.22
    fig, axes = plt.subplots(len(AGENTS), len(DOC_TYPES), figsize=(13, 3 * len(AGENTS)), sharey=False)
    for row_idx, agent in enumerate(AGENTS):
        for col_idx, doc_type in enumerate(DOC_TYPES):
            ax = axes[row_idx][col_idx]
            for p_idx, period in enumerate(PERIODS):
                rows = [r for r in results[(doc_type, period)] if r["Agent"] == agent.title()]
                if not rows:
                    continue
                windows = [r["Window"] for r in rows]
                betas   = [float(r["β"]) for r in rows]
                ses     = [float(r["SE"]) for r in rows]
                x = [i + p_idx * offset_step for i in range(len(windows))]
                ax.errorbar(x, betas, yerr=[1.96 * s for s in ses],
                            fmt="o", color=PERIOD_COLORS[period], capsize=3,
                            linewidth=1.2, label=period)
            ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
            center = (len(PERIODS) - 1) * offset_step / 2
            tick_x = [i + center for i in range(len(WINDOWS))]
            ax.set_xticks(tick_x)
            ax.set_xticklabels([f"{w}h" for w in WINDOWS], fontsize=7)
            ax.set_title(f"{agent.title()} — {doc_type}", fontsize=9)
            ax.set_ylabel("β")
    axes[0][0].legend(fontsize=7, loc="upper right")
    legend_desc = " | ".join(f"{c}={p}" for p, c in PERIOD_COLORS.items())
    fig.suptitle(f"OLS β ± 1.96·SE by period  ({legend_desc})", fontsize=9)
    fig.tight_layout()
    return fig_to_b64(fig)


def build_html(uni: dict, joint: dict) -> str:
    html = "<h3>Univariate regressions</h3>\n"
    html += '<p class="note">One regression per agent × window. HAC SE for ≥24h windows.</p>\n'
    for doc_type in DOC_TYPES:
        html += f"<h4>{doc_type}</h4>\n"
        for period in PERIODS:
            tdf = pd.DataFrame(uni[(doc_type, period)])
            if not tdf.empty:
                html += f"<p><strong>{period}</strong></p>\n"
                html += html_table(tdf, sig_col="Sig")

    html += img_tag(plot_coef(uni))

    html += "<h3>Joint model (all agents) — 1h and 24h</h3>\n"
    for doc_type in DOC_TYPES:
        html += f"<h4>{doc_type}</h4>\n"
        for period in PERIODS:
            for w, res in joint.get((doc_type, period), {}).items():
                html += f"<p><strong>{period} — {w}h</strong> — N={res['N']}, R²={res['R²']}, SE: {res['SE type']}</p>\n"
                html += html_table(res["table"], sig_col="Sig")
    return html


def run() -> str:
    df = load_data()
    uni   = run_univariate(df)
    joint = run_joint(df)
    return f"""
<h2>2. OLS Regression</h2>
<p class="note">
  Dependent variable: Bitcoin log return. Independent variable: net sentiment per agent.
  Stars: * p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01.
</p>
{build_html(uni, joint)}
"""


if __name__ == "__main__":
    from pathlib import Path
    html = f"<!DOCTYPE html><html><head><meta charset='UTF-8'><style>{CSS}</style></head><body>{run()}</body></html>"
    out = Path(__file__).parent / "outputs" / "ols_regression.html"
    out.write_text(html, encoding="utf-8")
    print(f"Saved: {out}")
