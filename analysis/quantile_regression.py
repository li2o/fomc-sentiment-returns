"""
Quantile regression: btc_log_return_Xh ~ net_sentiment_agent
  - Quantiles: 0.10, 0.25, 0.50, 0.75, 0.90
  - Focus windows: 1h, 24h
  - Separate per document type × period × agent
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from _common import (
    load_data, fig_to_b64, img_tag, html_table, sig_stars,
    AGENTS, DOC_TYPES, PERIODS, AGENT_COLORS, CSS
)

QUANTILES     = [0.10, 0.25, 0.50, 0.75, 0.90]
FOCUS_WINDOWS = [1, 24]
GROUPS        = [(dt, p) for dt in DOC_TYPES for p in PERIODS]


def run_qr(df: pd.DataFrame) -> dict:
    results = {}
    for doc_type, period in GROUPS:
        results[(doc_type, period)] = {}
        sub = df[(df["document_type"] == doc_type) & (df["period"] == period)].copy()
        for agent in AGENTS:
            results[(doc_type, period)][agent] = {}
            x_col = f"net_sentiment_{agent}".replace(" ", "_")
            sub = sub.rename(columns={f"net_sentiment_{agent}": x_col}, errors="ignore")
            for w in FOCUS_WINDOWS:
                y_col = f"btc_log_return_{w}h"
                data = sub[[x_col, y_col]].dropna()
                if len(data) < 10:
                    continue
                qr_res = {}
                for q in QUANTILES:
                    try:
                        model = smf.quantreg(f"{y_col} ~ {x_col}", data=data).fit(q=q)
                        qr_res[q] = {
                            "coef":  model.params[x_col],
                            "p":     model.pvalues[x_col],
                            "ci_lo": model.conf_int().loc[x_col, 0],
                            "ci_hi": model.conf_int().loc[x_col, 1],
                        }
                    except Exception:
                        pass
                results[(doc_type, period)][agent][w] = qr_res
    return results


def plot_qr(results: dict) -> str:
    n_cols = len(FOCUS_WINDOWS) * len(PERIODS)
    fig, axes = plt.subplots(len(AGENTS), n_cols, figsize=(14, 3 * len(AGENTS)), sharey=False)
    col_idx = 0
    for w in FOCUS_WINDOWS:
        for period in PERIODS:
            for row_idx, agent in enumerate(AGENTS):
                ax = axes[row_idx][col_idx]
                for doc_type in DOC_TYPES:
                    qr = results.get((doc_type, period), {}).get(agent, {}).get(w, {})
                    if not qr:
                        continue
                    qs    = list(qr.keys())
                    coefs = [qr[q]["coef"] for q in qs]
                    lo    = [qr[q]["coef"] - qr[q]["ci_lo"] for q in qs]
                    hi    = [qr[q]["ci_hi"] - qr[q]["coef"] for q in qs]
                    style = "o-" if doc_type == "Minutes" else "s--"
                    ax.errorbar(qs, coefs, yerr=[lo, hi], fmt=style,
                                color=AGENT_COLORS[agent], capsize=3, linewidth=1.2,
                                alpha=0.8, label=doc_type)
                ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
                ax.set_xticks(QUANTILES)
                ax.set_xticklabels([f"q{int(q*100)}" for q in QUANTILES], fontsize=7)
                ax.set_title(f"{agent.title()}\n{period} {w}h", fontsize=8)
                ax.set_ylabel("β")
            col_idx += 1

    axes[0][0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Quantile regression coefficients ± 95% CI (circle=Minutes, square=Statement)", fontsize=10)
    fig.tight_layout()
    return fig_to_b64(fig)


def build_tables(results: dict) -> str:
    html = ""
    for doc_type in DOC_TYPES:
        html += f"<h3>{doc_type}</h3>\n"
        for period in PERIODS:
            html += f"<h4>{period}</h4>\n"
            for w in FOCUS_WINDOWS:
                html += f"<p><strong>{w}h window</strong></p>\n"
                rows = []
                for agent in AGENTS:
                    qr = results.get((doc_type, period), {}).get(agent, {}).get(w, {})
                    row = {"Agent": agent.title()}
                    for q in QUANTILES:
                        if q in qr:
                            row[f"q{int(q*100)}"] = f"{qr[q]['coef']:.4f}{sig_stars(qr[q]['p'])}"
                        else:
                            row[f"q{int(q*100)}"] = "—"
                    rows.append(row)
                html += html_table(pd.DataFrame(rows))
    return html


def run() -> str:
    df = load_data()
    for agent in AGENTS:
        safe = f"net_sentiment_{agent}".replace(" ", "_")
        df[safe] = df[f"net_sentiment_{agent}"]
    results = run_qr(df)
    return f"""
<h2>3. Quantile Regression</h2>
<p class="note">
  Coefficients at q10/q25/q50/q75/q90 of the Bitcoin log return distribution.
  Focus windows: 1h and 24h. Split by period.
  Stars: * p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01.
</p>
{img_tag(plot_qr(results))}
{build_tables(results)}
"""


if __name__ == "__main__":
    from pathlib import Path
    html = f"<!DOCTYPE html><html><head><meta charset='UTF-8'><style>{CSS}</style></head><body>{run()}</body></html>"
    out = Path(__file__).parent / "outputs" / "quantile_regression.html"
    out.write_text(html, encoding="utf-8")
    print(f"Saved: {out}")
