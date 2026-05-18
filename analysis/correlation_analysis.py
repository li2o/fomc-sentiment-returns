"""
Spearman rank correlation between net_sentiment per agent and Bitcoin log returns.
Produces a heatmap per document type × period and a table with p-values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from _common import (
    load_data, fig_to_b64, img_tag, html_table, sig_stars,
    AGENTS, WINDOWS, DOC_TYPES, PERIODS, CSS
)

GROUPS = [(dt, p) for dt in DOC_TYPES for p in PERIODS]


def compute_correlations(df: pd.DataFrame) -> dict[tuple, pd.DataFrame]:
    results = {}
    for doc_type, period in GROUPS:
        sub = df[(df["document_type"] == doc_type) & (df["period"] == period)]
        rows = []
        for agent in AGENTS:
            x = sub[f"net_sentiment_{agent}"]
            row = {"Agent": agent.title()}
            for w in WINDOWS:
                y = sub[f"btc_log_return_{w}h"]
                mask = x.notna() & y.notna()
                if mask.sum() < 8:
                    row[f"{w}h_r"] = np.nan
                    row[f"{w}h_p"] = np.nan
                else:
                    r, p = stats.spearmanr(x[mask], y[mask])
                    row[f"{w}h_r"] = round(r, 3)
                    row[f"{w}h_p"] = round(p, 3)
            rows.append(row)
        results[(doc_type, period)] = pd.DataFrame(rows).set_index("Agent")
    return results


def plot_heatmaps(corr_dfs: dict[tuple, pd.DataFrame]) -> str:
    fig, axes = plt.subplots(2, len(PERIODS), figsize=(13, 8))
    for row_idx, doc_type in enumerate(DOC_TYPES):
        for col_idx, period in enumerate(PERIODS):
            ax = axes[row_idx][col_idx]
            key = (doc_type, period)
            r_cols = [c for c in corr_dfs[key].columns if c.endswith("_r")]
            p_cols = [c for c in corr_dfs[key].columns if c.endswith("_p")]
            r_df = corr_dfs[key][r_cols].copy().astype(float)
            p_df = corr_dfs[key][p_cols].copy().astype(float)
            r_df.columns = [f"{w}h" for w in WINDOWS]
            p_df.columns = [f"{w}h" for w in WINDOWS]
            n = len(corr_dfs[key])

            sns.heatmap(
                r_df, ax=ax, annot=True, fmt=".2f",
                center=0, vmin=-0.6, vmax=0.6,
                cmap="RdYlGn", linewidths=0.5, linecolor="#ddd",
                cbar_kws={"shrink": 0.8},
            )
            for i, agent in enumerate(r_df.index):
                for j, w_col in enumerate(r_df.columns):
                    p = p_df.loc[agent, w_col]
                    stars = sig_stars(p) if not np.isnan(p) else ""
                    if stars:
                        ax.text(j + 0.75, i + 0.25, stars, ha="center", va="center",
                                fontsize=7, color="black", fontweight="bold")
            sub = corr_dfs[key]
            n_docs = len(corr_dfs[key])
            ax.set_title(f"{doc_type} — {period}", fontsize=10)
            ax.set_xlabel("Return window")
            ax.set_ylabel("")
    fig.suptitle("Spearman ρ: net sentiment vs. Bitcoin log return (by period)", fontsize=12)
    fig.tight_layout()
    return fig_to_b64(fig)


def build_tables(corr_dfs: dict[tuple, pd.DataFrame]) -> str:
    html = ""
    for doc_type in DOC_TYPES:
        html += f"<h3>{doc_type}</h3>\n"
        for period in PERIODS:
            df = corr_dfs[(doc_type, period)]
            rows = []
            for agent in df.index:
                row = {"Agent": agent}
                for w in WINDOWS:
                    r = df.loc[agent, f"{w}h_r"]
                    p = df.loc[agent, f"{w}h_p"]
                    if np.isnan(r):
                        row[f"{w}h"] = "—"
                    else:
                        row[f"{w}h"] = f"{r:.3f}{sig_stars(p)}"
                rows.append(row)
            html += f"<h4>{period}</h4>\n"
            html += html_table(pd.DataFrame(rows))
    return html


def run() -> str:
    df = load_data()
    corr_dfs = compute_correlations(df)
    return f"""
<h2>1. Spearman Rank Correlation</h2>
<p class="note">
  Spearman ρ between net sentiment per agent and Bitcoin log return per window, split by period.
  Stars: * p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01 (uncorrected).
</p>
{img_tag(plot_heatmaps(corr_dfs))}
{build_tables(corr_dfs)}
"""


if __name__ == "__main__":
    from pathlib import Path
    html = f"<!DOCTYPE html><html><head><meta charset='UTF-8'><style>{CSS}</style></head><body>{run()}</body></html>"
    out = Path(__file__).parent / "outputs" / "correlation.html"
    out.write_text(html, encoding="utf-8")
    print(f"Saved: {out}")
