"""
Sign / directional analysis:
  Does positive net sentiment predict positive Bitcoin returns more often than chance?
  - Binomial test per agent × window × document type × period
  - Visualisation: hit-rate heatmap per period
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


def run_sign_tests(df: pd.DataFrame) -> dict:
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
                xm, ym = x[mask], y[mask]
                nonzero = (xm != 0) & (ym != 0)
                n = nonzero.sum()
                if n < 8:
                    row[f"{w}h_rate"] = np.nan
                    row[f"{w}h_p"]    = np.nan
                    continue
                agree = ((xm[nonzero] > 0) == (ym[nonzero] > 0)).sum()
                hit_rate = agree / n
                binom = stats.binomtest(int(agree), int(n), p=0.5, alternative="two-sided")
                row[f"{w}h_rate"] = round(float(hit_rate), 3)
                row[f"{w}h_p"]    = round(float(binom.pvalue), 3)
            rows.append(row)
        results[(doc_type, period)] = pd.DataFrame(rows).set_index("Agent")
    return results


def plot_heatmaps(sign_dfs: dict) -> str:
    fig, axes = plt.subplots(2, len(PERIODS), figsize=(13, 8))
    for row_idx, doc_type in enumerate(DOC_TYPES):
        for col_idx, period in enumerate(PERIODS):
            ax = axes[row_idx][col_idx]
            df = sign_dfs[(doc_type, period)]
            rate_cols = [c for c in df.columns if c.endswith("_rate")]
            p_cols    = [c for c in df.columns if c.endswith("_p")]
            r_df = df[rate_cols].copy().astype(float)
            p_df = df[p_cols].copy().astype(float)
            r_df.columns = [f"{w}h" for w in WINDOWS]
            p_df.columns = [f"{w}h" for w in WINDOWS]

            sns.heatmap(
                r_df, ax=ax, annot=True, fmt=".2f",
                center=0.5, vmin=0.2, vmax=0.8,
                cmap="RdYlGn", linewidths=0.5, linecolor="#ddd",
                cbar_kws={"shrink": 0.8, "label": "Hit rate"},
            )
            for i, agent in enumerate(r_df.index):
                for j, w_col in enumerate(r_df.columns):
                    p = p_df.loc[agent, w_col]
                    stars = sig_stars(p) if not np.isnan(p) else ""
                    if stars:
                        ax.text(j + 0.75, i + 0.25, stars, ha="center", va="center",
                                fontsize=7, color="black", fontweight="bold")
            ax.set_title(f"{doc_type} — {period}", fontsize=10)
            ax.set_xlabel("Return window")
    fig.suptitle("Directional hit rate by period (H₀: rate = 0.5)", fontsize=12)
    fig.tight_layout()
    return fig_to_b64(fig)


def build_tables(sign_dfs: dict) -> str:
    html = ""
    for doc_type in DOC_TYPES:
        html += f"<h3>{doc_type}</h3>\n"
        for period in PERIODS:
            df = sign_dfs[(doc_type, period)]
            rows = []
            for agent in df.index:
                row = {"Agent": agent}
                for w in WINDOWS:
                    rate = df.loc[agent, f"{w}h_rate"]
                    p    = df.loc[agent, f"{w}h_p"]
                    row[f"{w}h"] = "—" if np.isnan(rate) else f"{rate:.2f}{sig_stars(p)}"
                rows.append(row)
            html += f"<h4>{period}</h4>\n"
            html += html_table(pd.DataFrame(rows))
    return html


def run() -> str:
    df = load_data()
    sign_dfs = run_sign_tests(df)
    return f"""
<h2>4. Directional Sign Test</h2>
<p class="note">
  Hit rate = proportion of events where sign(net sentiment) matches sign(Bitcoin return).
  Binomial test against H₀: rate = 0.5. Split by period.
  Stars: * p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01.
</p>
{img_tag(plot_heatmaps(sign_dfs))}
{build_tables(sign_dfs)}
"""


if __name__ == "__main__":
    from pathlib import Path
    html = f"<!DOCTYPE html><html><head><meta charset='UTF-8'><style>{CSS}</style></head><body>{run()}</body></html>"
    out = Path(__file__).parent / "outputs" / "sign_test.html"
    out.write_text(html, encoding="utf-8")
    print(f"Saved: {out}")
