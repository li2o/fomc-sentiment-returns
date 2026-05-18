"""
Visualise the cross-agent correlation of net sentiment scores within FOMC Minutes.

Plots produced:
  1. Pearson correlation heatmap (full sample, pre-2020, post-2020)
  2. Scatter-plot matrix (pairplot) – full sample
  3. Rolling 12-meeting pairwise correlation over time for selected pairs
  4. Time series of all agents' net sentiment on one chart

Output: analysis/outputs/agent_sentiment_correlation.html
"""
from __future__ import annotations

import io, base64
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

# ── shared constants ──────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent.parent
DOC_CSV   = WORKSPACE / "llm_analysis/outputs/document_level/fomc_document_level.csv"
OUT       = Path(__file__).resolve().parent / "outputs" / "agent_sentiment_correlation.html"

AGENTS = ["households", "firms", "financial sector", "government", "central bank"]
AGENT_COLORS = {
    "central bank":    "#4E79A7",
    "households":      "#F28E2B",
    "firms":           "#59A14F",
    "financial sector":"#E15759",
    "government":      "#B07AA1",
}
PERIOD_CUT = 2020

CSS = """
body  { font-family: Arial, sans-serif; max-width: 1050px; margin: 40px auto;
        padding: 0 24px; color: #222; background: #fafafa; }
h1    { font-size: 1.6rem; border-bottom: 2px solid #ccc; padding-bottom: 8px; }
h2    { font-size: 1.2rem; margin-top: 36px; color: #333;
        border-left: 4px solid #4E79A7; padding-left: 10px; }
h3    { font-size: 1rem; color: #555; margin-top: 20px; }
img   { max-width: 100%; display: block; margin: 14px 0; }
.note { font-size: 0.8rem; color: #666; margin: 4px 0 14px 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.85rem; margin: 10px 0; }
th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: center; }
th    { background: #f0f0f0; }
"""


# ── helpers ───────────────────────────────────────────────────────────────────

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def img_tag(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}">'


def load_minutes() -> pd.DataFrame:
    df = pd.read_csv(DOC_CSV)
    df["document_date"] = pd.to_datetime(df["document_date"], format="%m/%d/%Y", errors="coerce")
    df = df[df["document_type"] == "Minutes"].copy()
    df["period"] = df["document_date"].dt.year.apply(
        lambda y: "2020–2025" if y >= PERIOD_CUT else "2012–2019"
    )
    df = df.sort_values("document_date").reset_index(drop=True)
    return df


def sentiment_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = {a: df[f"net_sentiment_{a}"] for a in AGENTS}
    return pd.DataFrame(cols)


# ── plot 1: correlation heatmaps (full / pre / post) ──────────────────────────

def plot_corr_heatmaps(df: pd.DataFrame) -> str:
    subsets = {
        "Full sample": df,
        "2012–2019":   df[df["period"] == "2012–2019"],
        "2020–2025":   df[df["period"] == "2020–2025"],
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = [a.title() for a in AGENTS]

    for ax, (title, sub) in zip(axes, subsets.items()):
        mat = sentiment_matrix(sub).dropna()
        corr = mat.corr(method="pearson")
        # compute p-values
        pvals = pd.DataFrame(np.ones_like(corr), index=corr.index, columns=corr.columns)
        for i, a1 in enumerate(AGENTS):
            for j, a2 in enumerate(AGENTS):
                if i != j:
                    _, p = stats.pearsonr(mat[a1], mat[a2])
                    pvals.iloc[i, j] = p

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # upper triangle (keep lower)
        sns.heatmap(
            corr, ax=ax, annot=True, fmt=".2f",
            center=0, vmin=-1, vmax=1,
            cmap="coolwarm", linewidths=0.5, linecolor="#ddd",
            xticklabels=labels, yticklabels=labels,
            cbar_kws={"shrink": 0.8},
            mask=np.triu(np.ones_like(corr, dtype=bool), k=1),
        )
        # stars on lower triangle
        for i in range(len(AGENTS)):
            for j in range(i):
                p = pvals.iloc[i, j]
                stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
                if stars:
                    ax.text(j + 0.75, i + 0.25, stars, ha="center", va="center",
                            fontsize=8, color="white", fontweight="bold")
        ax.set_title(f"{title}  (n={len(mat)})", fontsize=11)
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle("Pearson correlation between agent net sentiments (FOMC Minutes)", fontsize=12)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── plot 2: scatter matrix ─────────────────────────────────────────────────────

def plot_scatter_matrix(df: pd.DataFrame) -> str:
    mat = sentiment_matrix(df).dropna()
    mat.columns = [a.title() for a in AGENTS]

    g = sns.PairGrid(mat, height=2.0, aspect=1.0)
    g.map_lower(sns.regplot, scatter_kws={"alpha": 0.35, "s": 15}, line_kws={"color": "red", "lw": 1})
    g.map_diag(sns.histplot, bins=18, color="#4E79A7", alpha=0.7)
    g.map_upper(sns.kdeplot, fill=True, cmap="Blues", thresh=0.1, levels=5)
    g.figure.suptitle("Scatter matrix of agent net sentiments — FOMC Minutes", y=1.01, fontsize=11)
    g.figure.tight_layout()
    return fig_to_b64(g.figure)


# ── plot 3: rolling pairwise correlation ──────────────────────────────────────

def plot_rolling_correlation(df: pd.DataFrame, window: int = 12) -> str:
    mat = sentiment_matrix(df).copy()
    mat.index = df["document_date"].values

    # pick all unique pairs
    pairs = [(AGENTS[i], AGENTS[j]) for i in range(len(AGENTS)) for j in range(i + 1, len(AGENTS))]

    fig, axes = plt.subplots(5, 2, figsize=(13, 14), sharex=True)
    axes_flat = axes.flatten()

    for ax_idx, (a1, a2) in enumerate(pairs):
        ax = axes_flat[ax_idx]
        rolling = mat[a1].rolling(window).corr(mat[a2])
        dates = pd.to_datetime(mat.index)
        ax.plot(dates, rolling, color=AGENT_COLORS[a1], linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.fill_between(dates, rolling, 0,
                        where=rolling >= 0, alpha=0.15, color="green")
        ax.fill_between(dates, rolling, 0,
                        where=rolling < 0,  alpha=0.15, color="red")
        ax.set_ylim(-1.1, 1.1)
        ax.set_ylabel("ρ")
        ax.set_title(f"{a1.title()} × {a2.title()}", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    # hide unused axes if any
    for idx in range(len(pairs), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(f"Rolling {window}-meeting Pearson correlation between agent sentiments", fontsize=11)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── plot 4: time series of all agents ─────────────────────────────────────────

def plot_time_series(df: pd.DataFrame) -> str:
    dates = pd.to_datetime(df["document_date"].values)
    fig, ax = plt.subplots(figsize=(12, 5))

    for agent in AGENTS:
        vals = df[f"net_sentiment_{agent}"].values
        # raw (thin) + rolling mean (thick)
        roll = pd.Series(vals).rolling(6, min_periods=3).mean()
        ax.plot(dates, vals, color=AGENT_COLORS[agent], alpha=0.25, linewidth=0.8)
        ax.plot(dates, roll, color=AGENT_COLORS[agent], linewidth=2.0, label=agent.title())

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_ylabel("Net sentiment")
    ax.set_xlabel("")
    ax.set_title("Net sentiment over time by agent — FOMC Minutes\n"
                 "(thin=raw, thick=6-meeting rolling mean)", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return fig_to_b64(fig)


# ── correlation table ──────────────────────────────────────────────────────────

def corr_table_html(df: pd.DataFrame, title: str) -> str:
    mat = sentiment_matrix(df).dropna()
    corr = mat.corr(method="pearson")
    labels = [a.title() for a in AGENTS]
    rows = ["<table><thead><tr><th></th>" + "".join(f"<th>{l}</th>" for l in labels) + "</tr></thead><tbody>"]
    for i, a1 in enumerate(AGENTS):
        cells = [f"<td><strong>{a1.title()}</strong></td>"]
        for j, a2 in enumerate(AGENTS):
            if i == j:
                cells.append("<td>—</td>")
            else:
                r = corr.iloc[i, j]
                _, p = stats.pearsonr(mat[a1], mat[a2])
                stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
                cells.append(f"<td>{r:.3f}{stars}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    rows.append("</tbody></table>")
    return f"<h3>{title} (n={len(mat)})</h3>\n" + "\n".join(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df = load_minutes()
    date_str = pd.Timestamp.now().strftime("%Y-%m-%d")

    sections = []

    sections.append(f"""
<h1>Cross-Agent Sentiment Correlation — FOMC Minutes</h1>
<p class="note">
  Net sentiment per agent = (share positive sentences − share negative sentences)
  for sentences attributed to that agent within each FOMC Minutes document.<br>
  Sample: {len(df)} Minutes documents, {df["document_date"].min().year}–{df["document_date"].max().year}.
  Generated: {date_str}
</p>
""")

    sections.append(f"""
<h2>1. Correlation Heatmaps</h2>
<p class="note">Pearson correlation, lower triangle shown. Stars: * p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01.</p>
{img_tag(plot_corr_heatmaps(df))}
{corr_table_html(df, "Full sample")}
{corr_table_html(df[df["period"] == "2012–2019"], "2012–2019")}
{corr_table_html(df[df["period"] == "2020–2025"], "2020–2025")}
""")

    sections.append(f"""
<h2>2. Scatter Matrix</h2>
<p class="note">
  Lower triangle: scatter + OLS trend line. Diagonal: distribution.
  Upper triangle: kernel density estimate.
</p>
{img_tag(plot_scatter_matrix(df))}
""")

    sections.append(f"""
<h2>3. Rolling 12-Meeting Correlation</h2>
<p class="note">
  Pearson correlation computed over a rolling window of 12 consecutive minutes documents.
  Green shading = positive correlation; red = negative.
</p>
{img_tag(plot_rolling_correlation(df, window=12))}
""")

    sections.append(f"""
<h2>4. Net Sentiment Over Time</h2>
<p class="note">Thin line = per-document value. Thick line = 6-meeting rolling mean.</p>
{img_tag(plot_time_series(df))}
""")

    html = f"<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'><title>Agent Sentiment Correlation</title><style>{CSS}</style></head><body>{''.join(sections)}</body></html>"
    OUT.write_text(html, encoding="utf-8")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
