"""
Summary statistics for press-conference document-level sentiment data.
Outputs an HTML report to analysis/outputs/press_conferences_summary.html.

Usage:
    python analysis/summary_press_conferences.py
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

WORKSPACE = Path(__file__).resolve().parent.parent
DOC_CSV = WORKSPACE / "llm_analysis/outputs/document_level/press_conferences_document_level.csv"
OUT_HTML = Path(__file__).resolve().parent / "outputs/press_conferences_summary.html"

AGENTS = ["households", "firms", "financial sector", "government", "central bank"]
AGENT_COLORS = {
    "central bank": "#4E79A7",
    "households": "#F28E2B",
    "firms": "#59A14F",
    "financial sector": "#E15759",
    "government": "#B07AA1",
}


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def img_tag(b64: str, width: str = "100%") -> str:
    return f'<img src="data:image/png;base64,{b64}" style="width:{width};max-width:900px;">'


def plot_docs_per_year(df: pd.DataFrame) -> str:
    yearly = df.groupby("year").size()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(yearly.index.astype(str).tolist(), yearly.astype(float).tolist(), color="#4E79A7", width=0.7)
    ax.set_xlabel("Year")
    ax.set_ylabel("Documents")
    ax.set_title("Press conferences per year")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_sentence_counts(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["sentence_count"], bins=30, alpha=0.75, color="#4E79A7")
    ax.set_xlabel("Sentence count")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of sentence counts per press conference")
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_agent_share_avg(df: pd.DataFrame) -> str:
    means = [df[f"agent_share_{a}"].mean() for a in AGENTS]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(
        [a.title() for a in AGENTS],
        means,
        color=[AGENT_COLORS[a] for a in AGENTS],
    )
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set_title("Average agent share per press conference")
    ax.set_xlabel("Avg. share of sentences")
    ax.set_xlim(0, max(means) * 1.25)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_net_sentiment_over_time(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(len(AGENTS), 1, figsize=(11, 3 * len(AGENTS)), sharex=True)
    sub = df.sort_values("meeting_date")
    for ax, agent in zip(axes, AGENTS):
        ax.plot(
            sub["meeting_date"],
            sub[f"net_sentiment_{agent}"],
            color=AGENT_COLORS[agent],
            alpha=0.75,
            linewidth=1,
            label="Net sentiment",
        )
        rolling = sub[f"net_sentiment_{agent}"].rolling(6, min_periods=3).mean()
        ax.plot(
            sub["meeting_date"],
            rolling,
            color="black",
            linewidth=1.8,
            label="6-doc rolling mean",
        )
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_ylabel("Net")
        ax.set_title(agent.title(), fontsize=10)
        ax.set_ylim(-1.1, 1.1)
    axes[0].legend(loc="upper right", fontsize=8)
    plt.xticks(rotation=45, ha="right")
    fig.suptitle("Net sentiment over time by agent", fontsize=11)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_net_sentiment_boxplot(df: pd.DataFrame) -> str:
    data = [df[f"net_sentiment_{a}"].dropna().astype(float).tolist() for a in AGENTS]
    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
    for patch, agent in zip(bp["boxes"], AGENTS):
        patch.set_facecolor(AGENT_COLORS[agent])
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(AGENTS) + 1))
    ax.set_xticklabels([a.title() for a in AGENTS], rotation=20, ha="right")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_ylabel("Net sentiment")
    ax.set_title("Net sentiment distribution per agent")
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_dominant_agent_share(df: pd.DataFrame) -> str:
    dominant = df["dominant_agent"].value_counts()
    colors = [AGENT_COLORS.get(a, "#999") for a in dominant.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(
        dominant.astype(float).tolist(),
        labels=[a.title() for a in dominant.index],
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
    )
    ax.set_title("Dominant agent per press conference")
    fig.tight_layout()
    return fig_to_b64(fig)


def stats_table(df: pd.DataFrame) -> str:
    rows = []
    for agent in AGENTS:
        col = f"net_sentiment_{agent}"
        s = df[col].dropna()
        rows.append(
            {
                "Agent": agent.title(),
                "N": len(s),
                "Mean": f"{s.mean():.3f}",
                "Median": f"{s.median():.3f}",
                "Std": f"{s.std():.3f}",
                "Min": f"{s.min():.3f}",
                "Max": f"{s.max():.3f}",
                "% Net Positive": f"{(s > 0).mean():.1%}",
            }
        )
    tdf = pd.DataFrame(rows)
    return tdf.to_html(index=False, border=0, classes="stats-table")


def overview_table(df: pd.DataFrame) -> str:
    row = {
        "Document type": "Press Conference",
        "N documents": len(df),
        "Date range": f"{df['meeting_date'].min().date()} - {df['meeting_date'].max().date()}",
        "Avg sentences": f"{df['sentence_count'].mean():.0f}",
        "Median sentences": f"{df['sentence_count'].median():.0f}",
        "Total sentences": f"{df['sentence_count'].sum():,}",
    }
    return pd.DataFrame([row]).to_html(index=False, border=0, classes="stats-table")


CSS = """
body { font-family: Arial, sans-serif; max-width: 980px; margin: 40px auto;
       padding: 0 24px; color: #222; background: #fafafa; }
h1   { font-size: 1.7rem; border-bottom: 2px solid #ccc; padding-bottom: 8px; }
h2   { font-size: 1.2rem; margin-top: 36px; color: #333; }
h3   { font-size: 1rem; color: #555; margin-top: 24px; }
.stats-table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.88rem; }
.stats-table th, .stats-table td { border: 1px solid #ddd; padding: 6px 12px; text-align: left; }
.stats-table th { background: #f0f0f0; }
.stats-table tr:nth-child(even) { background: #f9f9f9; }
.note { font-size: 0.82rem; color: #666; margin-top: 4px; }
"""


def build_html(df: pd.DataFrame) -> str:
    sections = []
    sections.append(
        f"""
<h1>FOMC Press Conference Sentiment Analysis - Document-Level Summary</h1>
<p class="note">Data: CentralBankRoBERTa agent and sentiment classifier applied to FOMC press conferences. Sample: 2015–2023.<br>
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d')}</p>

<h2>1. Corpus Overview</h2>
{overview_table(df)}
{img_tag(plot_docs_per_year(df))}
{img_tag(plot_sentence_counts(df))}
"""
    )

    sections.append(
        f"""
<h2>2. Agent Classification</h2>
<p class=\"note\">Share of sentences attributed to each economic agent, averaged across press conferences.</p>
{img_tag(plot_agent_share_avg(df))}
{img_tag(plot_dominant_agent_share(df), width='70%')}
"""
    )

    sections.append(
        f"""
<h2>3. Net Sentiment by Agent</h2>
<p class=\"note\">Net sentiment = share of positive sentences - share of negative sentences, computed per agent per document.</p>
{img_tag(plot_net_sentiment_boxplot(df))}

<h3>Press Conferences</h3>
{stats_table(df)}
"""
    )

    sections.append(
        f"""
<h2>4. Net Sentiment Over Time</h2>
<p class=\"note\">Thin line = per-document value. Thick line = 6-document rolling mean.</p>
{img_tag(plot_net_sentiment_over_time(df))}
"""
    )

    body = "\n".join(sections)
    return (
        "<!DOCTYPE html><html><head><meta charset='UTF-8'><title>Press Conference Sentiment Summary</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )


def main() -> None:
    df = pd.read_csv(DOC_CSV)
    df["meeting_date"] = pd.to_datetime(df["meeting_date"], errors="coerce")
    df = df[
        (df["meeting_date"] >= pd.Timestamp("2015-01-01")) &
        (df["meeting_date"] <= pd.Timestamp("2023-12-31"))
    ].copy()
    df["year"] = df["meeting_date"].dt.year

    html = build_html(df)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Saved: {OUT_HTML}")


if __name__ == "__main__":
    main()
