"""
Summary statistics for the FOMC document-level sentiment data.
Outputs an HTML report to analysis/summary.html.

Usage:
    python analysis/summary.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import io, base64

WORKSPACE = Path(__file__).resolve().parent.parent
DOC_CSV   = WORKSPACE / "llm_analysis/outputs/document_level/fomc_document_level.csv"
OUT_HTML  = Path(__file__).resolve().parent / "summary.html"

AGENTS = ["households", "firms", "financial sector", "government", "central bank"]
AGENT_COLORS = {
    "central bank":    "#4E79A7",
    "households":      "#F28E2B",
    "firms":           "#59A14F",
    "financial sector":"#E15759",
    "government":      "#B07AA1",
}
DOC_TYPES   = ["Minutes", "Policy Statement"]
TYPE_COLORS = {"Minutes": "#4E79A7", "Policy Statement": "#F28E2B"}

# ── helpers ───────────────────────────────────────────────────────────────────

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def img_tag(b64: str, width: str = "100%") -> str:
    return f'<img src="data:image/png;base64,{b64}" style="width:{width};max-width:900px;">'


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_docs_per_year(df: pd.DataFrame) -> str:
    yearly = df.groupby(["year", "document_type"]).size().unstack(fill_value=0)
    yearly = yearly[[c for c in DOC_TYPES if c in yearly.columns]]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    yearly.plot(kind="bar", ax=ax, color=[TYPE_COLORS[c] for c in yearly.columns], width=0.7)
    ax.set_xlabel("Year"); ax.set_ylabel("Documents")
    ax.set_title("Documents per year by type")
    ax.legend(title="Type"); ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_sentence_counts(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    for doc_type in DOC_TYPES:
        sub = df[df["document_type"] == doc_type]["sentence_count"]
        ax.hist(sub, bins=30, alpha=0.65, label=doc_type, color=TYPE_COLORS[doc_type])
    ax.set_xlabel("Sentence count"); ax.set_ylabel("Frequency")
    ax.set_title("Distribution of sentence counts per document")
    ax.legend()
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_agent_share_avg(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, doc_type in zip(axes, DOC_TYPES):
        sub = df[df["document_type"] == doc_type]
        means = [sub[f"agent_share_{a}"].mean() for a in AGENTS]
        bars = ax.barh(
            [a.title() for a in AGENTS], means,
            color=[AGENT_COLORS[a] for a in AGENTS]
        )
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.set_title(doc_type); ax.set_xlabel("Avg. share of sentences")
        ax.set_xlim(0, max(means) * 1.25)
    fig.suptitle("Average agent share per document", fontsize=12)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_net_sentiment_over_time(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(len(AGENTS), 1, figsize=(11, 3 * len(AGENTS)), sharex=True)
    for ax, agent in zip(axes, AGENTS):
        for doc_type in DOC_TYPES:
            sub = df[df["document_type"] == doc_type].sort_values("meeting_date")
            ax.plot(sub["meeting_date"], sub[f"net_sentiment_{agent}"],
                    label=doc_type, color=TYPE_COLORS[doc_type], alpha=0.7, linewidth=0.9)
            # rolling mean
            rolling = sub[f"net_sentiment_{agent}"].rolling(6, min_periods=3).mean()
            ax.plot(sub["meeting_date"], rolling,
                    color=TYPE_COLORS[doc_type], linewidth=2)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_ylabel("Net sentiment")
        ax.set_title(f"{agent.title()}", fontsize=10)
        ax.set_ylim(-1.1, 1.1)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Net sentiment over time by agent (thin=raw, thick=6-doc rolling mean)", fontsize=11)
    # show only every 4th x-tick
    n_ticks = len(axes[-1].get_xticks())
    axes[-1].set_xticks(axes[-1].get_xticks()[::max(1, n_ticks // 12)])
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_net_sentiment_boxplot(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, doc_type in zip(axes, DOC_TYPES):
        sub = df[df["document_type"] == doc_type]
        data  = [sub[f"net_sentiment_{a}"].dropna().values for a in AGENTS]
        bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
        for patch, agent in zip(bp["boxes"], AGENTS):
            patch.set_facecolor(AGENT_COLORS[agent])
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(AGENTS) + 1))
        ax.set_xticklabels([a.title() for a in AGENTS], rotation=20, ha="right")
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_title(doc_type); ax.set_ylabel("Net sentiment")
    fig.suptitle("Net sentiment distribution per agent", fontsize=12)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_dominant_agent_pie(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, doc_type in zip(axes, DOC_TYPES):
        sub = df[df["document_type"] == doc_type]["dominant_agent"].value_counts()
        colors = [AGENT_COLORS.get(a, "#999") for a in sub.index]
        ax.pie(sub.values, labels=[a.title() for a in sub.index],
               colors=colors, autopct="%1.1f%%", startangle=140)
        ax.set_title(doc_type)
    fig.suptitle("Dominant agent per document", fontsize=12)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── stats tables ──────────────────────────────────────────────────────────────

def stats_table(df: pd.DataFrame, doc_type: str) -> str:
    sub = df[df["document_type"] == doc_type]
    rows = []
    for agent in AGENTS:
        col = f"net_sentiment_{agent}"
        s = sub[col].dropna()
        rows.append({
            "Agent": agent.title(),
            "N": len(s),
            "Mean": f"{s.mean():.3f}",
            "Median": f"{s.median():.3f}",
            "Std": f"{s.std():.3f}",
            "Min": f"{s.min():.3f}",
            "Max": f"{s.max():.3f}",
            "% Net Positive": f"{(s > 0).mean():.1%}",
        })
    tdf = pd.DataFrame(rows)
    return tdf.to_html(index=False, border=0, classes="stats-table")


def overview_table(df: pd.DataFrame) -> str:
    rows = []
    for doc_type in df["document_type"].unique():
        sub = df[df["document_type"] == doc_type]
        rows.append({
            "Document type": doc_type,
            "N documents": len(sub),
            "Date range": f"{sub['meeting_date'].min()} – {sub['meeting_date'].max()}",
            "Avg sentences": f"{sub['sentence_count'].mean():.0f}",
            "Median sentences": f"{sub['sentence_count'].median():.0f}",
            "Total sentences": f"{sub['sentence_count'].sum():,}",
        })
    return pd.DataFrame(rows).to_html(index=False, border=0, classes="stats-table")


# ── HTML assembly ─────────────────────────────────────────────────────────────

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

    sections.append(f"""
<h1>FOMC Sentiment Analysis — Document-Level Summary</h1>
<p class="note">Data: CentralBankRoBERTa agent &amp; sentiment classifier applied to FOMC minutes and policy statements (2012–2025).<br>
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d')}</p>

<h2>1. Corpus Overview</h2>
{overview_table(df)}
{img_tag(plot_docs_per_year(df))}
{img_tag(plot_sentence_counts(df))}
""")

    sections.append(f"""
<h2>2. Agent Classification</h2>
<p class="note">Share of sentences attributed to each economic agent, averaged across documents.</p>
{img_tag(plot_agent_share_avg(df))}
{img_tag(plot_dominant_agent_pie(df))}
""")

    sections.append(f"""
<h2>3. Net Sentiment by Agent</h2>
<p class="note">Net sentiment = share of positive sentences − share of negative sentences, computed per agent per document.</p>
{img_tag(plot_net_sentiment_boxplot(df))}

<h3>Minutes</h3>
{stats_table(df, "Minutes")}

<h3>Policy Statements</h3>
{stats_table(df, "Policy Statement")}
""")

    sections.append(f"""
<h2>4. Net Sentiment Over Time</h2>
<p class="note">Thin line = per-document value. Thick line = 6-document rolling mean.</p>
{img_tag(plot_net_sentiment_over_time(df))}
""")

    body = "\n".join(sections)
    return f"<!DOCTYPE html><html><head><meta charset='UTF-8'><title>FOMC Sentiment Summary</title><style>{CSS}</style></head><body>{body}</body></html>"


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df = pd.read_csv(DOC_CSV)
    df["meeting_date"] = pd.to_datetime(df["meeting_date"], format="%m/%d/%Y", errors="coerce")
    df["year"] = df["meeting_date"].dt.year
    df = df[df["document_type"].isin(DOC_TYPES)]

    html = build_html(df)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Saved: {OUT_HTML}")


if __name__ == "__main__":
    main()
