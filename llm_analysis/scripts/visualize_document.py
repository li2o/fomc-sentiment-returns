"""
Generate an HTML report for a single FOMC document showing each sentence
colour-coded by agent classification and annotated with sentiment.

Usage:
    python llm_analysis/scripts/visualize_document.py --document-id fomc_00134
    python llm_analysis/scripts/visualize_document.py --document-id fomc_00134 --output my_report.html
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# ── colour palette ────────────────────────────────────────────────────────────
AGENT_COLORS: dict[str, str] = {
    "households":       "#003D5C",
    "firms":            "#2CA58D",
    "financial sector": "#BC4C96",
    "government":       "#FF5F66",
    "central bank":     "#FFA600",
}
SENTIMENT_ICONS = {"positive": "▲", "negative": "▼"}
SENTIMENT_COLORS = {"positive": "#2ca02c", "negative": "#d62728"}

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{ font-family: Georgia, serif; max-width: 960px; margin: 40px auto;
          padding: 0 24px; color: #222; background: #fafafa; }}
  h1   {{ font-size: 1.6rem; border-bottom: 2px solid #ccc; padding-bottom: 8px; }}
  .meta {{ font-size: 0.85rem; color: #555; margin-bottom: 24px; }}
  .intro {{ background: #fff; border-left: 4px solid #aaa; padding: 14px 18px;
            margin-bottom: 28px; border-radius: 4px; font-size: 0.95rem; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 24px; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px;
                  font-size: 0.85rem; }}
  .swatch {{ width: 14px; height: 14px; border-radius: 3px; flex-shrink: 0; }}
  .sentences {{ line-height: 2.0; font-size: 0.97rem; }}
  .sent {{ display: inline; border-radius: 3px; padding: 1px 0px; }}
  .sent:hover .tooltip {{ display: block; }}
  .tooltip-wrap {{ position: relative; display: inline; }}
  .tooltip {{ display: none; position: absolute; bottom: 130%; left: 0;
              background: #333; color: #fff; padding: 5px 9px;
              border-radius: 4px; font-size: 0.78rem; white-space: nowrap;
              z-index: 10; pointer-events: none; }}
  .sent-text {{ border-bottom: 2px solid; padding-bottom: 1px; margin-right: 4px; }}
  .sentiment-tag {{ font-size: 0.72rem; font-weight: bold; vertical-align: super; }}
  .stats {{ margin-top: 32px; font-size: 0.88rem; color: #444; }}
  .stats table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
  .stats th, .stats td {{ border: 1px solid #ddd; padding: 6px 12px; text-align: left; }}
  .stats th {{ background: #f0f0f0; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">
  <strong>Document type:</strong> {doc_type} &nbsp;|&nbsp;
  <strong>Meeting date:</strong> {meeting_date} &nbsp;|&nbsp;
  <strong>Release date:</strong> {release_date} &nbsp;|&nbsp;
  <strong>Document ID:</strong> {document_id}<br>
  <strong>Source:</strong> <a href="{source_url}" target="_blank">{source_url}</a>
</div>

<div class="intro">
  <strong>About this visualisation:</strong>
  Each sentence is classified by the <em>CentralBankRoBERTa</em> model along two dimensions:
  (1) the <strong>economic agent</strong> the sentence primarily refers to
  (colour-coded, see legend), and
  (2) the <strong>sentiment</strong> expressed about that agent
  (<span style="color:{pos_color}">▲ positive</span> /
   <span style="color:{neg_color}">▼ negative</span>).
  Hover over any sentence for exact model probabilities.
  The document contains <strong>{n_sentences} sentences</strong>.
</div>

<div class="legend">
{legend_items}
  <div class="legend-item">
    <span style="color:{pos_color}; font-weight:bold;">▲</span> Positive sentiment
  </div>
  <div class="legend-item">
    <span style="color:{neg_color}; font-weight:bold;">▼</span> Negative sentiment
  </div>
</div>

<div class="sentences">
{sentence_html}
</div>

<div class="stats">
  <strong>Document-level summary</strong>
  <table>
    <tr><th>Agent</th><th>Sentence count</th><th>Share</th><th>Avg. confidence</th></tr>
{stats_rows}
  </table>
</div>

</body>
</html>
"""


def build_tooltip(row: pd.Series) -> str:
    agent_probs = " | ".join(
        f"{label}: {row[f'agent_prob_{label}']:.2f}"
        for label in ["central bank", "households", "firms", "financial sector", "government"]
    )
    sentiment_probs = (
        f"pos: {row['sentiment_prob_positive']:.2f} | neg: {row['sentiment_prob_negative']:.2f}"
    )
    return f"Agent — {agent_probs} || Sentiment — {sentiment_probs}"


def build_sentence_html(df: pd.DataFrame) -> str:
    parts: list[str] = []
    for _, row in df.iterrows():
        agent = row["agent_label"]
        sentiment = row["sentiment_label"]
        color = AGENT_COLORS.get(agent, "#999")
        sent_icon = SENTIMENT_ICONS.get(sentiment, "")
        sent_color = SENTIMENT_COLORS.get(sentiment, "#999")
        tooltip = build_tooltip(row)
        text = str(row["sentence_text"]).replace("<", "&lt;").replace(">", "&gt;")

        parts.append(
            f'<span class="tooltip-wrap sent" style="color:{color};">'
            f'<span class="sent-text" style="border-color:{color};">{text}</span>'
            f'<span class="sentiment-tag" style="color:{sent_color};">{sent_icon}</span>'
            f'<span class="tooltip">{tooltip}</span>'
            f"</span> "
        )
    return "\n".join(parts)


def build_legend_items() -> str:
    items = []
    for agent, color in AGENT_COLORS.items():
        items.append(
            f'  <div class="legend-item">'
            f'<div class="swatch" style="background:{color};"></div>{agent.title()}'
            f"</div>"
        )
    return "\n".join(items)


def build_stats_rows(df: pd.DataFrame) -> str:
    rows = []
    total = len(df)
    for agent, color in AGENT_COLORS.items():
        subset = df[df["agent_label"] == agent]
        count = len(subset)
        share = count / total if total else 0
        avg_conf = subset[f"agent_prob_{agent}"].mean() if count else 0.0
        rows.append(
            f'    <tr><td><span style="color:{color};">■</span> {agent.title()}</td>'
            f"<td>{count}</td><td>{share:.1%}</td><td>{avg_conf:.3f}</td></tr>"
        )
    return "\n".join(rows)


def generate_report(
    sentence_csv: Path,
    document_id: str,
    output_path: Path,
) -> None:
    df = pd.read_csv(sentence_csv)
    doc_df = df[df["document_id"] == document_id].copy()

    if doc_df.empty:
        raise SystemExit(f"Document '{document_id}' not found in {sentence_csv}")

    first = doc_df.iloc[0]
    title = str(first.get("meta_title", "FOMC Document"))
    meeting_date = str(first.get("meta_meeting_date", ""))
    release_date = str(first.get("meta_document_date", ""))
    source_url = str(first.get("meta_source_url", ""))
    doc_type = str(first.get("meta_document_type", ""))

    html = HTML_TEMPLATE.format(
        title=f"{doc_type} — {meeting_date}",
        doc_type=doc_type,
        meeting_date=meeting_date,
        release_date=release_date,
        document_id=document_id,
        source_url=source_url,
        n_sentences=len(doc_df),
        pos_color=SENTIMENT_COLORS["positive"],
        neg_color=SENTIMENT_COLORS["negative"],
        legend_items=build_legend_items(),
        sentence_html=build_sentence_html(doc_df),
        stats_rows=build_stats_rows(doc_df),
    )

    output_path.write_text(html, encoding="utf-8")
    print(f"Report saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Visualise a single FOMC document.")
    parser.add_argument("--document-id", default="fomc_00134",
                        help="Document ID from the sentence-level CSV (default: fomc_00134)")
    parser.add_argument("--sentence-csv", type=Path,
                        default=workspace / "llm_analysis/outputs/sentence_level/fomc_sentence_level.csv")
    parser.add_argument("--output", type=Path,
                        default=None,
                        help="Output HTML file path (default: llm_analysis/outputs/<document_id>.html)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(__file__).resolve().parents[2]
    output = args.output or workspace / f"llm_analysis/outputs/{args.document_id}_visualisation.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_report(args.sentence_csv, args.document_id, output)


if __name__ == "__main__":
    main()
