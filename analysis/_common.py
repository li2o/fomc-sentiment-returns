"""Shared constants and helpers for all analysis scripts."""
from __future__ import annotations

import io, base64
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

WORKSPACE = Path(__file__).resolve().parent.parent
DOC_CSV   = WORKSPACE / "llm_analysis/outputs/document_level/fomc_document_level.csv"
OUT_DIR   = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

AGENTS = ["households", "firms", "financial sector", "government", "central bank"]
AGENT_COLORS = {
    "central bank":    "#4E79A7",
    "households":      "#F28E2B",
    "firms":           "#59A14F",
    "financial sector":"#E15759",
    "government":      "#B07AA1",
}
WINDOWS    = [1, 3, 9, 24, 72, 144, 216, 288, 360]
DOC_TYPES  = ["Minutes", "Policy Statement"]
TYPE_COLORS = {"Minutes": "#4E79A7", "Policy Statement": "#F28E2B"}
PERIODS    = ["2012–2019", "2020–Jul2023", "Aug2023–2025"]
PERIOD_COLORS = {
    "2012–2019":     "#76b7b2",
    "2020–Jul2023":  "#e15759",
    "Aug2023–2025":  "#f1a340",
}

CSS = """
body  { font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto;
        padding: 0 24px; color: #222; background: #fafafa; }
h1    { font-size: 1.7rem; border-bottom: 2px solid #ccc; padding-bottom: 8px; }
h2    { font-size: 1.25rem; margin-top: 40px; color: #333;
        border-left: 4px solid #4E79A7; padding-left: 10px; }
h3    { font-size: 1rem; color: #555; margin-top: 24px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0;
        font-size: 0.85rem; }
th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }
th    { background: #f0f0f0; }
tr:nth-child(even) { background: #f9f9f9; }
.sig  { font-weight: bold; color: #c0392b; }
.note { font-size: 0.8rem; color: #666; margin: 4px 0 12px 0; }
img   { max-width: 100%; display: block; margin: 12px 0; }
"""


def _assign_period(date: pd.Timestamp) -> str:
    if pd.isna(date):
        return "unknown"
    if date < pd.Timestamp("2020-01-01"):
        return "2012–2019"
    if date <= pd.Timestamp("2023-07-31"):
        return "2020–Jul2023"
    return "Aug2023–2025"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DOC_CSV)
    df["document_date"] = pd.to_datetime(df["document_date"], format="%m/%d/%Y", errors="coerce")
    df = df[df["document_type"].isin(DOC_TYPES)].copy()
    df["period"] = df["document_date"].apply(_assign_period)
    df = df[df["period"] != "unknown"].copy()
    return df


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def img_tag(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}">'


def sig_stars(p: float) -> str:
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


def html_table(df: pd.DataFrame, sig_col: str | None = None) -> str:
    rows = ["<table><thead><tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr></thead><tbody>"]
    for _, row in df.iterrows():
        cells = []
        for col, val in row.items():
            cls = ' class="sig"' if (sig_col and col == sig_col and str(val).startswith("*")) else ""
            cells.append(f"<td{cls}>{val}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)
