#!/usr/bin/env python3
"""
FED FOMC Materials Scraper (Minutes, Policy Statements, and SEP) via materials page.

Uses the Federal Reserve materials page with date/type filters to capture
minutes, policy statements, and Summary of Economic Projections (SEP),
including both meeting date and document release date metadata.
"""

import argparse
import hashlib
import os
import re
import time
from datetime import date
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://www.federalreserve.gov"
MATERIALS_URL = BASE + "/monetarypolicy/materials/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

SPACE_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    return SPACE_RE.sub(" ", s).strip()


def fetch_html(url: str, timeout: int = 60) -> str:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or r.encoding
    return r.text


def stable_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def extract_date_from_text(text: str) -> Optional[str]:
    pattern = (
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{1,2})(?:-\d{1,2})?,\s+(\d{4})"
    )
    match = re.search(pattern, text)
    if not match:
        return None

    month_name, day, year = match.groups()
    try:
        from datetime import datetime

        dt = datetime.strptime(f"{month_name} {day} {year}", "%B %d %Y")
        return dt.strftime("%m/%d/%Y")
    except Exception:
        return None


def extract_all_dates_from_text(text: str) -> List[str]:
    pattern = re.compile(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{1,2})(?:-\d{1,2})?,\s+(\d{4})"
    )
    out: List[str] = []
    for match in pattern.finditer(text):
        month_name, day, year = match.groups()
        try:
            from datetime import datetime

            dt = datetime.strptime(f"{month_name} {day} {year}", "%B %d %Y")
            out.append(dt.strftime("%m/%d/%Y"))
        except Exception:
            continue
    return out


def save_text(texts_dir: str, material_date: str, material_url: str, text: str) -> str:
    ensure_dir(texts_dir)
    sid = stable_id(material_url)

    safe_date = "unknown"
    if material_date:
        parts = material_date.split("/")
        if len(parts) == 3:
            mm, dd, yyyy = parts
            safe_date = f"{yyyy.zfill(4)}-{mm.zfill(2)}-{dd.zfill(2)}"

    filename = f"{safe_date}_{sid}.txt"
    path = os.path.join(texts_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    content = (
        soup.select_one("#article")
        or soup.select_one("main")
        or soup.select_one("#content")
        or soup.body
        or soup
    )

    for tag in content.find_all(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    raw_text = content.get_text("\n")
    lines = [clean_text(line) for line in raw_text.splitlines()]
    return "\n".join(line for line in lines if line)


def _is_heading_like(tag: Any, text: str) -> bool:
    if not text:
        return False
    if tag.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return True

    # Many FED pages place section titles in short bold-only paragraphs.
    if tag.name in {"p", "div"}:
        strong_only = (
            len(tag.find_all(["strong", "b", "em"])) > 0
            and len([c for c in tag.children if getattr(c, "name", None) not in {"strong", "b", "em", None}])
            == 0
        )
        if strong_only and len(text) <= 140:
            return True
        if len(text) <= 90 and text.endswith(":"):
            return True
    return False


def extract_structured_sections(html: str) -> Tuple[List[Dict[str, str]], str]:
    soup = BeautifulSoup(html, "html.parser")
    content = (
        soup.select_one("#article")
        or soup.select_one("main")
        or soup.select_one("#content")
        or soup.body
        or soup
    )

    for tag in content.find_all(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    sections: List[Dict[str, Any]] = []
    current_heading = "Document Body"
    current_lines: List[str] = []

    def flush_current() -> None:
        nonlocal current_lines
        body = "\n".join(line for line in current_lines if line).strip()
        if body:
            sections.append({"heading": current_heading, "text": body})
        current_lines = []

    for tag in content.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "div"]):
        text = clean_text(tag.get_text(" ", strip=True))
        if not text:
            continue

        if _is_heading_like(tag, text):
            flush_current()
            current_heading = text
            continue

        if tag.name == "div":
            # Avoid pulling large container-level duplicates.
            continue

        current_lines.append(text)

    flush_current()

    if not sections:
        raw = extract_main_text(html)
        sections = [{"heading": "Document Body", "text": raw}]

    structured_text = []
    for sec in sections:
        structured_text.append(f"## {sec['heading']}")
        structured_text.append(sec["text"])
        structured_text.append("")

    return sections, "\n".join(structured_text).strip()


def discover_sep_links_from_minutes(minutes_url: str, minutes_html: str) -> List[str]:
    """Extract SEP page links embedded on a Minutes page.

    Older meetings often expose SEP as `fomcminutesYYYYMMDDep.htm` from the
    minutes page rather than as a separate, consistently typed materials-row.
    """
    soup = BeautifulSoup(minutes_html, "html.parser")
    found: List[str] = []

    for link in soup.find_all("a", href=True):
        href_attr = link.get("href")
        if isinstance(href_attr, list):
            href = str(href_attr[0]).strip() if href_attr else ""
        else:
            href = str(href_attr or "").strip()
        if not href or href.lower().endswith(".pdf"):
            continue

        full_url = urljoin(minutes_url, href)
        full_url_lower = full_url.lower()
        link_text = clean_text(link.get_text(" ", strip=True)).lower()

        is_legacy_sep_page = (
            "fomcminutes" in full_url_lower and full_url_lower.endswith("ep.htm")
        )
        is_sep_labeled_link = "summary of economic projections" in link_text

        if is_legacy_sep_page or is_sep_labeled_link:
            found.append(full_url)

    # Keep insertion order while deduplicating.
    return list(dict.fromkeys(found))


def parse_materials_page(html: str, start_year: int, end_year: int) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    items: List[Dict[str, Any]] = []

    meeting_rows = soup.find_all("div", class_="fomc-meeting")
    for row in meeting_rows:
        date_strong = row.find("strong")
        if not date_strong:
            continue

        meeting_date_str = clean_text(date_strong.get_text(strip=True))
        if meeting_date_str == "Meeting Date":
            continue

        meeting_date = extract_date_from_text(meeting_date_str)
        if not meeting_date:
            continue

        year = int(meeting_date.split("/")[-1])
        if year < start_year or year > end_year:
            continue

        cells = row.find_all("div", class_="fomc-meeting__month")
        if len(cells) < 3:
            continue

        type_text = clean_text(cells[1].get_text(" ", strip=True))
        if not type_text or type_text == "Type":
            continue

        document_date_text = clean_text(cells[-1].get_text(" ", strip=True))
        document_date = extract_date_from_text(document_date_text)

        if not document_date:
            row_text = clean_text(row.get_text(" ", strip=True))
            row_dates = extract_all_dates_from_text(row_text)
            document_date = row_dates[-1] if row_dates else meeting_date

        normalized_type = re.sub(r"\s+", " ", type_text).strip().lower()
        type_map = {
            "statement": "Policy Statement",
            "minutes": "Minutes",
            "summary of economic projections": "Summary of Economic Projections",
            "summary of economic projections (sep)": "Summary of Economic Projections",
            "sep: individual projections": "Summary of Economic Projections",
            "sep: accessible materials": "Summary of Economic Projections",
            "sep": "Summary of Economic Projections",
        }
        if normalized_type not in type_map:
            continue
        material_type = type_map[normalized_type]

        html_links: List[str] = []
        for link in row.find_all("a", href=True):
            href_attr = link.get("href")
            if isinstance(href_attr, list):
                href = str(href_attr[0]).strip() if href_attr else ""
            else:
                href = str(href_attr or "").strip()
            if not href or href.lower().endswith(".pdf"):
                continue
            html_links.append(urljoin(BASE, href))

        if not html_links:
            continue

        items.append(
            {
                "meeting_date": meeting_date,
                "document_date": document_date,
                "type": material_type,
                "title": material_type,
                "url": html_links[0],
            }
        )

    return items


def render_filtered_pages(start_date: str, end_date: str, type_labels: List[str]) -> List[str]:
    from playwright.sync_api import sync_playwright

    pages: List[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for target_type in type_labels:
            page.goto(MATERIALS_URL, wait_until="networkidle")
            time.sleep(1)

            start_input = page.locator("input#startmodel")
            end_input = page.locator("input#endmodel")
            start_input.fill(start_date)
            start_input.press("Tab")
            end_input.fill(end_date)
            end_input.press("Tab")

            all_cb = page.locator("label input[type='checkbox']")
            for i in range(all_cb.count()):
                cb = all_cb.nth(i)
                if cb.is_checked():
                    cb.uncheck(force=True)

            type_locator = page.locator(f"label:has-text('{target_type}') input")
            if type_locator.count() == 0:
                continue
            type_locator.first.check(force=True)

            page.evaluate(
                """
                () => {
                    const b = document.querySelector('button.fomc-filter');
                    if (b) {
                        b.disabled = false;
                        b.removeAttribute('disabled');
                    }
                }
                """
            )
            page.locator("button.fomc-filter").click()
            page.wait_for_load_state("networkidle")
            time.sleep(1)

            max_pages = 100
            for _ in range(max_pages):
                pages.append(page.content())

                next_li = page.locator("ul.pagination li").filter(has_text="Next")
                if next_li.count() == 0:
                    break
                next_classes = next_li.first.get_attribute("class") or ""
                if "disabled" in next_classes:
                    break
                next_li.first.locator("a").click()
                page.wait_for_load_state("networkidle")
                time.sleep(0.5)

        browser.close()

    return pages


def run(start_year: int, end_year: int, out_path: str, texts_dir: str, delay_s: float) -> None:
    rows: List[Dict[str, Any]] = []
    seen_urls = set()

    start_date = f"01/01/{start_year}"
    end_date = f"12/31/{end_year}"

    pages = render_filtered_pages(
        start_date,
        end_date,
        type_labels=[
            "Minutes (1993",
            "Policy Statements",
            "SEP: Individual Projections",
            "SEP: Accessible Materials",
        ],
    )
    items: List[Dict[str, Any]] = []
    for html in pages:
        items.extend(parse_materials_page(html, start_year, end_year))

    known_item_urls = {it["url"] for it in items}

    idx = 0
    while idx < len(items):
        it = items[idx]
        idx += 1
        url = it["url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)

        time.sleep(max(0.0, delay_s))
        try:
            html = fetch_html(url, timeout=60)
            text = extract_main_text(html)
        except Exception as e:
            text = f"[Error fetching document: {str(e)[:200]}]"

        if it["type"] == "Minutes":
            for sep_url in discover_sep_links_from_minutes(url, html):
                if sep_url in known_item_urls:
                    continue
                known_item_urls.add(sep_url)
                items.append(
                    {
                        "meeting_date": it["meeting_date"],
                        "document_date": it["document_date"],
                        "type": "Summary of Economic Projections",
                        "title": "Summary of Economic Projections",
                        "url": sep_url,
                    }
                )

        section_count = 0
        try:
            sections, structured_text = extract_structured_sections(html)
            if structured_text:
                text = structured_text
            section_count = len(sections)
        except Exception:
            # Fallback to plain text extraction if heading-structure parsing fails.
            section_count = 0

        text_path = save_text(texts_dir, it["document_date"], url, text)

        rows.append(
            {
                "meeting_date": it["meeting_date"],
                "document_date": it["document_date"],
                "document_type": it["type"],
                "title": it["title"],
                "source_url": url,
                "url": url,
                "text_path": text_path,
                "section_count": section_count,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["_document_dt"] = pd.to_datetime(df["document_date"], format="%m/%d/%Y", errors="coerce")
        df["_meeting_dt"] = pd.to_datetime(df["meeting_date"], format="%m/%d/%Y", errors="coerce")
        df = (
            df.sort_values(["_document_dt", "_meeting_dt"], na_position="last")
            .drop(columns=["_document_dt", "_meeting_dt"])
            .reset_index(drop=True)
        )

    out_lower = out_path.lower()
    if out_lower.endswith(".parquet"):
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    print(f"Done. Saved {len(df)} materials to: {out_path}")
    print(f"Full texts saved in: {os.path.abspath(texts_dir)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FED FOMC Materials Scraper (Minutes, Policy Statements, and SEP)"
    )
    p.add_argument("--start-year", type=int, default=2011, help="Start year (default: 2011)")
    p.add_argument("--end-year", type=int, default=date.today().year, help="End year (default: current year)")
    p.add_argument(
        "--out",
        default="data/metadata/fomc_minutes_statements.csv",
        help="Output CSV file (new dataset, does not overwrite existing canonical files)",
    )
    p.add_argument(
        "--texts-dir",
        default="data/raw/fed_texts/fomc_minutes_statements",
        help="Directory for section-structured document text files",
    )
    p.add_argument("--delay", type=float, default=0.25, help="Delay between requests (seconds)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        start_year=args.start_year,
        end_year=args.end_year,
        out_path=args.out,
        texts_dir=args.texts_dir,
        delay_s=args.delay,
    )
