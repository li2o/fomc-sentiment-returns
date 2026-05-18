#!/usr/bin/env python3
"""
FOMC press conference transcript scraper.

Discovers meeting dates from:
  - 2012-2020: https://www.federalreserve.gov/monetarypolicy/fomchistorical{YYYY}.htm
  - 2021+:     https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm

Downloads transcript PDFs from:
  https://www.federalreserve.gov/mediacenter/files/FOMCpresconf{YYYYMMDD}.pdf

Output follows the same convention as existing scrapers:
  - Metadata CSV:  data/metadata/fomc_press_conferences.csv
  - Raw TXT files: data/raw/fed_texts/fomc_press_conferences/YYYY-MM-DD_<hash12>.txt
"""

import argparse
import hashlib
import io
import os
import re
import time
from datetime import date
from typing import Dict, List, Tuple

import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup

BASE = "https://www.federalreserve.gov"
HISTORICAL_URL = BASE + "/monetarypolicy/fomchistorical{year}.htm"
CALENDAR_URL = BASE + "/monetarypolicy/fomccalendars.htm"
PDF_URL = BASE + "/mediacenter/files/FOMCpresconf{yyyymmdd}.pdf"

PRESCONF_HREF_RE = re.compile(r"/monetarypolicy/fomcpresconf(\d{8})\.htm", re.IGNORECASE)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def fetch(url: str, timeout: int = 60, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or r.encoding
            return r.text
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < retries - 1:
                time.sleep((attempt + 1) * 2)
            else:
                raise
        except requests.exceptions.HTTPError:
            raise


def fetch_bytes(url: str, timeout: int = 120, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.content
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < retries - 1:
                time.sleep((attempt + 1) * 2)
            else:
                raise
        except requests.exceptions.HTTPError:
            raise


def pdf_exists(url: str, timeout: int = 15) -> bool:
    try:
        r = requests.head(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_bytes: bytes) -> Tuple[str, int]:
    """Extract plain text from PDF bytes. Returns (text, page_count)."""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
        page_count = len(pages)
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    return text, page_count


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def stable_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def save_text(texts_dir: str, date_iso: str, source_url: str, text: str) -> str:
    ensure_dir(texts_dir)
    sid = stable_id(source_url)
    filename = f"{date_iso}_{sid}.txt"
    path = os.path.join(texts_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# ---------------------------------------------------------------------------
# Index page parsing
# ---------------------------------------------------------------------------

def extract_dates_from_historical(html: str, year: int) -> List[str]:
    """
    Parse fomchistorical{YYYY}.htm and return YYYYMMDD strings for meetings
    that have a press conference link.
    """
    soup = BeautifulSoup(html, "lxml")
    dates = []
    for a in soup.find_all("a", href=PRESCONF_HREF_RE):
        m = PRESCONF_HREF_RE.search(a["href"])
        if m and m.group(1)[:4] == str(year):
            dates.append(m.group(1))
    return sorted(set(dates))


def extract_dates_from_calendar(html: str, years: List[int]) -> List[str]:
    """
    Parse fomccalendars.htm and return YYYYMMDD strings for press conferences
    in the requested years.
    """
    soup = BeautifulSoup(html, "lxml")
    year_set = {str(y) for y in years}
    dates = []
    for a in soup.find_all("a", href=PRESCONF_HREF_RE):
        m = PRESCONF_HREF_RE.search(a["href"])
        if m and m.group(1)[:4] in year_set:
            dates.append(m.group(1))
    return sorted(set(dates))


def yyyymmdd_to_iso(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    start_year: int,
    end_year: int,
    out_path: str,
    texts_dir: str,
    delay_s: float,
) -> None:
    ensure_dir(texts_dir)
    if os.path.dirname(out_path):
        ensure_dir(os.path.dirname(out_path))

    # --- Step A: Discover all YYYYMMDD dates with press conferences ---
    all_dates: List[str] = []

    historical_years = [y for y in range(start_year, min(end_year, 2020) + 1)]
    calendar_years = [y for y in range(max(start_year, 2021), end_year + 1)]

    for year in historical_years:
        url = HISTORICAL_URL.format(year=year)
        print(f"Fetching historical index for {year}: {url}")
        try:
            html = fetch(url)
            dates = extract_dates_from_historical(html, year)
            print(f"  Found {len(dates)} press conference(s)")
            all_dates.extend(dates)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"  No historical page for {year} (404), skipping")
            else:
                print(f"  HTTP error for {year}: {e}")
        except Exception as e:
            print(f"  Error fetching {year}: {e}")
        time.sleep(delay_s)

    if calendar_years:
        print(f"Fetching calendar index for {calendar_years[0]}-{calendar_years[-1]}: {CALENDAR_URL}")
        try:
            html = fetch(CALENDAR_URL)
            dates = extract_dates_from_calendar(html, calendar_years)
            print(f"  Found {len(dates)} press conference(s)")
            all_dates.extend(dates)
        except Exception as e:
            print(f"  Error fetching calendar: {e}")
        time.sleep(delay_s)

    all_dates = sorted(set(all_dates))
    print(f"\nTotal press conferences discovered: {len(all_dates)}")

    # --- Step B: Download PDF and extract text for each meeting ---
    rows: List[Dict] = []

    for yyyymmdd in all_dates:
        date_iso = yyyymmdd_to_iso(yyyymmdd)
        pdf_url = PDF_URL.format(yyyymmdd=yyyymmdd)

        print(f"  [{date_iso}] {pdf_url}")

        if not pdf_exists(pdf_url):
            print(f"    PDF not found, skipping")
            continue

        try:
            pdf_bytes = fetch_bytes(pdf_url)
        except Exception as e:
            print(f"    Download failed: {e}")
            continue

        try:
            text, page_count = extract_pdf_text(pdf_bytes)
        except Exception as e:
            print(f"    Text extraction failed: {e}")
            continue

        text_path = save_text(texts_dir, date_iso, pdf_url, text)

        rows.append({
            "meeting_date": date_iso,
            "document_type": "press_conference",
            "title": f"FOMC Press Conference - {date_iso}",
            "source_url": pdf_url,
            "text_path": text_path,
            "page_count": page_count,
        })
        print(f"    {page_count} pages -> {os.path.basename(text_path)}")

        time.sleep(delay_s)

    # --- Step C: Write metadata CSV / Parquet ---
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("meeting_date").reset_index(drop=True)

    if out_path.lower().endswith(".parquet"):
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    print(f"\nDone. {len(df)} transcripts saved to: {out_path}")
    print(f"Text files in: {os.path.abspath(texts_dir)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape FOMC press conference transcripts.")
    p.add_argument("--start-year", type=int, default=2012,
                   help="First year to scrape (default: 2012).")
    p.add_argument("--end-year", type=int, default=date.today().year,
                   help="Last year to scrape (default: current year).")
    p.add_argument("--out", default="data/metadata/fomc_press_conferences.csv",
                   help="Output metadata path (.csv or .parquet).")
    p.add_argument("--texts-dir", default="data/raw/fed_texts/fomc_press_conferences",
                   help="Directory for per-transcript .txt files.")
    p.add_argument("--delay", type=float, default=1.0,
                   help="Delay between requests in seconds (default: 1.0).")
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
