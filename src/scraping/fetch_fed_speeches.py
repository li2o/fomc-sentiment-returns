#!/usr/bin/env python3
"""
Improved FED speeches and testimony scraper (all speakers)

Directly scrapes from the index pages instead of relying on complex HTML parsing.
Collects speeches and testimony from all Federal Reserve speakers.
"""

import argparse
import hashlib
import os
import re
import time
from datetime import date
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

BASE = "https://www.federalreserve.gov"
YEAR_URL = BASE + "/newsevents/speech/{year}-speeches.htm"
TESTIMONY_URL = BASE + "/newsevents/testimony/{year}-testimony.htm"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

SPACE_RE = re.compile(r"\s+")
FED_SPEAKER_RE = re.compile(r"\b(Fed|Federal Reserve|Bank|Committee|Board|Chair|Governor|President)\b", re.IGNORECASE)

# Speech type inference from title
TYPE_KEYWORDS = [
    "Opening Remarks",
    "Welcoming Remarks",
    "Brief Remarks",
    "Prepared Remarks",
    "Keynote",
    "Address",
    "Statement",
    "Remarks",
    "Speech",
]


def clean_text(s: str) -> str:
    return SPACE_RE.sub(" ", s).strip()


def fetch(url: str, timeout: int = 60, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or r.encoding
            return r.text
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < retries - 1:
                wait_time = (attempt + 1) * 2
                time.sleep(wait_time)
            else:
                raise
        except requests.exceptions.HTTPError as e:
            raise


def infer_type_from_title(title: str) -> str:
    t = title.strip()
    for kw in TYPE_KEYWORDS:
        if kw.lower() in t.lower():
            return kw
    return "Speech"


def stable_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def extract_date_from_url(url: str) -> Optional[str]:
    """Extract date from speech URL like powell20230110a.htm or bowman20240315b.htm -> 01/10/2023"""
    # Match any speaker pattern: /newsevents/speech/SPEAKER(\d{8})LETTER.htm
    m = re.search(r'/newsevents/(speech|testimony)/[a-z]+(\d{4})(\d{2})(\d{2})', url, re.IGNORECASE)
    if m:
        year, month, day = m.groups()[1:]
        return f"{month}/{day}/{year}"
    return None


def parse_year_page_improved(year_html: str, year: int, content_type: str = "speech") -> List[Dict[str, Any]]:
    """
    Extract all FED speech or testimony links from a yearly speeches/testimony page.
    content_type: 'speech' or 'testimony'
    Looks for direct links to individual speech/testimony pages by URL pattern.
    """
    soup = BeautifulSoup(year_html, "html.parser")
    
    items: List[Dict[str, Any]] = []
    seen_urls = set()
    
    # Find all links that point to FED speech/testimony pages
    for link in soup.find_all('a', href=True):
        href = link.get('href', '')
        text = clean_text(link.get_text(' ', strip=True))
        
        # Check if this is a FED speech/testimony link by pattern
        # Should match: /newsevents/speech/lastname20190101a.htm etc.
        if f'/newsevents/{content_type}/' in href.lower() and '.htm' in href.lower():
            full_url = urljoin(BASE, href)
            
            # Skip duplicates
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)
            
            # Skip video/other non-article links
            if text.lower() in {'video', 'watch live', 'slides', 'share', 'pdf', ''}:
                continue
            
            # Extract date from URL
            date_str = extract_date_from_url(href)
            if not date_str:
                continue
            
            items.append({
                'url': full_url,
                'title': text,
                'date': date_str,
                'type': infer_type_from_title(text),
                'content_type': content_type,
            })
    
    return items


def parse_speech_detail_page(speech_url: str, html: str) -> Dict[str, Any]:
    """
    Extract full text and metadata from the detail page.
    Extracts complete speech content with minimal filtering.
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # Find main content area - try specific containers first
    main = None
    for selector in ['div#article', 'main', '#content', '.col-xs-12', '.col-sm-8', 'body']:
        elem = soup.select_one(selector)
        if elem:
            main = elem
            break
    
    if not main:
        main = soup.body or soup
    
    # Title on detail page
    h = main.find(["h1", "h2", "h3"])
    detail_title = clean_text(h.get_text(" ", strip=True)) if h else ""
    
    # Speaker/location lines - extract from first few short paragraphs before the main text
    speaker_detail: Optional[str] = None
    location_detail: Optional[str] = None
    
    paras_checked = 0
    for p in main.find_all("p"):
        if paras_checked > 5:
            break
        tx = clean_text(p.get_text(" ", strip=True))
        if not tx or len(tx) < 5:
            continue
        
        paras_checked += 1
        
        # Speaker: Look for FED speaker reference
        if speaker_detail is None and FED_SPEAKER_RE.search(tx) and len(tx) < 150:
            speaker_detail = tx
        
        # Location: Starts with "At " or is a location-like phrase
        if location_detail is None and (tx.lower().startswith("at ") or tx.lower().startswith("at the")):
            if len(tx) < 300:  # Keep it reasonable length
                location_detail = tx
    
    # Full text: gather ALL paragraphs from main content
    paras: List[str] = []
    seen_texts = set()  # Track to avoid duplicates
    
    for p in main.find_all("p"):
        tx = clean_text(p.get_text(" ", strip=True))
        if not tx:
            continue
        
        low = tx.lower()
        
        # Skip navigation, buttons, and boilerplate at page level
        skip_keywords = [
            'official website of the',
            '.gov website',
            'secure .gov',
            'federal open market committee',
            'board of governors of the federal reserve',
            'about the fed',
            'news & events',
            'back to top',
            'last update',
            'share',
            'watch live',
        ]
        
        if any(low.startswith(kw) for kw in skip_keywords):
            continue
        
        # Skip very short lines that are likely metadata (but keep footnotes which start with numbers)
        if len(tx) < 15 and not tx[0].isdigit():
            continue
        
        # Avoid duplicates
        if tx in seen_texts:
            continue
        seen_texts.add(tx)
        
        paras.append(tx)
    
    full_text = "\n\n".join(paras).strip()
    
    return {
        'detail_title': detail_title,
        'speaker': speaker_detail,
        'location': location_detail,
        'full_text': full_text,
    }


def save_text(texts_dir: str, speech_date: str, speech_url: str, text: str) -> str:
    ensure_dir(texts_dir)
    sid = stable_id(speech_url)
    
    mm, dd, yyyy = speech_date.split("/")
    safe_date = f"{yyyy.zfill(4)}-{mm.zfill(2)}-{dd.zfill(2)}"
    
    filename = f"{safe_date}_{sid}.txt"
    path = os.path.join(texts_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def run(
    start_year: int,
    end_year: int,
    out_path: str,
    texts_dir: str,
    delay_s: float,
    include_testimony: bool = False,
) -> None:
    ensure_dir(texts_dir)
    
    rows: List[Dict[str, Any]] = []
    seen_urls = set()
    
    # Collect speeches
    for y in range(start_year, end_year + 1):
        year_page_url = YEAR_URL.format(year=y)
        print(f"Fetching speeches for {y}: {year_page_url}...")
        try:
            year_html = fetch(year_page_url, retries=1)
            items = parse_year_page_improved(year_html, y, content_type="speech")
            print(f"  Found {len(items)} speeches in {y}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"  Speeches page for {y} not available (404)")
                items = []
            else:
                print(f"  Error fetching speeches for {y}: {e}")
                items = []
        except Exception as e:
            print(f"  Error fetching speeches for {y}: {e}")
            items = []
        
        for it in items:
            speech_url = it['url']
            
            if speech_url in seen_urls:
                continue
            seen_urls.add(speech_url)
            
            time.sleep(max(0.0, delay_s))
            print(f"  Fetching: {it['title'][:50]}")
            
            try:
                detail_html = fetch(speech_url, timeout=30, retries=1)
                detail = parse_speech_detail_page(speech_url, detail_html)
            except requests.exceptions.RequestException as e:
                print(f"    Skipping {speech_url}: {str(e)[:60]}")
                continue
            except Exception as e:
                print(f"    Error parsing {speech_url}: {e}")
                continue
            
            # Build final row
            title = it['title'] or detail.get('detail_title', '')
            speaker = detail.get('speaker') or ''
            location = detail.get('location') or ''
            full_text = detail.get('full_text') or ''
            
            text_path = save_text(texts_dir, it['date'], speech_url, full_text)
            
            rows.append({
                'date': it['date'],
                'type': it['type'],
                'content_type': it['content_type'],
                'location': location,
                'speaker': speaker,
                'title': title,
                'speech_url': speech_url,
                'text_path': text_path,
            })
    
    # Optionally collect testimony
    if include_testimony:
        for y in range(start_year, end_year + 1):
            year_page_url = TESTIMONY_URL.format(year=y)
            print(f"Fetching testimony for {y}: {year_page_url}...")
            try:
                year_html = fetch(year_page_url, retries=1)
                items = parse_year_page_improved(year_html, y, content_type="testimony")
                print(f"  Found {len(items)} testimonies in {y}")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    print(f"  Testimony page for {y} not available (404)")
                    items = []
                else:
                    print(f"  Error fetching testimony for {y}: {e}")
                    items = []
            except Exception as e:
                print(f"  Error fetching testimony for {y}: {e}")
                items = []
            
            for it in items:
                speech_url = it['url']
                
                if speech_url in seen_urls:
                    continue
                seen_urls.add(speech_url)
                
                time.sleep(max(0.0, delay_s))
                print(f"  Fetching: {it['title'][:50]}")
                
                try:
                    detail_html = fetch(speech_url, timeout=30, retries=1)
                    detail = parse_speech_detail_page(speech_url, detail_html)
                except requests.exceptions.RequestException as e:
                    print(f"    Skipping {speech_url}: {str(e)[:60]}")
                    continue
                except Exception as e:
                    print(f"    Error parsing {speech_url}: {e}")
                    continue
                
                # Build final row
                title = it['title'] or detail.get('detail_title', '')
                speaker = detail.get('speaker') or ''
                location = detail.get('location') or ''
                full_text = detail.get('full_text') or ''
                
                text_path = save_text(texts_dir, it['date'], speech_url, full_text)
                
                rows.append({
                    'date': it['date'],
                    'type': it['type'],
                    'content_type': it['content_type'],
                    'location': location,
                    'speaker': speaker,
                    'title': title,
                    'speech_url': speech_url,
                    'text_path': text_path,
                })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['date', 'speech_url']).reset_index(drop=True)
    
    out_lower = out_path.lower()
    if out_lower.endswith(".parquet"):
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    
    print(f"\nDone. Saved {len(df)} speeches/testimony to: {out_path}")
    print(f"Full texts saved in: {os.path.abspath(texts_dir)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=2017)
    p.add_argument("--end-year", type=int, default=date.today().year)
    p.add_argument("--out", required=True, help="Output table path (.csv or .parquet).")
    p.add_argument("--texts-dir", default="texts", help="Directory to store per-speech .txt files.")
    p.add_argument("--delay", type=float, default=0.5, help="Delay between page requests (seconds).")
    p.add_argument("--include-testimony", action="store_true", help="Also scrape testimony pages.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        start_year=args.start_year,
        end_year=args.end_year,
        out_path=args.out,
        texts_dir=args.texts_dir,
        delay_s=args.delay,
        include_testimony=args.include_testimony,
    )
