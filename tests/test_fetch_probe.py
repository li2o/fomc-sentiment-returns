#!/usr/bin/env python3
"""
tests/test_fetch_probe.py
-------------------------
Lightweight "probe" tests that make a small number of live HTTP requests to
verify that each fetch script still works against the real Federal Reserve
website.

These tests catch the most common source of silent breakage in scrapers:
the upstream website changes its HTML structure, but the scripts appear to
run without errors while silently producing empty or malformed output.

What is tested (per scraper):
  fetch_fed_speeches.py
    - The yearly speech-index page is reachable and returns HTTP 200.
    - At least one speech item is parsed with a valid URL and date.
    - The detail page of one parsed item is reachable and yields non-empty text.

  fetch_fomc_minutes_statements.py
    - A known stable FOMC minutes page is reachable and yields non-empty text.
    - extract_structured_sections finds at least one section.
    - A known stable policy statement page is reachable and yields non-empty text.

Requirements:
  - Network access to www.federalreserve.gov.
  - Tests are skipped automatically when the site is unreachable.

Run from repo root:
    .venv/Scripts/python.exe tests/test_fetch_probe.py

These tests intentionally fetch only ONE page per check so they are fast
(~5–15 seconds total) and do not burden the server.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Load fetch modules via importlib (avoids package-path issues)
# ---------------------------------------------------------------------------

def _load(rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


speeches_mod = _load("src/scraping/fetch_fed_speeches.py")
fomc_mod = _load("src/scraping/fetch_fomc_minutes_statements.py")


# ---------------------------------------------------------------------------
# Connectivity guard — skip all tests if the site is unreachable
# ---------------------------------------------------------------------------

def _site_reachable() -> bool:
    try:
        import requests
        r = requests.get(
            "https://www.federalreserve.gov",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        return r.status_code < 500
    except Exception:
        return False


SITE_UP = _site_reachable()
skip_if_offline = unittest.skipUnless(SITE_UP, "federalreserve.gov is unreachable — skipping live probe tests")


# ---------------------------------------------------------------------------
# Speech scraper probe
# ---------------------------------------------------------------------------

# Use a completed year so the index page is stable.
PROBE_YEAR = 2023
PROBE_SPEECH_INDEX_URL = f"https://www.federalreserve.gov/newsevents/speech/{PROBE_YEAR}-speeches.htm"


@skip_if_offline
class TestSpeechScraperProbe(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Fetch the index page once and parse items — shared across all tests in this class."""
        html = speeches_mod.fetch(PROBE_SPEECH_INDEX_URL, timeout=30, retries=2)
        cls.items = speeches_mod.parse_year_page_improved(html, PROBE_YEAR, content_type="speech")

    def test_index_page_returns_items(self):
        """The yearly index page must yield at least one speech."""
        self.assertGreater(
            len(self.items), 0,
            f"No speeches found on {PROBE_SPEECH_INDEX_URL} — the page structure may have changed.",
        )

    def test_items_have_required_fields(self):
        """Every parsed item must have url, date, and title fields."""
        for item in self.items:
            with self.subTest(url=item.get("url")):
                self.assertIn("url", item)
                self.assertIn("date", item)
                self.assertIn("title", item)
                self.assertTrue(item["url"].startswith("https://"), f"Unexpected URL: {item['url']}")

    def test_item_dates_look_valid(self):
        """Dates must follow MM/DD/YYYY format."""
        import re
        date_re = re.compile(r"^\d{2}/\d{2}/\d{4}$")
        for item in self.items:
            with self.subTest(url=item.get("url")):
                self.assertRegex(
                    item["date"], date_re,
                    f"Unexpected date format '{item['date']}' for {item.get('url')}",
                )

    def test_detail_page_yields_text(self):
        """The detail page of the first parsed speech must return non-empty full text."""
        first = self.items[0]
        detail_html = speeches_mod.fetch(first["url"], timeout=30, retries=2)
        detail = speeches_mod.parse_speech_detail_page(first["url"], detail_html)
        full_text = detail.get("full_text", "")
        self.assertGreater(
            len(full_text), 200,
            f"Detail page returned very short text ({len(full_text)} chars) for {first['url']}",
        )


# ---------------------------------------------------------------------------
# FOMC materials scraper probe
# ---------------------------------------------------------------------------

# Use a stable, completed FOMC meeting. These URLs are permanent.
PROBE_MINUTES_URL = "https://www.federalreserve.gov/monetarypolicy/fomcminutes20231101.htm"
PROBE_STATEMENT_URL = "https://www.federalreserve.gov/newsevents/pressreleases/monetary20231101a.htm"


@skip_if_offline
class TestFomcScraperProbe(unittest.TestCase):

    def test_minutes_page_yields_text(self):
        """A known FOMC minutes page must return substantial plain text."""
        html = fomc_mod.fetch_html(PROBE_MINUTES_URL, timeout=30)
        text = fomc_mod.extract_main_text(html)
        self.assertGreater(
            len(text), 1_000,
            f"extract_main_text returned very short text ({len(text)} chars) for minutes page.",
        )

    def test_minutes_page_yields_sections(self):
        """extract_structured_sections must find at least one named section in a minutes page."""
        html = fomc_mod.fetch_html(PROBE_MINUTES_URL, timeout=30)
        sections, structured_text = fomc_mod.extract_structured_sections(html)
        self.assertGreater(
            len(sections), 0,
            "extract_structured_sections returned no sections for minutes page.",
        )
        self.assertGreater(
            len(structured_text), 1_000,
            f"Structured text is unexpectedly short ({len(structured_text)} chars).",
        )

    def test_statement_page_yields_text(self):
        """A known FOMC policy statement page must return substantial plain text."""
        html = fomc_mod.fetch_html(PROBE_STATEMENT_URL, timeout=30)
        text = fomc_mod.extract_main_text(html)
        self.assertGreater(
            len(text), 200,
            f"extract_main_text returned very short text ({len(text)} chars) for statement page.",
        )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not SITE_UP:
        print("WARNING: federalreserve.gov is unreachable. All live probe tests will be skipped.")
    unittest.main(verbosity=2)
