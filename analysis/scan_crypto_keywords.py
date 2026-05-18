"""
scan_crypto_keywords.py
-----------------------
Scan all FOMC minutes files for crypto/bitcoin-related keywords and write
results to analysis/outputs/fomc_crypto_keyword_matches.csv.

Columns: date, keyword_hit, one_line_quote, web_link
  - date:          document release date (YYYY-MM-DD)
  - keyword_hit:   keyword(s) found in the sentence
  - one_line_quote: the sentence containing the keyword (trimmed)
  - web_link:      source URL from metadata
"""

import csv
import re
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA = os.path.join(BASE, "data", "metadata", "fomc_minutes_statements.csv")
OUTPUT   = os.path.join(BASE, "analysis", "outputs", "fomc_crypto_keyword_matches.csv")

# ---------------------------------------------------------------------------
# Keywords to search for (each is a regex pattern, case-insensitive)
# ---------------------------------------------------------------------------
KEYWORD_PATTERNS = {
    "bitcoin":               re.compile(r"\bbitcoin\b", re.IGNORECASE),
    "cryptocurrency":        re.compile(r"\bcryptocurrenc(?:y|ies)\b", re.IGNORECASE),
    "crypto-asset":          re.compile(r"\bcrypto-assets?\b", re.IGNORECASE),
    "crypto-asset exchange": re.compile(r"\bcrypto-asset\s+exchange\b", re.IGNORECASE),
    "stablecoin":            re.compile(r"\bstablecoins?\b", re.IGNORECASE),
    "stablecoin arrangement":re.compile(r"\bstablecoin\s+arrangement\b", re.IGNORECASE),
    "payment stablecoin":    re.compile(r"\bpayment\s+stablecoins?\b", re.IGNORECASE),
    "digital asset":         re.compile(r"\bdigital\s+assets?\b", re.IGNORECASE),
    "digital currency":      re.compile(r"\bdigital\s+currenc(?:y|ies)\b", re.IGNORECASE),
    "digital asset entity":  re.compile(r"\bdigital\s+asset\s+entit(?:y|ies)\b", re.IGNORECASE),
    "virtual currency":      re.compile(r"\bvirtual\s+currenc(?:y|ies)\b", re.IGNORECASE),
    "decentralized finance": re.compile(r"\bdecentralized\s+finance\b", re.IGNORECASE),
    "DeFi":                  re.compile(r"\bDeFi\b"),
    "blockchain":            re.compile(r"\bblockchain\b", re.IGNORECASE),
    "tokenization":          re.compile(r"\btokenization\b", re.IGNORECASE),
    "crypto exchange":       re.compile(r"\bcrypto\s+exchange\b", re.IGNORECASE),
}

# Priority order: more specific labels listed first
PRIORITY_ORDER = [
    "payment stablecoin",
    "stablecoin arrangement",
    "crypto-asset exchange",
    "digital asset entity",
    "decentralized finance",
    "digital currency",
    "virtual currency",
    "crypto exchange",
    "cryptocurrency",
    "crypto-asset",
    "digital asset",
    "stablecoin",
    "bitcoin",
    "DeFi",
    "blockchain",
    "tokenization",
]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on '.', '!', '?' followed by whitespace or end."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def find_keywords_in_sentence(sentence: str) -> list[str]:
    """Return list of keyword labels that match in this sentence (priority-ordered)."""
    found = set()
    for label, pattern in KEYWORD_PATTERNS.items():
        if pattern.search(sentence):
            found.add(label)

    result = []
    for label in PRIORITY_ORDER:
        if label in found:
            result.append(label)
            found.discard(label)
    result.extend(sorted(found))
    return result


def truncate_sentence(s: str, max_len: int = 300) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[:max_len].rstrip() + "..."


def main():
    rows = []

    with open(METADATA, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        minutes_rows = [r for r in reader if r["document_type"] == "Minutes"]

    print(f"Scanning {len(minutes_rows)} minutes files...")

    for meta in minutes_rows:
        doc_date   = meta["document_date"]   # e.g. "02/15/2012"
        source_url = meta["source_url"]
        text_path  = os.path.join(BASE, meta["text_path"].replace("/", os.sep))

        # Parse date to YYYY-MM-DD
        parts = doc_date.split("/")
        date_str = f"{parts[2]}-{parts[0]}-{parts[1]}"

        if not os.path.exists(text_path):
            print(f"  MISSING: {text_path}")
            continue

        with open(text_path, encoding="utf-8") as f:
            text = f.read()

        sentences = split_sentences(text)

        seen_sentences = set()

        for sentence in sentences:
            keywords = find_keywords_in_sentence(sentence)
            if not keywords:
                continue

            cleaned = truncate_sentence(sentence)
            if cleaned in seen_sentences:
                continue
            seen_sentences.add(cleaned)

            keyword_label = "; ".join(keywords)
            rows.append({
                "date":           date_str,
                "keyword_hit":    keyword_label,
                "one_line_quote": cleaned,
                "web_link":       source_url,
            })

    rows.sort(key=lambda r: r["date"])

    print(f"Found {len(rows)} keyword matches across all minutes.")

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "keyword_hit", "one_line_quote", "web_link"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written to {OUTPUT}")


if __name__ == "__main__":
    main()
