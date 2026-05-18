#!/usr/bin/env python3
"""
tools/generate_manifest.py
--------------------------
Compute SHA-256 checksums for all raw text files and save them to JSON
manifest files under data/metadata/.

This allows you to detect:
  - Files that changed unexpectedly after a re-fetch
  - Missing files (incomplete fetch run)
  - New files added since the manifest was generated

One manifest is generated per corpus (speeches / fomc_materials), keeping
the two data sources separate.

Usage (from repo root):

  Generate manifests (after a successful fetch run):
      .venv/Scripts/python.exe tools/generate_manifest.py

  Verify current files match an existing manifest:
      .venv/Scripts/python.exe tools/generate_manifest.py --verify

Exit code 0 on success, 1 if verification finds mismatches.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Raw data directories — one entry per corpus.
# Key   : short corpus name (used in manifest filename and output messages)
# Value : path to raw text directory
CORPORA: dict[str, Path] = {
    "fed_speeches": ROOT / "data/raw/fed_texts/fed_speeches",
    "fomc_minutes_statements": ROOT / "data/raw/fed_texts/fomc_minutes_statements",
}

MANIFEST_DIR = ROOT / "data/metadata"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_path(corpus_name: str) -> Path:
    return MANIFEST_DIR / f"manifest_{corpus_name}.json"


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

def generate(corpus_name: str, corpus_dir: Path) -> None:
    files = sorted(corpus_dir.glob("*.txt"))
    hashes = {f.name: sha256(f) for f in files}

    record = {
        "corpus": corpus_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "file_count": len(hashes),
        "files": hashes,
    }

    out = manifest_path(corpus_name)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print(f"[{corpus_name}] Manifest saved: {out} ({len(hashes)} files)")


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def verify(corpus_name: str, corpus_dir: Path) -> bool:
    """Return True only if every file in the manifest is present and unchanged."""
    mpath = manifest_path(corpus_name)
    if not mpath.exists():
        print(
            f"[{corpus_name}] No manifest found at {mpath}.\n"
            f"  Run without --verify first to generate one."
        )
        return False

    with open(mpath, encoding="utf-8") as f:
        saved = json.load(f)

    saved_hashes: dict[str, str] = saved.get("files", {})
    current_files = sorted(corpus_dir.glob("*.txt"))
    current_hashes = {f.name: sha256(f) for f in current_files}

    changed = [n for n, h in saved_hashes.items() if n in current_hashes and current_hashes[n] != h]
    missing = [n for n in saved_hashes if n not in current_hashes]
    new_files = sorted(set(current_hashes) - set(saved_hashes))

    for name in missing:
        print(f"  [MISSING] {name}")
    for name in changed:
        print(f"  [CHANGED] {name}")
    for name in new_files:
        print(f"  [NEW]     {name}")

    saved_count = saved.get("file_count", "?")
    print(
        f"[{corpus_name}] {len(current_files)} files now vs {saved_count} at manifest time "
        f"(generated {saved.get('generated_at', '?')})"
    )

    ok = not missing and not changed
    if ok and not new_files:
        print(f"[{corpus_name}] All {len(current_files)} files match the manifest. OK")
    elif ok and new_files:
        print(
            f"[{corpus_name}] All saved files match, but {len(new_files)} new file(s) present. "
            f"Re-run without --verify to update the manifest."
        )
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify raw files against existing manifests instead of generating new ones.",
    )
    args = parser.parse_args()

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for corpus_name, corpus_dir in CORPORA.items():
        if not corpus_dir.exists():
            print(f"[{corpus_name}] Raw directory not found: {corpus_dir}. Skipping.")
            continue

        if args.verify:
            ok = verify(corpus_name, corpus_dir)
            all_ok = all_ok and ok
        else:
            generate(corpus_name, corpus_dir)

    if args.verify and not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
