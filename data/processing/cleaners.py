"""
data/processing/cleaners.py
---------------------------
Modular text-cleaning functions for Fed speeches and FOMC minutes/statements.

Each function:
  - Takes a raw text string
  - Returns a cleaned text string
  - Is independently togglable

Use PIPELINE_SPEECHES and PIPELINE_FOMC to select which steps apply to each corpus.
The audit notebook imports these directly so you can inspect step-by-step changes.
"""

import re


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _strip_trailing_whitespace(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines)


def _collapse_blank_lines(text: str, max_blank: int = 2) -> str:
    """Collapse runs of more than `max_blank` consecutive blank lines."""
    pattern = r"(\n\s*){" + str(max_blank + 1) + r",}"
    return re.sub(pattern, "\n" * max_blank, text).strip()


# ---------------------------------------------------------------------------
# Shared cleaners (both corpora)
# ---------------------------------------------------------------------------

def remove_figure_references(text: str) -> str:
    """
    Remove inline figure-link artifacts from HTML scraping, e.g.:
        '... over the past 12 months ( figure 1 ).'
    These appear as '( figure N )' or '(figure N)' in the text.
    """
    return re.sub(r"\(\s*figure\s+\d+\s*\)", "", text, flags=re.IGNORECASE)


def remove_orphaned_footnote_numbers(text: str) -> str:
    """
    Remove lone digits on their own line that are PDF-to-text footnote markers:
        'Eric M. Engen\n1\nand Daniel M. Covitz'
    Only removes lines that are a single integer (1–99), not numeric data lines.
    """
    return re.sub(r"(?m)^\s*(\d{1,2})\s*$", "", text)


def normalize_whitespace(text: str) -> str:
    """
    - Strip trailing spaces from every line
    - Collapse 3+ consecutive blank lines to 2
    - Strip leading/trailing whitespace from the whole document
    """
    text = _strip_trailing_whitespace(text)
    text = _collapse_blank_lines(text, max_blank=2)
    return text


def fix_encoding_artifacts(text: str) -> str:
    """
    Replace common Windows-1252 / PDF-extraction encoding artifacts
    that survive as literal Unicode replacement characters or known sequences.
    """
    replacements = {
        "\u2013": "-",   # en-dash → hyphen
        "\u2014": "--",  # em-dash → double hyphen
        "\u2019": "'",   # right single quotation mark
        "\u2018": "'",   # left single quotation mark
        "\u201c": '"',   # left double quotation mark
        "\u201d": '"',   # right double quotation mark
        "\u00a0": " ",   # non-breaking space
        "\ufffd": "",    # Unicode replacement character (scraping error)
        "\x96":   "-",   # Windows-1252 en-dash
        "\x97":   "--",  # Windows-1252 em-dash
            "\u00c2": "",    # Spurious Â mojibake before multi-byte chars (structured scraper artifact)
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


# ---------------------------------------------------------------------------
# Speech-specific cleaners
# ---------------------------------------------------------------------------

def remove_speech_header(text: str) -> str:
    """
    Remove the top metadata block that appears in every speech file:
        Line 1: date           e.g. 'August 23, 2024'  or 'January 06, 2012'
        Line 2: speaker        e.g. 'Chair Jerome H. Powell'
        Line 3: venue/event    e.g. 'At "Reassessing the Effectiveness...'
        [optional blank line]

    Strategy: skip lines until we hit the first non-empty line that does NOT
    look like a date, speaker attribution, or venue line, then start there.

    The heuristic:
      - A line starting with a month name  → date line
      - A line starting with 'Governor'|'Chair'|'President'|'Vice' → speaker
      - A line starting with 'At '         → venue
      - A blank line after the above       → skip
    """
    date_pattern   = re.compile(
        r"^(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},\s+\d{4}"
    )
    speaker_pattern = re.compile(
        r"^(Governor|Chairman|Chair|President|Vice Chairman|Vice Chair|Deputy|"
        r"Acting|Member|Director|Secretary)\b",
        re.IGNORECASE,
    )
    venue_pattern = re.compile(r"^At\s+", re.IGNORECASE)

    lines = text.splitlines()
    skip_until = 0
    state = "header"  # header → body

    for i, line in enumerate(lines):
        stripped = line.strip()
        if state == "header":
            if (not stripped
                    or date_pattern.match(stripped)
                    or speaker_pattern.match(stripped)
                    or venue_pattern.match(stripped)):
                skip_until = i + 1
            else:
                state = "body"
                break

    return "\n".join(lines[skip_until:]).strip()


def remove_speech_accessibility_artifacts(text: str) -> str:
    """
    Remove accessibility/navigation artifacts commonly present in scraped speech
    pages and video transcript pages.

    Removes:
      - Standalone "Accessible Version" lines
      - The "Accessible Keys for Video" instruction block
      - Bracketed keybinding lines such as "[Space Bar] ..."
      - The specific "Tab/Enter" navigation instruction line
    """
    # Remove single-line accessibility markers.
    text = re.sub(r"(?mi)^\s*Accessible Version\s*$", "", text)
    text = re.sub(r"(?mi)\bView speech charts and figures\s+Accessible Version\b", "", text)
    text = re.sub(r"(?mi)\bAccessible Version\b", "", text)

    # Remove the keyboard-help block that starts with "Accessible Keys for Video".
    text = re.sub(
        r"(?mi)^\s*Accessible Keys for Video\s*\n(?:\s*\n|\s*\[[^\n]*\n|.*?(?:toggles|seeks|increase/decrease|navigate and activate control buttons).*\n)+",
        "",
        text,
    )

    # Remove any remaining standalone bracketed keybinding instructions.
    text = re.sub(r"(?mi)^\s*\[[^\n\]]+\]\s+[^\n]*$", "", text)

    # Remove any lingering navigation sentence variant.
    text = re.sub(
        r"(?mi)^\s*The\s+\[Tab\]\s+key.*activate\s+control\s+buttons[^\n]*$",
        "",
        text,
    )
    return text


def remove_speech_footnotes(text: str) -> str:
    """
    Remove the footnote reference block at the end of speeches.

    Footnotes look like:
        1. Some citation text... Return to text
        2. Another citation...   Return to text

    Strategy: find the last occurrence of a block that starts with a line
    matching '^\\d+\\.\\s+' and whose entries end with 'Return to text'.
    Everything from that point to the end of the document is removed.
    """
    # Pattern: a line like "1. Something..." followed eventually by "Return to text"
    # We find the START of the footnotes block.
    footnote_block_start = re.compile(
        r"(?m)^1\.\s+.{5,}",   # line starting with "1. " and at least 5 chars
    )
    return_to_text = re.compile(r"Return to text", re.IGNORECASE)

    lines = text.splitlines()
    candidate_starts = []

    for i, line in enumerate(lines):
        if footnote_block_start.match(line.strip()):
            # Check that within the next 20 lines there is a "Return to text"
            window = "\n".join(lines[i: i + 20])
            if return_to_text.search(window):
                candidate_starts.append(i)

    if not candidate_starts:
        return text  # no footnote block found

    # Use the LAST candidate start (footnotes are always at the end)
    cut = candidate_starts[-1]
    return "\n".join(lines[:cut]).strip()


def remove_return_to_text_markers(text: str) -> str:
    """Remove leftover inline or standalone 'Return to text' markers."""
    text = re.sub(r"(?mi)^\s*Return to text\s*$", "", text)
    text = re.sub(r"\s*Return to text\b", "", text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------------------
# FOMC-specific cleaners (minutes + statements)
# ---------------------------------------------------------------------------

def remove_fomc_statement_header(text: str) -> str:
    """
    Remove the administrative header from policy statement files:
        'January 25, 2012'
        'Federal Reserve issues FOMC statement'
        'For immediate release'
        'Share'
        [blank line]

    Strategy: strip lines until the first substantive paragraph.
    """
    boilerplate = re.compile(
        r"^(Federal Reserve issues|For immediate release|Share\s*$)",
        re.IGNORECASE,
    )
    date_pattern = re.compile(
        r"^(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},\s+\d{4}"
    )

    lines = text.splitlines()
    skip_until = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or date_pattern.match(stripped) or boilerplate.match(stripped):
            skip_until = i + 1
        else:
            break

    return "\n".join(lines[skip_until:]).strip()


def remove_fomc_attendee_list(text: str) -> str:
    """
    Remove the attendee roster from FOMC minutes files.

    The roster starts with 'PRESENT:' (or 'Present:') and ends just before
    the first heading line that is a section title (all-caps word, or a line
    that introduces substantive content like 'Role of...' / 'Staff Review...'
    / 'Staff Economic Outlook...').
    """
    present_pattern = re.compile(r"(?m)^PRESENT:\s*$", re.IGNORECASE)
    # Section headings in minutes follow specific patterns:
    section_heading = re.compile(
        r"^(Role of|Staff Review|Staff Economic|Financial Markets|"
        r"Developments in|Review of|Economic Situation|Summary of|"
        r"Committee Policy|Participants'|Participants Discussed|"
        r"Meeting participants|In their discussion)",
        re.IGNORECASE,
    )

    lines = text.splitlines()
    present_idx = None
    for i, line in enumerate(lines):
        if present_pattern.match(line.strip()):
            present_idx = i
            break

    if present_idx is None:
        return text  # no attendee list found

    # Find first section heading after PRESENT:
    resume_idx = None
    for i in range(present_idx + 1, len(lines)):
        if section_heading.match(lines[i].strip()):
            resume_idx = i
            break

    if resume_idx is None:
        return text  # safety: don't strip if we can't find the boundary

    return "\n".join(lines[:present_idx] + lines[resume_idx:]).strip()


def remove_fomc_voting_records(text: str) -> str:
    """
    Remove the formal voting record lines from statements and minutes.

    Statement format (single long paragraph at end of file):
        'Voting for the FOMC monetary policy action were: Ben S. Bernanke...'

    Minutes format (multi-line block after policy discussion):
        'Voting for this action:\nBen Bernanke...\nVoting against this action:\nJ. Lacker.'

    Strategy: once we hit a 'Voting for...' line that starts a new paragraph,
    remove from that point to the end of the document — these lines are always
    at the tail of the file (in statements) or already inside the block removed
    by remove_fomc_duplicate_statement_in_minutes (in minutes).
    """
    # Match from the first "Voting for/against" paragraph to end of document
    return re.sub(
        r"\n+Voting (for|against).+",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )


def remove_fomc_policy_directive(text: str) -> str:
    """
    Remove the verbatim policy directive boilerplate from minutes, e.g.:
        '"The Federal Open Market Committee seeks monetary and financial
         conditions that will foster price stability...'
    This is standard legal language repeated every meeting with minimal edits.
    Removes the quoted block from '"The Federal Open Market Committee seeks'
    to the closing '"'.
    """
    return re.sub(
        r'"The Federal Open Market Committee seeks monetary[^"]*?"',
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )


def remove_fomc_duplicate_statement_in_minutes(text: str) -> str:
    """
    FOMC minutes files reproduce the full policy statement verbatim under:
        'The vote encompassed approval of the statement below...'
        'The vote also encompassed approval of the statement below...'  (newer variant)
    Remove everything from that line to the end of the document.
    """
    trigger = re.compile(
        r"The vote(?:\s+also)?\s+encompassed approval of the statement below",
        re.IGNORECASE,
    )
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if trigger.search(line):
            return "\n".join(lines[:i]).strip()
    return text


# ---------------------------------------------------------------------------
# Pipelines: ordered list of (name, function) tuples
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Structured FOMC materials cleaners (fomc_minutes_statements corpus)
# ---------------------------------------------------------------------------

def remove_fomc_structured_statement_preamble(text: str) -> str:
    """
    Remove the boilerplate preamble from structured Policy Statement files:
        ## Document Body
        January 25, 2012
        ## Federal Reserve issues FOMC statement
        For immediate release Share
        For release at 2:00 p.m. EDT Share   (time-stamped release line variant)
        Share
    Removes lines until the first substantive paragraph.
    """
    boilerplate = re.compile(
        r"^(##\s*Document Body|##\s*Federal Reserve issues|"
        r"Federal Reserve issues|For immediate release|For release at \d|Share\s*$)",
        re.IGNORECASE,
    )
    date_pattern = re.compile(
        r"^(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},\s+\d{4}"
    )
    lines = text.splitlines()
    skip_until = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or date_pattern.match(stripped) or boilerplate.match(stripped):
            skip_until = i + 1
        else:
            break
    return "\n".join(lines[skip_until:]).strip()


def remove_fomc_structured_minutes_preamble(text: str) -> str:
    """
    Remove the date heading, FOMC Minutes title, and attendee/staff roster
    from structured FOMC minutes files. The block runs from the top of the
    file until the first recognisable substantive section heading.
    """
    section_start = re.compile(
        r"^(Developments in|Staff Review|Staff Economic|Financial Markets|"
        r"Review of|Economic Situation|Participants|In their discussion|"
        r"Committee Policy|The information reviewed|The Manager of the System)",
        re.IGNORECASE,
    )
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if section_start.match(line.strip()):
            return "\n".join(lines[i:]).strip()
    return text


def remove_sep_navigation_artifacts(text: str) -> str:
    """
    Remove figure/table navigation noise from structured SEP files:
      - Standalone ## Figure N. and ## Table N. caption headings
      - Accessible version of figure N | Return to figure N lines
      - Inline Return to table / Return to figure references
      - Return to top footer line
    """
    text = re.sub(
        r"(?m)^##\s+(Figure|Table)\s+[\d.A-Za-z]+[^\n]*\n?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?m)^Accessible version of figure[^\n]*\n?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*Return to (table|figure[\s\d.A-Za-z]*)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?m)^Return to top\s*$", "", text, flags=re.IGNORECASE)
    return text


def remove_fomc_release_time_line(text: str) -> str:
    """
    Remove timestamped release lines from FOMC policy statement files, e.g.:
        'For release at 2:00 p.m. EDT Share'
        'For release at 11:00 a.m. EST Share'
    and any resulting standalone 'Share' lines left behind.

    These are website navigation elements injected during scraping and are
    not caught by remove_fomc_structured_statement_preamble in edge cases
    where the document starts with a non-standard heading.
    """
    text = re.sub(
        r"(?m)^For release at \d+:\d+ [ap]\.m\. [A-Z]{2,3}\s*(?:Share)?\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?m)^Share\s*$", "", text)
    return text


def remove_fomc_minutes_tail(text: str) -> str:
    """
    Remove the administrative tail block from structured FOMC minutes files.

    After the Committee Policy Action section, FOMC minutes end with:
      - Voting records ('Voting for this action:', '## Voting against this action:')
      - Board of Governors action ('Consistent with the Committee's decision to...')
      - Adjournment notice ('It was agreed that the next meeting...')
      - Notation vote ('Notation Vote By notation vote...')
      - Secretary signature (underscore separator line + '[Name] Secretary')
      - Numbered footnotes ('N. ... Return to text')

    Strategy: find the first line matching any tail trigger and cut everything
    from that point to the end of the document.
    """
    tail_trigger = re.compile(
        r"^(##\s*Voting\s+(for|against)|Voting\s+(for|against)|"
        r"Consistent with the Committee.s decision to\b|"
        r"It was agreed that the next meeting\b|"
        r"Notation Vote\b|"
        r"_{5,})",
        re.IGNORECASE,
    )
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if tail_trigger.match(line.strip()):
            return "\n".join(lines[:i]).strip()
    return text


# ---------------------------------------------------------------------------
# FOMC press conference cleaners
# ---------------------------------------------------------------------------

def remove_fomc_press_conference_page_artifacts(text: str) -> str:
    """
    Remove repeated PDF page headers and page counters from press conference
    transcripts, e.g.:
        'January 30, 2019 Chairman Powell’s Press Conference FINAL'
        'Page 1 of 24'
        '1 of 29'
    """
    text = re.sub(
        r"(?mi)^.*press conference\s+final\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?mi)^\s*Transcript of .*?Press Conference\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?mi)^\s*(?:Page\s+)?\d{1,3}\s+of\s+\d{1,3}\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text


def trim_fomc_press_conference_to_opening(text: str) -> str:
    """
    Start the transcript at the first substantive chair opening line, typically:
        'CHAIRMAN BERNANKE. ...'
        'CHAIR YELLEN. ...'
        'CHAIRMAN POWELL: ...'

    This removes the title/date block that precedes the spoken transcript while
    preserving the full Q&A that follows.
    """
    speaker_start = re.compile(
        r"^(?:CHAIRMAN|CHAIR)\s+[A-Z][A-Z'’.\-]+(?:\s+[A-Z][A-Z'’.\-]+)*[.:]\s+",
        re.IGNORECASE,
    )
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if speaker_start.match(line.strip()):
            return "\n".join(lines[i:]).strip()
    return text.strip()


def remove_fomc_press_conference_reporter_questions(text: str) -> str:
    """
    Remove reporter-question turns while keeping the chair's answers.

    After the opening remarks, press conference transcripts alternate between
    speaker-labelled turns such as:
        'ROBIN HARDING. ...'          -> drop
        'CHAIR YELLEN. ...'           -> keep
        'CHAIRMAN POWELL. ...'        -> keep

    Strategy:
      - detect speaker labels at the start of a line,
      - keep only turns whose label contains CHAIR/CHAIRMAN,
      - preserve unlabeled continuation lines only when they belong to a kept
        chair segment.
    """
    speaker_line = re.compile(
        r"^(?P<label>(?:VICE\s+CHAIRMAN|VICE\s+CHAIR|CHAIRMAN|CHAIR|"
        r"[A-Z][A-Z'’.\-]+(?:\s+[A-Z][A-Z'’.\-]+){0,4}))[.:]\s+"
    )

    kept_lines: list[str] = []
    keep_current_block = True

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            if keep_current_block and kept_lines and kept_lines[-1] != "":
                kept_lines.append("")
            continue

        match = speaker_line.match(stripped)
        if match:
            label = match.group("label").upper()
            keep_current_block = bool(re.search(r"\bCHAIR(MAN)?\b", label))
            if keep_current_block:
                kept_lines.append(stripped)
            continue

        if keep_current_block:
            kept_lines.append(line.rstrip())

    return "\n".join(kept_lines).strip()


def reflow_fomc_press_conference_paragraphs(text: str) -> str:
    """
    Reflow PDF-style hard line breaks into readable paragraphs.

    Press conference transcripts are extracted with short wrapped lines and
    inconsistent blank lines, including page-break gaps that may split a
    sentence in the middle. This step:
      - joins wrapped lines within a paragraph into a single line,
      - removes spurious blank paragraphs caused by page breaks,
      - keeps true paragraph boundaries as blank lines,
      - starts a fresh paragraph whenever a new chair speaker label appears.
    """
    speaker_line = re.compile(
        r"^(?:VICE\s+CHAIRMAN|VICE\s+CHAIR|CHAIRMAN|CHAIR)\s+"
        r"[A-Z][A-Z'’.\-]+(?:\s+[A-Z][A-Z'’.\-]+)*[.:]\s+"
    )

    def _ends_paragraph(line: str) -> bool:
        return bool(re.search(r"[.!?:\"')\]]$|--$", line))

    raw_lines = [line.strip() for line in text.splitlines()]
    paragraphs: list[str] = []
    current_lines: list[str] = []
    i = 0

    while i < len(raw_lines):
        stripped = raw_lines[i]

        if not stripped:
            j = i + 1
            while j < len(raw_lines) and not raw_lines[j]:
                j += 1

            if not current_lines:
                i = j
                continue

            next_line = raw_lines[j] if j < len(raw_lines) else ""
            if next_line and not _ends_paragraph(current_lines[-1]) and not speaker_line.match(next_line):
                i = j
                continue

            paragraphs.append(" ".join(current_lines).strip())
            current_lines = []
            i = j
            continue

        if speaker_line.match(stripped):
            if current_lines:
                paragraphs.append(" ".join(current_lines).strip())
            current_lines = [stripped]
            i += 1
            continue

        if current_lines and current_lines[-1].endswith("-") and not current_lines[-1].endswith("--"):
            current_lines[-1] = current_lines[-1] + stripped
        else:
            current_lines.append(stripped)

        i += 1

    if current_lines:
        paragraphs.append(" ".join(current_lines).strip())

    return "\n\n".join(paragraphs).strip()


#: Steps applied to Fed speech files, in order.
PIPELINE_SPEECHES = [
    ("fix_encoding_artifacts",     fix_encoding_artifacts),
    ("remove_speech_header",       remove_speech_header),
    ("remove_speech_accessibility_artifacts", remove_speech_accessibility_artifacts),
    ("remove_speech_footnotes",    remove_speech_footnotes),
    ("remove_return_to_text_markers", remove_return_to_text_markers),
    ("remove_figure_references",   remove_figure_references),
    ("remove_orphaned_footnote_numbers", remove_orphaned_footnote_numbers),
    ("normalize_whitespace",       normalize_whitespace),
]

#: Steps applied to FOMC press conference files, in order.
PIPELINE_FOMC_PRESS_CONFERENCES = [
    ("fix_encoding_artifacts",                   fix_encoding_artifacts),
    ("remove_fomc_press_conference_page_artifacts", remove_fomc_press_conference_page_artifacts),
    ("trim_fomc_press_conference_to_opening",   trim_fomc_press_conference_to_opening),
    ("remove_fomc_press_conference_reporter_questions", remove_fomc_press_conference_reporter_questions),
    ("reflow_fomc_press_conference_paragraphs", reflow_fomc_press_conference_paragraphs),
    ("normalize_whitespace",                    normalize_whitespace),
]

#: Steps applied to FOMC minutes files, in order.
PIPELINE_FOMC_MINUTES = [
    ("fix_encoding_artifacts",               fix_encoding_artifacts),
    ("remove_fomc_attendee_list",            remove_fomc_attendee_list),
    ("remove_fomc_duplicate_statement_in_minutes", remove_fomc_duplicate_statement_in_minutes),
    ("remove_fomc_policy_directive",         remove_fomc_policy_directive),
    ("remove_fomc_voting_records",           remove_fomc_voting_records),
    ("remove_orphaned_footnote_numbers",     remove_orphaned_footnote_numbers),
    ("normalize_whitespace",                 normalize_whitespace),
]

#: Steps applied to FOMC policy statement files, in order.
PIPELINE_FOMC_STATEMENTS = [
    ("fix_encoding_artifacts",       fix_encoding_artifacts),
    ("remove_fomc_statement_header", remove_fomc_statement_header),
    ("remove_fomc_voting_records",   remove_fomc_voting_records),
    ("normalize_whitespace",         normalize_whitespace),
]


def apply_pipeline(text: str, pipeline: list) -> str:
    """Apply a full pipeline to a text string. Returns the final cleaned text."""
    for _name, fn in pipeline:
        text = fn(text)
    return text


# ---------------------------------------------------------------------------
# Structured FOMC materials pipelines (fomc_minutes_statements corpus)
# ---------------------------------------------------------------------------

#: Steps applied to structured FOMC minutes files.
PIPELINE_FOMC_STRUCTURED_MINUTES = [
    ("fix_encoding_artifacts",                    fix_encoding_artifacts),
    ("remove_fomc_structured_minutes_preamble",   remove_fomc_structured_minutes_preamble),
    ("remove_fomc_duplicate_statement_in_minutes",remove_fomc_duplicate_statement_in_minutes),
    ("remove_fomc_minutes_tail",                  remove_fomc_minutes_tail),
    ("remove_fomc_policy_directive",              remove_fomc_policy_directive),
    ("remove_sep_navigation_artifacts",           remove_sep_navigation_artifacts),
    ("remove_orphaned_footnote_numbers",          remove_orphaned_footnote_numbers),
    ("normalize_whitespace",                      normalize_whitespace),
]

#: Steps applied to structured FOMC policy statement files.
PIPELINE_FOMC_STRUCTURED_STATEMENTS = [
    ("fix_encoding_artifacts",                     fix_encoding_artifacts),
    ("remove_fomc_structured_statement_preamble",  remove_fomc_structured_statement_preamble),
    ("remove_fomc_release_time_line",              remove_fomc_release_time_line),
    ("remove_fomc_voting_records",                 remove_fomc_voting_records),
    ("remove_sep_navigation_artifacts",            remove_sep_navigation_artifacts),
    ("normalize_whitespace",                       normalize_whitespace),
]

#: Steps applied to structured Summary of Economic Projections (SEP) files.
PIPELINE_FOMC_STRUCTURED_SEP = [
    ("fix_encoding_artifacts",          fix_encoding_artifacts),
    ("remove_sep_navigation_artifacts", remove_sep_navigation_artifacts),
    ("remove_figure_references",        remove_figure_references),
    ("normalize_whitespace",            normalize_whitespace),
]


def apply_pipeline_stepwise(text: str, pipeline: list) -> list[dict]:
    """
    Apply a pipeline step by step, recording intermediate results.

    Returns a list of dicts, one per step:
        {
            "step":          step name (str),
            "text_before":   text before this step,
            "text_after":    text after this step,
            "chars_removed": int,
            "lines_removed": int,
        }
    """
    results = []
    current = text
    for name, fn in pipeline:
        before = current
        after  = fn(current)
        results.append(
            {
                "step":          name,
                "text_before":   before,
                "text_after":    after,
                "chars_removed": len(before) - len(after),
                "lines_removed": before.count("\n") - after.count("\n"),
            }
        )
        current = after
    return results
