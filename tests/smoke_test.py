#!/usr/bin/env python3
"""
tests/smoke_test.py
-------------------
Smoke tests for the text-cleaning pipelines.

Run from repo root:
    .venv/Scripts/python.exe tests/smoke_test.py

What is tested:
  1. Each pipeline (speeches, FOMC minutes, FOMC statements, FOMC SEP) produces
     non-empty output from known sample inputs.
  2. Specific artifacts are removed (header, footnotes, voting records, etc.)
  3. Substantive content is preserved.
  4. All pipelines are idempotent (running twice gives identical output).
  5. The cleaners module exposes all expected pipelines and functions.
  6. Both fetch scripts are importable (i.e. their dependencies are installed
     and the files contain no syntax errors).

No network access is required.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.processing.cleaners import (
    PIPELINE_FOMC_PRESS_CONFERENCES,
    PIPELINE_FOMC_STRUCTURED_MINUTES,
    PIPELINE_FOMC_STRUCTURED_STATEMENTS,
    PIPELINE_FOMC_STRUCTURED_SEP,
    PIPELINE_SPEECHES,
    apply_pipeline,
)

# ---------------------------------------------------------------------------
# Minimal sample inputs — sufficient to exercise each pipeline's key cleaners
# ---------------------------------------------------------------------------

SAMPLE_SPEECH = """\
August 23, 2024
Chair Jerome H. Powell
At the Economic Symposium, Jackson Hole, Wyoming

The time has come for policy to adjust. The direction of travel is clear.

Inflation has declined significantly. ( figure 1 ) Labor market conditions
have cooled considerably. This reflects supply and demand rebalancing.

1. See Federal Reserve Board (2024), "Economic Projections." Return to text
2. See Powell (2023), "Inflation and Labor Markets." Return to text
"""

SAMPLE_FOMC_MINUTES = """\
## Document Body
January 25-26, 2022

## Minutes of the Federal Open Market Committee

PRESENT:
Jerome H. Powell, Chair
John C. Williams, Vice Chair

Staff Review of the Economic Situation
The information reviewed for this meeting suggested that economic activity
had been expanding at a solid pace.

The vote also encompassed approval of the statement below to be released at 2:00 p.m.:
This is the duplicated policy statement text.

Voting for this action: Jerome H. Powell, Chair; John C. Williams, Vice Chair.

## Voting against this action: None.
Consistent with the Committee's decision to leave the target range unchanged,
the Board of Governors voted unanimously to leave interest rates unchanged.
It was agreed that the next meeting would be held on March 15-16, 2022.
The meeting adjourned at 10:30 a.m. on January 26, 2022.
Notation Vote By notation vote completed on January 5, 2022, the Committee
unanimously approved the minutes of the December meeting.
_______________________
James A. Clouse Secretary
1. The Federal Open Market Committee is referenced as the "FOMC". Return to text
2. Attended Tuesday's session only. Return to text
"""

SAMPLE_FOMC_STATEMENT = """\
## Document Body
January 26, 2022
## Federal Reserve issues FOMC statement
For release at 2:00 p.m. EST Share
Share

The labor market continues to strengthen and inflation remains elevated,
reflecting supply and demand imbalances related to the pandemic.

Voting for the FOMC monetary policy action were: Jerome H. Powell, Chair;
John C. Williams, Vice Chair.
"""

SAMPLE_FOMC_SEP = """\
## Summary of Economic Projections

The following table shows the projections of FOMC participants.

## Figure 1. GDP Projections
Accessible version of figure 1
GDP is projected to grow at a moderate pace. Return to figure 1

## Table A. Economic Projections
Inflation is projected to decline toward the 2 percent objective. Return to table A

Return to top
"""

SAMPLE_FOMC_PRESS_CONFERENCE = """\
January 30, 2019 Chairman Powell’s Press Conference FINAL
Transcript of Chairman Powell’s Press Conference
January 30, 2019
CHAIRMAN POWELL. Good afternoon, everyone, and welcome.
The economy is in a good place.
Page 1 of 24

January 30, 2019 Chairman Powell’s Press Conference FINAL
We always emphasize that our policies are data dependent.
Short-term
2 of 24
January 30, 2019 Chairman Powell’s Press Conference FINAL
money market rates matter for transmission.
ROBIN HARDING. Robin Harding, Financial Times. Why are you changing your stance?
Would you say risks are rising?
CHAIRMAN POWELL. We remain data dependent and will act as appropriate.
"""


# ---------------------------------------------------------------------------
# Tests: Fed speeches pipeline
# ---------------------------------------------------------------------------

class TestSpeechPipeline(unittest.TestCase):

    def setUp(self):
        self.result = apply_pipeline(SAMPLE_SPEECH, PIPELINE_SPEECHES)

    def test_output_nonempty(self):
        self.assertGreater(len(self.result.strip()), 0)

    def test_date_header_removed(self):
        self.assertNotIn("August 23, 2024", self.result)

    def test_speaker_header_removed(self):
        self.assertNotIn("Chair Jerome H. Powell", self.result)

    def test_venue_header_removed(self):
        self.assertNotIn("At the Economic Symposium", self.result)

    def test_footnotes_removed(self):
        self.assertNotIn("Return to text", self.result)
        self.assertNotIn("See Federal Reserve Board", self.result)

    def test_figure_references_removed(self):
        self.assertNotIn("( figure 1 )", self.result)

    def test_substantive_content_preserved(self):
        self.assertIn("time has come for policy to adjust", self.result)

    def test_idempotent(self):
        second_pass = apply_pipeline(self.result, PIPELINE_SPEECHES)
        self.assertEqual(self.result, second_pass)


# ---------------------------------------------------------------------------
# Tests: FOMC structured minutes pipeline
# ---------------------------------------------------------------------------

class TestFomcMinutesPipeline(unittest.TestCase):

    def setUp(self):
        self.result = apply_pipeline(SAMPLE_FOMC_MINUTES, PIPELINE_FOMC_STRUCTURED_MINUTES)

    def test_output_nonempty(self):
        self.assertGreater(len(self.result.strip()), 0)

    def test_attendee_preamble_removed(self):
        self.assertNotIn("PRESENT:", self.result)

    def test_duplicate_statement_removed(self):
        # Covers both "vote encompassed" and "vote also encompassed" variants
        self.assertNotIn("vote also encompassed approval", self.result)
        self.assertNotIn("duplicated policy statement text", self.result)

    def test_voting_records_removed(self):
        self.assertNotIn("Voting for this action", self.result)
        self.assertNotIn("Voting against this action", self.result)

    def test_tail_boilerplate_removed(self):
        self.assertNotIn("Board of Governors voted unanimously", self.result)
        self.assertNotIn("It was agreed that the next meeting", self.result)
        self.assertNotIn("Notation Vote", self.result)
        self.assertNotIn("James A. Clouse Secretary", self.result)
        self.assertNotIn("Return to text", self.result)

    def test_substantive_content_preserved(self):
        self.assertIn("information reviewed for this meeting", self.result)

    def test_idempotent(self):
        second_pass = apply_pipeline(self.result, PIPELINE_FOMC_STRUCTURED_MINUTES)
        self.assertEqual(self.result, second_pass)


# ---------------------------------------------------------------------------
# Tests: FOMC structured policy statements pipeline
# ---------------------------------------------------------------------------

class TestFomcStatementPipeline(unittest.TestCase):

    def setUp(self):
        self.result = apply_pipeline(SAMPLE_FOMC_STATEMENT, PIPELINE_FOMC_STRUCTURED_STATEMENTS)

    def test_output_nonempty(self):
        self.assertGreater(len(self.result.strip()), 0)

    def test_for_immediate_release_removed(self):
        self.assertNotIn("For immediate release", self.result)

    def test_release_time_line_removed(self):
        self.assertNotIn("For release at", self.result)

    def test_share_boilerplate_removed(self):
        # "Share" on its own line is a navigation button artifact
        lines = self.result.splitlines()
        self.assertNotIn("Share", [l.strip() for l in lines])

    def test_fed_issues_header_removed(self):
        self.assertNotIn("Federal Reserve issues FOMC statement", self.result)

    def test_voting_records_removed(self):
        self.assertNotIn("Voting for the FOMC", self.result)

    def test_substantive_content_preserved(self):
        self.assertIn("labor market continues to strengthen", self.result)

    def test_idempotent(self):
        second_pass = apply_pipeline(self.result, PIPELINE_FOMC_STRUCTURED_STATEMENTS)
        self.assertEqual(self.result, second_pass)


# ---------------------------------------------------------------------------
# Tests: FOMC structured SEP pipeline
# ---------------------------------------------------------------------------

class TestFomcSepPipeline(unittest.TestCase):

    def setUp(self):
        self.result = apply_pipeline(SAMPLE_FOMC_SEP, PIPELINE_FOMC_STRUCTURED_SEP)

    def test_output_nonempty(self):
        self.assertGreater(len(self.result.strip()), 0)

    def test_figure_headings_removed(self):
        self.assertNotIn("## Figure 1.", self.result)

    def test_table_headings_removed(self):
        self.assertNotIn("## Table A.", self.result)

    def test_accessible_version_removed(self):
        self.assertNotIn("Accessible version of figure", self.result)

    def test_return_to_figure_removed(self):
        self.assertNotIn("Return to figure", self.result)

    def test_return_to_table_removed(self):
        self.assertNotIn("Return to table", self.result)

    def test_return_to_top_removed(self):
        self.assertNotIn("Return to top", self.result)

    def test_substantive_content_preserved(self):
        self.assertIn("GDP is projected to grow", self.result)

    def test_idempotent(self):
        second_pass = apply_pipeline(self.result, PIPELINE_FOMC_STRUCTURED_SEP)
        self.assertEqual(self.result, second_pass)


# ---------------------------------------------------------------------------
# Tests: FOMC press conference pipeline
# ---------------------------------------------------------------------------

class TestFomcPressConferencePipeline(unittest.TestCase):

    def setUp(self):
        self.result = apply_pipeline(SAMPLE_FOMC_PRESS_CONFERENCE, PIPELINE_FOMC_PRESS_CONFERENCES)

    def test_header_removed(self):
        self.assertNotIn("Press Conference FINAL", self.result)
        self.assertNotIn("Transcript of Chairman Powell’s Press Conference", self.result)

    def test_page_numbers_removed(self):
        self.assertNotIn("Page 1 of 24", self.result)
        self.assertNotIn("2 of 24", self.result)

    def test_starts_at_first_chair_line(self):
        self.assertTrue(self.result.startswith("CHAIRMAN POWELL. Good afternoon"))

    def test_substantive_content_preserved(self):
        self.assertIn("The economy is in a good place", self.result)
        self.assertIn("policies are data dependent", self.result)

    def test_reporter_question_removed_but_chair_answer_kept(self):
        self.assertNotIn("ROBIN HARDING.", self.result)
        self.assertNotIn("Why are you changing your stance?", self.result)
        self.assertIn("CHAIRMAN POWELL. We remain data dependent", self.result)

    def test_paragraphs_are_reflowed_for_readability(self):
        self.assertIn(
            "CHAIRMAN POWELL. Good afternoon, everyone, and welcome. The economy is in a good place.",
            self.result,
        )
        self.assertNotIn("welcome.\nThe economy is in a good place.", self.result)
        self.assertIn(
            "good place.\n\nWe always emphasize that our policies are data dependent.",
            self.result,
        )
        self.assertIn("Short-term money market rates matter for transmission.", self.result)
        self.assertNotIn("Short-term\n\nmoney market rates matter", self.result)

    def test_idempotent(self):
        second_pass = apply_pipeline(self.result, PIPELINE_FOMC_PRESS_CONFERENCES)
        self.assertEqual(self.result, second_pass)


# ---------------------------------------------------------------------------
# Tests: cleaners module structure
# ---------------------------------------------------------------------------

class TestCleanersModule(unittest.TestCase):
    """Verify that the cleaners module exposes all expected pipelines."""

    def _check_pipeline(self, pipeline):
        self.assertIsInstance(pipeline, list)
        self.assertGreater(len(pipeline), 0)
        for name, fn in pipeline:
            self.assertIsInstance(name, str)
            self.assertTrue(callable(fn))

    def test_pipeline_speeches(self):
        self._check_pipeline(PIPELINE_SPEECHES)

    def test_pipeline_fomc_press_conferences(self):
        self._check_pipeline(PIPELINE_FOMC_PRESS_CONFERENCES)

    def test_pipeline_fomc_structured_minutes(self):
        self._check_pipeline(PIPELINE_FOMC_STRUCTURED_MINUTES)

    def test_pipeline_fomc_structured_statements(self):
        self._check_pipeline(PIPELINE_FOMC_STRUCTURED_STATEMENTS)

    def test_pipeline_fomc_structured_sep(self):
        self._check_pipeline(PIPELINE_FOMC_STRUCTURED_SEP)


# ---------------------------------------------------------------------------
# Tests: fetch scripts importable (dependencies installed, no syntax errors)
# ---------------------------------------------------------------------------

class TestFetchScriptsImportable(unittest.TestCase):
    """Load fetch scripts via importlib to avoid package-path issues."""

    def _load(self, rel_path: str):
        path = ROOT / rel_path
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    def test_fetch_fed_speeches_importable(self):
        mod = self._load("src/scraping/fetch_fed_speeches.py")
        self.assertTrue(hasattr(mod, "run"), "fetch_fed_speeches.py is missing a `run` function")

    def test_fetch_fomc_minutes_statements_importable(self):
        mod = self._load("src/scraping/fetch_fomc_minutes_statements.py")
        self.assertTrue(hasattr(mod, "run"), "fetch_fomc_minutes_statements.py is missing a `run` function")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
