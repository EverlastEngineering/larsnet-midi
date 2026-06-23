"""
WebUI JS tests for PGA filter functions (2026-06-22: minimal
recreation after legacy-code-removal pass deleted the previous
file). Each filter added via .github/skills/add-filter appends a
test class here. The tests verify the JS function exists and is
called from applyTuningFilter.

Previous content (test_snap_delta_mask.py before legacy-code
removal) covered:
  1. Slider HTML structure
  2. updateThresholdDisplay rounding
  3. Filter chain order
  4. Spectral event filter exemption
  5. applyPgaFilterFunctions integration
  6. The server-side rebuild_core._apply_show_only_snap_events
     and rebuild_core._apply_band_max_ratio_max functions existed
     with callable signatures

The 2026-06-22 cleanup deleted those snap-mask / band_max_ratio
filters along with the rest of the legacy non-PGA filter chain.
The file is kept (with this minimal header) as the target for
the add-filter script's JS test scaffolding.
"""
import re
from pathlib import Path

import pytest


WEBUI_JS_PATH = (
    Path(__file__).resolve().parent
    / 'static'
    / 'js'
    / 'threshold-tuning.js'
)


@pytest.fixture
def threshold_tuning_js_text():
    """Read the WebUI threshold-tuning.js source as text."""
    return WEBUI_JS_PATH.read_text()


class TestPgaMinEnvelopeValue:
    """applyPgaMinEnvelopeValue — added by .github/skills/add-filter.

    Skeleton. Fill in real test cases — the boilerplate
    below mirrors TestPgaFilterFunctions and
    TestAttackRiseFilter.
    """

    def test_function_exists(self, threshold_tuning_js_text):
        m = re.search(
            r"function\s+applyPgaMinEnvelopeValue\s*\(\s*events\s*,\s*threshold",
            threshold_tuning_js_text,
        )
        assert m is not None, (
            f"expected `function applyPgaMinEnvelopeValue(events, threshold, "
            f"disabledIds)` in threshold-tuning.js"
        )

    def test_filter_wired_into_apply_tuning_filter(self, threshold_tuning_js_text):
        """applyPgaMinEnvelopeValue must be called from applyTuningFilter
        for the stems in applies_to_stems."""
        m = re.search(
            'function\\s*applyTuningFilter\\s*\\(\\s*\\)\\s*\\{(.*?)\\n\\}\\n',
            threshold_tuning_js_text,
            re.DOTALL,
        )
        assert m is not None
        body = m.group(1)
        assert 'applyPgaMinEnvelopeValue' in body, (
            f"applyTuningFilter must call applyPgaMinEnvelopeValue for "
            f"the stems in applies_to_stems (['toms', 'snare', 'hihat', 'kick', 'cymbals'])."
        )
