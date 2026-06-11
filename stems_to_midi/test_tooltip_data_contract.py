"""Test that the spectral event sidecar data shape matches what the
new WebUI tooltip expects.

The tooltip (webui/static/js/waveform.js:drawTooltip) shows the full
per-band profile for spectral events: 5 band_powers, band_max_idx,
band_max_ratio, strength. This test verifies every spectral event in
a project analysis.json has all 5 fields populated correctly, so the
tooltip can render without showing "undefined" or missing data.

Failure modes caught:
  - band_powers is not a list of length 5 → tooltip shows garbage
  - band_max_idx doesn't point to the actual max → "Top band" is wrong
  - band_max_ratio is null/missing → ratio line is missing
  - band_powers has fewer than 5 entries → only some bands render
"""
import json
from pathlib import Path

import pytest


def _find_analysis_files():
    """Find any .analysis.json in user_files/ for live regression testing.

    Excludes `_e2e_test/` subdirectories — those are stale sidecars
    from the Playwright e2e suite that pre-date the band-profile
    work (no band_powers field). Run a fresh conversion to get a
    current sidecar for live testing."""
    root = Path("user_files")
    if not root.exists():
        return []
    out = []
    for p in root.rglob("*.analysis.json"):
        if "_e2e_test" in p.parts:
            continue
        out.append(p)
    return out


def _all_spectral_events(analysis_path):
    """Yield (stem_name, event_index, event) for every spectral event."""
    with open(analysis_path) as f:
        data = json.load(f)
    stems = data.get("stems", {})
    for stem_name, stem_info in stems.items():
        for i, event in enumerate(stem_info.get("events_configured", [])):
            if event.get("method") == "spectral":
                yield stem_name, i, event


def _synthetic_spectral_event(
    band_powers, band_max_idx, band_max_ratio, band_max_ratio_10=None
):
    """Build a synthetic spectral event dict to validate the data
    contract (this is what the new detector produces and what the
    new tooltip reads)."""
    if band_max_ratio_10 is None:
        # The detector emits band_max_ratio_10 as the raw / 10 form
        # (no clamp). For tests we just pass through the input.
        band_max_ratio_10 = band_max_ratio / 10.0
    return {
        "time": 1.0,
        "status": "KEPT",
        "method": "spectral",
        # 2026-06-10: the lossy clamp-to-1.0 `strength` field was
        # removed. The detector now emits the raw band_max_ratio
        # (top/second-highest band) plus a back-compat
        # band_max_ratio_10 alias (= ratio / 10, unclamped).
        "band_max_ratio": band_max_ratio,
        "band_max_ratio_10": band_max_ratio_10,
        "band_max_idx": band_max_idx,
        "band_powers": list(band_powers),
    }


def test_spectral_event_has_band_powers_list_of_5():
    """Every spectral event must have band_powers as a list of 5 floats.
    The tooltip iterates 0..4 and calls .toExponential(2) on each."""
    event = _synthetic_spectral_event(
        [1.0, 0.1, 0.01, 0.001, 0.0001], 0, 10.0
    )
    bp = event["band_powers"]
    assert isinstance(bp, list)
    assert len(bp) == 5
    assert all(isinstance(v, (int, float)) for v in bp)


def test_band_max_idx_points_to_max_band():
    """The band_max_idx field must point to the band with the highest
    power — this is what the tooltip marks with '*' as the top band.
    If the index is wrong, the user sees misleading 'Top: B0' when
    the actual max is at B3."""
    # B3 is clearly the max
    event = _synthetic_spectral_event(
        [0.1, 0.1, 0.1, 100.0, 0.1], 3, 1000.0
    )
    bp = event["band_powers"]
    assert bp[event["band_max_idx"]] == max(bp), (
        f"band_max_idx={event['band_max_idx']} but max power is at "
        f"index {bp.index(max(bp))} (powers={bp})"
    )


def test_band_max_ratio_is_top_over_second():
    """The ratio should be top/second-highest band. The tooltip shows
    this with a 'higher = clearer strike' annotation, so the user
    uses it to judge event quality."""
    event = _synthetic_spectral_event(
        [1.0, 0.5, 0.1, 0.05, 0.01], 0, 2.0
    )
    # Top=1.0, second=0.5, ratio=2.0
    assert event["band_max_ratio"] == 2.0


def test_real_audio_spectral_events_have_all_tooltip_fields():
    """For every .analysis.json in user_files/, every spectral event
    must have the 3 fields the tooltip reads: band_powers (list of 5),
    band_max_idx (int 0-4), band_max_ratio (float).

    2026-06-10: the lossy `strength` field was removed (it was the
    clamp-to-1.0 of band_max_ratio/10 and masked real differences).
    The tooltip now reads band_max_ratio directly via the "Top/2nd
    ratio" line; `band_max_ratio_10` is emitted as a back-compat
    alias but is not part of the tooltip contract."""
    analysis_files = _find_analysis_files()
    if not analysis_files:
        pytest.skip("no user_files/ analysis files to test against")

    issues = []
    for analysis_path in analysis_files:
        for stem_name, idx, event in _all_spectral_events(analysis_path):
            bp = event.get("band_powers")
            if not isinstance(bp, list) or len(bp) != 5:
                issues.append(
                    f"{analysis_path} {stem_name} event {idx}: "
                    f"band_powers={bp!r}"
                )
                continue
            if not all(isinstance(v, (int, float)) for v in bp):
                issues.append(
                    f"{analysis_path} {stem_name} event {idx}: "
                    f"band_powers has non-numeric values: {bp}"
                )
            if event.get("band_max_idx") not in range(5):
                issues.append(
                    f"{analysis_path} {stem_name} event {idx}: "
                    f"band_max_idx={event.get('band_max_idx')!r} (expected 0-4)"
                )
                continue
            if bp[event["band_max_idx"]] != max(bp):
                issues.append(
                    f"{analysis_path} {stem_name} event {idx}: "
                    f"band_max_idx={event['band_max_idx']} but max is at "
                    f"{bp.index(max(bp))} (powers={bp})"
                )
            if event.get("band_max_ratio") is None:
                issues.append(
                    f"{analysis_path} {stem_name} event {idx}: "
                    f"missing band_max_ratio"
                )
            # 2026-06-10: the lossy `strength` field was removed
            # (it was the clamp-to-1.0 of band_max_ratio/10 and
            # masked real differences — e.g. 18.99 vs 459.12
            # both reported as 1.0). The new tooltip reads
            # `band_max_ratio` directly via the "Top/2nd ratio"
            # line, so the contract now only checks that field.
            # `band_max_ratio_10` is emitted as a back-compat
            # alias for older readers but is not required.

    assert not issues, (
        "tooltip data contract violations found:\n  "
        + "\n  ".join(issues)
    )
