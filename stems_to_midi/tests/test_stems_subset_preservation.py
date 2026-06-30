"""
Tests for the load-merge step in stems_to_midi_cli.py (2026-06-30).

When the CLI is invoked with `--stems <subset>` (e.g. `--stems snare`),
the orchestration function should preserve the other stems' data in
the sidecar — only re-process the requested stems. This is the fix
for the bug where `--stems snare` erased kick/toms/hihat/cymbals from
both the sidecar JSON and the resulting MIDI.

The merge logic lives in:
  - `_deserialize_sidecar_stems_for_merge` (pure helper):
      Takes a sidecar data dict (output of `load_analysis_sidecar`)
      and a list of stems to preserve. Returns two dicts in the
      shape `save_analysis_sidecar` expects:
        - `midi_events_by_stem`: KEPT events reconstructed as MIDI
        - `analysis_by_stem`: existing `events_pga` re-wrapped as
          `pga_onset_data` for the save pipeline.

These tests pin:
  1. Empty sidecar → empty merge output (no error, current behavior preserved).
  2. KEPT events become MIDI events; FILTERED events do not.
  3. MIDI event fields (note, velocity, duration, hihat_state) round-trip
     correctly from the existing sidecar's KEPT events.
  4. Only the requested stems appear in the output (filter by stem list).
  5. Duration is clamped to `config.midi.max_note_duration` (same as the
     live pipeline).
"""

import os
import sys
from pathlib import Path

import pytest

# Match the existing tests/ subdirectory sys.path trick.
_TEST_DIR = Path(__file__).resolve().parent
_PKG_PARENT = _TEST_DIR.parent.parent
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

# Import the helper from stems_to_midi_cli. The CLI imports its
# heavyweight deps lazily, so importing just the helper function
# shouldn't pull librosa / soundfile at module load time.
from stems_to_midi_cli import _deserialize_sidecar_stems_for_merge  # noqa: E402


# --- Helpers --------------------------------------------------------------


def _make_kept_event(
    time: float = 1.0,
    note: int = 38,
    midi_velocity: int = 100,
    duration_ms: float = 150.0,
    hihat_state: str = None,
    status: str = 'KEPT',
    **extras,
) -> dict:
    """Build a sidecar-shaped KEPT event dict."""
    ev = {
        'time': time,
        'status': status,
        'note': note,
        'midi_velocity': midi_velocity,
        'duration_ms': duration_ms,
    }
    if hihat_state is not None:
        ev['hihat_state'] = hihat_state
    ev.update(extras)
    return ev


def _make_sidecar(stems_data: dict) -> dict:
    """Build a sidecar-shaped top-level dict."""
    return {
        'version': '3.0',
        'tempo_bpm': 120.0,
        'stems': stems_data,
    }


# --- Tests ----------------------------------------------------------------


class TestDeserializeSidecarStemsForMerge:
    """Tests for the load-merge helper used by --stems <subset>."""

    def test_empty_sidecar_returns_empty_dicts(self):
        """No existing sidecar → empty merge output. The CLI handles
        this case by skipping the merge entirely; the helper itself
        should still be safe to call on an empty dict."""
        sidecar = _make_sidecar({})
        midi_events, analysis = _deserialize_sidecar_stems_for_merge(
            sidecar, stems_to_preserve=['kick'],
        )
        assert midi_events == {}
        assert analysis == {}

    def test_preserves_ke_events_only(self):
        """The helper filters to KEPT events. FILTERED events stay
        in events_pga (so the sidecar retains them for diagnostic
        display) but are NOT included in the reconstructed MIDI
        events list — the MIDI only carries notes for KEPT events."""
        sidecar = _make_sidecar({
            'snare': {
                'events_pga': [
                    _make_kept_event(time=0.5, note=38, status='KEPT'),
                    _make_kept_event(time=1.0, note=37, status='KEPT'),
                    _make_kept_event(time=1.5, note=39, status='FILTERED'),
                    _make_kept_event(time=2.0, note=38, status='FILTERED'),
                ],
            },
        })
        midi_events, _analysis = _deserialize_sidecar_stems_for_merge(
            sidecar, stems_to_preserve=['snare'],
        )
        assert 'snare' in midi_events
        assert len(midi_events['snare']) == 2  # only KEPT
        assert midi_events['snare'][0]['time'] == 0.5
        assert midi_events['snare'][1]['time'] == 1.0

    def test_extracts_midi_event_fields_correctly(self):
        """The MIDI event dict has exactly the fields the rebuild +
        create_midi_file pipeline expects: time, note, velocity
        (mapped from midi_velocity), duration (mapped from
        duration_ms / 1000), hihat_state (if present)."""
        sidecar = _make_sidecar({
            'snare': {
                'events_pga': [
                    _make_kept_event(
                        time=0.5, note=38, midi_velocity=112,
                        duration_ms=200.0,
                    ),
                ],
            },
        })
        midi_events, _analysis = _deserialize_sidecar_stems_for_merge(
            sidecar, stems_to_preserve=['snare'],
        )
        ev = midi_events['snare'][0]
        assert ev['time'] == 0.5
        assert ev['note'] == 38
        assert ev['velocity'] == 112
        assert abs(ev['duration'] - 0.2) < 1e-9
        # hihat_state is only set for hihat; should be absent for snare
        assert 'hihat_state' not in ev

    def test_hihat_state_preserved(self):
        """For hihat events, hihat_state ('open' or 'closed') must
        be carried through so the create_midi_file loop can pick
        the right note (46 vs 42)."""
        sidecar = _make_sidecar({
            'hihat': {
                'events_pga': [
                    _make_kept_event(time=1.0, note=46, hihat_state='open'),
                    _make_kept_event(time=2.0, note=42, hihat_state='closed'),
                ],
            },
        })
        midi_events, _analysis = _deserialize_sidecar_stems_for_merge(
            sidecar, stems_to_preserve=['hihat'],
        )
        states = [e['hihat_state'] for e in midi_events['hihat']]
        assert states == ['open', 'closed']
        notes = [e['note'] for e in midi_events['hihat']]
        assert notes == [46, 42]

    def test_stems_to_preserve_filters_correctly(self):
        """Only stems in stems_to_preserve appear in the output.
        Other stems (e.g. snare when only hihat is requested) are
        dropped — they will be re-detected in the current run, no
        need to preserve them from the old sidecar."""
        sidecar = _make_sidecar({
            'kick': {'events_pga': [_make_kept_event(time=1.0, note=36)]},
            'snare': {'events_pga': [_make_kept_event(time=1.0, note=38)]},
            'hihat': {'events_pga': [_make_kept_event(time=1.0, note=42)]},
        })
        midi_events, analysis = _deserialize_sidecar_stems_for_merge(
            sidecar, stems_to_preserve=['kick', 'hihat'],
        )
        assert set(midi_events.keys()) == {'kick', 'hihat'}
        assert set(analysis.keys()) == {'kick', 'hihat'}
        # And the analysis for kick includes the full events_pga
        assert len(analysis['kick']['pga_onset_data']) == 1
        assert analysis['kick']['pga_onset_data'][0]['note'] == 36

    def test_clamps_duration_to_max_note_duration(self):
        """A KEPT event with duration_ms > max_note_duration should
        be clamped. Mirrors the behavior of the live pipeline in
        process_percentile_gated.py."""
        sidecar = _make_sidecar({
            'cymbals': {
                'events_pga': [
                    _make_kept_event(
                        time=1.0, note=49, midi_velocity=100,
                        duration_ms=5000.0,  # way too long
                    ),
                ],
            },
        })
        midi_events, _analysis = _deserialize_sidecar_stems_for_merge(
            sidecar, stems_to_preserve=['cymbals'],
            config={'midi': {'max_note_duration': 0.5}},
        )
        ev = midi_events['cymbals'][0]
        assert ev['duration'] == 0.5  # clamped from 5.0

    def test_clamps_duration_per_stem_max_note_duration(self):
        """When the per-stem max_note_duration is set (cymbals has
        2.0 in the live yaml), it wins over the global default."""
        sidecar = _make_sidecar({
            'cymbals': {
                'events_pga': [
                    _make_kept_event(
                        time=1.0, note=49, midi_velocity=100,
                        duration_ms=5000.0,
                    ),
                ],
            },
        })
        midi_events, _analysis = _deserialize_sidecar_stems_for_merge(
            sidecar, stems_to_preserve=['cymbals'],
            config={
                'midi': {'max_note_duration': 0.5},
                'cymbals': {'max_note_duration': 2.0},
            },
        )
        ev = midi_events['cymbals'][0]
        assert ev['duration'] == 2.0  # per-stem wins

    def test_preserves_extras_in_pga_onset_data(self):
        """The full events_pga list (KEPT + FILTERED + all
        per-event fields like stereo_width, pitch_hz, etc.) must
        pass through to pga_onset_data unchanged. The re-serialize
        step in save_analysis_sidecar is what persists these
        fields back to JSON."""
        kept = _make_kept_event(
            time=1.0, note=38, midi_velocity=100, duration_ms=150.0,
            stereo_width=0.123, pitch_hz=72.5, classification=0,
        )
        filtered = _make_kept_event(
            time=1.5, note=None, status='FILTERED',
            filter_reason='below pga_min_prominence',
        )
        sidecar = _make_sidecar({
            'snare': {'events_pga': [kept, filtered]},
        })
        _midi_events, analysis = _deserialize_sidecar_stems_for_merge(
            sidecar, stems_to_preserve=['snare'],
        )
        pga_data = analysis['snare']['pga_onset_data']
        assert len(pga_data) == 2  # both KEPT and FILTERED preserved
        assert pga_data[0]['stereo_width'] == 0.123
        assert pga_data[0]['pitch_hz'] == 72.5
        assert pga_data[1]['status'] == 'FILTERED'
        assert 'below pga_min_prominence' in pga_data[1]['filter_reason']

    def test_no_config_uses_safe_defaults(self):
        """When config is None or missing max_note_duration, the
        helper uses the live pipeline's default (0.5s) instead of
        crashing. Mirrors process_percentile_gated.py:218-220 which
        uses ``config.get('midi', {}).get('max_note_duration', 0.5)``
        as the global default."""
        sidecar = _make_sidecar({
            'snare': {
                'events_pga': [
                    _make_kept_event(time=1.0, note=38, midi_velocity=100,
                                     duration_ms=2000.0),
                ],
            },
        })
        # No config at all
        midi_events, _analysis = _deserialize_sidecar_stems_for_merge(
            sidecar, stems_to_preserve=['snare'],
        )
        # duration_ms=2000 → duration=2.0; default max=0.5 clamps to 0.5
        assert midi_events['snare'][0]['duration'] == 0.5

    def test_missing_duration_ms_defaults_to_default_note_duration(self):
        """Older sidecars may have events without duration_ms
        (the field was added 2026-06-19). For those, fall back to
        config.audio.default_note_duration (0.1 by default)."""
        sidecar = _make_sidecar({
            'snare': {
                'events_pga': [
                    {'time': 1.0, 'status': 'KEPT', 'note': 38,
                     'midi_velocity': 100},  # no duration_ms
                ],
            },
        })
        midi_events, _analysis = _deserialize_sidecar_stems_for_merge(
            sidecar, stems_to_preserve=['snare'],
            config={'audio': {'default_note_duration': 0.15}},
        )
        assert abs(midi_events['snare'][0]['duration'] - 0.15) < 1e-9