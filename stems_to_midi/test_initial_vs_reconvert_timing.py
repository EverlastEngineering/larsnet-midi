"""
End-to-end regression test for bug E — MIDI timing parity.

Bug E: the initial conversion produced MIDI that played too fast,
while 'Save & Reconvert' produced correct MIDI. The two paths are
required to produce identical MIDI event times when run on the same
stems and config.

The two paths under test:

  1. Initial conversion (stems_to_midi_cli._process_stems_to_midi):
     - Detect onsets in audio
     - Save analysis sidecar (Detection Output Contract v3)
     - Load sidecar and call rebuild_events_from_analysis()
     - Create MIDI from rebuilt events

  2. Reconvert (stems_to_midi.rebuild_shell.rebuild_midi_for_project):
     - Load sidecar
     - Call rebuild_events_from_analysis() (no overrides, no threshold
       change)
     - Create MIDI from rebuilt events

After commit 15a5461 both paths route through the same rebuild
function, but the test guards against any future regression that
introduces a parallel code path with different timing. The test
asserts:

  - Both paths produce the same number of MIDI events
  - Event times match within 1ms
  - Event notes match
  - Event velocities match

A tolerance of 1ms is generous — the underlying detection has
hop_length=512 / sr=44100 ≈ 11.6ms resolution, but mid-event time
arithmetic should round identically in both paths.
"""

import json
import shutil
import tempfile
import wave
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import yaml

from stems_to_midi.midi import read_midi_notes


# ============================================================================
# Helpers
# ============================================================================


def _make_synthetic_audio_with_onsets(
    onset_times: List[float],
    sr: int = 22050,
    duration: float = 4.0,
) -> np.ndarray:
    """
    Build a mono audio signal with sharp transients at the given times.

    Uses a ~10ms attack followed by a longer ringing decay so the
    energy envelope has a clear peak above the floor. The amplitude
    is set to fill most of the int16 range so the energy-based
    detection's min_absolute_energy gate (default 0.01) is easily
    cleared.
    """
    audio = np.zeros(int(sr * duration), dtype=np.float32)
    for t in onset_times:
        idx = int(t * sr)
        # 2000-sample ringing decay (90ms at 22050)
        n = 2000
        # Envelope: instant attack, exponential decay
        envelope = np.exp(-np.linspace(0, 6, n))
        # Sum two tones (kick: low body + click; snare/toms/etc: mid body)
        tone1 = np.sin(2 * np.pi * 80 * np.linspace(0, n / sr, n))   # 80Hz body
        tone2 = np.sin(2 * np.pi * 400 * np.linspace(0, n / sr, n))  # 400Hz click
        transient = envelope * (0.6 * tone1 + 0.4 * tone2)
        end = min(idx + n, len(audio))
        actual_n = end - idx
        if actual_n <= 0:
            continue
        audio[idx:end] = transient[:actual_n].astype(np.float32) * 0.95
    return audio


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    """Write a mono float32 array as a 16-bit PCM WAV file."""
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def _write_stems(
    stems_dir: Path,
    project_name: str,
    onset_times_by_stem: Dict[str, List[float]],
) -> None:
    """Write one stereo stem file per stem type with the given onset times."""
    stems_dir.mkdir(parents=True, exist_ok=True)
    sr = 22050
    duration = 4.0
    for stem_type, times in onset_times_by_stem.items():
        audio = _make_synthetic_audio_with_onsets(times, sr=sr, duration=duration)
        # Make it "stereo" by duplicating the mono channel so the pan_confidence
        # code path doesn't error.
        stereo = np.stack([audio, audio * 0.95], axis=-1)
        # Save stereo as 16-bit WAV
        path = stems_dir / f"{project_name}-{stem_type}.wav"
        # wave module only writes mono directly, so we go through soundfile
        # if available, else save 2 channels via a simple 2-channel writer.
        try:
            import soundfile as sf
            sf.write(str(path), stereo, sr, subtype='PCM_16')
        except ImportError:
            # Fall back: just write mono (the pipeline should still work,
            # the parity check is about timing not stereo features).
            _write_wav(path, audio, sr)


def _minimal_config() -> dict:
    """Minimal config that won't trip over missing keys in the pipeline."""
    return {
        'midi': {
            'default_tempo': 120.0,
            'max_note_duration': 0.5,
            'min_velocity': 80,
            'max_velocity': 110,
        },
        'audio': {'default_note_duration': 0.1},
        'onset_detection': {
            'hop_length': 256,
            'threshold': 0.3,
            'delta': 0.1,
            'wait': 3,
        },
        'filtering': {
            'reverb_continuation_attack_threshold': 0.4,
        },
        # Per-stem sections with the spectral features the pipeline needs.
        'kick': {
            'geomean_threshold': 50.0,
            'onset_threshold': 0.3, 'onset_delta': 0.1, 'onset_wait': 3,
            'fundamental_freq_min': 30, 'fundamental_freq_max': 80,
            'body_freq_min': 100, 'body_freq_max': 300,
            'attack_freq_min': 2000, 'attack_freq_max': 5000,
        },
        'snare': {
            'geomean_threshold': 40.0,
            'onset_threshold': 0.3, 'onset_delta': 0.1, 'onset_wait': 3,
            'low_freq_min': 100, 'low_freq_max': 300,
            'body_freq_min': 200, 'body_freq_max': 800,
            'wire_freq_min': 4000, 'wire_freq_max': 8000,
            'min_sustain_ms': 25,
        },
        'toms': {
            'geomean_threshold': 80.0,
            'onset_threshold': 0.3, 'onset_delta': 0.1, 'onset_wait': 3,
            'fundamental_freq_min': 80, 'fundamental_freq_max': 300,
            'body_freq_min': 2000, 'body_freq_max': 6000,
            'min_sustain_ms': 25,
        },
        'hihat': {
            'geomean_threshold': 8.0,
            'onset_threshold': 0.3, 'onset_delta': 0.1, 'onset_wait': 3,
            'body_freq_min': 500, 'body_freq_max': 4000,
            'sizzle_freq_min': 8000, 'sizzle_freq_max': 16000,
            'min_sustain_ms': 25,
            'open_sustain_ms': 100.0,
        },
        'cymbals': {
            'geomean_threshold': 100.0,
            'onset_threshold': 0.3, 'onset_delta': 0.1, 'onset_wait': 3,
            'body_freq_min': 500, 'body_freq_max': 4000,
            'brilliance_freq_min': 8000, 'brilliance_freq_max': 16000,
            'min_sustain_ms': 150,
        },
        'drum_mapping': {},
        'learning_mode': {'enabled': False},
    }


def _read_midi_event_times(midi_path: Path, target_note: int = None) -> List[float]:
    """Read all note-on times from a MIDI file. If target_note is None, return all."""
    import mido
    mid = mido.MidiFile(str(midi_path))
    tpb = mid.ticks_per_beat
    tempo = 500000  # default 120 BPM
    times = []
    for track in mid.tracks:
        current_time = 0.0
        for msg in track:
            current_time += mido.tick2second(msg.time, tpb, tempo)
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'note_on' and msg.velocity > 0:
                if target_note is None or msg.note == target_note:
                    times.append(current_time)
    return sorted(times)


def _read_all_midi_events(midi_path: Path) -> List[Dict]:
    """Read all note-on events from a MIDI file (time, note, velocity)."""
    import mido
    mid = mido.MidiFile(str(midi_path))
    tpb = mid.ticks_per_beat
    tempo = 500000
    events = []
    for track in mid.tracks:
        current_time = 0.0
        for msg in track:
            current_time += mido.tick2second(msg.time, tpb, tempo)
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'note_on' and msg.velocity > 0:
                events.append({
                    'time': current_time,
                    'note': msg.note,
                    'velocity': msg.velocity,
                })
    events.sort(key=lambda e: (e['time'], e['note']))
    return events


# ============================================================================
# Bug E regression test
# ============================================================================


@pytest.fixture
def funk_like_project(tmp_path):
    """
    Build a temp project mimicking the funk test ground truth:
    5 stems (kick, snare, toms, hihat, cymbals), each with a few synthetic
    onsets, plus a midiconfig.yaml.
    """
    project_dir = tmp_path / "1 - test_funk_80_beat_4-4_4"
    stems_dir = project_dir / "stems"
    midi_dir = project_dir / "midi"
    project_dir.mkdir(parents=True)
    midi_dir.mkdir(parents=True)

    onset_times_by_stem = {
        'kick':   [0.50, 1.50, 2.50, 3.50],
        'snare':  [1.00, 2.00, 3.00],
        'toms':   [0.75, 1.75, 2.75],
        'hihat':  [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75],
        'cymbals':[0.00, 2.00],
    }
    project_name = "2_funk_80_beat_4-4_4"
    _write_stems(stems_dir, project_name, onset_times_by_stem)

    config_path = project_dir / "midiconfig.yaml"
    with open(config_path, 'w') as f:
        yaml.safe_dump(_minimal_config(), f)

    return project_dir, stems_dir, midi_dir, project_name


class TestInitialConversionEqualsReconvert:
    """
    Initial conversion and reconvert must produce identical MIDI timing
    for the same input stems and config (bug E).
    """

    def test_initial_conversion_creates_midi(self, funk_like_project):
        """Smoke test: initial conversion actually produces a MIDI file."""
        project_dir, stems_dir, midi_dir, project_name = funk_like_project

        from stems_to_midi_cli import _process_stems_to_midi
        with open(project_dir / "midiconfig.yaml") as f:
            config = yaml.safe_load(f)

        _process_stems_to_midi(
            stems_source=stems_dir,
            midi_dir=midi_dir,
            project_name=project_name,
            config=config,
            stems_to_process=['kick', 'snare', 'toms', 'hihat', 'cymbals'],
            max_duration=None,
            learning_mode=False,
        )

        midi_files = list(midi_dir.glob("*.mid"))
        assert len(midi_files) == 1, f"Expected 1 MIDI, got {midi_files}"
        assert midi_files[0].stat().st_size > 0, "MIDI file is empty"

    def test_reconvert_after_initial_produces_same_midi(self, funk_like_project):
        """Full bug E regression: initial then reconvert → identical MIDI timing."""
        project_dir, stems_dir, midi_dir, project_name = funk_like_project

        from stems_to_midi_cli import _process_stems_to_midi
        from stems_to_midi.rebuild_shell import rebuild_midi_for_project

        with open(project_dir / "midiconfig.yaml") as f:
            config = yaml.safe_load(f)

        # Step 1: Initial conversion (creates MIDI + analysis.json)
        _process_stems_to_midi(
            stems_source=stems_dir,
            midi_dir=midi_dir,
            project_name=project_name,
            config=config,
            stems_to_process=['kick', 'snare', 'toms', 'hihat', 'cymbals'],
            max_duration=None,
            learning_mode=False,
        )
        initial_midi = list(midi_dir.glob("*.mid"))[0]

        # Snapshot the initial MIDI content
        initial_events = _read_all_midi_events(initial_midi)
        assert len(initial_events) > 0, "Initial conversion produced no MIDI events"

        # Step 2: Reconvert (Save & Reconvert path)
        result = rebuild_midi_for_project(
            project_dir=project_dir,
            honor_overrides=False,
        )
        assert result['success'], f"Reconvert failed: {result.get('error')}"

        # Step 3: Read both MIDIs and compare event times within 1ms
        reconvert_events = _read_all_midi_events(initial_midi)
        assert len(reconvert_events) == len(initial_events), (
            f"Event count differs: initial={len(initial_events)}, "
            f"reconvert={len(reconvert_events)}. "
            f"Initial: {initial_events[:5]} ... Reconvert: {reconvert_events[:5]}"
        )

        for i, (init_ev, re_ev) in enumerate(zip(initial_events, reconvert_events)):
            time_diff = abs(init_ev['time'] - re_ev['time'])
            assert time_diff <= 0.001, (
                f"Event {i} time differs by {time_diff*1000:.3f}ms "
                f"(initial={init_ev['time']:.4f}s, reconvert={re_ev['time']:.4f}s)"
            )
            assert init_ev['note'] == re_ev['note'], (
                f"Event {i} note differs: initial={init_ev['note']}, "
                f"reconvert={re_ev['note']}"
            )
            assert init_ev['velocity'] == re_ev['velocity'], (
                f"Event {i} velocity differs: initial={init_ev['velocity']}, "
                f"reconvert={re_ev['velocity']}"
            )

    def test_reconvert_is_idempotent(self, funk_like_project):
        """Multiple reconverts should produce identical MIDI."""
        project_dir, stems_dir, midi_dir, project_name = funk_like_project

        from stems_to_midi_cli import _process_stems_to_midi
        from stems_to_midi.rebuild_shell import rebuild_midi_for_project

        with open(project_dir / "midiconfig.yaml") as f:
            config = yaml.safe_load(f)

        _process_stems_to_midi(
            stems_source=stems_dir,
            midi_dir=midi_dir,
            project_name=project_name,
            config=config,
            stems_to_process=['kick', 'snare', 'toms', 'hihat', 'cymbals'],
            max_duration=None,
            learning_mode=False,
        )
        midi_path = list(midi_dir.glob("*.mid"))[0]
        first = _read_all_midi_events(midi_path)

        for n in range(3):
            rebuild_midi_for_project(
                project_dir=project_dir,
                honor_overrides=False,
            )
            nth = _read_all_midi_events(midi_path)
            assert len(nth) == len(first), (
                f"Iteration {n}: event count {len(nth)} != {len(first)}"
            )
            for i, (a, b) in enumerate(zip(first, nth)):
                assert abs(a['time'] - b['time']) <= 0.001, (
                    f"Iteration {n} event {i}: time {a['time']} vs {b['time']}"
                )

    def test_reconvert_does_not_lose_stems(self, funk_like_project):
        """Reconvert should produce the same set of stems as initial."""
        project_dir, stems_dir, midi_dir, project_name = funk_like_project

        from stems_to_midi_cli import _process_stems_to_midi
        from stems_to_midi.rebuild_shell import rebuild_midi_for_project

        with open(project_dir / "midiconfig.yaml") as f:
            config = yaml.safe_load(f)

        _process_stems_to_midi(
            stems_source=stems_dir,
            midi_dir=midi_dir,
            project_name=project_name,
            config=config,
            stems_to_process=['kick', 'snare', 'toms', 'hihat', 'cymbals'],
            max_duration=None,
            learning_mode=False,
        )
        initial_midi = list(midi_dir.glob("*.mid"))[0]
        initial_notes = {ev['note'] for ev in _read_all_midi_events(initial_midi)}

        rebuild_midi_for_project(project_dir=project_dir, honor_overrides=False)
        reconvert_notes = {ev['note'] for ev in _read_all_midi_events(initial_midi)}

        # No stems disappeared after the rebuild
        assert reconvert_notes == initial_notes, (
            f"Note set changed: initial={initial_notes}, reconvert={reconvert_notes}"
        )

    def test_reconvert_preserves_analysis_json(self, funk_like_project):
        """Reconvert should keep the analysis.json in sync (events_configured
        statuses should match what was just saved to MIDI)."""
        project_dir, stems_dir, midi_dir, project_name = funk_like_project

        from stems_to_midi_cli import _process_stems_to_midi
        from stems_to_midi.rebuild_shell import rebuild_midi_for_project

        with open(project_dir / "midiconfig.yaml") as f:
            config = yaml.safe_load(f)

        _process_stems_to_midi(
            stems_source=stems_dir,
            midi_dir=midi_dir,
            project_name=project_name,
            config=config,
            stems_to_process=['kick', 'snare', 'toms', 'hihat', 'cymbals'],
            max_duration=None,
            learning_mode=False,
        )

        # Run two reconverts and compare the second analysis.json to the first
        rebuild_midi_for_project(project_dir=project_dir, honor_overrides=False)
        first_sidecar = json.loads(
            (midi_dir / f"{project_name}.analysis.json").read_text()
        )

        rebuild_midi_for_project(project_dir=project_dir, honor_overrides=False)
        second_sidecar = json.loads(
            (midi_dir / f"{project_name}.analysis.json").read_text()
        )

        # Same stems, same number of events per stem
        for stem_type in first_sidecar['stems']:
            ce1 = first_sidecar['stems'][stem_type]['events_configured']
            ce2 = second_sidecar['stems'][stem_type]['events_configured']
            assert len(ce1) == len(ce2), (
                f"Stem {stem_type} event count drifted: "
                f"{len(ce1)} → {len(ce2)}"
            )
            for i, (a, b) in enumerate(zip(ce1, ce2)):
                assert abs(a['time'] - b['time']) <= 0.001, (
                    f"Stem {stem_type} event {i} time drifted"
                )
                assert a.get('status') == b.get('status'), (
                    f"Stem {stem_type} event {i} status changed: "
                    f"{a.get('status')} → {b.get('status')}"
                )
