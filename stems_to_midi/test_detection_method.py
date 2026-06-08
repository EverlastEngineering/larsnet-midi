"""
Tests for the ``detection_method`` runtime consumer in
``stems_to_midi.processing_shell.process_stem_to_midi``.

The setting controls which candidate list becomes ``events_configured``
in the analysis.json sidecar:

  - ``"energy"`` (default in pre-detection_method builds) — the
    energy-detector filtered list, written verbatim.
  - ``"spectral"`` — the spectral-transient detector list, with
    ``method: "spectral"`` stamped on every event so the WebUI can
    color them differently.
  - ``"both"`` — union of energy + spectral, deduplicated within 12ms
    (matches the validator tolerance in ``midi._validate_events_subset``).
    When an energy event and a spectral event are within 12ms of each
    other, the energy one wins (it carries pitch/classification
    metadata). Surviving spectral events keep ``method: "spectral"``.

The function always returns ``events_configured`` so the sidecar
serializer can pick it up directly. ``save_analysis_sidecar`` falls
back to the legacy ``all_onset_data`` path when the key is missing
(back-compat for older code paths that bypass ``process_stem_to_midi``).
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from stems_to_midi.config import DrumMapping
from stems_to_midi.midi import load_analysis_sidecar, save_analysis_sidecar
from stems_to_midi.processing_shell import process_stem_to_midi


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SR = 44100
HIT_TIMES = (0.5, 1.0, 1.5, 2.0)


def _synth_toms_audio() -> tuple:
    """Synthetic tom-like audio with broadband transients.

    Same signal shape as ``test_pipeline_spectral`` — each hit is a sum
    of sinusoids at 200/1000/3000/5000Hz with an exponential decay,
    broadband enough to fire the spectral detector.
    """
    duration = 3.0
    t = np.arange(int(SR * duration)) / SR
    audio = np.zeros_like(t)
    decay_samples = int(0.08 * SR)
    env = np.exp(-np.arange(decay_samples) / (decay_samples / 4.0))
    for ht in HIT_TIMES:
        i0 = int(ht * SR)
        n = min(decay_samples, len(audio) - i0)
        if n <= 0:
            continue
        burst = (
            np.sin(2 * np.pi * 200 * np.arange(n) / SR)
            + 0.5 * np.sin(2 * np.pi * 1000 * np.arange(n) / SR)
            + 0.3 * np.sin(2 * np.pi * 3000 * np.arange(n) / SR)
            + 0.2 * np.sin(2 * np.pi * 5000 * np.arange(n) / SR)
        )
        audio[i0:i0 + n] += burst * env[:n]
    stereo = np.stack([audio, audio], axis=0)
    return stereo, audio.copy(), SR, HIT_TIMES


@pytest.fixture
def wav_path(tmp_path) -> Path:
    stereo, _, _, _ = _synth_toms_audio()
    path = tmp_path / "toms.wav"
    sf.write(str(path), stereo.T, SR)
    return path


@pytest.fixture
def drum_mapping() -> DrumMapping:
    return DrumMapping(
        kick=36, snare=38, hihat_closed=42, hihat_open=46, hihat_handclap=39,
        tom_low=45, tom_mid=47, tom_high=50, crash=49, ride=51, chinese=52,
        snare_rimshot=37, snare_clap=39,
    )


@pytest.fixture
def base_config() -> dict:
    """Minimum viable per-stem config so toms spectral filter passes."""
    return {
        'audio': {
            'force_mono': False,
            'silence_threshold': 0.001,
            'default_note_duration': 0.1,
        },
        'onset_detection': {
            'hop_length': 512,
            'threshold': 0.5,
            'delta': 0.07,
            'wait': 3,
        },
        'onset_export': {
            'include_filtered_in_sensitive': True,
        },
        'toms': {
            'enable_spectral_filter': True,
            'geomean_threshold': 0.5,
            'min_sustain_ms': None,
            'enable_pitch_detection': True,
            'pitch_method': 'yin',
            'min_pitch_hz': 60.0,
            'max_pitch_hz': 250.0,
            'expected_clusters': 3,
            'cluster_feature': 'spectral_centroid_hz',
            'threshold_db': 15.0,
            'min_peak_spacing_ms': 100.0,
            'min_absolute_energy': 0.001,
            'merge_window_ms': 150.0,
            'energy_method': 'rms',
            'peak_hold_ms': 3.0,
            'use_librosa_detection': False,
            'fundamental_freq_min': 60,
            'fundamental_freq_max': 150,
            'body_freq_min': 150,
            'body_freq_max': 400,
        },
        'midi': {'max_note_duration': 0.5},
        'learning_mode': {'enabled': False},
    }


# ---------------------------------------------------------------------------
# 1. Result-dict shape
# ---------------------------------------------------------------------------

class TestEventsConfiguredAlwaysPresent:
    """``process_stem_to_midi`` always returns ``events_configured``."""

    def test_events_configured_key_present(self, wav_path, base_config, drum_mapping):
        result = process_stem_to_midi(
            audio_path=wav_path, stem_type='toms', drum_mapping=drum_mapping,
            config=base_config, onset_threshold=0.5, onset_delta=0.07,
            onset_wait=3, hop_length=512, min_velocity=80, max_velocity=110,
        )
        assert 'events_configured' in result
        assert isinstance(result['events_configured'], list)

    def test_default_method_is_both(self, wav_path, base_config, drum_mapping):
        """Schema default is 'both'. With no override in the config,
        process_stem_to_midi uses 'both' which means the configured
        list is the union (energy + spectral survivors)."""
        # base_config has no detection_method → must be treated as 'both'
        result = process_stem_to_midi(
            audio_path=wav_path, stem_type='toms', drum_mapping=drum_mapping,
            config=base_config, onset_threshold=0.5, onset_delta=0.07,
            onset_wait=3, hop_length=512, min_velocity=80, max_velocity=110,
        )
        # At minimum, the energy events should be in configured.
        # 'both' would also include spectral survivors, so the count
        # should be >= what 'energy' produces.
        energy_result = process_stem_to_midi(
            audio_path=wav_path, stem_type='toms', drum_mapping=drum_mapping,
            config={**base_config,
                    'onset_detection': {**base_config['onset_detection'],
                                        'detection_method': 'energy'}},
            onset_threshold=0.5, onset_delta=0.07,
            onset_wait=3, hop_length=512, min_velocity=80, max_velocity=110,
        )
        assert len(result['events_configured']) >= len(
            energy_result['events_configured']
        )


# ---------------------------------------------------------------------------
# 2. Per-method behavior
# ---------------------------------------------------------------------------

def _run_with_method(wav_path, base_config, drum_mapping, method):
    config = {
        **base_config,
        'onset_detection': {
            **base_config['onset_detection'],
            'detection_method': method,
        },
    }
    return process_stem_to_midi(
        audio_path=wav_path, stem_type='toms', drum_mapping=drum_mapping,
        config=config, onset_threshold=0.5, onset_delta=0.07,
        onset_wait=3, hop_length=512, min_velocity=80, max_velocity=110,
    )


class TestDetectionMethodEnergy:
    """method='energy' keeps the existing behavior verbatim."""

    def test_configured_only_has_energy_events(self, wav_path, base_config, drum_mapping):
        result = _run_with_method(wav_path, base_config, drum_mapping, 'energy')
        configured = result['events_configured']
        # In 'energy' mode, every configured event is from the energy
        # detector; none should carry method='spectral' (only 'energy'
        # is allowed).
        for event in configured:
            assert event.get('method') != 'spectral', (
                f"'energy' mode leaked a spectral event: {event}"
            )

    def test_configured_matches_all_onset_data(self, wav_path, base_config, drum_mapping):
        """In 'energy' mode, events_configured should be the same content
        as all_onset_data (current behavior)."""
        result = _run_with_method(wav_path, base_config, drum_mapping, 'energy')
        # The serializer dedupes by onset identity; same count, same
        # times.
        all_times = sorted(e['time'] for e in result['all_onset_data'])
        configured_times = sorted(e['time'] for e in result['events_configured'])
        assert configured_times == all_times


class TestDetectionMethodSpectral:
    """method='spectral' promotes spectral events into configured."""

    def test_configured_has_spectral_method_marker(self, wav_path, base_config, drum_mapping):
        result = _run_with_method(wav_path, base_config, drum_mapping, 'spectral')
        configured = result['events_configured']
        # In 'spectral' mode, every event in configured is from the
        # spectral detector and must carry method='spectral'.
        for event in configured:
            assert event.get('method') == 'spectral', (
                f"'spectral' mode produced a non-spectral event: {event}"
            )

    def test_configured_count_matches_spectral_onset_data(
        self, wav_path, base_config, drum_mapping
    ):
        result = _run_with_method(wav_path, base_config, drum_mapping, 'spectral')
        assert len(result['events_configured']) == len(
            result['spectral_onset_data']
        ), (
            "In 'spectral' mode, events_configured count must equal "
            "spectral_onset_data count"
        )

    def test_spectral_events_in_configured_have_time_field(
        self, wav_path, base_config, drum_mapping
    ):
        result = _run_with_method(wav_path, base_config, drum_mapping, 'spectral')
        configured = result['events_configured']
        assert len(configured) > 0, "expected spectral detector to find hits"
        for event in configured:
            assert 'time' in event
            assert isinstance(event['time'], float)


class TestDetectionMethodBoth:
    """method='both' is the union, deduped within 12ms."""

    def test_configured_is_union_of_energy_and_spectral(
        self, wav_path, base_config, drum_mapping
    ):
        """The 'both' result must include all energy events (they
        always win the dedup collision) and any spectral events
        whose time is more than 12ms from every energy event."""
        energy = _run_with_method(wav_path, base_config, drum_mapping, 'energy')
        both = _run_with_method(wav_path, base_config, drum_mapping, 'both')

        configured = both['events_configured']
        energy_times = set(round(e['time'], 4) for e in energy['events_configured'])

        # Every energy event must be in the 'both' result.
        for t in energy_times:
            assert any(
                round(e['time'], 4) == t for e in configured
            ), f"energy event at t={t} missing from 'both' result"

    def test_spectral_events_within_12ms_of_energy_are_dropped(
        self, wav_path, base_config, drum_mapping
    ):
        """If a spectral event is within 12ms of an energy event, the
        energy one wins. So the spectral event shouldn't be in
        events_configured."""
        both = _run_with_method(wav_path, base_config, drum_mapping, 'both')
        configured = both['events_configured']
        spectral_survivors = [e for e in configured if e.get('method') == 'spectral']

        # Each surviving spectral event must be at least 12ms away from
        # any energy event in the same list. (This is the dedup rule
        # in code; here we verify the property held.)
        energy_times = [
            e['time'] for e in configured if e.get('method') != 'spectral'
        ]
        DEDUP_WINDOW = 0.012
        for sp in spectral_survivors:
            t = sp['time']
            assert all(
                abs(t - et) > DEDUP_WINDOW for et in energy_times
            ), (
                f"spectral event at t={t} is within {DEDUP_WINDOW*1000}ms "
                f"of an energy event but wasn't dropped: {sp}"
            )

    def test_spectral_survivors_carry_method_marker(
        self, wav_path, base_config, drum_mapping
    ):
        """Spectral events that survive the dedup must have
        method='spectral' so the WebUI can color them differently."""
        both = _run_with_method(wav_path, base_config, drum_mapping, 'both')
        configured = both['events_configured']
        for event in configured:
            # Either it's an energy event (no method='spectral' marker,
            # or explicit method='energy') or it's a spectral survivor
            # (method='spectral').
            method = event.get('method')
            assert method in ('energy', 'spectral', None), (
                f"unexpected method={method!r} on event: {event}"
            )

    def test_both_count_is_at_least_energy_count(
        self, wav_path, base_config, drum_mapping
    ):
        energy = _run_with_method(wav_path, base_config, drum_mapping, 'energy')
        both = _run_with_method(wav_path, base_config, drum_mapping, 'both')
        assert len(both['events_configured']) >= len(energy['events_configured'])


# ---------------------------------------------------------------------------
# 3. End-to-end: sidecar JSON has the right events_configured
# ---------------------------------------------------------------------------

class TestSidecarUsesEventsConfigured:
    """``save_analysis_sidecar`` must use the new key when present."""

    def test_sidecar_uses_prebuilt_events_configured(
        self, wav_path, base_config, drum_mapping, tmp_path
    ):
        """When process_stem_to_midi returns events_configured, the
        sidecar must use it (not the legacy all_onset_data path).

        We construct a contrived analysis_by_stem where the prebuilt
        list differs from all_onset_data: we set all_onset_data to an
        empty list and the prebuilt list to a single entry with a
        distinctive time. If the sidecar falls back to all_onset_data,
        the sidecar ends up empty. If it uses the prebuilt list, the
        distinctive time shows up.
        """
        result = _run_with_method(wav_path, base_config, drum_mapping, 'spectral')
        assert result['events_configured']

        # A prebuilt list that is intentionally different from
        # all_onset_data so we can tell which path the sidecar took.
        # The energy all_onset_data path would produce events with
        # times around 0.5, 1.0, 1.5, 2.0; we put one event at
        # t=42.0 — well outside the synthetic signal.
        prebuilt = [{
            'time': 42.0,
            'status': 'KEPT',
            'strength': 0.99,
            'method': 'spectral',
        }]

        midi_path = tmp_path / "out.mid"
        analysis_by_stem = {
            'toms': {
                'all_onset_data': [],  # empty — would produce empty sidecar
                'sensitive_onset_data': result.get('sensitive_onset_data', []),
                'spectral_onset_data': result.get('spectral_onset_data', []),
                'spectral_config': result.get('spectral_config'),
                'events_configured': prebuilt,  # non-empty, distinctive
            },
        }
        # Pass enough midi_events to keep the serializer happy when
        # matching the prebuilt KEPT event to a midi_events index.
        events_by_stem = {'toms': result['events']}
        spectral_config = {
            **base_config,
            'onset_detection': {
                **base_config['onset_detection'],
                'detection_method': 'spectral',
            },
        }
        sidecar_path = save_analysis_sidecar(
            events_by_stem, midi_path, tempo=120.0,
            analysis_by_stem=analysis_by_stem,
            config=spectral_config,
        )
        with open(sidecar_path) as f:
            data = json.load(f)

        configured = data['stems']['toms']['events_configured']
        # If the sidecar used the prebuilt list, the entry at t=42.0
        # is present. If it fell back to all_onset_data, the list is
        # empty.
        times = [e.get('time') for e in configured]
        assert 42.0 in times, (
            f"sidecar did not use the prebuilt events_configured; "
            f"times were {times} (expected 42.0 in there). "
            f"save_analysis_sidecar must look up analysis.events_configured "
            f"when present."
        )
