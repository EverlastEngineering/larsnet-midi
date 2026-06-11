"""
Tests for the spectral-transient detector being wired into the main
processing pipeline (process_stem_to_midi).

The spectral detector is a complementary signal that always runs
alongside the energy detector. Its candidate events are written to
``stems.<stem>.events_spectral`` in the analysis.json sidecar.

The two detectors are independent — the spectral detector does not
affect which events become ``events_configured`` (that's still driven
by the energy detector + the detection_method selection).
"""

import json
import numpy as np
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from stems_to_midi.processing_shell import (
    process_stem_to_midi,
    _run_spectral_detection,
)
from stems_to_midi.midi import save_analysis_sidecar, load_analysis_sidecar
from stems_to_midi.spectral_transient_core import SpectralTransientConfig


# --- Fixtures ---------------------------------------------------------------

@pytest.fixture
def synthetic_toms_audio():
    """Generate synthetic tom-like audio with broadband transients.

    Each hit is a sum of sinusoids at 200/1000/3000/5000Hz with an
    exponential decay — broadband enough to fire the spectral detector
    (which looks for high-freq content above the noise floor in
    800-8000Hz).

    Returns (stereo, mono, sr, hit_times).
    """
    sr = 44100
    duration = 3.0
    hit_times = (0.5, 1.0, 1.5, 2.0)
    t = np.arange(int(sr * duration)) / sr
    audio = np.zeros_like(t)
    decay_samples = int(0.08 * sr)  # 80ms decay
    env = np.exp(-np.arange(decay_samples) / (decay_samples / 4.0))
    for ht in hit_times:
        i0 = int(ht * sr)
        n = min(decay_samples, len(audio) - i0)
        if n <= 0:
            continue
        burst = (
            np.sin(2 * np.pi * 200 * np.arange(n) / sr) +
            0.5 * np.sin(2 * np.pi * 1000 * np.arange(n) / sr) +
            0.3 * np.sin(2 * np.pi * 3000 * np.arange(n) / sr) +
            0.2 * np.sin(2 * np.pi * 5000 * np.arange(n) / sr)
        )
        audio[i0:i0 + n] += burst * env[:n]
    mono = audio.copy()
    stereo = np.stack([audio, audio], axis=0)
    return stereo, mono, sr, hit_times


@pytest.fixture
def toms_stem_config():
    """Minimal per-stem config for the toms stem — enables spectral filter
    and sets the geomean threshold low enough that the synthetic hits
    will pass.

    The spectral detector's defaults (1024/256, 800-8000Hz, -50dB floor)
    are the right starting point; we don't override them here.
    """
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
            # Spectral analysis band edges (must match what the
            # spectral_utils helper expects, otherwise the test crashes
            # in filter_onsets_by_spectral).
            'fundamental_freq_min': 60,
            'fundamental_freq_max': 150,
            'body_freq_min': 150,
            'body_freq_max': 400,
        },
        'midi': {
            'max_note_duration': 0.5,
        },
        'learning_mode': {'enabled': False},
    }


@pytest.fixture
def drum_mapping():
    """Minimal drum mapping for tests."""
    from stems_to_midi.config import DrumMapping
    return DrumMapping(
        kick=36, snare=38, hihat_closed=42, hihat_open=46, hihat_handclap=39,
        tom_low=45, tom_mid=47, tom_high=50, crash=49, ride=51, chinese=52,
        snare_rimshot=37, snare_clap=39,
    )


@pytest.fixture
def tmp_midi_path(tmp_path):
    """Temp MIDI path used for save_analysis_sidecar sidecar resolution."""
    return tmp_path / "test.mid"


# --- _run_spectral_detection direct tests ----------------------------------

class TestRunSpectralDetection:
    """The new _run_spectral_detection helper."""

    def test_returns_list_of_event_dicts_with_required_fields(
        self, synthetic_toms_audio
    ):
        """Should return a list of dicts with time, strength, band_powers,
        band_max_idx, band_max_ratio, method='spectral'."""
        stereo, mono, sr, hit_times = synthetic_toms_audio
        result = _run_spectral_detection(
            audio=stereo,
            audio_mono=mono,
            sr=sr,
            is_stereo=True,
            stem_type='toms',
        )

        assert isinstance(result, list)
        assert len(result) > 0, "spectral detector should find the 4 hits"
        required_fields = {'time', 'band_powers',
                           'band_max_idx', 'band_max_ratio', 'method'}
        for event in result:
            assert required_fields.issubset(event.keys()), (
                f"missing fields: {required_fields - event.keys()}"
            )
            assert event['method'] == 'spectral'
            # 2026-06-10: `strength` was removed (it was the lossy
            # clamp-to-1.0 of band_max_ratio/10). band_max_ratio
            # itself is the raw top/second-highest band ratio —
            # always >= 1 by construction. The back-compat alias
            # `band_max_ratio_10` is also emitted (= band_max_ratio
            # / 10) but is intentionally NOT in required_fields
            # since no current filter consumes it.
            assert event['band_max_ratio'] >= 1.0
            assert isinstance(event['band_powers'], list)
            assert len(event['band_powers']) == 5
            assert 0 <= event['band_max_idx'] <= 4

    def test_band_powers_and_band_max_consistent(self, synthetic_toms_audio):
        """For each event, band_max_idx must equal argmax(band_powers)."""
        stereo, mono, sr, _ = synthetic_toms_audio
        result = _run_spectral_detection(
            audio=stereo, audio_mono=mono, sr=sr,
            is_stereo=True, stem_type='toms',
        )
        for event in result:
            bp = event['band_powers']
            # argmax may return any of the ties; we check the value at
            # band_max_idx is the maximum (loose — could be tied)
            max_val = max(bp)
            assert bp[event['band_max_idx']] == max_val, (
                f"band_max_idx {event['band_max_idx']} doesn't point to "
                f"the max of band_powers {bp}"
            )

    def test_finds_4_hits_in_synthetic_audio(self, synthetic_toms_audio):
        """Synthetic toms: 4 evenly-spaced hits should all be detected.

        With the band-power detector (2026-06-09), the synthetic
        broadband burst's ratio peak may trail the strike by up to
        ~100ms (the spectral shape changes during decay). We allow
        100ms tolerance for synthetic audio; real-audio tests
        (test_project_4_toms_finds_six_known_hits_in_73_77s) are
        stricter.
        """
        stereo, mono, sr, hit_times = synthetic_toms_audio
        result = _run_spectral_detection(
            audio=stereo, audio_mono=mono, sr=sr,
            is_stereo=True, stem_type='toms',
        )
        detected_times = sorted(e['time'] for e in result)
        # Each hit should be near at least one detected event.
        for ht in hit_times:
            nearest = min(abs(t - ht) for t in detected_times)
            assert nearest < 0.100, (
                f"No detected event within 100ms of hit at {ht}s "
                f"(nearest: {nearest * 1000:.1f}ms)"
            )

    def test_silent_audio_returns_empty_list(self):
        """Silent input produces no events (no need to crash)."""
        sr = 22050
        silent = np.zeros(sr * 2)
        stereo = np.stack([silent, silent], axis=0)
        result = _run_spectral_detection(
            audio=stereo, audio_mono=silent, sr=sr,
            is_stereo=True, stem_type='toms',
        )
        assert result == []

    def test_works_with_mono_input(self, synthetic_toms_audio):
        """Mono input path works (is_stereo=False)."""
        _, mono, sr, _ = synthetic_toms_audio
        result = _run_spectral_detection(
            audio=mono, audio_mono=mono, sr=sr,
            is_stereo=False, stem_type='toms',
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_accepts_custom_config(self, synthetic_toms_audio):
        """Caller can override the SpectralTransientConfig."""
        stereo, mono, sr, _ = synthetic_toms_audio
        custom_cfg = SpectralTransientConfig(
            n_fft=1024, hop=256,
            min_band_ratio=1.5,  # lower threshold = more sensitive
        )
        result = _run_spectral_detection(
            audio=stereo, audio_mono=mono, sr=sr,
            is_stereo=True, stem_type='toms',
            config=custom_cfg,
        )
        assert isinstance(result, list)


# --- process_stem_to_midi integration tests --------------------------------

class TestProcessStemToMidiReturnsSpectralEvents:
    """process_stem_to_midi must stash spectral events in the result dict."""

    def test_result_dict_has_spectral_onset_data_key(
        self, synthetic_toms_audio, toms_stem_config, drum_mapping, tmp_path
    ):
        """Result dict has 'spectral_onset_data' key alongside the existing
        'sensitive_onset_data' key."""
        stereo, mono, sr, _ = synthetic_toms_audio
        # Write the synthetic audio to a temp WAV
        import soundfile as sf
        wav_path = tmp_path / "toms.wav"
        # soundfile expects (frames, channels) for 2D
        sf.write(str(wav_path), stereo.T, sr)

        result = process_stem_to_midi(
            audio_path=wav_path,
            stem_type='toms',
            drum_mapping=drum_mapping,
            config=toms_stem_config,
            onset_threshold=0.5,
            onset_delta=0.07,
            onset_wait=3,
            hop_length=512,
            min_velocity=80,
            max_velocity=110,
        )

        assert 'spectral_onset_data' in result, (
            "process_stem_to_midi must include 'spectral_onset_data' "
            "in its return dict"
        )
        # The energy-based 'sensitive_onset_data' must still be there.
        assert 'sensitive_onset_data' in result


# --- save_analysis_sidecar serialization test ------------------------------

class TestSaveAnalysisSidecarWritesSpectral:
    """save_analysis_sidecar must write stems.<stem>.events_spectral."""

    def test_events_spectral_key_present_in_json(
        self, tmp_midi_path, toms_stem_config
    ):
        """The sidecar JSON has events_spectral on every stem."""
        events_by_stem = {
            'toms': [{'time': 0.5, 'note': 47, 'velocity': 100}],
        }
        analysis_by_stem = {
            'toms': {
                'all_onset_data': [],
                'sensitive_onset_data': [],
                'spectral_onset_data': [
                    {
                        'time': 0.502,
                        # 2026-06-10: `strength` was the lossy
                        # clamp-to-1.0 field. The detector now
                        # emits the raw band_max_ratio and a
                        # back-compat `band_max_ratio_10` alias.
                        # Test fixtures follow the same shape.
                        'band_max_ratio_10': 0.95,
                        'band_powers': [1.0e+00, 5.0e-04, 1.0e-04, 2.0e-05, 1.0e-05],
                        'band_max_idx': 0,
                        'band_max_ratio': 2000.0,
                        'method': 'spectral',
                    },
                ],
                'spectral_config': {
                    'geomean_threshold': 0.5,
                    'min_sustain_ms': None,
                    'geomean_bands': [],
                },
            },
        }

        path = save_analysis_sidecar(
            events_by_stem, tmp_midi_path, tempo=120.0,
            analysis_by_stem=analysis_by_stem,
            config=toms_stem_config,
        )

        with open(path) as f:
            data = json.load(f)

        assert 'events_spectral' in data['stems']['toms']
        spec_events = data['stems']['toms']['events_spectral']
        assert len(spec_events) == 1
        assert spec_events[0]['time'] == 0.502
        assert spec_events[0]['method'] == 'spectral'
        assert spec_events[0]['band_max_idx'] == 0
        assert spec_events[0]['band_max_ratio'] == 2000.0
        assert spec_events[0]['band_powers'] == [1.0e+00, 5.0e-04, 1.0e-04, 2.0e-05, 1.0e-05]

    def test_empty_spectral_produces_empty_list(
        self, tmp_midi_path, toms_stem_config
    ):
        """When no spectral data is available, events_spectral is []."""
        events_by_stem = {'toms': []}
        analysis_by_stem = {
            'toms': {
                'all_onset_data': [],
                'sensitive_onset_data': [],
                'spectral_onset_data': [],
                'spectral_config': None,
            },
        }

        path = save_analysis_sidecar(
            events_by_stem, tmp_midi_path, tempo=120.0,
            analysis_by_stem=analysis_by_stem,
            config=toms_stem_config,
        )

        with open(path) as f:
            data = json.load(f)

        assert data['stems']['toms']['events_spectral'] == []


# --- end-to-end test -------------------------------------------------------

class TestEndToEndPipelineEmitsSpectralEvents:
    """Full pipeline: process_stem_to_midi → save_analysis_sidecar →
    load_analysis_sidecar must show both events_sensitive and
    events_spectral."""

    def test_both_event_lists_present_after_full_run(
        self, synthetic_toms_audio, toms_stem_config, drum_mapping, tmp_path
    ):
        """After a full pipeline run, the sidecar JSON has both
        events_sensitive and events_spectral on the toms stem."""
        stereo, mono, sr, _ = synthetic_toms_audio
        import soundfile as sf
        wav_path = tmp_path / "toms.wav"
        sf.write(str(wav_path), stereo.T, sr)

        result = process_stem_to_midi(
            audio_path=wav_path,
            stem_type='toms',
            drum_mapping=drum_mapping,
            config=toms_stem_config,
            onset_threshold=0.5,
            onset_delta=0.07,
            onset_wait=3,
            hop_length=512,
            min_velocity=80,
            max_velocity=110,
        )

        # Pack into analysis_by_stem (mirrors stems_to_midi_cli.py)
        midi_path = tmp_path / "out.mid"
        analysis_by_stem = {
            'toms': {
                'all_onset_data': result.get('all_onset_data', []),
                'sensitive_onset_data': result.get('sensitive_onset_data', []),
                'spectral_onset_data': result.get('spectral_onset_data', []),
                'spectral_config': result.get('spectral_config'),
            }
        }
        events_by_stem = (
            {'toms': result['events']} if result.get('events') else {'toms': []}
        )

        sidecar_path = save_analysis_sidecar(
            events_by_stem, midi_path, tempo=120.0,
            analysis_by_stem=analysis_by_stem,
            config=toms_stem_config,
        )

        with open(sidecar_path) as f:
            data = json.load(f)

        stem_data = data['stems']['toms']
        assert 'events_sensitive' in stem_data
        assert 'events_spectral' in stem_data
        # Both must be non-empty lists for this synthetic signal.
        assert isinstance(stem_data['events_sensitive'], list)
        assert isinstance(stem_data['events_spectral'], list)
        assert len(stem_data['events_spectral']) > 0, (
            "spectral detector should find at least one event in synthetic "
            "toms audio"
        )
        # Each spectral event has the required shape.
        for event in stem_data['events_spectral']:
            assert 'time' in event
            # 2026-06-10: `strength` was removed in favor of
            # the raw `band_max_ratio` (and a back-compat
            # `band_max_ratio_10` alias). No current filter
            # consumes the alias; the test only checks that
            # the raw ratio is present so the new sidecar
            # ratio slider has data to read.
            assert 'band_max_ratio' in event
            assert 'method' in event
            assert event['method'] == 'spectral'
