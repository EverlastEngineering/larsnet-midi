"""
Tests for Step 3: Dual-Sensitivity Detection

Verifies:
- _run_sensitive_detection() finds events with max-sensitivity params
- _serialize_onset_events() correctly rounds and serializes onset data
- save_analysis_sidecar() writes v3 format with events_configured + events_sensitive
- Sensitive detection produces >= configured event count
- All sensitive events have spectral features pre-computed
"""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from stems_to_midi.processing_shell import _run_sensitive_detection
from stems_to_midi.midi import save_analysis_sidecar, _serialize_onset_events


# --- Fixtures ---

@pytest.fixture
def synthetic_kick_audio():
    """Generate synthetic kick-like audio with clear transients."""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 4 transients at known times
    hit_times = [0.2, 0.6, 1.0, 1.5]
    audio = np.zeros_like(t)
    for ht in hit_times:
        idx = int(ht * sr)
        # Short burst of low-frequency energy (kick-like)
        burst_len = int(0.05 * sr)
        end = min(idx + burst_len, len(audio))
        burst_t = np.arange(end - idx) / sr
        audio[idx:end] += 0.8 * np.sin(2 * np.pi * 60 * burst_t) * np.exp(-burst_t * 30)

    # Stereo: duplicate
    stereo = np.stack([audio, audio], axis=0)
    return stereo, audio, sr, hit_times


@pytest.fixture
def sample_config():
    """Minimal config dict for testing."""
    return {
        'kick': {
            'threshold_db': 15.0,
            'min_peak_spacing_ms': 100.0,
            'min_absolute_energy': 0.01,
            'merge_window_ms': 150.0,
            'energy_method': 'rms',
            'peak_hold_ms': 3.0,
            'enable_spectral_filter': True,
            'geomean_threshold': 0.5,
            'fundamental_freq_min': 40,
            'fundamental_freq_max': 100,
            'body_freq_min': 80,
            'body_freq_max': 300,
            'attack_freq_min': 1500,
            'attack_freq_max': 5000,
        },
        'snare': {
            'threshold_db': 15.0,
            'min_peak_spacing_ms': 100.0,
            'min_absolute_energy': 0.01,
            'merge_window_ms': 150.0,
            'energy_method': 'rms',
            'peak_hold_ms': 3.0,
            'enable_spectral_filter': True,
            'geomean_threshold': 0.5,
        },
    }


@pytest.fixture
def sample_onset_data():
    """Sample onset data dicts as produced by filter_onsets_by_spectral."""
    return [
        {
            'time': 0.20012345,
            'strength': 0.85432,
            'amplitude': 0.72345,
            'geomean': 12.567,
            'total_energy': 45.123,
            'sustain_ms': 150.789,
            'geomean_bands': ['fundamental', 'body', 'attack'],
            'fundamental_energy': 20.123,
            'body_energy': 15.456,
            'attack_energy': 9.789,
            'status': 'KEPT',
            'duration_sec': 0.123456,
            'amplitude_at_start': 0.654321,
        },
        {
            'time': 0.60056789,
            'strength': 0.35678,
            'amplitude': 0.21234,
            'geomean': 2.345,
            'total_energy': 8.901,
            'geomean_bands': ['fundamental', 'body', 'attack'],
            'fundamental_energy': 3.456,
            'body_energy': 2.789,
            'attack_energy': 2.123,
            'status': 'FILTERED',
        },
    ]


@pytest.fixture
def tmp_midi_path(tmp_path):
    """Create a temporary MIDI file path."""
    return tmp_path / "test_output.mid"


# --- Tests for _serialize_onset_events ---

class TestSerializeOnsetEvents:
    """Test the onset data serialization helper."""

    def test_basic_serialization(self, sample_onset_data):
        """Events are serialized with proper rounding."""
        result = _serialize_onset_events(sample_onset_data)
        assert len(result) == 2

        # Check time rounding (4 decimals)
        assert result[0]['time'] == round(0.20012345, 4)
        assert result[1]['time'] == round(0.60056789, 4)

        # Check feature rounding (2 decimals)
        assert result[0]['strength'] == round(0.85432, 2)
        assert result[0]['geomean'] == round(12.567, 2)
        assert result[0]['fundamental_energy'] == round(20.123, 2)

        # Check status preserved
        assert result[0]['status'] == 'KEPT'
        assert result[1]['status'] == 'FILTERED'

    def test_midi_events_attached_to_kept(self, sample_onset_data):
        """MIDI note/velocity attached to KEPT events when provided."""
        midi_events = [{'note': 36, 'velocity': 100}]
        result = _serialize_onset_events(sample_onset_data, midi_events=midi_events)

        assert result[0]['note'] == 36
        assert result[0]['velocity'] == 100
        assert 'note' not in result[1]  # FILTERED event gets no MIDI info

    def test_no_midi_events(self, sample_onset_data):
        """Without midi_events, no note/velocity fields added."""
        result = _serialize_onset_events(sample_onset_data)
        assert 'note' not in result[0]
        assert 'velocity' not in result[0]

    def test_phase2_metadata_rounding(self, sample_onset_data):
        """Phase 2 metadata fields use 4-decimal rounding."""
        result = _serialize_onset_events(sample_onset_data)
        assert result[0]['duration_sec'] == round(0.123456, 4)
        assert result[0]['amplitude_at_start'] == round(0.654321, 4)

    def test_empty_list(self):
        """Empty input produces empty output."""
        result = _serialize_onset_events([])
        assert result == []


# --- Tests for save_analysis_sidecar v3 ---

class TestSidecarV3Format:
    """Test the v3 sidecar format with events_configured + events_sensitive."""

    def test_v3_version_and_structure(self, tmp_midi_path, sample_onset_data):
        """Sidecar v3 has correct version and both event keys per stem."""
        events_by_stem = {'kick': [{'time': 0.2, 'note': 36, 'velocity': 100}]}
        analysis_by_stem = {
            'kick': {
                'all_onset_data': sample_onset_data,
                'sensitive_onset_data': sample_onset_data,  # Reuse for test
                'spectral_config': {
                    'geomean_threshold': 5.0,
                    'min_sustain_ms': None,
                    'geomean_bands': ['fundamental', 'body', 'attack'],
                },
            }
        }

        path = save_analysis_sidecar(events_by_stem, tmp_midi_path, tempo=120.0,
                                      analysis_by_stem=analysis_by_stem)

        with open(path) as f:
            data = json.load(f)

        assert data['version'] == '3.0'
        assert 'kick' in data['stems']
        stem = data['stems']['kick']
        assert 'events_configured' in stem
        assert 'events_sensitive' in stem
        assert 'logic' in stem

    def test_events_configured_has_midi_fields(self, tmp_midi_path, sample_onset_data):
        """Configured events have MIDI note/velocity for KEPT onsets."""
        events_by_stem = {'kick': [{'time': 0.2, 'note': 36, 'velocity': 100}]}
        analysis_by_stem = {
            'kick': {
                'all_onset_data': sample_onset_data,
                'sensitive_onset_data': [],
                'spectral_config': {
                    'geomean_threshold': 5.0,
                    'min_sustain_ms': None,
                    'geomean_bands': ['fundamental', 'body', 'attack'],
                },
            }
        }

        path = save_analysis_sidecar(events_by_stem, tmp_midi_path, tempo=120.0,
                                      analysis_by_stem=analysis_by_stem)

        with open(path) as f:
            data = json.load(f)

        configured = data['stems']['kick']['events_configured']
        kept_events = [e for e in configured if e['status'] == 'KEPT']
        assert len(kept_events) == 1
        assert kept_events[0]['note'] == 36

    def test_events_sensitive_no_midi_fields(self, tmp_midi_path, sample_onset_data):
        """Sensitive events have spectral features but no MIDI note/velocity."""
        events_by_stem = {'kick': [{'time': 0.2, 'note': 36, 'velocity': 100}]}
        analysis_by_stem = {
            'kick': {
                'all_onset_data': [],
                'sensitive_onset_data': sample_onset_data,
                'spectral_config': {
                    'geomean_threshold': 5.0,
                    'min_sustain_ms': None,
                    'geomean_bands': ['fundamental', 'body', 'attack'],
                },
            }
        }

        path = save_analysis_sidecar(events_by_stem, tmp_midi_path, tempo=120.0,
                                      analysis_by_stem=analysis_by_stem)

        with open(path) as f:
            data = json.load(f)

        sensitive = data['stems']['kick']['events_sensitive']
        assert len(sensitive) == 2
        # Sensitive events should have spectral features
        assert 'geomean' in sensitive[0]
        assert 'fundamental_energy' in sensitive[0]
        # But no MIDI fields (no note/velocity attached to sensitive events)
        assert 'note' not in sensitive[0]

    def test_empty_sensitive_produces_empty_list(self, tmp_midi_path, sample_onset_data):
        """When no sensitive data exists, events_sensitive is empty list."""
        events_by_stem = {'kick': [{'time': 0.2, 'note': 36, 'velocity': 100}]}
        analysis_by_stem = {
            'kick': {
                'all_onset_data': sample_onset_data,
                'sensitive_onset_data': [],
                'spectral_config': {
                    'geomean_threshold': 5.0,
                    'min_sustain_ms': None,
                    'geomean_bands': ['fundamental', 'body', 'attack'],
                },
            }
        }

        path = save_analysis_sidecar(events_by_stem, tmp_midi_path, tempo=120.0,
                                      analysis_by_stem=analysis_by_stem)

        with open(path) as f:
            data = json.load(f)

        assert data['stems']['kick']['events_sensitive'] == []

    def test_logic_block_preserved(self, tmp_midi_path, sample_onset_data):
        """Logic block still contains threshold/band metadata."""
        events_by_stem = {'kick': [{'time': 0.2, 'note': 36, 'velocity': 100}]}
        analysis_by_stem = {
            'kick': {
                'all_onset_data': sample_onset_data,
                'sensitive_onset_data': [],
                'spectral_config': {
                    'geomean_threshold': 5.0,
                    'min_sustain_ms': None,
                    'geomean_bands': ['fundamental', 'body', 'attack'],
                },
            }
        }

        path = save_analysis_sidecar(events_by_stem, tmp_midi_path, tempo=120.0,
                                      analysis_by_stem=analysis_by_stem)

        with open(path) as f:
            data = json.load(f)

        logic = data['stems']['kick']['logic']
        assert logic['geomean_threshold'] == 5.0
        assert logic['freq_bands'] == ['fundamental', 'body', 'attack']


# --- Tests for _run_sensitive_detection ---

class TestRunSensitiveDetection:
    """Test the sensitive detection helper function."""

    def test_returns_list_of_onset_dicts(self, synthetic_kick_audio, sample_config):
        """Sensitive detection returns a list of onset dicts with spectral features."""
        stereo, mono, sr, hit_times = synthetic_kick_audio
        result = _run_sensitive_detection(
            audio=stereo,
            audio_mono=mono,
            sr=sr,
            is_stereo=True,
            hop_length=512,
            stem_type='kick',
            config=sample_config,
        )

        assert isinstance(result, list)
        # Should detect at least some events from the synthetic audio
        if len(result) > 0:
            # Each onset should have spectral features
            assert 'time' in result[0]
            assert 'geomean' in result[0]
            assert 'status' in result[0]
            # learning_mode=True marks all as KEPT
            assert all(d['status'] == 'KEPT' for d in result)

    def test_sensitive_finds_more_than_configured(self, synthetic_kick_audio, sample_config):
        """Sensitive detection (threshold_db=1.0) finds >= events vs configured (threshold_db=15.0)."""
        stereo, mono, sr, hit_times = synthetic_kick_audio
        from stems_to_midi.energy_detection_shell import detect_onsets_energy_based

        # Configured detection
        configured_times, _, _ = detect_onsets_energy_based(
            stereo, sr,
            threshold_db=15.0,
            min_peak_spacing_ms=100.0,
            min_absolute_energy=0.01,
        )

        # Sensitive detection
        sensitive_result = _run_sensitive_detection(
            audio=stereo,
            audio_mono=mono,
            sr=sr,
            is_stereo=True,
            hop_length=512,
            stem_type='kick',
            config=sample_config,
        )

        # Sensitive should find at least as many as configured
        assert len(sensitive_result) >= len(configured_times)

    def test_silent_audio_returns_empty(self, sample_config):
        """Silent audio produces empty result."""
        sr = 22050
        silent = np.zeros(sr * 2)
        stereo = np.stack([silent, silent], axis=0)

        result = _run_sensitive_detection(
            audio=stereo,
            audio_mono=silent,
            sr=sr,
            is_stereo=True,
            hop_length=512,
            stem_type='kick',
            config=sample_config,
        )

        assert result == []

    def test_mono_audio_path(self, sample_config):
        """Works with mono audio (is_stereo=False)."""
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Single transient
        audio = np.zeros_like(t)
        idx = int(0.3 * sr)
        burst_len = int(0.05 * sr)
        end = min(idx + burst_len, len(audio))
        burst_t = np.arange(end - idx) / sr
        audio[idx:end] = 0.9 * np.sin(2 * np.pi * 60 * burst_t) * np.exp(-burst_t * 30)

        result = _run_sensitive_detection(
            audio=audio,  # mono passed directly
            audio_mono=audio,
            sr=sr,
            is_stereo=False,
            hop_length=512,
            stem_type='kick',
            config=sample_config,
        )

        assert isinstance(result, list)

    def test_all_sensitive_events_have_spectral_features(self, synthetic_kick_audio, sample_config):
        """Every sensitive event has pre-computed spectral features for client-side filtering."""
        stereo, mono, sr, _ = synthetic_kick_audio
        result = _run_sensitive_detection(
            audio=stereo,
            audio_mono=mono,
            sr=sr,
            is_stereo=True,
            hop_length=512,
            stem_type='kick',
            config=sample_config,
        )

        required_fields = ['time', 'strength', 'amplitude', 'geomean', 'total_energy', 'status']
        for onset in result:
            for field in required_fields:
                assert field in onset, f"Missing field '{field}' in sensitive onset data"
