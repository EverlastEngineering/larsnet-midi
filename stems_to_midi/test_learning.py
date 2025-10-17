"""
Tests for learning.py module - threshold learning from edited MIDI files.
"""

import pytest
import numpy as np
import tempfile
import soundfile as sf
from pathlib import Path
from mido import MidiFile, MidiTrack, Message, MetaMessage
import yaml

from stems_to_midi.learning import learn_threshold_from_midi, save_calibrated_config
from stems_to_midi.config import DrumMapping


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'audio': {
            'force_mono': True,
            'peak_window_sec': 0.05,
            'min_segment_length': 512,
            'sustain_window_sec': 0.2,
            'envelope_threshold': 0.1,
            'envelope_smooth_kernel': 51
        },
        'learning_mode': {
            'enabled': True,
            'match_tolerance_sec': 0.05
        },
        'snare': {
            'low_freq_min': 50,
            'low_freq_max': 150,
            'body_freq_min': 150,
            'body_freq_max': 400,
            'wire_freq_min': 2000,
            'wire_freq_max': 8000,
            'geomean_threshold': 1000.0
        },
        'kick': {
            'fundamental_freq_min': 40,
            'fundamental_freq_max': 80,
            'body_freq_min': 80,
            'body_freq_max': 150,
            'geomean_threshold': 500.0
        },
        'toms': {
            'fundamental_freq_min': 60,
            'fundamental_freq_max': 150,
            'body_freq_min': 150,
            'body_freq_max': 400,
            'geomean_threshold': 800.0
        },
        'hihat': {
            'body_freq_min': 500,
            'body_freq_max': 2000,
            'sizzle_freq_min': 6000,
            'sizzle_freq_max': 12000,
            'geomean_threshold': 2000.0,
            'min_sustain_ms': 25
        },
        'cymbals': {
            'geomean_threshold': 3000.0,
            'min_sustain_ms': 150
        }
    }


@pytest.fixture
def drum_mapping():
    """Standard drum mapping."""
    return DrumMapping()


def create_test_audio(duration_sec=2.0, sr=22050, onsets=None):
    """
    Create synthetic audio with onsets at specified times.
    
    Args:
        duration_sec: Duration in seconds
        sr: Sample rate
        onsets: List of onset times in seconds
    
    Returns:
        Audio array (mono)
    """
    if onsets is None:
        onsets = [0.5, 1.0, 1.5]
    
    # Create silence
    audio = np.zeros(int(duration_sec * sr))
    
    # Add impulses at onset times with varying frequencies
    for i, onset_time in enumerate(onsets):
        onset_sample = int(onset_time * sr)
        
        # Create a short burst with different spectral content for each onset
        burst_duration = int(0.05 * sr)  # 50ms burst
        t = np.arange(burst_duration) / sr
        
        # Different frequency content for each onset
        # First onset: low frequency (kick-like)
        # Second onset: mid frequency (snare-like)
        # Third onset: high frequency (cymbal-like)
        if i == 0:
            # Low frequency fundamental + body
            burst = np.sin(2 * np.pi * 60 * t) * 0.8  # Fundamental
            burst += np.sin(2 * np.pi * 100 * t) * 0.4  # Body
        elif i == 1:
            # Mid frequency body + high wire
            burst = np.sin(2 * np.pi * 200 * t) * 0.6  # Body
            burst += np.sin(2 * np.pi * 3000 * t) * 0.3  # Wire
        else:
            # High frequency cymbal-like
            burst = np.sin(2 * np.pi * 2000 * t) * 0.5  # Body
            burst += np.sin(2 * np.pi * 6000 * t) * 0.4  # Brilliance
            # Add decay envelope for sustain
            decay = np.exp(-5 * t)
            burst = burst * decay
        
        # Apply envelope
        envelope = np.exp(-10 * t)
        burst = burst * envelope
        
        # Add to audio
        end_sample = min(onset_sample + burst_duration, len(audio))
        actual_burst_len = end_sample - onset_sample
        audio[onset_sample:end_sample] = burst[:actual_burst_len]
    
    return audio


def create_midi_file(output_path, note_events, tempo=120):
    """
    Create a MIDI file with specified note events.
    
    Args:
        output_path: Path to save MIDI file
        note_events: List of (time_sec, note, velocity) tuples
        tempo: Tempo in BPM
    """
    mid = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    mid.tracks.append(track)
    
    # Set tempo
    track.append(MetaMessage('set_tempo', tempo=int(60000000 / tempo), time=0))
    
    # Convert events to MIDI messages
    # Sort by time
    sorted_events = sorted(note_events, key=lambda x: x[0])
    
    current_time = 0
    for time_sec, note, velocity in sorted_events:
        # Convert seconds to ticks
        ticks = int(time_sec * mid.ticks_per_beat * tempo / 60)
        delta_ticks = ticks - current_time
        
        # Note on
        track.append(Message('note_on', note=note, velocity=velocity, time=delta_ticks))
        
        # Note off (very short duration)
        track.append(Message('note_off', note=note, velocity=0, time=10))
        
        current_time = ticks + 10
    
    mid.save(output_path)


# ============================================================================
# TEST: learn_threshold_from_midi
# ============================================================================

class TestLearnThresholdFromMidi:
    """Tests for learn_threshold_from_midi function."""
    
    def test_learn_threshold_snare_basic(self, sample_config, drum_mapping, tmp_path):
        """Test basic threshold learning for snare."""
        # Create test audio with 3 onsets
        audio = create_test_audio(duration_sec=2.0, onsets=[0.5, 1.0, 1.5])
        audio_path = tmp_path / "test_snare.wav"
        sf.write(audio_path, audio, 22050)
        
        # Create original MIDI with all 3 detections
        original_midi_path = tmp_path / "original.mid"
        snare_note = drum_mapping.snare
        create_midi_file(original_midi_path, [
            (0.5, snare_note, 100),
            (1.0, snare_note, 100),
            (1.5, snare_note, 100)
        ])
        
        # Create edited MIDI with only 2 kept (removed the third one)
        edited_midi_path = tmp_path / "edited.mid"
        create_midi_file(edited_midi_path, [
            (0.5, snare_note, 100),
            (1.0, snare_note, 100)
        ])
        
        # Learn threshold
        result = learn_threshold_from_midi(
            audio_path, original_midi_path, edited_midi_path,
            'snare', sample_config, drum_mapping
        )
        
        # Check result structure
        assert 'geomean_threshold' in result
        assert 'kept_range' in result
        assert 'removed_range' in result
        
        # Check that threshold is reasonable
        assert result['geomean_threshold'] > 0
        
        # Check that ranges are tuples with min/max
        kept_min, kept_max = result['kept_range']
        removed_min, removed_max = result['removed_range']
        assert kept_min <= kept_max
        assert removed_min <= removed_max
    
    def test_learn_threshold_kick(self, sample_config, drum_mapping, tmp_path):
        """Test threshold learning for kick."""
        # Create test audio
        audio = create_test_audio(duration_sec=2.0, onsets=[0.5, 1.0, 1.5])
        audio_path = tmp_path / "test_kick.wav"
        sf.write(audio_path, audio, 22050)
        
        # Create MIDI files
        kick_note = drum_mapping.kick
        original_midi_path = tmp_path / "original.mid"
        create_midi_file(original_midi_path, [
            (0.5, kick_note, 100),
            (1.0, kick_note, 100),
            (1.5, kick_note, 100)
        ])
        
        edited_midi_path = tmp_path / "edited.mid"
        create_midi_file(edited_midi_path, [
            (0.5, kick_note, 100),
            (1.0, kick_note, 100)
        ])
        
        result = learn_threshold_from_midi(
            audio_path, original_midi_path, edited_midi_path,
            'kick', sample_config, drum_mapping
        )
        
        assert 'geomean_threshold' in result
        assert result['geomean_threshold'] > 0
    
    def test_learn_threshold_toms(self, sample_config, drum_mapping, tmp_path):
        """Test threshold learning for toms."""
        audio = create_test_audio(duration_sec=2.0, onsets=[0.5, 1.0, 1.5])
        audio_path = tmp_path / "test_toms.wav"
        sf.write(audio_path, audio, 22050)
        
        # Use tom_mid (which is what getattr(drum_mapping, 'toms') returns)
        tom_note = drum_mapping.toms  # This returns tom_mid = 47
        original_midi_path = tmp_path / "original.mid"
        create_midi_file(original_midi_path, [
            (0.5, tom_note, 100),
            (1.0, tom_note, 100),
            (1.5, tom_note, 100)
        ])
        
        edited_midi_path = tmp_path / "edited.mid"
        create_midi_file(edited_midi_path, [
            (0.5, tom_note, 100),
            (1.0, tom_note, 100)
        ])
        
        result = learn_threshold_from_midi(
            audio_path, original_midi_path, edited_midi_path,
            'toms', sample_config, drum_mapping
        )
        
        assert 'geomean_threshold' in result
    
    def test_learn_threshold_hihat(self, sample_config, drum_mapping, tmp_path):
        """Test threshold learning for hihat."""
        audio = create_test_audio(duration_sec=2.0, onsets=[0.5, 1.0, 1.5])
        audio_path = tmp_path / "test_hihat.wav"
        sf.write(audio_path, audio, 22050)
        
        hihat_note = drum_mapping.hihat_closed
        original_midi_path = tmp_path / "original.mid"
        create_midi_file(original_midi_path, [
            (0.5, hihat_note, 100),
            (1.0, hihat_note, 100),
            (1.5, hihat_note, 100)
        ])
        
        edited_midi_path = tmp_path / "edited.mid"
        create_midi_file(edited_midi_path, [
            (0.5, hihat_note, 100),
            (1.0, hihat_note, 100)
        ])
        
        result = learn_threshold_from_midi(
            audio_path, original_midi_path, edited_midi_path,
            'hihat', sample_config, drum_mapping
        )
        
        assert 'geomean_threshold' in result
    
    def test_learn_threshold_cymbals_with_sustain(self, sample_config, drum_mapping, tmp_path):
        """Test threshold learning for cymbals including sustain threshold."""
        # Create audio with longer sustain for cymbals
        audio = create_test_audio(duration_sec=2.0, onsets=[0.5, 1.0, 1.5])
        audio_path = tmp_path / "test_cymbals.wav"
        sf.write(audio_path, audio, 22050)
        
        cymbal_note = drum_mapping.crash
        original_midi_path = tmp_path / "original.mid"
        create_midi_file(original_midi_path, [
            (0.5, cymbal_note, 100),
            (1.0, cymbal_note, 100),
            (1.5, cymbal_note, 100)
        ])
        
        edited_midi_path = tmp_path / "edited.mid"
        create_midi_file(edited_midi_path, [
            (0.5, cymbal_note, 100),
            (1.0, cymbal_note, 100)
        ])
        
        result = learn_threshold_from_midi(
            audio_path, original_midi_path, edited_midi_path,
            'cymbals', sample_config, drum_mapping
        )
        
        assert 'geomean_threshold' in result
        # Cymbals should also return sustain threshold
        assert 'min_sustain_ms' in result
        assert result['min_sustain_ms'] > 0
    
    def test_learn_threshold_all_kept(self, sample_config, drum_mapping, tmp_path):
        """Test when user keeps all detections (no false positives)."""
        audio = create_test_audio(duration_sec=2.0, onsets=[0.5, 1.0, 1.5])
        audio_path = tmp_path / "test_snare.wav"
        sf.write(audio_path, audio, 22050)
        
        snare_note = drum_mapping.snare
        midi_path = tmp_path / "all_kept.mid"
        create_midi_file(midi_path, [
            (0.5, snare_note, 100),
            (1.0, snare_note, 100),
            (1.5, snare_note, 100)
        ])
        
        # Use same file for both original and edited
        result = learn_threshold_from_midi(
            audio_path, midi_path, midi_path,
            'snare', sample_config, drum_mapping
        )
        
        # Should return empty dict when no removed hits
        assert result == {}
    
    def test_learn_threshold_all_removed(self, sample_config, drum_mapping, tmp_path):
        """Test when user removes all detections."""
        audio = create_test_audio(duration_sec=2.0, onsets=[0.5, 1.0, 1.5])
        audio_path = tmp_path / "test_snare.wav"
        sf.write(audio_path, audio, 22050)
        
        snare_note = drum_mapping.snare
        original_midi_path = tmp_path / "original.mid"
        create_midi_file(original_midi_path, [
            (0.5, snare_note, 100),
            (1.0, snare_note, 100),
            (1.5, snare_note, 100)
        ])
        
        # Empty edited file
        edited_midi_path = tmp_path / "edited.mid"
        create_midi_file(edited_midi_path, [])
        
        result = learn_threshold_from_midi(
            audio_path, original_midi_path, edited_midi_path,
            'snare', sample_config, drum_mapping
        )
        
        # Should return empty dict when no kept hits
        assert result == {}
    
    def test_learn_threshold_stereo_audio(self, sample_config, drum_mapping, tmp_path):
        """Test with stereo audio (should be converted to mono)."""
        # Create stereo audio
        mono_audio = create_test_audio(duration_sec=2.0, onsets=[0.5, 1.0, 1.5])
        stereo_audio = np.stack([mono_audio, mono_audio], axis=1)
        
        audio_path = tmp_path / "test_stereo.wav"
        sf.write(audio_path, stereo_audio, 22050)
        
        snare_note = drum_mapping.snare
        original_midi_path = tmp_path / "original.mid"
        create_midi_file(original_midi_path, [
            (0.5, snare_note, 100),
            (1.0, snare_note, 100),
            (1.5, snare_note, 100)
        ])
        
        edited_midi_path = tmp_path / "edited.mid"
        create_midi_file(edited_midi_path, [
            (0.5, snare_note, 100),
            (1.0, snare_note, 100)
        ])
        
        result = learn_threshold_from_midi(
            audio_path, original_midi_path, edited_midi_path,
            'snare', sample_config, drum_mapping
        )
        
        assert 'geomean_threshold' in result
    
    def test_learn_threshold_custom_tolerance(self, sample_config, drum_mapping, tmp_path):
        """Test with custom match tolerance."""
        audio = create_test_audio(duration_sec=2.0, onsets=[0.5, 1.0, 1.5])
        audio_path = tmp_path / "test_snare.wav"
        sf.write(audio_path, audio, 22050)
        
        # Modify config with tighter tolerance
        config = sample_config.copy()
        config['learning_mode']['match_tolerance_sec'] = 0.01  # 10ms tolerance
        
        snare_note = drum_mapping.snare
        original_midi_path = tmp_path / "original.mid"
        create_midi_file(original_midi_path, [
            (0.5, snare_note, 100),
            (1.0, snare_note, 100),
            (1.5, snare_note, 100)
        ])
        
        # Edited with slightly different timing
        edited_midi_path = tmp_path / "edited.mid"
        create_midi_file(edited_midi_path, [
            (0.501, snare_note, 100),  # Slightly off
            (1.0, snare_note, 100)
        ])
        
        result = learn_threshold_from_midi(
            audio_path, original_midi_path, edited_midi_path,
            'snare', config, drum_mapping
        )
        
        # Should still match with tight tolerance
        assert 'geomean_threshold' in result
    
    def test_learn_threshold_onset_at_end_of_audio(self, sample_config, drum_mapping, tmp_path):
        """Test with onset very close to end of audio (onset_frame >= len(onset_env))."""
        # Create very short audio
        audio = create_test_audio(duration_sec=0.1, onsets=[0.05, 0.09])
        audio_path = tmp_path / "test_short.wav"
        sf.write(audio_path, audio, 22050)
        
        snare_note = drum_mapping.snare
        original_midi_path = tmp_path / "original.mid"
        create_midi_file(original_midi_path, [
            (0.05, snare_note, 100),
            (0.09, snare_note, 100)  # Very close to end
        ])
        
        edited_midi_path = tmp_path / "edited.mid"
        create_midi_file(edited_midi_path, [
            (0.05, snare_note, 100)
        ])
        
        result = learn_threshold_from_midi(
            audio_path, original_midi_path, edited_midi_path,
            'snare', sample_config, drum_mapping
        )
        
        # Should handle edge case gracefully
        assert 'geomean_threshold' in result or result == {}
    
    def test_learn_threshold_very_short_segment(self, sample_config, drum_mapping, tmp_path):
        """Test with onset that produces segment too short for analysis."""
        # Create audio where onset is at the very end (segment will be too short)
        audio = np.zeros(int(0.1 * 22050))  # 100ms of silence
        audio[-10:] = 0.5  # Tiny burst at the very end
        
        audio_path = tmp_path / "test_tiny.wav"
        sf.write(audio_path, audio, 22050)
        
        snare_note = drum_mapping.snare
        original_midi_path = tmp_path / "original.mid"
        # Place onset at 0.099 seconds (very close to end)
        create_midi_file(original_midi_path, [
            (0.099, snare_note, 100)
        ])
        
        edited_midi_path = tmp_path / "edited.mid"
        create_midi_file(edited_midi_path, [])
        
        result = learn_threshold_from_midi(
            audio_path, original_midi_path, edited_midi_path,
            'snare', sample_config, drum_mapping
        )
        
        # Should return empty dict (not enough data)
        assert result == {}
    
    def test_learn_threshold_onset_beyond_envelope(self, sample_config, drum_mapping, tmp_path):
        """Test with onset time that produces frame beyond onset envelope length."""
        # Create very short audio with a single note at a specific time
        # that will cause onset_frame to be >= len(onset_env)
        sr = 22050
        hop_length = 512
        # Create audio that's exactly long enough that the onset frame calculation
        # will be at or beyond the onset envelope length
        duration_samples = sr // 4  # 0.25 seconds
        audio = np.zeros(duration_samples)
        
        # Add an impulse near the end
        impulse_time = 0.24  # Very close to end
        impulse_sample = int(impulse_time * sr)
        if impulse_sample < len(audio) - 100:
            audio[impulse_sample:impulse_sample+100] = np.sin(2 * np.pi * 200 * np.arange(100) / sr) * 0.8
        
        audio_path = tmp_path / "test_edge.wav"
        sf.write(audio_path, audio, sr)
        
        snare_note = drum_mapping.snare
        original_midi_path = tmp_path / "original.mid"
        create_midi_file(original_midi_path, [
            (impulse_time, snare_note, 100)
        ])
        
        edited_midi_path = tmp_path / "edited.mid"
        create_midi_file(edited_midi_path, [])
        
        result = learn_threshold_from_midi(
            audio_path, original_midi_path, edited_midi_path,
            'snare', sample_config, drum_mapping
        )
        
        # Function should handle this gracefully
        # May return empty dict or a result, both are valid
        assert isinstance(result, dict)


# ============================================================================
# TEST: save_calibrated_config
# ============================================================================

class TestSaveCalibratedConfig:
    """Tests for save_calibrated_config function."""
    
    def test_save_calibrated_config_basic(self, sample_config, tmp_path):
        """Test saving a calibrated config."""
        learned_thresholds = {
            'snare': {'geomean_threshold': 1234.5},
            'kick': {'geomean_threshold': 678.9}
        }
        
        output_path = tmp_path / "calibrated_config.yaml"
        save_calibrated_config(sample_config, learned_thresholds, output_path)
        
        # Check file was created
        assert output_path.exists()
        
        # Load and verify
        with open(output_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        # Check thresholds were updated
        assert loaded_config['snare']['geomean_threshold'] == 1234.5
        assert loaded_config['kick']['geomean_threshold'] == 678.9
        
        # Check learning mode was disabled
        assert loaded_config['learning_mode']['enabled'] == False
    
    def test_save_calibrated_config_preserves_other_settings(self, sample_config, tmp_path):
        """Test that other config settings are preserved."""
        learned_thresholds = {
            'snare': {'geomean_threshold': 1500.0}
        }
        
        output_path = tmp_path / "calibrated_config.yaml"
        save_calibrated_config(sample_config, learned_thresholds, output_path)
        
        with open(output_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        # Check other settings preserved
        assert loaded_config['audio']['force_mono'] == sample_config['audio']['force_mono']
        assert loaded_config['snare']['body_freq_min'] == sample_config['snare']['body_freq_min']
        
        # Check snare threshold was updated
        assert loaded_config['snare']['geomean_threshold'] == 1500.0
    
    def test_save_calibrated_config_multiple_stems(self, sample_config, tmp_path):
        """Test saving thresholds for multiple stems."""
        learned_thresholds = {
            'snare': {'geomean_threshold': 1000.0},
            'kick': {'geomean_threshold': 500.0},
            'toms': {'geomean_threshold': 800.0},
            'hihat': {'geomean_threshold': 2000.0}
        }
        
        output_path = tmp_path / "calibrated_config.yaml"
        save_calibrated_config(sample_config, learned_thresholds, output_path)
        
        with open(output_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config['snare']['geomean_threshold'] == 1000.0
        assert loaded_config['kick']['geomean_threshold'] == 500.0
        assert loaded_config['toms']['geomean_threshold'] == 800.0
        assert loaded_config['hihat']['geomean_threshold'] == 2000.0
    
    def test_save_calibrated_config_no_learning_mode_section(self, sample_config, tmp_path):
        """Test saving config when learning_mode section doesn't exist."""
        # Remove learning_mode from config
        config_copy = sample_config.copy()
        del config_copy['learning_mode']
        
        learned_thresholds = {
            'snare': {'geomean_threshold': 1234.5}
        }
        
        output_path = tmp_path / "calibrated_config.yaml"
        save_calibrated_config(config_copy, learned_thresholds, output_path)
        
        # Should not raise error
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config['snare']['geomean_threshold'] == 1234.5
    
    def test_save_calibrated_config_empty_thresholds(self, sample_config, tmp_path):
        """Test saving config with empty learned thresholds."""
        learned_thresholds = {}
        
        output_path = tmp_path / "calibrated_config.yaml"
        save_calibrated_config(sample_config, learned_thresholds, output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        # Original thresholds should be preserved
        assert loaded_config['snare']['geomean_threshold'] == sample_config['snare']['geomean_threshold']
        
        # Learning mode should still be disabled
        assert loaded_config['learning_mode']['enabled'] == False
    
    def test_save_calibrated_config_with_sustain_threshold(self, sample_config, tmp_path):
        """Test saving config with sustain threshold for cymbals."""
        learned_thresholds = {
            'cymbals': {
                'geomean_threshold': 3500.0,
                'min_sustain_ms': 175.0
            }
        }
        
        output_path = tmp_path / "calibrated_config.yaml"
        save_calibrated_config(sample_config, learned_thresholds, output_path)
        
        with open(output_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        # Note: Currently save_calibrated_config only updates geomean_threshold
        # This test documents current behavior
        assert loaded_config['cymbals']['geomean_threshold'] == 3500.0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestLearningIntegration:
    """Integration tests for the learning workflow."""
    
    def test_full_learning_workflow(self, sample_config, drum_mapping, tmp_path):
        """Test complete workflow: learn threshold -> save config -> verify."""
        # Create test audio
        audio = create_test_audio(duration_sec=2.0, onsets=[0.5, 1.0, 1.5])
        audio_path = tmp_path / "test_snare.wav"
        sf.write(audio_path, audio, 22050)
        
        # Create MIDI files
        snare_note = drum_mapping.snare
        original_midi_path = tmp_path / "original.mid"
        create_midi_file(original_midi_path, [
            (0.5, snare_note, 100),
            (1.0, snare_note, 100),
            (1.5, snare_note, 100)
        ])
        
        edited_midi_path = tmp_path / "edited.mid"
        create_midi_file(edited_midi_path, [
            (0.5, snare_note, 100),
            (1.0, snare_note, 100)
        ])
        
        # Learn threshold
        result = learn_threshold_from_midi(
            audio_path, original_midi_path, edited_midi_path,
            'snare', sample_config, drum_mapping
        )
        
        # Save calibrated config
        learned_thresholds = {'snare': result}
        output_config_path = tmp_path / "calibrated_config.yaml"
        save_calibrated_config(sample_config, learned_thresholds, output_config_path)
        
        # Verify the saved config
        with open(output_config_path, 'r') as f:
            calibrated_config = yaml.safe_load(f)
        
        # Check that threshold was updated
        assert calibrated_config['snare']['geomean_threshold'] == result['geomean_threshold']
        
        # Check that learning mode is disabled
        assert calibrated_config['learning_mode']['enabled'] == False
        
        # Check that the new threshold is different from original
        assert calibrated_config['snare']['geomean_threshold'] != sample_config['snare']['geomean_threshold']
