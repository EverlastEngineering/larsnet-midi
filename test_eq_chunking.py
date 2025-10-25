"""
Test EQ chunking to ensure it doesn't crash on large audio files.

This test validates that the chunked processing approach prevents memory
issues and can handle large audio buffers that previously caused crashes.
"""

import pytest
import torch
from separation_utils import apply_frequency_cleanup, _apply_filters_to_chunk


class TestEQChunking:
    """Test chunked EQ processing for large audio files"""
    
    def test_small_audio_no_chunking(self):
        """Small audio should be processed without chunking"""
        # 1 second of stereo audio at 44.1kHz
        waveform = torch.randn(2, 44100)
        sr = 44100
        
        eq_config = {
            'kick': {'highpass': 30.0, 'lowpass': 8000.0}
        }
        
        result = apply_frequency_cleanup(waveform, sr, 'kick', eq_config)
        
        assert result.shape == waveform.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
    
    def test_large_audio_with_chunking(self):
        """Large audio should be chunked and processed successfully"""
        # 5 minutes of stereo audio at 44.1kHz (matches crash scenario)
        num_samples = 44100 * 60 * 5  # 13,230,000 samples
        waveform = torch.randn(2, num_samples)
        sr = 44100
        
        eq_config = {
            'kick': {'highpass': 30.0, 'lowpass': 8000.0}
        }
        
        # Use 30-second chunks (default)
        result = apply_frequency_cleanup(waveform, sr, 'kick', eq_config, chunk_size=44100 * 30)
        
        assert result.shape == waveform.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
    
    def test_chunking_preserves_continuity(self):
        """Chunked processing should not introduce major discontinuities"""
        # 2 minutes of audio
        num_samples = 44100 * 120
        waveform = torch.randn(2, num_samples)
        sr = 44100
        
        eq_config = {
            'kick': {'highpass': 30.0, 'lowpass': 8000.0}
        }
        
        # Process with chunking
        chunked = apply_frequency_cleanup(waveform, sr, 'kick', eq_config, chunk_size=44100 * 30)
        
        # Process without chunking
        unchunked = apply_frequency_cleanup(waveform, sr, 'kick', eq_config, chunk_size=num_samples + 1)
        
        # IIR filters have internal state, so chunking will introduce some differences
        # Allow 15% difference (biquad filters lose state between chunks)
        max_diff = torch.abs(chunked - unchunked).max()
        relative_diff = max_diff / torch.abs(waveform).max()
        
        assert relative_diff < 0.15, f"Chunked vs unchunked difference too large: {relative_diff}"
    
    def test_chunk_filter_application(self):
        """Test that _apply_filters_to_chunk correctly applies filters"""
        chunk = torch.randn(2, 44100)
        sr = 44100
        
        stem_eq = {'highpass': 30.0, 'lowpass': 8000.0}
        
        result = _apply_filters_to_chunk(chunk, sr, 'kick', stem_eq)
        
        assert result.shape == chunk.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
    
    def test_edge_case_exact_chunk_boundary(self):
        """Test audio length exactly equals chunk size"""
        chunk_size = 44100 * 30
        waveform = torch.randn(2, chunk_size)
        sr = 44100
        
        eq_config = {
            'kick': {'highpass': 30.0, 'lowpass': 8000.0}
        }
        
        result = apply_frequency_cleanup(waveform, sr, 'kick', eq_config, chunk_size=chunk_size)
        
        assert result.shape == waveform.shape
    
    def test_edge_case_one_sample_over_chunk(self):
        """Test audio that's just one sample over chunk boundary"""
        chunk_size = 44100 * 30
        waveform = torch.randn(2, chunk_size + 1)
        sr = 44100
        
        eq_config = {
            'kick': {'highpass': 30.0, 'lowpass': 8000.0}
        }
        
        result = apply_frequency_cleanup(waveform, sr, 'kick', eq_config, chunk_size=chunk_size)
        
        assert result.shape == waveform.shape
    
    def test_all_stem_types_with_chunking(self):
        """Test all stem types can be processed with chunking"""
        # 2 minutes of audio
        waveform = torch.randn(2, 44100 * 120)
        sr = 44100
        
        eq_config = {
            'kick': {'highpass': 30.0, 'lowpass': 8000.0},
            'snare': {'highpass': 100.0},
            'toms': {'highpass': 60.0, 'lowpass': 1500.0},
            'hihat': {'highpass': 5000.0},
            'cymbals': {'highpass': 3000.0}
        }
        
        for stem_type in ['kick', 'snare', 'toms', 'hihat', 'cymbals']:
            result = apply_frequency_cleanup(
                waveform, sr, stem_type, eq_config, chunk_size=44100 * 30
            )
            assert result.shape == waveform.shape, f"Failed for {stem_type}"
            assert not torch.isnan(result).any(), f"NaN in {stem_type}"
    
    def test_very_low_frequency_highpass(self):
        """Test that very low frequency highpass (30Hz) works with chunking"""
        # This is the specific case that was causing crashes
        waveform = torch.randn(2, 44100 * 300)  # 5 minutes
        sr = 44100
        
        eq_config = {
            'kick': {'highpass': 30.0}  # Very low cutoff that was problematic
        }
        
        result = apply_frequency_cleanup(waveform, sr, 'kick', eq_config, chunk_size=44100 * 30)
        
        assert result.shape == waveform.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
