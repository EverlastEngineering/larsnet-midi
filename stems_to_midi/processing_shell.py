"""
Stem Processing Module

Handles the main processing pipeline for converting audio stems to MIDI events.

PGA-only since 2026-06-20. The energy/peak_hold/spectral/geomean paths
were wrapped in CLEANUP markers and are queued for hard-deletion in
phase 7 of agent-plans/pga-cleanup-2026-06.plan.md. New code should
add functionality in ``processing_shell_percentile_gated`` (the live
PGA pipeline) or as a per-event filter (see .github/skills/add-filter).
"""

from pathlib import Path
from typing import Union, List, Dict, Optional

import numpy as np  # type: ignore  # used by _load_and_validate_audio (live)
import soundfile as sf  # type: ignore  # used by _load_and_validate_audio (live)

from .analysis_core import ensure_mono  # used by _load_and_validate_audio (live)
from .config import DrumMapping
from .processing_shell_percentile_gated import process_percentile_gated

__all__ = [
    'process_stem_to_midi',
]


def _load_and_validate_audio(
    audio_path: Union[str, Path],
    config: Dict,
    stem_type: str,
    max_duration: Optional[float] = None
) -> tuple[Optional[np.ndarray], Optional[int]]:
    """
    Load audio file and validate it's usable.
    
    Helper function for process_stem_to_midi (imperative shell).
    
    Args:
        audio_path: Path to audio file
        config: Configuration dictionary
        stem_type: Type of stem (for logging)
        max_duration: Maximum duration in seconds to load (None = load all)
    
    Returns:
        Tuple of (audio, sample_rate) or (None, None) if invalid
    """
    
    print(f"Status Update: Generating MIDI from {stem_type.capitalize()}")
    print(f"    from: {audio_path.name}")
    
    # Load audio (I/O)
    audio, sr = sf.read(str(audio_path))
    
    # Truncate to max_duration if specified
    if max_duration is not None and max_duration > 0:
        max_samples = int(max_duration * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            print(f"    Truncated to {max_duration:.1f} seconds for faster processing")

    # Debug: Print audio shape and sample rate
    print(f"    Audio shape: {audio.shape}, Sample rate: {sr}")
    print(f"    Audio min: {audio.min():.6f}, max: {audio.max():.6f}, mean: {audio.mean():.6f}")
    if audio.shape[0] > sr:
        print(f"    First second min: {audio[:sr].min():.6f}, max: {audio[:sr].max():.6f}, mean: {audio[:sr].mean():.6f}")

    # Handle stereo/mono conversion based on per-stem or global settings
    # Priority: per-stem use_stereo > global force_mono
    stem_config = config.get(stem_type, {})
    use_stereo = stem_config.get('use_stereo', None)
    
    # If per-stem setting not specified, fall back to global force_mono
    if use_stereo is None:
        # Legacy behavior: respect global force_mono setting
        use_stereo = not config['audio'].get('force_mono', True)
    
    if not use_stereo and audio.ndim == 2:
        # Convert to mono
        audio = ensure_mono(audio)
        print("    Converted stereo to mono")
    elif use_stereo and audio.ndim == 2:
        # Keep stereo for spatial analysis
        print("    Keeping stereo for spatial analysis")
    elif use_stereo and audio.ndim == 1:
        # Mono file but stereo requested - just keep mono
        print("    Audio is mono (no stereo info available)")

    # Check if audio is essentially silent
    max_amplitude = np.max(np.abs(audio))
    print(f"    Max amplitude: {max_amplitude:.6f}")

    silence_threshold = config.get('audio', {}).get('silence_threshold', 0.001)
    if max_amplitude < silence_threshold:
        print("    Audio is silent, skipping...")
        return None, None

    # Amplitude normalization (per-stem or global setting)
    # This ensures spectral energy thresholds work consistently across quiet/loud recordings
    normalize_amplitude = stem_config.get('normalize_amplitude', 
                                          config.get('audio', {}).get('normalize_amplitude', False))
    
    if normalize_amplitude and max_amplitude > 0:
        target_amplitude = config.get('audio', {}).get('target_amplitude', 0.8)
        if max_amplitude < target_amplitude:  # Only normalize if too quiet
            scale_factor = target_amplitude / max_amplitude
            audio = audio * scale_factor
            print(f"    Normalized amplitude: {max_amplitude:.6f} -> {target_amplitude:.2f} (scale: {scale_factor:.2f}x)")

    # Stereo channel balance normalization (per-stem or global setting)
    # This equalizes L/R channel levels for fair detection when one channel is louder
    normalize_stereo_balance = stem_config.get('normalize_stereo_balance',
                                               config.get('audio', {}).get('normalize_stereo_balance', False))
    
    if normalize_stereo_balance and audio.ndim == 2:
        left_rms = np.sqrt(np.mean(audio[:, 0] ** 2))
        right_rms = np.sqrt(np.mean(audio[:, 1] ** 2))
        
        if left_rms > 0 and right_rms > 0:
            # Normalize each channel to equal RMS
            target_rms = (left_rms + right_rms) / 2
            left_scale = target_rms / left_rms
            right_scale = target_rms / right_rms
            
            audio[:, 0] *= left_scale
            audio[:, 1] *= right_scale
            
            print(f"    Balanced stereo channels: L×{left_scale:.2f}, R×{right_scale:.2f} (R/L was {right_rms/left_rms:.2f}x)")

    return audio, sr


def process_stem_to_midi(
    audio_path: Union[str, Path],
    stem_type: str,
    drum_mapping: DrumMapping,
    config: Dict,
    min_velocity: int = 80,
    max_velocity: int = 110,
    max_duration: Optional[float] = None
) -> List[Dict]:
    """
    Process a drum stem and extract MIDI events.

    This is a thin coordinator that short-circuits to the PGA pipeline
    (processing_shell_percentile_gated.process_percentile_gated) when
    ``<stem_type>.use_pga_detection`` is true (the default for all 5
    stems since 2026-06-20). The legacy energy/spectral pipeline that
    used to live below this function is queued for deletion; see the
    module docstring for the cleanup status.

    Args:
        audio_path: Path to audio file
        stem_type: Type of stem ('kick', 'snare', 'toms', 'hihat', 'cymbals')
        drum_mapping: MIDI note mapping
        min_velocity: Minimum MIDI velocity
        max_velocity: Maximum MIDI velocity
        max_duration: Maximum duration in seconds to analyze (None = all)
    
    Returns:
        Dict with:
            'events': List of MIDI events
            'all_onset_data': List of all detected onsets (kept + filtered)
            'sensitive_onset_data': Onsets from max-sensitivity energy
                                     detection (interactive tuning)
            'spectral_onset_data': Onsets from the spectral-transient
                                    detector (complementary signal,
                                    always present even if empty)
            'spectral_config': Spectral config used for this stem
            'envelope_data': Energy envelope for waveform visualization
    """
    # Percentile-gated shortcut (2026-06-15; config-driven
    # 2026-06-18). Stems that opt into the PGA-only pipeline
    # via ``<stem_type>.use_pga_detection: true`` delegate
    # to the dedicated function and skip the
    # energy/spectral/pan paths entirely. Default is
    # ``False`` — the legacy energy-based pipeline still
    # runs for any stem that doesn't opt in. The old
    # hard-coded ``stem_type == 'toms'`` check was replaced
    # when the toms stem became configurable: a project
    # enables PGA for its toms by setting
    # ``toms.use_pga_detection: true`` in its
    # per-project midiconfig.yaml.
    if config.get(stem_type, {}).get('use_pga_detection', False):
        return process_percentile_gated(
            audio_path, drum_mapping, config,
            min_velocity, max_velocity, stem_type=stem_type,
        )
