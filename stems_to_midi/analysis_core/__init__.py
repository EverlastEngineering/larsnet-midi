"""
Analysis Package

Modular functional core components for audio analysis, spectral processing,
onset filtering, and threshold learning.

Modules:
- audio_utils: Audio analysis utilities
- spectral_utils: Spectral calculations and configuration
- classification: Instrument classification functions
- onset_filtering: Onset filtering and analysis
- threshold_learning: Threshold learning functions
- time_utils: Time and MIDI conversion utilities
"""

# Audio utilities
from .audio_utils import (
    ensure_mono,
    calculate_peak_amplitude,
    calculate_sustain_duration,
    calculate_event_durations,
    calculate_amplitude_at_time,
    calculate_attack_sharpness,
    calculate_envelope_continuity,
    calculate_peak_prominence,
    calculate_spectral_centroid,
    calculate_spectral_flux,
    detect_pitch_autocorrelation,
    detect_pitch,
    calculate_gap_from_previous,
    calculate_spectral_energies,
    analyze_cymbal_decay_pattern,
    time_to_sample,
    extract_audio_segment,
)

# Spectral utilities
from .spectral_utils import (
    get_spectral_config_for_stem,
    calculate_geomean,
    calculate_statistical_params,
    calculate_badness_score,
    should_keep_onset,
    normalize_values,
)

# Classification functions
from .classification import (
    classify_tom_pitch,
    classify_cymbal_pitch,
    classify_snare_pitch,
    classify_cymbal_by_pan,
    extract_onset_features,
)

# Onset filtering
from .onset_filtering import (
    mark_reverb_continuations,
    filter_onsets_by_spectral,
    analyze_onset_spectral,
)

# Threshold learning
from .threshold_learning import (
    calculate_velocities_from_features,
    calculate_threshold_from_distributions,
    calculate_classification_accuracy,
    predict_classification,
    analyze_threshold_performance,
    estimate_velocity,
)

# Time utilities
from .time_utils import (
    seconds_to_beats,
    prepare_midi_events_for_writing,
)


__all__ = [
    # Audio utilities
    'ensure_mono',
    'calculate_peak_amplitude',
    'calculate_sustain_duration',
    'calculate_event_durations',
    'calculate_amplitude_at_time',
    'calculate_attack_sharpness',
    'calculate_envelope_continuity',
    'calculate_peak_prominence',
    'calculate_spectral_centroid',
    'calculate_spectral_flux',
    'detect_pitch_autocorrelation',
    'detect_pitch',
    'calculate_gap_from_previous',
    'calculate_spectral_energies',
    'analyze_cymbal_decay_pattern',
    'time_to_sample',
    'extract_audio_segment',
    
    # Spectral utilities
    'get_spectral_config_for_stem',
    'calculate_geomean',
    'calculate_statistical_params',
    'calculate_badness_score',
    'should_keep_onset',
    'normalize_values',
    
    # Classification
    'classify_tom_pitch',
    'classify_cymbal_pitch',
    'classify_snare_pitch',
    'classify_cymbal_by_pan',
    'extract_onset_features',
    
    # Onset filtering
    'mark_reverb_continuations',
    'filter_onsets_by_spectral',
    'analyze_onset_spectral',
    
    # Threshold learning
    'calculate_velocities_from_features',
    'calculate_threshold_from_distributions',
    'calculate_classification_accuracy',
    'predict_classification',
    'analyze_threshold_performance',
    'estimate_velocity',
    
    # Time utilities
    'seconds_to_beats',
    'prepare_midi_events_for_writing',
]
