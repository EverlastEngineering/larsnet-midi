"""
Spectral Utilities and Configuration

Pure helper functions for spectral analysis configuration and calculations.

Functions:
- get_spectral_config_for_stem: Get spectral config for specific stem type
- calculate_geomean: Calculate geometric mean of energy values
- calculate_statistical_params: Compute normalization parameters from onset data
- calculate_badness_score: Compute normalized badness score for outlier detection
- should_keep_onset: Determine if onset should be kept based on thresholds
- normalize_values: Normalize array to 0-1 range
"""

import numpy as np
from typing import Dict, List, Optional

from .audio_utils import calculate_spectral_energies


def get_spectral_config_for_stem(stem_type: str, config: Dict) -> Dict:
    """
    Get spectral configuration for a specific stem type.
    
    Uses domain-specific frequency band names so that downstream data
    (onset_data dicts, analysis.json) is self-documenting.
    
    Pure function - extracts config without side effects.
    
    Args:
        stem_type: Type of stem ('kick', 'snare', 'toms', 'hihat', 'cymbals')
        config: Full configuration dictionary
    
    Returns:
        Dict with:
        - freq_ranges: Dict of frequency ranges keyed by domain-specific names
        - energy_labels: Dict mapping band names to display labels
        - geomean_bands: Ordered list of band names used for geomean calculation
        - geomean_threshold: Threshold for filtering (or None)
        - min_sustain_ms: Minimum sustain duration (or None)
        - min_strength_threshold: Minimum onset strength (or None)
        - display_hints: List of context strings for debug output (empty if none)
        
        Capability flags (drive pipeline flow without stem_type checks):
        - velocity_source: 'geomean' | 'onset_strength' | 'peak_amplitude'
        - has_sustain_analysis: bool — collect sustain durations during filtering
        - use_sustain_duration: bool — use sustain envelope for MIDI note duration
        - has_spectral_data: bool — collect per-band spectral data for classification
        - filter_mode: 'require_both' | 'geomean_only' — how geomean + sustain combine
    """
    stem_config = config.get(stem_type, {})
    
    if stem_type == 'snare':
        return {
            'freq_ranges': {
                'low': (stem_config['low_freq_min'], stem_config['low_freq_max']),
                'body': (stem_config['body_freq_min'], stem_config['body_freq_max']),
                'wire': (stem_config['wire_freq_min'], stem_config['wire_freq_max'])
            },
            'energy_labels': {
                'body': 'Body',
                'wire': 'Wire'
            },
            'geomean_bands': ['body', 'wire'],
            'geomean_threshold': stem_config.get('geomean_threshold'),
            'min_sustain_ms': None,
            'min_strength_threshold': stem_config.get('min_strength_threshold'),
            'display_hints': [],
            'velocity_source': 'geomean',
            'has_sustain_analysis': False,
            'use_sustain_duration': False,
            'has_spectral_data': False,
            'filter_mode': 'geomean_only'
        }
    
    elif stem_type == 'kick':
        return {
            'freq_ranges': {
                'fundamental': (stem_config['fundamental_freq_min'], stem_config['fundamental_freq_max']),
                'body': (stem_config['body_freq_min'], stem_config['body_freq_max']),
                'attack': (stem_config['attack_freq_min'], stem_config['attack_freq_max'])
            },
            'energy_labels': {
                'fundamental': 'Fundamental',
                'body': 'Body',
                'attack': 'Attack'
            },
            'geomean_bands': ['fundamental', 'body', 'attack'],
            'geomean_threshold': stem_config.get('geomean_threshold'),
            'min_sustain_ms': None,
            'min_strength_threshold': stem_config.get('min_strength_threshold'),
            'display_hints': [],
            'velocity_source': 'geomean',
            'has_sustain_analysis': False,
            'use_sustain_duration': False,
            'has_spectral_data': False,
            'filter_mode': 'geomean_only'
        }
    
    elif stem_type == 'toms':
        return {
            'freq_ranges': {
                'fundamental': (stem_config['fundamental_freq_min'], stem_config['fundamental_freq_max']),
                'body': (stem_config['body_freq_min'], stem_config['body_freq_max'])
            },
            'energy_labels': {
                'fundamental': 'Fundamental',
                'body': 'Body'
            },
            'geomean_bands': ['fundamental', 'body'],
            'geomean_threshold': stem_config.get('geomean_threshold'),
            'min_sustain_ms': None,
            'min_strength_threshold': stem_config.get('min_strength_threshold'),
            'display_hints': [],
            'velocity_source': 'geomean',
            'has_sustain_analysis': False,
            'use_sustain_duration': False,
            'has_spectral_data': False,
            'filter_mode': 'geomean_only'
        }
    
    elif stem_type == 'hihat':
        min_sustain = stem_config.get('min_sustain_ms', 25)
        open_sustain = stem_config.get('open_sustain_ms', 100)
        hints = [
            f"Minimum sustain duration: {min_sustain}ms (filters out handclap bleed)",
            f"Open/Closed threshold: {open_sustain}ms (>={open_sustain}ms = open hihat)"
        ]
        return {
            'freq_ranges': {
                'body': (stem_config['body_freq_min'], stem_config['body_freq_max']),
                'sizzle': (stem_config['sizzle_freq_min'], stem_config['sizzle_freq_max'])
            },
            'energy_labels': {
                'body': 'Body',
                'sizzle': 'Sizzle'
            },
            'geomean_bands': ['body', 'sizzle'],
            'geomean_threshold': stem_config.get('geomean_threshold'),
            'min_sustain_ms': min_sustain,
            'min_strength_threshold': stem_config.get('min_strength_threshold'),
            'display_hints': hints,
            'velocity_source': 'onset_strength',
            'has_sustain_analysis': True,
            'use_sustain_duration': False,
            'has_spectral_data': True,
            'filter_mode': 'geomean_only'  # Sustain filter applied at end, after reverb
        }
    
    elif stem_type == 'cymbals':
        min_sustain = stem_config.get('min_sustain_ms', 150)
        hints = [
            f"Minimum sustain duration: {min_sustain}ms"
        ]
        return {
            'freq_ranges': {
                'body': (stem_config.get('body_freq_min', 1000), stem_config.get('body_freq_max', 4000)),
                'brilliance': (stem_config.get('brilliance_freq_min', 4000), stem_config.get('brilliance_freq_max', 10000))
            },
            'energy_labels': {
                'body': 'Body',
                'brilliance': 'Brilliance'
            },
            'geomean_bands': ['body', 'brilliance'],
            'geomean_threshold': stem_config.get('geomean_threshold'),
            'min_sustain_ms': min_sustain,
            'min_strength_threshold': stem_config.get('min_strength_threshold'),
            'display_hints': hints,
            'velocity_source': 'geomean',
            'has_sustain_analysis': True,
            'use_sustain_duration': True,
            'has_spectral_data': False,
            'filter_mode': 'require_both'
        }
    
    else:
        raise ValueError(f"Unknown stem type: {stem_type}")


def calculate_geomean(
    primary_energy: float,
    secondary_energy: float,
    tertiary_energy: Optional[float] = None
) -> float:
    """
    Calculate geometric mean of energy values.
    
    Pure function - no side effects.
    
    Args:
        primary_energy: First energy value
        secondary_energy: Second energy value
        tertiary_energy: Optional third energy value (for 3-way geomean)
    
    Returns:
        Geometric mean (sqrt of product for 2 values, cube root for 3 values)
    """
    if tertiary_energy is not None and tertiary_energy > 0:
        # 3-way geometric mean: cube root of product
        return float(np.cbrt(primary_energy * secondary_energy * tertiary_energy))
    else:
        # 2-way geometric mean: square root of product
        return float(np.sqrt(primary_energy * secondary_energy))


def calculate_statistical_params(onset_data_list: List[Dict]) -> Dict[str, float]:
    """
    Analyze full dataset of onsets to compute normalization parameters.
    
    Used for statistical outlier detection to identify snare bleed in kicks
    by comparing the first two geomean band energies' ratio and total energy
    against dataset medians.
    
    Pure function - no side effects.
    
    Args:
        onset_data_list: List of onset data dicts with domain-specific energy keys
                         (e.g. 'fundamental_energy', 'body_energy') and 'total_energy'.
                         Each dict must have 'geomean_bands' listing band names.
    
    Returns:
        Dict with median and spread values:
        - median_ratio: Median band1/band2 ratio across all events
        - median_total: Median total energy across all events
        - ratio_spread: Standard deviation of ratios
        - total_spread: Standard deviation of total energies
    """
    if not onset_data_list:
        return {
            'median_ratio': 1.0,
            'median_total': 100.0,
            'ratio_spread': 1.0,
            'total_spread': 1.0
        }
    
    # Extract first two geomean band energies using domain-specific names
    bands = onset_data_list[0].get('geomean_bands', [])
    if len(bands) >= 2:
        band1_key = f'{bands[0]}_energy'
        band2_key = f'{bands[1]}_energy'
    else:
        # Fallback: shouldn't happen with proper data
        band1_key = 'fundamental_energy'
        band2_key = 'body_energy'
    
    band1_energies = np.array([d.get(band1_key, 0.0) for d in onset_data_list])
    band2_energies = np.array([d.get(band2_key, 0.0) for d in onset_data_list])
    total_energies = np.array([d['total_energy'] for d in onset_data_list])
    
    # Calculate band1/band2 ratios (with safety for division by zero)
    ratios = band1_energies / np.maximum(band2_energies, 1e-9)
    
    params = {
        'median_ratio': float(np.median(ratios)),
        'median_total': float(np.median(total_energies)),
        'ratio_spread': float(np.std(ratios)) if len(ratios) > 1 else 1.0,
        'total_spread': float(np.std(total_energies)) if len(total_energies) > 1 else 1.0
    }
    
    # Ensure non-zero spreads to avoid division by zero
    if params['ratio_spread'] < 1e-9:
        params['ratio_spread'] = 1e-9
    if params['total_spread'] < 1e-9:
        params['total_spread'] = 1e-9
    
    return params


def calculate_badness_score(
    onset_data: Dict,
    statistical_params: Dict[str, float],
    ratio_weight: float = 0.7,
    total_weight: float = 0.3
) -> float:
    """
    Compute normalized badness score for a single onset.
    
    Measures how much an onset deviates from typical kicks in the dataset.
    Snare bleed typically has lower Primary/Secondary ratio and different total energy.
    
    Pure function - no side effects.
    
    Args:
        onset_data: Dict with domain-specific energy keys and 'geomean_bands'
        statistical_params: Dict from calculate_statistical_params()
        ratio_weight: Weight for ratio deviation (0-1)
        total_weight: Weight for total energy deviation (0-1)
    
    Returns:
        Badness score in range [0, 1]:
        - 0.0 = perfectly typical kick
        - 1.0 = maximum deviation (likely artifact/bleed)
    """
    # Calculate this onset's ratio using first two geomean bands
    bands = onset_data.get('geomean_bands', [])
    if len(bands) >= 2:
        band1 = onset_data.get(f'{bands[0]}_energy', 0.0)
        band2 = onset_data.get(f'{bands[1]}_energy', 0.0)
    else:
        band1 = 0.0
        band2 = 0.0
    ratio = band1 / max(band2, 1e-9)
    total = onset_data['total_energy']
    
    # Calculate normalized deviations
    # Ratio deviation: how much lower is this ratio compared to median?
    ratio_dev = (statistical_params['median_ratio'] - ratio) / (statistical_params['median_ratio'] + 1e-9)
    ratio_dev = float(np.clip(ratio_dev, 0, 1))  # Only penalize lower ratios, not higher
    
    # Total energy deviation: how different is total energy from median?
    total_dev = abs(statistical_params['median_total'] - total) / (statistical_params['median_total'] + 1e-9)
    total_dev = float(np.clip(total_dev, 0, 1))
    
    # Weighted combination
    score = ratio_weight * ratio_dev + total_weight * total_dev
    
    return float(np.clip(score, 0, 1))


def should_keep_onset(
    geomean: float,
    sustain_ms: Optional[float],
    geomean_threshold: Optional[float],
    min_sustain_ms: Optional[float],
    filter_mode: str = 'geomean_only',
    strength: Optional[float] = None,
    min_strength_threshold: Optional[float] = None,
    # Deprecated: use filter_mode instead
    stem_type: Optional[str] = None
) -> bool:
    """
    Determine if an onset should be kept based on spectral/sustain/strength criteria.
    
    Pure function - decision logic without side effects.
    
    Args:
        geomean: Geometric mean of primary and secondary energy
        sustain_ms: Sustain duration in milliseconds (None if not calculated)
        geomean_threshold: Threshold for geomean filtering (None to disable)
        min_sustain_ms: Minimum sustain threshold (None to disable)
        filter_mode: How geomean and sustain thresholds combine:
            - 'require_both': require BOTH geomean AND sustain (if both thresholds set)
            - 'geomean_only': use geomean threshold only (sustain ignored for filtering)
        strength: Onset strength value (0-1, normalized)
        min_strength_threshold: Minimum onset strength required (None to disable)
        stem_type: Deprecated — infers filter_mode if filter_mode not explicitly set
    
    Returns:
        True if onset should be kept, False if it should be rejected
    """
    # Backward compatibility: infer filter_mode from stem_type if stem_type is provided
    # and filter_mode is at default. This allows callers to migrate incrementally.
    if stem_type is not None and filter_mode == 'geomean_only':
        if stem_type == 'cymbals':
            filter_mode = 'require_both'
        # All other stems use 'geomean_only' (the default), so no change needed
    
    # Check strength first (applies to all filter modes)
    if min_strength_threshold is not None and strength is not None:
        if strength < min_strength_threshold:
            return False
    
    # If no filtering enabled, keep everything
    if geomean_threshold is None and min_sustain_ms is None:
        return True
    
    # require_both: require BOTH geomean AND sustain (if both thresholds set)
    if filter_mode == 'require_both':
        if geomean_threshold is not None and min_sustain_ms is not None:
            geomean_ok = geomean > geomean_threshold
            sustain_ok = (sustain_ms is not None) and (sustain_ms >= min_sustain_ms)
            return geomean_ok and sustain_ok
        elif min_sustain_ms is not None:
            return (sustain_ms is not None) and (sustain_ms >= min_sustain_ms)
        elif geomean_threshold is not None:
            return geomean > geomean_threshold
    
    # geomean_only: use geomean threshold only
    else:
        if geomean_threshold is not None:
            if geomean <= geomean_threshold:
                return False
        return True


def normalize_values(values: np.ndarray) -> np.ndarray:
    """
    Normalize array of values to 0-1 range.
    
    Pure function - no side effects.
    
    Args:
        values: Array of values to normalize
    
    Returns:
        Normalized array (0-1 range)
    """
    if len(values) == 0:
        return values
    
    max_val = np.max(values)
    if max_val > 0:
        return values / max_val
    else:
        return np.ones_like(values)
