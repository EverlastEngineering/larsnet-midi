"""
Threshold Learning and Analysis

Pure helper functions for threshold learning and performance analysis.

Functions:
- calculate_velocities_from_features: Convert feature values to MIDI velocities
- calculate_threshold_from_distributions: Calculate optimal threshold from data
- calculate_classification_accuracy: Calculate classification accuracy
- predict_classification: Predict classification based on thresholds
- analyze_threshold_performance: Analyze threshold performance on dataset
"""

import numpy as np
from typing import Dict, List, Optional

from .spectral_utils import should_keep_onset


def calculate_velocities_from_features(
    feature_values: np.ndarray,
    min_velocity: int,
    max_velocity: int
) -> np.ndarray:
    """
    Calculate MIDI velocities from normalized feature values.
    
    Pure function - no side effects.
    
    Args:
        feature_values: Normalized feature values (0-1 range, can be any feature like geomean or amplitude)
        min_velocity: Minimum MIDI velocity
        max_velocity: Maximum MIDI velocity
    
    Returns:
        Array of MIDI velocities (1-127)
    """
    if len(feature_values) == 0:
        return np.array([], dtype=int)
    
    # Calculate velocities using estimate_velocity for each value
    velocities = np.array([
        estimate_velocity(value, min_velocity, max_velocity)
        for value in feature_values
    ])
    
    return velocities


def estimate_velocity(strength: float, min_vel: int = 40, max_vel: int = 127) -> int:
    """
    Convert onset strength to MIDI velocity.
    
    Pure function - no side effects.
    
    Args:
        strength: Onset strength (0-1)
        min_vel: Minimum MIDI velocity
        max_vel: Maximum MIDI velocity
    
    Returns:
        MIDI velocity (1-127)
    """
    velocity = int(min_vel + strength * (max_vel - min_vel))
    return np.clip(velocity, 1, 127)


def calculate_threshold_from_distributions(
    kept_values: List[float],
    removed_values: List[float]
) -> Optional[float]:
    """
    Calculate optimal threshold as midpoint between max removed and min kept.
    
    Pure function - no side effects.
    
    Args:
        kept_values: List of values that should be kept (true positives)
        removed_values: List of values that should be removed (false positives)
    
    Returns:
        Suggested threshold (midpoint), or None if insufficient data
    """
    if not kept_values or not removed_values:
        return None
    
    min_kept = min(kept_values)
    max_removed = max(removed_values)
    
    # Threshold is midpoint between max removed and min kept
    suggested_threshold = (max_removed + min_kept) / 2.0
    
    return suggested_threshold


def calculate_classification_accuracy(
    user_actions: List[str],
    predictions: List[str]
) -> Dict[str, float]:
    """
    Calculate classification accuracy between user actions and predictions.
    
    Pure function - no side effects.
    
    Args:
        user_actions: List of 'KEPT' or 'REMOVED' (ground truth)
        predictions: List of 'KEPT' or 'REMOVED' (predicted by threshold)
    
    Returns:
        Dict with:
        - correct_count: Number of correct predictions
        - total_count: Total number of predictions
        - accuracy: Accuracy percentage (0-100)
    """
    if len(user_actions) != len(predictions) or len(user_actions) == 0:
        return {
            'correct_count': 0,
            'total_count': 0,
            'accuracy': 0.0
        }
    
    correct_count = sum(1 for user, pred in zip(user_actions, predictions) if user == pred)
    total_count = len(user_actions)
    accuracy = (correct_count / total_count) * 100.0
    
    return {
        'correct_count': correct_count,
        'total_count': total_count,
        'accuracy': accuracy
    }


def predict_classification(
    geomean: float,
    geomean_threshold: float,
    sustain_ms: Optional[float] = None,
    sustain_threshold: Optional[float] = None,
    filter_mode: str = 'geomean_only',
    # Deprecated: use filter_mode instead
    stem_type: Optional[str] = None
) -> str:
    """
    Predict classification (KEPT/REMOVED) based on thresholds.
    
    Pure function - no side effects.
    
    Args:
        geomean: Geometric mean value
        geomean_threshold: Threshold for geomean
        sustain_ms: Sustain duration in milliseconds (optional)
        sustain_threshold: Threshold for sustain (optional)
        filter_mode: 'require_both' | 'geomean_only'
        stem_type: Deprecated — infers filter_mode if provided
    
    Returns:
        'KEPT' or 'REMOVED'
    """
    # Backward compatibility
    if stem_type is not None and filter_mode == 'geomean_only':
        if stem_type == 'cymbals':
            filter_mode = 'require_both'
    
    # require_both: require geomean AND sustain if both thresholds provided
    if filter_mode == 'require_both' and sustain_threshold is not None and sustain_ms is not None:
        geomean_ok = geomean > geomean_threshold
        sustain_ok = sustain_ms > sustain_threshold
        return 'KEPT' if (geomean_ok and sustain_ok) else 'REMOVED'
    else:
        return 'KEPT' if geomean > geomean_threshold else 'REMOVED'


def analyze_threshold_performance(
    analysis_data: List[Dict],
    geomean_threshold: float,
    sustain_threshold: Optional[float] = None,
    filter_mode: str = 'geomean_only',
    # Deprecated: use filter_mode instead
    stem_type: Optional[str] = None
) -> Dict:
    """
    Analyze threshold performance on a dataset.
    
    Pure function - no side effects.
    
    Args:
        analysis_data: List of dicts with 'is_kept', 'geomean', 'sustain_ms' (optional)
        geomean_threshold: Threshold to test
        sustain_threshold: Sustain threshold (optional)
        filter_mode: 'require_both' | 'geomean_only'
        stem_type: Deprecated — infers filter_mode if provided
    
    Returns:
        Dict with:
        - user_actions: List[str] ('KEPT' or 'REMOVED')
        - predictions: List[str] ('KEPT' or 'REMOVED')
        - results: List[str] (comparison results like '✓ Both OK')
        - accuracy: Dict from calculate_classification_accuracy
    """
    user_actions = []
    predictions = []
    results = []
    
    for data in analysis_data:
        user_action = 'KEPT' if data['is_kept'] else 'REMOVED'
        
        prediction = predict_classification(
            data['geomean'],
            geomean_threshold,
            data.get('sustain_ms'),
            sustain_threshold,
            filter_mode=filter_mode,
            stem_type=stem_type
        )
        
        user_actions.append(user_action)
        predictions.append(prediction)
        
        # Determine result string
        if user_action == prediction:
            results.append('✓ Correct')
        else:
            results.append('✗ Wrong')
    
    accuracy = calculate_classification_accuracy(user_actions, predictions)
    
    return {
        'user_actions': user_actions,
        'predictions': predictions,
        'results': results,
        'accuracy': accuracy
    }
