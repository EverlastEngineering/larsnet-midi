"""
Temporal filtering for MIDI event post-processing.

Functional Core: Pure functions for filtering events based on temporal criteria
(time gaps between events), independent of audio analysis.

This module provides post-processing filters that operate on detected event times
after spectral analysis has completed. These filters examine the inter-event
intervals to identify and remove spurious detections (e.g., bleed, double-triggers).

Usage:
    from stems_to_midi.filters.temporal_filter import filter_by_min_interval
    
    kept_times = filter_by_min_interval(
        event_times=kept_times,
        min_interval_ms=100.0
    )
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class TemporalFilterConfig:
    """
    Configuration for temporal filtering.
    
    Attributes:
        min_interval_ms: Minimum time gap between events in milliseconds.
            Events closer than this to the previous event will be filtered.
        stem_type: The type of stem being filtered (e.g., 'hihat', 'snare').
            Used for logging and configuration inheritance.
    """
    min_interval_ms: float
    stem_type: str = "unknown"


@dataclass(frozen=True)
class TemporalFilterResult:
    """
    Result of temporal filtering operation.
    
    Attributes:
        kept_times: Times of events that passed the filter.
        filtered_times: Times of events that were filtered out.
        intervals_ms: Time gaps between consecutive kept events (in ms).
    """
    kept_times: np.ndarray
    filtered_times: np.ndarray
    intervals_ms: np.ndarray


def calculate_inter_event_intervals(
    event_times: Sequence[float],
    unit: str = "ms"
) -> np.ndarray:
    """
    Calculate time intervals between consecutive events.
    
    Args:
        event_times: Sorted array of event times (in seconds).
        unit: Output unit - 'ms' for milliseconds, 's' for seconds.
            Defaults to 'ms'.
    
    Returns:
        Array of inter-event intervals. First event has no predecessor,
        so returns empty array for single events.
    
    Raises:
        ValueError: If unit is not 'ms' or 's'.
    """
    if len(event_times) < 2:
        return np.array([])
    
    times = np.asarray(event_times)
    intervals = np.diff(times)
    
    if unit == "ms":
        intervals = intervals * 1000.0
    elif unit != "s":
        raise ValueError(f"Invalid unit '{unit}'. Must be 'ms' or 's'.")
    
    return intervals


def filter_by_min_interval(
    event_times: Sequence[float],
    min_interval_ms: float,
    reference_times: Optional[Sequence[float]] = None
) -> np.ndarray:
    """
    Filter events based on minimum time gap from previous event.
    
    This is a "catch" filter that runs after spectral analysis. It examines
    the time gap between each event previous kept event. Events that and the
    occur too close to their predecessor (likely bleed or double-triggers)
    are filtered out.
    
    The filter uses a simple greedy approach: if an event is within
    min_interval_ms of the previously kept event, it is discarded.
    This ensures a minimum temporal separation between all kept events.
    
    Args:
        event_times: Times of events to filter (in seconds). Should be sorted.
        min_interval_ms: Minimum required interval in milliseconds.
            Events with gap < this value from previous event are filtered.
        reference_times: Optional reference times to filter against (e.g., 
            original detection times before other filters). If None, uses
            event_times as reference. Useful when event_times has already
            been filtered but you want to check against original detections.
    
    Returns:
        Array of event times that pass the minimum interval filter.
    
    Example:
        >>> times = [0.0, 0.05, 0.12, 0.25, 0.30]  # seconds
        >>> filter_by_min_interval(times, min_interval_ms=100)
        array([0.  , 0.12, 0.25])  # 50ms and 30ms gaps filtered
    
    Note:
        - The first event is always kept (no previous event to compare)
        - Times are assumed to be in seconds; min_interval_ms converts to seconds
        - The filter is order-dependent; events must be sorted by time
    """
    if len(event_times) == 0:
        return np.array([])
    
    if min_interval_ms <= 0:
        return np.asarray(event_times)
    
    # Use reference times if provided, otherwise use event_times
    ref = reference_times if reference_times is not None else event_times
    
    times = np.asarray(event_times)
    ref = np.asarray(ref)
    
    min_interval_sec = min_interval_ms / 1000.0
    
    kept = [times[0]]  # First event always kept
    
    for i in range(1, len(times)):
        # Calculate gap from last kept event
        gap = times[i] - kept[-1]
        
        if gap >= min_interval_sec:
            kept.append(times[i])
        # Else: event is too close, filter it out
    
    return np.array(kept)


def filter_by_min_interval_with_config(
    event_times: Sequence[float],
    config: TemporalFilterConfig
) -> TemporalFilterResult:
    """
    Filter events using TemporalFilterConfig dataclass.
    
    Convenience wrapper that returns full result with metadata.
    
    Args:
        event_times: Times of events to filter (in seconds).
        config: Configuration dataclass with min_interval_ms and stem_type.
    
    Returns:
        TemporalFilterResult with kept/filtered times and intervals.
    """
    kept_times = filter_by_min_interval(
        event_times=event_times,
        min_interval_ms=config.min_interval_ms
    )
    
    # Find filtered times
    all_times = np.asarray(event_times)
    kept_set = set(kept_times)
    filtered_times = np.array([t for t in all_times if t not in kept_set])
    
    # Calculate intervals between kept events
    intervals = calculate_inter_event_intervals(kept_times, unit="ms")
    
    return TemporalFilterResult(
        kept_times=kept_times,
        filtered_times=filtered_times,
        intervals_ms=intervals
    )


def filter_short_gaps(
    event_times: Sequence[float],
    event_strengths: Optional[Sequence[float]] = None,
    min_interval_ms: float = 50.0,
    keep_stronger: bool = True
) -> np.ndarray:
    """
    Filter events with short time gaps, optionally keeping the stronger one.
    
    When two events are closer than min_interval_ms, this filter can either:
    - Keep the first one (default)
    - Keep the one with higher strength (if event_strengths provided)
    
    This is more sophisticated than filter_by_min_interval as it considers
    event strength when resolving conflicts.
    
    Args:
        event_times: Times of events to filter (in seconds). Sorted.
        event_strengths: Optional strength values for each event.
            If provided and keep_stronger=True, keeps the stronger event
            when there's a conflict.
        min_interval_ms: Minimum required interval in milliseconds.
        keep_stronger: If True and strengths provided, keep the stronger
            of two conflicting events. If False, keep the first.
    
    Returns:
        Array of event times that pass the filter.
    
    Example:
        >>> times = [0.0, 0.05, 0.12]
        >>> strengths = [0.5, 0.8, 0.3]
        >>> filter_short_gaps(times, strengths, min_interval_ms=100, keep_stronger=True)
        array([0.05, 0.12])  # 0.8 > 0.5, so keep 50ms event
    """
    if len(event_times) == 0:
        return np.array([])
    
    if min_interval_ms <= 0:
        return np.asarray(event_times)
    
    times = np.asarray(event_times)
    min_interval_sec = min_interval_ms / 1000.0
    
    if event_strengths is None or not keep_stronger:
        # Simple greedy: just keep if gap is sufficient
        return filter_by_min_interval(times, min_interval_ms)
    
    # With strengths: resolve conflicts by keeping stronger
    strengths = np.asarray(event_strengths)
    
    kept_indices = [0]  # Always keep first
    
    for i in range(1, len(times)):
        gap = times[i] - times[kept_indices[-1]]
        
        if gap >= min_interval_sec:
            # Sufficient gap, keep it
            kept_indices.append(i)
        else:
            # Conflict: decide based on strength
            # If current is stronger than last kept, replace the last
            if strengths[i] > strengths[kept_indices[-1]]:
                kept_indices[-1] = i
            # Else keep the previous one, discard current
    
    return times[kept_indices]


def merge_close_events(
    event_times: Sequence[float],
    merge_threshold_ms: float = 20.0
) -> np.ndarray:
    """
    Merge events that are very close together into a single event.
    
    Unlike filter_by_min_interval which discards close events, this function
    merges them by taking the time of the first event in a cluster.
    
    Args:
        event_times: Times of events to merge (in seconds). Sorted.
        merge_threshold_ms: Events within this many ms are merged into one.
    
    Returns:
        Array of merged event times.
    
    Example:
        >>> times = [0.0, 0.01, 0.02, 0.10, 0.12]  # seconds
        >>> merge_close_events(times, merge_threshold_ms=20)
        array([0.  , 0.10])  # First 3 merged into one
    """
    if len(event_times) == 0:
        return np.array([])
    
    if merge_threshold_ms <= 0:
        return np.asarray(event_times)
    
    times = np.asarray(event_times)
    threshold_sec = merge_threshold_ms / 1000.0
    
    merged = [times[0]]
    
    for i in range(1, len(times)):
        if times[i] - merged[-1] < threshold_sec:
            # Within merge threshold: skip (already merged into last)
            continue
        else:
            merged.append(times[i])
    
    return np.array(merged)


def validate_temporal_filter_config(
    config: TemporalFilterConfig,
    stem_type: str
) -> TemporalFilterConfig:
    """
    Validate and apply defaults for temporal filter configuration.
    
    Args:
        config: User-provided config (may have None values for defaults).
        stem_type: Stem type for stem-specific defaults.
    
    Returns:
        Validated config with defaults applied.
    """
    # Apply stem-type specific defaults
    defaults = {
        'hihat': 50.0,   # 50ms minimum for hihat
        'snare': 30.0,   # 30ms for snare
        'kick': 20.0,   # 20ms for kick
        'toms': 30.0,   # 30ms for toms
        'cymbal': 50.0,  # 50ms for cymbals
    }
    
    min_interval = config.min_interval_ms
    if min_interval is None or min_interval <= 0:
        min_interval = defaults.get(stem_type, 50.0)
    
    return TemporalFilterConfig(
        min_interval_ms=min_interval,
        stem_type=stem_type
    )


# ============================================================================
# Backward compatibility: simple function interface
# ============================================================================

def apply_temporal_filter(
    event_times: List[float],
    min_interval_ms: float
) -> List[float]:
    """
    Apply temporal filtering with simple list interface.
    
    This function provides backward compatibility with existing code that
    uses list-based interfaces. For new code, prefer filter_by_min_interval
    with numpy arrays.
    
    Args:
        event_times: List of event times in seconds.
        min_interval_ms: Minimum interval in milliseconds.
    
    Returns:
        List of filtered event times.
    """
    if not event_times:
        return []
    
    filtered = filter_by_min_interval(event_times, min_interval_ms)
    return filtered.tolist()
