"""
Filtering modules for MIDI event post-processing.

This package provides functional core modules for filtering detected events
based on various criteria. The filters operate on event times after spectral
analysis has completed.

Modules:
    - temporal_filter: Time-based filtering (min interval between events)
"""

from stems_to_midi.filters.temporal_filter import (
    TemporalFilterConfig,
    TemporalFilterResult,
    filter_by_min_interval,
    filter_by_min_interval_with_config,
    filter_short_gaps,
    merge_close_events,
    calculate_inter_event_intervals,
    apply_temporal_filter,
    validate_temporal_filter_config,
)

__all__ = [
    'TemporalFilterConfig',
    'TemporalFilterResult',
    'filter_by_min_interval',
    'filter_by_min_interval_with_config',
    'filter_short_gaps',
    'merge_close_events',
    'calculate_inter_event_intervals',
    'apply_temporal_filter',
    'validate_temporal_filter_config',
]
