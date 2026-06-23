"""
Analysis Package (PGA-only)

2026-06-22: trimmed to the live, PGA-pipeline-only surface. The
legacy geomean / energy / statistical / learning modules
(``spectral_utils``, ``onset_filtering``, ``threshold_learning``,
``classification``, ``audio_utils`` minus ``ensure_mono``,
``time_utils`` minus ``prepare_midi_events_for_writing``,
``learning.py``, ``optimization_core.py``,
``energy_detection_core.py``, ``energy_detection_shell.py``,
``detection_shell.py``) were deleted as part of the
PGA-universal cleanup. See
``agent-plans/legacy-code-removal.plan.md`` for the audit and
removal plan.

Modules:
- audio_utils: Audio utilities (ensure_mono only)
- time_utils: Time and MIDI conversion utilities
"""

from .audio_utils import (
    ensure_mono,
)

from .time_utils import (
    seconds_to_beats,
    prepare_midi_events_for_writing,
)


__all__ = [
    # Audio utilities
    'ensure_mono',

    # Time utilities
    'seconds_to_beats',
    'prepare_midi_events_for_writing',
]
