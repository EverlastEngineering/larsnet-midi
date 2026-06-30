"""
User event overrides — utility for the WebUI's
``event_overrides.json`` sidecar.

When the user clicks (or click-cycles) an event in the WebUI
waveform panel, the panel writes a per-project override file at::

    user_files/<project>/midi/event_overrides.json

with the shape::

    {
        "<stem_type>": {
            "<time_str>": {
                "status": "KEPT",
                [optional "classification": int]
            },
            ...
        }
    }

The override record carries at minimum a ``status`` field
(``"KEPT"`` or ``"FILTERED"``). The optional ``classification``
is the user's per-event note override (snare body vs rimshot vs
clap, toms low vs mid vs high, etc.) — the rebuild path uses it
to drive the per-event MIDI note via the standard
classify_notes path.

This module is the read+write utility for that file. The
rebuild path uses ``clean_overrides`` after every Save to
prune entries whose state now matches the sidecar's natural
state — keeping the file intentionally minimal.

Pure functions. File I/O is the only side-effect (functional
core / imperative shell split).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


__all__ = [
    'load_event_overrides',
    'save_event_overrides',
    'clean_overrides',
    'EventOverridesError',
]


class EventOverridesError(RuntimeError):
    """Raised when the override file is present but malformed
    (e.g. not a JSON object, or values are not the expected
    shape). The WebUI surfaces this as a toast — the file is
    left in place so the user can hand-edit it if needed."""


def load_event_overrides(
    project_dir: str | Path,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Read the per-project ``midi/event_overrides.json`` file.

    Args:
        project_dir: Path to the project directory (the one
            under ``user_files/`` that contains ``midi/``,
            ``stems/``, ``midiconfig.yaml``, etc.). Can be a
            string or a ``pathlib.Path``; the function resolves
            ``midi/event_overrides.json`` underneath it.

    Returns:
        ``None`` when the file does not exist (the common case
        — most projects have no overrides). A ``dict`` mapping
        stem type -> override record dict when the file exists.
        Each override record is a dict with at minimum a
        ``status`` key (``"FILTERED"`` or ``"KEPT"``); other keys
        are preserved verbatim.

    Raises:
        EventOverridesError: The file exists but is not valid
            JSON, or its top-level value is not a dict, or any
            per-stem value is not a dict.
    """
    project_dir = Path(project_dir)
    override_path = project_dir / 'midi' / 'event_overrides.json'

    if not override_path.exists():
        return None

    try:
        with open(override_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise EventOverridesError(
            f"event_overrides.json at {override_path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise EventOverridesError(
            f"event_overrides.json at {override_path} must be a JSON "
            f"object (stem -> time -> override), got {type(data).__name__}"
        )

    # Light schema check: stem values must be dicts (time -> override).
    for stem, stem_overrides in data.items():
        if not isinstance(stem_overrides, dict):
            raise EventOverridesError(
                f"event_overrides.json at {override_path}: stem "
                f"{stem!r} value must be an object, got "
                f"{type(stem_overrides).__name__}"
            )
        # Per-time values must be dicts (override record). The
        # legacy file format used strings ("KEPT" / "FILTERED")
        # — per the user's "nuke the old files and start fresh"
        # direction (2026-06-30), legacy files are unsupported
        # and must be deleted or rewritten by hand. Catching the
        # legacy shape here is the safety net.
        for time_key, override_record in stem_overrides.items():
            if not isinstance(override_record, dict):
                raise EventOverridesError(
                    f"event_overrides.json at {override_path}: stem "
                    f"{stem!r} time {time_key!r} value must be an "
                    f"object (override record), got "
                    f"{type(override_record).__name__}. Legacy "
                    f"string-valued entries are no longer supported "
                    f"as of 2026-06-30 — please rewrite the file "
                    f"with {{ status: 'KEPT'|'FILTERED', "
                    f"[classification]: N }} entries."
                )

    return data


def save_event_overrides(
    project_dir: str | Path,
    overrides: Dict[str, Dict[str, Any]],
) -> Path:
    """Write the per-project override file.

    Args:
        project_dir: Path to the project directory.
        overrides: The override dict in the canonical shape
            (stem -> time -> override record).

    Returns:
        The path of the written file.
    """
    project_dir = Path(project_dir)
    midi_dir = project_dir / 'midi'
    midi_dir.mkdir(parents=True, exist_ok=True)
    override_path = midi_dir / 'event_overrides.json'

    with open(override_path, 'w') as f:
        json.dump(overrides, f, indent=2)

    return override_path


def _event_key(ev: Dict) -> str:
    """The override key for an event. Uses ``ev['frame']``
    (integer frame index from the PGA detector) when
    available; falls back to a 4-decimal time string for
    legacy data without a frame field.

    2026-06-30: switched from time-string to frame-integer to
    fix the user-reported "time: 2.954 vs '2.9540' mismatch" —
    a file with non-4-decimal time keys would never match the
    lookup. Frame is always an integer (no float-precision
    issues) and is the canonical per-event identifier.
    """
    frame = ev.get('frame')
    if frame is not None:
        return str(frame)
    time = ev.get('time')
    if time is not None:
        return f"{time:.4f}"
    return ''


def clean_overrides(
    overrides: Dict[str, Dict[str, Any]],
    analysis_data: Dict,
) -> Dict[str, Dict[str, Any]]:
    """Drop entries whose effective state now matches the
    sidecar's natural state. The cleaned dict is structurally
    equivalent to the input (same dict-of-dict shape) but with
    redundant entries removed.

    An entry is redundant when:
      - The override's status matches the sidecar event's
        status (KEPT or FILTERED), AND
      - Either the override has no classification (so the
        status alone is the comparison), OR the override's
        classification matches the sidecar event's classification
        (so classification is also consistent).

    Args:
        overrides: The full override dict in the canonical
            shape (stem -> time_str -> override record).
        analysis_data: The current sidecar data (v3 format).
            Used to look up each event's natural status and
            classification.

    Returns:
        A new dict with redundant entries removed. If no entries
        are redundant, the input dict is returned (a new dict
        is always returned for caller safety — the input is
        not mutated).
    """
    if not overrides:
        return {}

    cleaned: Dict[str, Dict[str, Any]] = {}
    for stem_type, stem_overrides in (overrides or {}).items():
        stem_data = analysis_data.get('stems', {}).get(stem_type, {})
        events_pga = stem_data.get('events_pga', [])

        # Build a quick lookup: frame_str (or time_str fallback)
        # -> event. events_pga is the canonical post-2026-06-15
        # source for status + classification.
        #
        # 2026-06-30: switched from time-string to frame-integer
        # (with time fallback) — the user reported that the
        # 4-decimal time key in the WebUI ("2.9540") didn't match
        # manually-edited files with 3-decimal keys ("2.954").
        # Frame is always an integer, no rounding issues.
        events_by_key: Dict[str, Dict] = {}
        for ev in events_pga:
            key = _event_key(ev)
            if not key:
                continue
            events_by_key[key] = ev

        kept: Dict[str, Any] = {}
        for over_key, override in (stem_overrides or {}).items():
            sidecar_event = events_by_key.get(over_key)
            if sidecar_event is None:
                # Event no longer in the sidecar. Drop the
                # override — it's referencing a ghost event.
                continue

            # Compare override to sidecar's natural state.
            override_status = override.get('status')
            sidecar_status = sidecar_event.get('status')
            if override_status != sidecar_status:
                # Override disagrees with the filter — keep.
                kept[over_key] = override
                continue

            # Status matches. Check classification if the
            # override sets one.
            override_class = override.get('classification')
            if override_class is None:
                # Status-only override, and it matches the
                # sidecar's natural state. Drop it.
                continue

            sidecar_class = sidecar_event.get('classification')
            if override_class == sidecar_class:
                # Both status and classification match. Drop.
                continue

            # Classification override disagrees with sidecar.
            # Keep.
            kept[over_key] = override

        if kept:
            cleaned[stem_type] = kept

    return cleaned
