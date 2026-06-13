"""
User event overrides — read-side loader for the WebUI's
``event_overrides.json`` sidecar.

When the user toggles an event on / off in the WebUI tuning
panel, the panel writes a per-project override file at::

    user_files/<project>/midi/event_overrides.json

with the shape::

    {
        "<event_id>": {
            "status": "FILTERED",
            "reason": "manually disabled via WebUI"
        },
        ...
    }

This module is the read-side: it loads the file (if present)
and returns the dict. The write path is owned by the WebUI
(``webui/static/js/waveform.js``) and is intentionally out of
scope for this refactor — the goal here is to make the PGA
filter pipeline able to consume the file when the WebUI
re-applies the filter, not to change how the file is written.

Design constraints:
  - Pure read path. The function returns the dict verbatim
    after a light schema check (must be a dict; values must
    be dicts); malformed files raise ``EventOverridesError``
    with a clear message so the WebUI can show a toast.
  - No mutation of the loaded dict — the caller is free to
    re-use the keys for ``apply_pga_prominence_filter``'s
    ``disabled_ids`` argument.
  - File I/O is the only side-effect (functional core /
    imperative shell split — the read function is a thin
    imperative shell around a tiny pure parser).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


__all__ = [
    'load_event_overrides',
    'EventOverridesError',
]


class EventOverridesError(RuntimeError):
    """Raised when the override file is present but malformed
    (e.g. not a JSON object, or values are not dicts). The
    WebUI surfaces this as a toast — the file is left in
    place so the user can hand-edit it if needed."""


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
        event id -> override record when the file exists. The
        override record is a dict with at minimum a ``status``
        key (``"FILTERED"`` or ``"KEPT"``); other keys are
        preserved verbatim so future fields (``timestamp``,
        ``user_note``, etc.) can be added without breaking
        this loader.

    Raises:
        EventOverridesError: The file exists but is not a
            valid JSON object, or any value is not a dict.
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
            f"object (event_id -> override), got {type(data).__name__}"
        )

    # Light schema check — values must be dicts (an override
    # record carries status + reason; we don't validate the
    # inner keys here, that would couple us to the WebUI's
    # write format). Bad records surface as a clear error
    # rather than a silent None.
    for k, v in data.items():
        if not isinstance(v, dict):
            raise EventOverridesError(
                f"event_overrides.json at {override_path}: override "
                f"for event id {k!r} must be an object, got "
                f"{type(v).__name__}"
            )

    return data
