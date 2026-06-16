"""
Filter registry — Python side. 2026-06-15.

Single source of truth for per-event filters shared between
Python (rebuild_core, pga_event_builder) and the WebUI
(threshold-tuning.js). The registry lives in
``stems_to_midi/filter_registry.json``. Both languages read
the same file — Python loads it via ``json.load`` (this
module), the WebUI fetches it via the ``/api/filters/schema``
API endpoint. Adding a new filter is a JSON-only change; the
parity problem (Python and JS implementing the same filter
logic in two places) is gone.

Each filter entry in the JSON has:
  - id: unique identifier
  - label, description: human-readable
  - default, min, max, step, unit: numeric bounds for the UI
  - ui_control: slider, number, checkbox, etc.
  - applies_to_stems: which stems use this filter
  - yaml_paths: where the threshold is stored (per_stem > global)
  - filter: {kind, field, reason_template, value_format}
    The ``kind`` is one of the closed enum below.

Closed enum of filter kinds (this PR implements all 6 — the
2 PGA filters in this PR only use min_value; the others are
here for the migration of the other 5 filters in Phase 6):

  - min_value: KEPT if event[field] >= threshold. Returns
    None if field is missing. Covers PGA prominence,
    decay_col_min, geomean, sustain, strength, attack_sharpness.

  - max_value: KEPT if event[field] <= threshold. Returns
    None if field is missing. Covers band_max_ratio_max.

  - nonzero_when_enabled: KEPT if not enabled, OR if enabled
    and event[field] > 0. Returns False if enabled and
    field is null/None or <= 0. Covers show_only_snap_events.

  - and: KEPT if all child filters are KEPT. Short-circuits on
    first False. Empty children = True (no constraint).
    Future: covers geomean+sustain+strength composition.

  - or: KEPT if any child filter is KEPT. Returns None
    (treat as pass-through) if all children return None.
    Empty children = True.

  - not: inverts a single child filter. None -> None.

The Python and JS evaluators are tiny (~50 lines each) and
sit side-by-side in this module and filter_kinds.js. Adding
a new KIND requires a small change in both. Adding a new
filter INSTANCE is JSON-only.

value_format controls how value and threshold are rendered
in reason_template. Python uses f-strings (f"{value:.0f}",
f"{value:.1f}"); JS uses toFixed(). The JSON is
language-agnostic — the placeholders are simple {value},
{threshold}, {field}.
"""
import json
import os
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------


def _get_registry_path() -> str:
    """Path to filter_registry.json, sibling to this module."""
    return os.path.join(os.path.dirname(__file__), 'filter_registry.json')


_REGISTRY_CACHE: Optional[Dict[str, Any]] = None


def load_filter_registry(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the filter registry from JSON. Cached at module level
    on first call (the file is read-only at runtime — no need
    to re-read on every call). Tests can pass a custom path
    to load a fixture."""
    global _REGISTRY_CACHE
    if path is None and _REGISTRY_CACHE is not None:
        return _REGISTRY_CACHE
    if path is None:
        path = _get_registry_path()
    with open(path) as f:
        registry = json.load(f)
    if path is None:
        _REGISTRY_CACHE = registry
    return registry


def find_filter(
    filter_id: str,
    registry: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Find a filter entry by id, or None if not found."""
    if registry is None:
        registry = load_filter_registry()
    for f in registry.get('filters', []):
        if f.get('id') == filter_id:
            return f
    return None


def list_filters_for_stem(
    stem_type: str,
    registry: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Return the filter entries whose applies_to_stems contains
    the given stem."""
    if registry is None:
        registry = load_filter_registry()
    return [
        f for f in registry.get('filters', [])
        if stem_type in f.get('applies_to_stems', [])
    ]


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------


def _format_value(value: Any, value_format: str) -> str:
    """Format value for the filter reason template. Handles None
    (returns 'N/A' so the template still substitutes cleanly)."""
    if value is None:
        return 'N/A'
    if value_format == 'int':
        return f"{int(round(float(value)))}"
    if value_format == 'float1':
        return f"{float(value):.1f}"
    if value_format == 'float2':
        return f"{float(value):.2f}"
    return str(value)


# ---------------------------------------------------------------------------
# AST evaluation
# ---------------------------------------------------------------------------


def _evaluate_node(
    filter_node: Dict[str, Any],
    event: Dict[str, Any],
    threshold: Any,
    enabled: bool,
) -> Optional[bool]:
    """Walk the filter AST. Returns:
      - True: KEPT (event passes the filter)
      - False: FILTERED (event fails the filter)
      - None: cannot evaluate (field missing, etc.) — caller
        decides whether to keep or filter None cases

    The enabled flag is the toggle for nonzero_when_enabled.
    For threshold-based kinds (min_value, max_value), enabled
    is unused (the filter is always active when called).
    """
    kind = filter_node.get('kind')

    if kind == 'min_value':
        value = event.get(filter_node.get('field'))
        if value is None:
            return None
        return value >= threshold

    if kind == 'max_value':
        value = event.get(filter_node.get('field'))
        if value is None:
            return None
        return value <= threshold

    if kind == 'nonzero_when_enabled':
        if not enabled:
            return True
        value = event.get(filter_node.get('field'))
        if value is None or value <= 0:
            return False
        return True

    if kind == 'and':
        children = filter_node.get('filters', [])
        if not children:
            return True  # empty AND = no constraint
        for child in children:
            if _evaluate_node(child, event, threshold, enabled) is False:
                return False
        return True

    if kind == 'or':
        children = filter_node.get('filters', [])
        if not children:
            return True  # empty OR = no constraint
        saw_none = False
        for child in children:
            result = _evaluate_node(child, event, threshold, enabled)
            if result is True:
                return True
            if result is None:
                saw_none = True
        return None if saw_none else False

    if kind == 'not':
        child = filter_node.get('filter')
        if child is None:
            return True
        result = _evaluate_node(child, event, threshold, enabled)
        if result is None:
            return None
        return not result

    raise ValueError(f"Unknown filter kind: {kind!r}")


def evaluate_filter(
    filter_spec: Dict[str, Any],
    event: Dict[str, Any],
    threshold: Any,
    enabled: bool = True,
) -> Optional[bool]:
    """Public entry point. Evaluates the filter against a single
    event. The filter_spec is the FULL entry from the registry
    (the kind, field, and reason_template live under
    filter_spec['filter']).

    Returns True/False/None per _evaluate_node.
    """
    return _evaluate_node(
        filter_spec['filter'], event, threshold, enabled,
    )


def build_filter_reason(
    filter_spec: Dict[str, Any],
    event: Dict[str, Any],
    threshold: Any,
) -> str:
    """Build the filter_reason string from the filter spec's
    reason_template. Substitutes {value}, {threshold}, {field}
    with formatted strings.

    The value comes from event[field] for the min_value and
    max_value kinds. For other kinds, value is N/A (the
    template is expected to be specific to the kind — e.g.,
    nonzero_when_enabled's template would just say
    "snap_delta is zero or null").
    """
    inner = filter_spec.get('filter', {})
    template = inner.get('reason_template', '')
    value_format = inner.get('value_format', 'float2')
    field = inner.get('field', '?')
    kind = inner.get('kind')

    if kind in ('min_value', 'max_value'):
        value = event.get(field)
    else:
        value = None

    return template.format(
        value=_format_value(value, value_format),
        threshold=_format_value(threshold, value_format),
        field=field,
    )


# ---------------------------------------------------------------------------
# Threshold resolution (per-stem > global > default)
# ---------------------------------------------------------------------------


def _lookup_yaml_path(config: Dict[str, Any], path: List[str]) -> Any:
    """Walk the config dict along the path. Returns None if any
    key is missing (so the caller can fall through to the
    next resolution tier)."""
    current = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def resolve_threshold(
    filter_spec: Dict[str, Any],
    stem_type: str,
    config: Dict[str, Any],
) -> Any:
    """Resolve the threshold for a filter given the current config.
    Precedence: per_stem > global > default.

    Returns the resolved value (number, bool, or None).
    Returns None if no resolution was found AND the JSON
    default is None (rare; the JSON defaults are always set
    for the 2 PGA filters in this PR).
    """
    yaml_paths = filter_spec.get('yaml_paths', {})
    per_stem = yaml_paths.get('per_stem', {})
    global_path = yaml_paths.get('global')

    # Try per-stem first
    if stem_type in per_stem:
        val = _lookup_yaml_path(config, per_stem[stem_type])
        if val is not None:
            return val

    # Then global
    if global_path:
        val = _lookup_yaml_path(config, global_path)
        if val is not None:
            return val

    # Fall back to the JSON default
    return filter_spec.get('default')
