"""
Filter Registry API (2026-06-15).

Serves the filter registry JSON to the WebUI. The registry is
the single source of truth for per-event filters shared between
Python (stems_to_midi.filter_kinds) and the WebUI
(webui.static.js.filter_kinds). Both languages read the SAME
file — Python loads it from disk, the WebUI fetches it via
this endpoint. No static-file duplication, no drift.

The response is the verbatim contents of
``stems_to_midi/filter_registry.json`` (no transformation).
The WebUI caches the result and re-evaluates filters
client-side as the user drags sliders.

Endpoints:
  GET /api/filters/schema  — full filter registry
  GET /api/filters/<id>     — single filter entry by id
                              (404 if not found)
  GET /api/filters/stem/<stem_type>  — filters applicable
                              to a given stem (e.g., 'toms')

The Python filter_kinds module does the same lookup
functions server-side (find_filter, list_filters_for_stem),
so any new endpoint added here should have a corresponding
helper in filter_kinds.
"""
from flask import Blueprint, jsonify

from stems_to_midi.filter_kinds import (
    find_filter,
    list_filters_for_stem,
    load_filter_registry,
)


filters_bp = Blueprint('filters', __name__, url_prefix='/api/filters')


@filters_bp.route('/schema', methods=['GET'])
def get_filter_schema():
    """
    Return the full filter registry as JSON.

    Response: the verbatim contents of
    ``stems_to_midi/filter_registry.json``. The WebUI caches
    this on load and uses it for both slider rendering
    (the metadata) and filter evaluation (the AST).

    The registry is the single source of truth — Python
    loads the same file via ``stems_to_midi.filter_kinds``.
    """
    try:
        registry = load_filter_registry()
        return jsonify(registry), 200
    except Exception as e:
        return jsonify({
            'error': 'Failed to load filter registry',
            'message': str(e),
        }), 500


@filters_bp.route('/<filter_id>', methods=['GET'])
def get_filter(filter_id: str):
    """
    Return a single filter entry by id, or 404 if not found.

    Useful for WebUI components that need to render a
    specific filter's slider without loading the whole
    registry. (Most components prefer the bulk endpoint
    and look up by id client-side.)
    """
    spec = find_filter(filter_id)
    if spec is None:
        return jsonify({
            'error': f"Filter {filter_id!r} not found",
        }), 404
    return jsonify(spec), 200


@filters_bp.route('/stem/<stem_type>', methods=['GET'])
def get_filters_for_stem(stem_type: str):
    """
    Return the filter entries that apply to a given stem.

    Used by the WebUI tuning panel to render the right
    sliders for the current stem. The result is a list
    of filter specs (the same objects that /schema
    returns, just filtered).
    """
    filters = list_filters_for_stem(stem_type)
    return jsonify(filters), 200
