"""
Tests for stems_to_midi.filter_kinds (2026-06-15).

The filter registry is the single source of truth for
per-event filters shared between Python and the WebUI.
These tests lock the Python evaluator's contract so adding
a new filter (via a JSON entry) is a one-step process.

Test layout:
  1. Registry loading
  2. Filter lookup (find_filter, list_filters_for_stem)
  3. evaluate_filter for the min_value kind (the only kind
     the 2 PGA filters use in this PR; the others are
     covered in Phase 6 when those filters are migrated)
  4. evaluate_filter for the and / or / not combinators
     (used by the future geomean+sustain+strength composition)
  5. evaluate_filter for the nonzero_when_enabled kind
     (used by show_only_snap_events)
  6. build_filter_reason for value_format (int, float1, float2)
  7. resolve_threshold for per-stem > global > default
  8. JSON schema validation (every filter kind is in the
     kinds enum; every yaml_paths entry is a list of strings)
"""
import os
import sys
from pathlib import Path

import pytest

# Ensure repo root is on sys.path so ``stems_to_midi`` resolves
# consistently with the existing test_* modules. The new
# tests/ subdirectory changes the import resolution relative
# to the existing tests, so we explicitly add the parent
# directory.
_TEST_DIR = Path(__file__).resolve().parent
_PKG_PARENT = _TEST_DIR.parent.parent
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from stems_to_midi.filter_kinds import (  # noqa: E402
    load_filter_registry,
    find_filter,
    list_filters_for_stem,
    evaluate_filter,
    build_filter_reason,
    resolve_threshold,
    _evaluate_node,
    _format_value,
    _lookup_yaml_path,
)


# ---------------------------------------------------------------------------
# 1. Registry loading
# ---------------------------------------------------------------------------


class TestRegistryLoading:
    def test_load_returns_dict(self):
        reg = load_filter_registry()
        assert isinstance(reg, dict)

    def test_load_has_version(self):
        reg = load_filter_registry()
        assert 'version' in reg

    def test_load_has_filters_list(self):
        reg = load_filter_registry()
        assert isinstance(reg.get('filters'), list)
        assert len(reg['filters']) >= 2  # at least the 2 PGA filters

    def test_load_has_kinds_enum(self):
        reg = load_filter_registry()
        assert isinstance(reg.get('kinds'), list)
        assert 'min_value' in reg['kinds']
        assert 'max_value' in reg['kinds']
        assert 'nonzero_when_enabled' in reg['kinds']
        assert 'and' in reg['kinds']


# ---------------------------------------------------------------------------
# 2. Filter lookup
# ---------------------------------------------------------------------------


class TestFilterLookup:
    def test_find_filter_existing(self):
        spec = find_filter('pga_min_prominence')
        assert spec is not None
        assert spec['id'] == 'pga_min_prominence'
        assert spec['filter']['kind'] == 'min_value'

    def test_find_filter_missing_returns_none(self):
        spec = find_filter('does_not_exist')
        assert spec is None

    def test_list_filters_for_toms(self):
        filters = list_filters_for_stem('toms')
        ids = [f['id'] for f in filters]
        assert 'pga_min_prominence' in ids
        assert 'min_decay_col_min_db' in ids

    def test_list_filters_for_other_stem_empty(self):
        # The 2 PGA filters apply to toms only. Other stems
        # (e.g., kick) have no entries yet — that's expected
        # for this PR; the other 5 filters migrate in Phase 6.
        filters = list_filters_for_stem('kick')
        assert filters == []


# ---------------------------------------------------------------------------
# 3. evaluate_filter — min_value
# ---------------------------------------------------------------------------


class TestEvaluateMinValue:
    def test_value_at_threshold_is_kept(self):
        spec = find_filter('pga_min_prominence')
        ev = {'time': 1.0, 'prominence': 1000}
        # At threshold: KEPT (>= threshold, not strictly <).
        assert evaluate_filter(spec, ev, 1000) is True

    def test_value_above_threshold_is_kept(self):
        spec = find_filter('pga_min_prominence')
        ev = {'time': 1.0, 'prominence': 5000}
        assert evaluate_filter(spec, ev, 1000) is True

    def test_value_below_threshold_is_filtered(self):
        spec = find_filter('pga_min_prominence')
        ev = {'time': 1.0, 'prominence': 500}
        assert evaluate_filter(spec, ev, 1000) is False

    def test_missing_field_returns_none(self):
        spec = find_filter('pga_min_prominence')
        ev = {'time': 1.0}  # no prominence
        assert evaluate_filter(spec, ev, 1000) is None

    def test_decay_col_min_at_threshold_is_kept(self):
        spec = find_filter('min_decay_col_min_db')
        ev = {'time': 1.0, 'decay_col_min_median_db': -80.0}
        assert evaluate_filter(spec, ev, -80.0) is True

    def test_decay_col_min_above_threshold_is_kept(self):
        spec = find_filter('min_decay_col_min_db')
        ev = {'time': 1.0, 'decay_col_min_median_db': -70.0}
        assert evaluate_filter(spec, ev, -80.0) is True

    def test_decay_col_min_below_threshold_is_filtered(self):
        spec = find_filter('min_decay_col_min_db')
        ev = {'time': 1.0, 'decay_col_min_median_db': -90.0}
        assert evaluate_filter(spec, ev, -80.0) is False


# ---------------------------------------------------------------------------
# 4. evaluate_filter — combinators (and / or / not)
# ---------------------------------------------------------------------------


class TestEvaluateCombinators:
    def test_and_all_pass(self):
        node = {
            'kind': 'and',
            'filters': [
                {'kind': 'min_value', 'field': 'a'},
                {'kind': 'min_value', 'field': 'b'},
            ],
        }
        ev = {'a': 100, 'b': 100}
        assert _evaluate_node(node, ev, threshold=10, enabled=True) is True

    def test_and_one_fails(self):
        node = {
            'kind': 'and',
            'filters': [
                {'kind': 'min_value', 'field': 'a'},
                {'kind': 'min_value', 'field': 'b'},
            ],
        }
        ev = {'a': 100, 'b': 5}  # b fails
        assert _evaluate_node(node, ev, threshold=10, enabled=True) is False

    def test_and_short_circuits(self):
        # First child fails, second is never evaluated. Verify
        # no exception is raised even if second would error.
        node = {
            'kind': 'and',
            'filters': [
                {'kind': 'min_value', 'field': 'a'},
                {'kind': 'NONEXISTENT_KIND'},
            ],
        }
        ev = {'a': 5}  # a fails; second child would raise
        assert _evaluate_node(node, ev, threshold=10, enabled=True) is False

    def test_and_empty_is_true(self):
        node = {'kind': 'and', 'filters': []}
        ev = {}
        assert _evaluate_node(node, ev, threshold=10, enabled=True) is True

    def test_or_one_passes(self):
        node = {
            'kind': 'or',
            'filters': [
                {'kind': 'min_value', 'field': 'a'},
                {'kind': 'min_value', 'field': 'b'},
            ],
        }
        ev = {'a': 5, 'b': 100}  # only b passes
        assert _evaluate_node(node, ev, threshold=10, enabled=True) is True

    def test_or_all_fail(self):
        node = {
            'kind': 'or',
            'filters': [
                {'kind': 'min_value', 'field': 'a'},
                {'kind': 'min_value', 'field': 'b'},
            ],
        }
        ev = {'a': 5, 'b': 5}
        assert _evaluate_node(node, ev, threshold=10, enabled=True) is False

    def test_or_all_none_returns_none(self):
        # All children return None (missing fields). OR returns
        # None (treat as pass-through — caller decides).
        node = {
            'kind': 'or',
            'filters': [
                {'kind': 'min_value', 'field': 'a'},
                {'kind': 'min_value', 'field': 'b'},
            ],
        }
        ev = {}  # both fields missing
        assert _evaluate_node(node, ev, threshold=10, enabled=True) is None

    def test_not_inverts(self):
        node = {'kind': 'not', 'filter': {'kind': 'min_value', 'field': 'a'}}
        ev = {'a': 5}  # min_value says False (a < 10)
        assert _evaluate_node(node, ev, threshold=10, enabled=True) is True
        ev = {'a': 100}  # min_value says True
        assert _evaluate_node(node, ev, threshold=10, enabled=True) is False

    def test_not_of_none_is_none(self):
        node = {'kind': 'not', 'filter': {'kind': 'min_value', 'field': 'a'}}
        ev = {}  # missing field -> None
        assert _evaluate_node(node, ev, threshold=10, enabled=True) is None

    def test_not_empty_is_true(self):
        node = {'kind': 'not', 'filter': None}
        ev = {}
        assert _evaluate_node(node, ev, threshold=10, enabled=True) is True

    def test_unknown_kind_raises(self):
        node = {'kind': 'NONEXISTENT_KIND'}
        ev = {}
        with pytest.raises(ValueError, match='Unknown filter kind'):
            _evaluate_node(node, ev, threshold=10, enabled=True)


# ---------------------------------------------------------------------------
# 5. evaluate_filter — nonzero_when_enabled
# ---------------------------------------------------------------------------


class TestEvaluateNonzeroWhenEnabled:
    def test_disabled_passes_everything(self):
        node = {'kind': 'nonzero_when_enabled', 'field': 'snap_delta'}
        # When disabled, the filter is a no-op — even zero/null
        # values pass.
        assert _evaluate_node(node, {'snap_delta': 0}, threshold=None, enabled=False) is True
        assert _evaluate_node(node, {'snap_delta': None}, threshold=None, enabled=False) is True
        assert _evaluate_node(node, {'snap_delta': -1}, threshold=None, enabled=False) is True

    def test_enabled_positive_passes(self):
        node = {'kind': 'nonzero_when_enabled', 'field': 'snap_delta'}
        assert _evaluate_node(node, {'snap_delta': 1.0}, threshold=None, enabled=True) is True

    def test_enabled_zero_filtered(self):
        node = {'kind': 'nonzero_when_enabled', 'field': 'snap_delta'}
        assert _evaluate_node(node, {'snap_delta': 0}, threshold=None, enabled=True) is False

    def test_enabled_negative_filtered(self):
        node = {'kind': 'nonzero_when_enabled', 'field': 'snap_delta'}
        assert _evaluate_node(node, {'snap_delta': -0.1}, threshold=None, enabled=True) is False

    def test_enabled_null_filtered(self):
        node = {'kind': 'nonzero_when_enabled', 'field': 'snap_delta'}
        assert _evaluate_node(node, {}, threshold=None, enabled=True) is False


# ---------------------------------------------------------------------------
# 6. build_filter_reason
# ---------------------------------------------------------------------------


class TestBuildFilterReason:
    def test_int_format(self):
        spec = find_filter('pga_min_prominence')
        ev = {'time': 1.0, 'prominence': 500}
        reason = build_filter_reason(spec, ev, 1000)
        assert reason == 'below pga_min_prominence (500 < 1000)'

    def test_int_format_rounds(self):
        spec = find_filter('pga_min_prominence')
        ev = {'time': 1.0, 'prominence': 500.7}
        # int value_format rounds to nearest int
        reason = build_filter_reason(spec, ev, 1000.4)
        assert reason == 'below pga_min_prominence (501 < 1000)'

    def test_float1_format(self):
        spec = find_filter('min_decay_col_min_db')
        ev = {'time': 1.0, 'decay_col_min_median_db': -90.123}
        reason = build_filter_reason(spec, ev, -80.0)
        assert reason == 'below min_decay_col_min_db (-90.1dB < -80.0dB)'

    def test_missing_field_renders_na(self):
        spec = find_filter('pga_min_prominence')
        ev = {'time': 1.0}  # no prominence
        reason = build_filter_reason(spec, ev, 1000)
        # The {value} placeholder renders as N/A when the
        # field is missing.
        assert 'N/A' in reason
        assert '1000' in reason

    def test_format_value_helper_int(self):
        assert _format_value(500.7, 'int') == '501'
        assert _format_value(-90.4, 'int') == '-90'
        assert _format_value(0, 'int') == '0'
        assert _format_value(None, 'int') == 'N/A'

    def test_format_value_helper_float1(self):
        assert _format_value(-90.123, 'float1') == '-90.1'
        assert _format_value(1000.456, 'float1') == '1000.5'
        assert _format_value(None, 'float1') == 'N/A'

    def test_format_value_helper_float2(self):
        assert _format_value(-90.123, 'float2') == '-90.12'
        assert _format_value(1000.456, 'float2') == '1000.46'

    def test_format_value_helper_unknown_format_falls_back(self):
        # Unknown format enum falls back to str(value) — defensive.
        assert _format_value(500, 'unknown') == '500'


# ---------------------------------------------------------------------------
# 7. resolve_threshold
# ---------------------------------------------------------------------------


class TestResolveThreshold:
    def test_per_stem_wins_over_global(self):
        spec = find_filter('pga_min_prominence')
        config = {
            'toms': {'pga_min_prominence': 5000},
            'onset_detection': {'pga_min_prominence': 1000},
        }
        assert resolve_threshold(spec, 'toms', config) == 5000

    def test_global_used_when_per_stem_absent(self):
        spec = find_filter('pga_min_prominence')
        config = {
            'toms': {},  # no per-stem override
            'onset_detection': {'pga_min_prominence': 1000},
        }
        assert resolve_threshold(spec, 'toms', config) == 1000

    def test_default_used_when_both_absent(self):
        spec = find_filter('pga_min_prominence')
        config = {'toms': {}, 'onset_detection': {}}
        assert resolve_threshold(spec, 'toms', config) == 1000

    def test_partial_per_stem_path_returns_global(self):
        # If the per-stem path is partially specified (key
        # missing), fall through to global.
        spec = find_filter('pga_min_prominence')
        config = {
            'toms': {'other_key': 99},
            'onset_detection': {'pga_min_prominence': 1000},
        }
        assert resolve_threshold(spec, 'toms', config) == 1000

    def test_other_stem_falls_through(self):
        # If asked about a stem that has no per-stem override,
        # fall through to global.
        spec = find_filter('pga_min_prominence')
        config = {
            'toms': {'pga_min_prominence': 5000},
            'onset_detection': {'pga_min_prominence': 1000},
        }
        # Asking for a non-toms stem with no per-stem path
        # falls through to global (and then to default if
        # global is also missing).
        assert resolve_threshold(spec, 'kick', config) == 1000

    def test_lookup_yaml_path_missing_key(self):
        assert _lookup_yaml_path({}, ['a', 'b']) is None
        assert _lookup_yaml_path({'a': 1}, ['a', 'b']) is None
        assert _lookup_yaml_path({'a': {'b': 2}}, ['a', 'b']) == 2
        # Non-dict in the middle
        assert _lookup_yaml_path({'a': 'string'}, ['a', 'b']) is None


# ---------------------------------------------------------------------------
# 8. JSON schema validation
# ---------------------------------------------------------------------------


class TestJsonSchemaValidation:
    """Every entry in the registry must be well-formed:
    - filter.kind is in the kinds enum
    - yaml_paths entries are lists of strings
    - value_format is in the value_formats enum (if present)
    - ui_control is one of the known UI controls
    """

    def test_all_kinds_are_in_enum(self):
        reg = load_filter_registry()
        kinds_enum = set(reg['kinds'])
        for f in reg['filters']:
            kind = f.get('filter', {}).get('kind')
            if kind not in ('and', 'or', 'not'):
                # Combinators are evaluated against their
                # children, so they may use 'min_value' / etc.
                # recursively. The kind on a top-level filter
                # node is what we validate here.
                assert kind in kinds_enum, (
                    f"filter {f.get('id')!r} uses unknown kind {kind!r}"
                )

    def test_all_yaml_paths_are_lists_of_strings(self):
        reg = load_filter_registry()
        for f in reg['filters']:
            yaml_paths = f.get('yaml_paths', {})
            global_path = yaml_paths.get('global')
            if global_path is not None:
                assert isinstance(global_path, list)
                for key in global_path:
                    assert isinstance(key, str)
            per_stem = yaml_paths.get('per_stem', {})
            assert isinstance(per_stem, dict)
            for stem, path in per_stem.items():
                assert isinstance(stem, str)
                assert isinstance(path, list)
                for key in path:
                    assert isinstance(key, str)

    def test_all_value_formats_are_in_enum(self):
        reg = load_filter_registry()
        formats = set(reg.get('value_formats', {}).keys())
        for f in reg['filters']:
            value_format = f.get('filter', {}).get('value_format')
            if value_format is not None:
                assert value_format in formats, (
                    f"filter {f.get('id')!r} uses unknown "
                    f"value_format {value_format!r}"
                )

    def test_all_filters_have_required_fields(self):
        reg = load_filter_registry()
        required = {'id', 'label', 'description', 'default', 'filter'}
        for f in reg['filters']:
            missing = required - set(f.keys())
            assert not missing, (
                f"filter {f.get('id')!r} is missing required "
                f"fields: {missing}"
            )

    def test_all_filter_ids_are_unique(self):
        reg = load_filter_registry()
        ids = [f.get('id') for f in reg['filters']]
        assert len(ids) == len(set(ids)), (
            f"duplicate filter ids: {ids}"
        )
