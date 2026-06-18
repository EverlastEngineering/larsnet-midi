# Filter Kinds Reference

The closed enum of filter kinds in `stems_to_midi/filter_registry.json`.
Both Python (`stems_to_midi/filter_kinds.py`) and JavaScript
(`webui/static/js/filter_kinds.js`) implement these. Adding a
new KIND requires code in both evaluators (~10 lines each) — see
`stems_to_midi/filter_kinds.py::_evaluate_node` and
`webui/static/js/filter_kinds.js::_evaluateNode`. **Adding a
new FILTER INSTANCE is just a JSON entry** — that's what this
skill is for.

## Closed Enum

| Kind | Predicate | Returns | Used by |
|---|---|---|---|
| `min_value` | `value >= threshold` | `true` / `false` / `null` (if field missing) | `pga_min_prominence`, `min_decay_col_min_db`, `attack_rise_max_ms` (inverted — see below) |
| `max_value` | `value <= threshold` | `true` / `false` / `null` | `attack_rise_max_ms`, future: `band_max_ratio_max` |
| `nonzero_when_enabled` | when enabled: `value > 0`; when disabled: always `true` | `true` / `false` (never `null` — missing fields return `false` when enabled) | future: `show_only_snap_events` |
| `and` | `all(children pass)` | `true` / `false` (short-circuits on first `false`) | future: geomean + sustain + strength composition |
| `or` | `any(child passes)` | `true` / `false` / `null` (None if all children return None) | rare |
| `not` | `not(child)` | `true` / `false` / `null` | rare |

## Predicate Semantics in Detail

### `min_value`
**Use when**: the filter should DROP events where `event[field]` is below some threshold (e.g., prominence is too low).

**Python**: `value is not None and value >= threshold`
**JS**: `value != null && value >= threshold`

**Returns `null`** when `event[field]` is missing — caller
treats `null` as "can't evaluate" (KEPT).

### `max_value`
**Use when**: the filter should DROP events where `event[field]` is ABOVE some threshold (e.g., attack_rise is too long).

**Python**: `value is not None and value <= threshold`
**JS**: `value != null && value <= threshold`

**Returns `null`** when `event[field]` is missing.

### `nonzero_when_enabled`
**Use when**: there's a TOGGLE in the UI (on/off), and when ON, the filter should drop events where `event[field]` is null or `<= 0`.

**Python**:
```python
if not enabled:
    return True
value = event.get(field)
return value is not None and value > 0  # False when None or 0
```

**JS**:
```javascript
if (!enabled) return true;
const value = event[filterNode.field];
if (value == null || value <= 0) return false;
return true;
```

**Takes the `enabled` flag** (passed by the caller from the
toggle's YAML value, e.g., `toms.show_only_snap_events`).

### `and`
**Use when**: the filter is the AND of multiple conditions.

**Children shape**:
```json
{
    "kind": "and",
    "filters": [
        {"kind": "min_value", "field": "geomean"},
        {"kind": "min_value", "field": "sustain_ms"},
        {"kind": "min_value", "field": "strength"}
    ]
}
```

**Semantics**:
- Empty `filters` array → `true` (no constraint).
- One child returns `false` → returns `false` immediately
  (short-circuits).
- All children return `true` → returns `true`.
- All children return `None` → returns `None` (caller
  treats as KEPT).
- Mix of `true` and `None` → returns `true` (None is
  treated as "couldn't evaluate", which is a pass).

### `or`
**Use when**: the filter is the OR of multiple conditions (rare).

**Children shape**: same as `and`.

**Semantics**:
- Empty `filters` array → `true`.
- One child returns `true` → returns `true` immediately.
- One child returns `None` → continues checking (None is
  treated as "couldn't evaluate").
- All children return `false` → returns `false`.
- All children return `None` → returns `None`.

### `not`
**Use when**: the filter is the negation of a single condition (rare).

**Child shape**:
```json
{
    "kind": "not",
    "filter": {"kind": "min_value", "field": "prominence"}
}
```

**Semantics**:
- Empty child → `true`.
- Child returns `None` → `None`.
- Otherwise → `not(child)`.

## The `value_format` Enum

Controls how `{value}` and `{threshold}` are rendered in the
`reason_template`. The JSON is language-agnostic — each language
formats per its own conventions.

| Format | Python | JS |
|---|---|---|
| `int` | `f"{int(round(float(value)))}"` | `String(Math.round(Number(value)))` |
| `float1` | `f"{float(value):.1f}"` | `Number(value).toFixed(1)` |
| `float2` | `f"{float(value):.2f}"` | `Number(value).toFixed(2)` |

**Pick**:
- `int` for slider values that are always whole numbers
  (e.g., prominence at 0-10000).
- `float1` for values where 1 decimal place matters (e.g.,
  dB values like -79.3).
- `float2` for values that need 2 decimal places (rare;
  frequency values are the usual use case).

## Threshold Resolution Order

The filter's threshold is resolved at runtime with the
following precedence (highest to lowest):

1. `config[stem_type][<filter_id>]` — per-stem override
2. `config['onset_detection'][<filter_id>]` — global fallback
3. `filter_spec['default']` — JSON default (last resort)

The same pattern is used in Python (`filter_kinds.py::resolve_threshold`)
and JS (`filter_kinds.js::resolveThreshold`).

For the YAML keys:
- Per-stem: `<stem>.<filter_id>` (e.g., `toms.pga_min_prominence`)
- Global: `onset_detection.<filter_id>` (e.g.,
  `onset_detection.pga_min_prominence`)

The JSON `yaml_paths` block records both:

```json
{
    "yaml_paths": {
        "global": ["onset_detection", "pga_min_prominence"],
        "per_stem": {
            "toms": ["toms", "pga_min_prominence"]
        }
    }
}
```

## Adding a New KIND (rare — needs code in both evaluators)

If none of the 6 existing kinds fit your filter, you can add a
new kind. This requires:

1. Add the kind name to the `kinds` enum in
   `stems_to_midi/filter_registry.json`.
2. Implement the predicate in
   `stems_to_midi/filter_kinds.py::_evaluate_node` (~5 lines).
3. Implement the predicate in
   `webui/static/js/filter_kinds.js::_evaluateNode` (~5 lines).
4. Add unit tests for the new kind in
   `stems_to_midi/tests/test_filter_kinds.py::TestEvaluateMinValue`
   (or a new test class).
5. Document the new kind in
   `.github/skills/add-filter/references/filter_kinds.md`
   (this file).

Adding a new KIND is a more invasive change than adding a new
filter instance. Try to express new filters as compositions of
existing kinds first (e.g., `min_value` + `and` + `not` covers
many cases).