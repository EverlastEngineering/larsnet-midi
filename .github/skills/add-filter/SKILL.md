---
name: add-filter
description: 'Add a new per-event filter to the larsnet toms (and other stems) PGA pipeline. Use when adding a filter for prominence, decay_col_min, attack_rise, geomean, sustain, strength, band_max_ratio, or any new diagnostic threshold. Covers the filter registry JSON, midiconfig.yaml, Python wrapper, JavaScript wrapper, API endpoint, and test scaffolding. Triggers on: add a filter, new filter, filter for X, threshold for Y, wire a filter.'
user-invocable: true
argument-hint: '[--spec path/to/spec.json] [--id <id> --label <l> --kind <k> --field <f> ...]'
---

# Add a Filter

Add a new per-event filter to the larsnet pipeline. The filter
registry (`stems_to_midi/filter_registry.json`) is the **single
source of truth** — both Python and JavaScript consume it, so
the parity problem is gone. This skill adds a filter via the
registry; the auto-generated Python and JavaScript wrappers stay
in lockstep.

## When to Use

Use this skill when adding a new per-event threshold filter.
Triggers:
- "Add a filter for X"
- "New filter for Y"
- "Wire a threshold for Z"
- "Add a slider for [diagnostic name]"
- "I need a filter that drops events where [field] is above/below N"

Do NOT use this skill for:
- Per-stem config that's NOT a filter (e.g., `midi.min_velocity` —
  those go in `webui/settings_schema.py`, not the filter registry).
- New filter KINDS (e.g., a new combinator like `xor`); those
  require editing `stems_to_midi/filter_kinds.py` and
  `webui/static/js/filter_kinds.js` (the generic evaluators).
- Detection-time fields (e.g., a new computed feature on events);
  see `docs/stems_toms_prominence_and_decay_col_min.md` first.

## Procedure

### Step 1 — Decide the filter's shape

Choose:
- **id**: snake_case, unique across the registry (e.g.,
  `min_decay_col_min_db`, `attack_rise_max_ms`).
- **kind**: one of the closed enum — see
  [references/filter_kinds.md](./references/filter_kinds.md).
- **field**: the event dict key to compare against the threshold
  (e.g., `prominence`, `decay_col_min_median_db`,
  `attack_rise_ms`, `geomean`, `band_max_ratio`).
- **applies_to_stems**: list of stems that show this slider in
  the WebUI. Currently most toms filters are toms-only.
- **default**: the initial threshold value. Pick from empirical
  cluster boundaries (see `agent-plans/calibration-data.md`).
- **min / max / step**: slider bounds. Use coarse steps for
  values that don't need precision (e.g., step=1 ms for
  attack_rise) and finer steps for dB values (step=0.5).
- **reason_template**: language-agnostic template with
  `{value}`, `{threshold}`, `{field}` placeholders. Keep it
  terse: `"below my_filter ({value} < {threshold})"`.
- **value_format**: one of `int`, `float1`, `float2`. Controls
  how `{value}` and `{threshold}` are rendered.

See [templates/filter_spec.example.json](./templates/filter_spec.example.json)
for a complete example spec file.

### Step 2 — Run the script

The script handles the boilerplate (registry JSON, YAML,
Python wrapper, JS wrapper, API endpoint, test scaffolding).
The wiring (where in the apply chain to call the new filter)
is **always manual** because it depends on context.

```bash
# With a JSON spec file (recommended for production):
conda run -n drumtomidi python .github/skills/add-filter/scripts/new_filter.py \
    --spec .github/skills/add-filter/templates/filter_spec.example.json

# Or with CLI flags (quick experimentation):
conda run -n drumtomidi python .github/skills/add-filter/scripts/new_filter.py \
    --id my_filter \
    --label "My Filter" \
    --description "What this filter does." \
    --kind min_value \
    --field prominence \
    --default 1000 --min 0 --max 10000 --step 100 --unit "" \
    --stems toms \
    --reason-template "below my_filter ({value} < {threshold})" \
    --value-format int
```

The script will:
1. Add the entry to `stems_to_midi/filter_registry.json`.
2. Add the threshold key to `midiconfig.yaml` under each stem
   in `applies_to_stems` AND under `onset_detection` (global
   fallback).
3. Add the resolved entry to `webui/api/projects.py`
   (`get_project_tuning_config`).
4. Add a Python wrapper to `stems_to_midi/pga_event_builder.py`
   (function `apply_<id>`).
5. Add a JavaScript function to
   `webui/static/js/threshold-tuning.js` (function
   `apply<IdCamelCase>`).
6. Add test scaffolding to
   `stems_to_midi/tests/test_pga_event_builder.py` and
   `webui/test_snap_delta_mask.py`.

The script is idempotent: re-running with the same spec
overwrites the previous entry.

### Step 3 — Wire the filter into the apply chain

The script does NOT automatically call the new filter from the
apply chains. **You must wire it in manually** — this is the
one place that requires context-aware decisions.

For the **toms** stem, there are two places to wire:

#### A. `stems_to_midi/pga_event_builder.py` — `_build_pga_events_with_filter`

This is the detect-time path (runs when the CLI generates
the analysis.json). Find the existing chain:

```python
events_kept, decay_filtered = apply_pga_decay_col_min_filter(
    events_kept, decay_col_min_threshold,
)
events_filtered = events_filtered + decay_filtered
# 2026-06-17: attack_rise filter (third PGA pass). ...
attack_rise_threshold = float(...)
events_kept, attack_filtered = apply_attack_rise_max_filter(
    events_kept, attack_rise_threshold,
)
events_filtered = events_filtered + attack_filtered
```

Add your filter as the next pass. **Chaining rule** (2026-06-17
bug fix): pass `events_kept` (the kept list from the previous
filter), NOT `events_filtered` and NOT the raw events. This
prevents the second filter from overwriting the first's
FILTERED status with KEPT.

Resolve the threshold with the same per-stem > global >
default precedence:

```python
my_threshold = float(
    toms_cfg.get('my_filter_id')
    if toms_cfg.get('my_filter_id') is not None
    else onset_cfg.get('my_filter_id', <default_value>)
)
```

Stamp the resolved threshold on every event's
`pga_filter_config` so the sidecar tooltip shows the live
value:

```python
for ev in raw:
    pga_filter_config = dict(ev.get('pga_filter_config', {}))
    pga_filter_config['pga_min_prominence'] = prom_threshold
    pga_filter_config['min_decay_col_min_db'] = decay_col_min_threshold
    pga_filter_config['attack_rise_max_ms'] = attack_rise_threshold
    pga_filter_config['my_filter_id'] = my_threshold  # NEW
    ev['pga_filter_config'] = pga_filter_config
```

#### B. `stems_to_midi/rebuild_core.py` — toms re-filter branch

This is the rebuild path (runs when the WebUI re-filters
without re-detecting). Find the existing chain:

```python
pga_kept, col_min_filtered = apply_pga_decay_col_min_filter(
    pga_kept,
    col_min_threshold,
)
pga_filtered = pga_filtered + col_min_filtered
# 2026-06-17: attack_rise filter (third PGA pass). ...
pga_kept, attack_filtered = apply_attack_rise_max_filter(
    pga_kept,
    attack_rise_threshold,
)
pga_filtered = pga_filtered + attack_filtered
```

Add your filter as the next pass. Same chaining rule. Import
the new wrapper at the top of the function (`from
.pga_event_builder import apply_my_filter` — same pattern
as `apply_attack_rise_max_filter`).

Also update the `filter_reason` heuristic so events filtered
by your filter keep their reason from the wrapper (currently
the heuristic looks for `'min_decay_col_min_db'` and
`'attack_rise_max_ms'` in the existing reason — add your
filter's substring if needed):

```python
if (
    'min_decay_col_min_db' in existing_reason
    or 'attack_rise_max_ms' in existing_reason
    or 'my_filter_id' in existing_reason  # NEW
):
    pass  # reason already set by the wrapper
else:
    # Build the prominence-filter reason
    ...
```

#### C. `webui/static/js/threshold-tuning.js` — `applyTuningFilter`

This is the live-preview path (runs as the user drags sliders).
Find the existing chain:

```javascript
const decayColMinThreshold = params.min_decay_col_min_db;
if (decayColMinThreshold != null) {
    const [kept2, filtered2] = applyPgaDecayColMinFilter(
        pgaKept, decayColMinThreshold
    );
    pgaKept = kept2;
    pgaFiltered = pgaFiltered.concat(filtered2);
}
const attackRiseThreshold = params.attack_rise_max_ms;
if (attackRiseThreshold != null) {
    const [kept3, filtered3] = applyAttackRiseMaxFilter(
        pgaKept, attackRiseThreshold
    );
    pgaKept = kept3;
    pgaFiltered = pgaFiltered.concat(filtered3);
}
```

Add your filter as the next pass. **Chaining rule** (2026-06-17
bug fix): pass `pgaKept` (the kept list from the previous
filter), NOT `tuningBaseEvents` and NOT `pgaFiltered`. This
prevents the bug the user reported on 2026-06-17.

### Step 4 — Verify

Run the test suite. New tests were added by the script in
Step 2; they should pass without modification:

```bash
conda run -n drumtomidi pytest \
    stems_to_midi/tests/test_filter_kinds.py \
    stems_to_midi/tests/test_pga_event_builder.py \
    webui/test_snap_delta_mask.py::TestPgaFilterFunctions \
    webui/test_snap_delta_mask.py::TestAttackRiseFilter \
    -k "not Real and not real" -v
```

End-to-end smoke on project 6 (Taylor Swift toms):

```bash
conda run -n drumtomidi python -m stems_to_midi.rebuild_cli 6 --stems toms
```

Verify the new slider appears in the WebUI by opening the
tune slideout and confirming the new control is present with
the default value.

### Step 5 — Commit

Use a Conventional Commits message:

```bash
git add <all changed files>
git commit -m "feat(<stem>): add <filter_id> filter

<one-line description of what the filter does and why>.

- filter_registry.json: new entry for <id>
- midiconfig.yaml: <stem>.<id>: <default>
- pga_event_builder.py: apply_<id> wrapper (registry-driven)
- rebuild_core.py: wired into toms re-filter branch (Pass N)
- threshold-tuning.js: apply<IdCamelCase> (registry-driven)
- webui/api/projects.py: get_project_tuning_config includes <id>
- tests: <N> new tests in test_pga_event_builder.py and test_snap_delta_mask.py"
```

## Files Touched (every filter addition)

| File | What changes |
|---|---|
| `stems_to_midi/filter_registry.json` | New entry in `filters[]` |
| `midiconfig.yaml` | New key under each stem + `onset_detection` global |
| `webui/api/projects.py` | New entry in `get_project_tuning_config` |
| `stems_to_midi/pga_event_builder.py` | New wrapper function + `__all__` entry |
| `webui/static/js/threshold-tuning.js` | New filter function (registry-driven) |
| `stems_to_midi/tests/test_pga_event_builder.py` | New test class `Test<IdPascalCase>` |
| `webui/test_snap_delta_mask.py` | New test class `Test<IdPascalCase>` |
| `_build_pga_events_with_filter` (in pga_event_builder.py) | **Manual wire** — add the new filter call |
| `rebuild_core.py` toms re-filter branch | **Manual wire** — add the new filter call |
| `applyTuningFilter` (in threshold-tuning.js) | **Manual wire** — add the new filter call |

## Common Pitfalls

- **Wrong chaining order** (2026-06-17 bug): always pass the
  KEPT list from the previous filter, not the raw events.
  The bug was that `applyPgaDecayColMinFilter` was called on
  `tuningBaseEvents` (the full list), overwriting the
  prominence filter's FILTERED status.
- **Missing YAML key**: forgetting to add the threshold key to
  `midiconfig.yaml` under both the stem section and the
  `onset_detection` global section. The script handles this;
  manual wirings must remember both.
- **Wrong `applies_to_stems`**: the slider appears in the WebUI
  for every stem in `applies_to_stems`. If you only want toms,
  use `["toms"]`. If you want kick/snare/etc., list them — but
  remember that the Python rebuild path is currently
  toms-only (Phase 6 will migrate the other stems).
- **Hardcoded threshold default**: the JSON `default` field is
  the fallback when the YAML is missing the key. Make sure it
  matches the empirical cluster boundary (see
  `agent-plans/calibration-data.md`).
- **Missing `pga_filter_config` update**: every event's
  `pga_filter_config` must include the new threshold so the
  WebUI tooltip shows the live value. The Python wiring does
  this automatically in the loop at the end of
  `_build_pga_events_with_filter`.

## References

- [Filter kinds reference](./references/filter_kinds.md)
- [Example spec file](./templates/filter_spec.example.json)
- [Script source](./scripts/new_filter.py)
- `docs/stems_toms_prominence_and_decay_col_min.md` — the
  source-of-truth docs for the prominence and decay_col_min
  calculations that the filter chain depends on.
- `agent-plans/calibration-data.md` — empirical cluster data
  for setting defaults.