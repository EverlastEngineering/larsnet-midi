#!/usr/bin/env python3
"""
new_filter.py — Add a new per-event filter to the larsnet pipeline.

This is the automation half of the .github/skills/add-filter
skill. It handles the boilerplate (registry JSON, midiconfig.yaml,
API endpoint, Python wrapper, JS wrapper, test scaffolding). The
wiring step (calling the new filter from the apply chain) is
always manual — see SKILL.md Step 3.

Usage:
    # With a JSON spec file (recommended):
    python .github/skills/add-filter/scripts/new_filter.py --spec <spec.json>

    # Or with CLI flags (quick experimentation):
    python .github/skills/add-filter/scripts/new_filter.py \
        --id <id> --label <label> --kind <kind> --field <field> \
        --default <v> --min <v> --max <v> --step <v> \
        --stems <stems> --reason-template <tmpl> --value-format <fmt>

The script is idempotent: re-running with the same spec
overwrites the previous entry.

Spec fields (all required unless noted):
    id: snake_case unique identifier (e.g., "min_decay_col_min_db")
    label: human-readable slider label
    description: longer description for tooltip / docs
    kind: one of the closed enum — see filter_kinds.md
    field: event dict key to compare against threshold
    default: default threshold value (number, bool, or null)
    min: slider min (number)
    max: slider max (number)
    step: slider step (number)
    unit: slider unit string (optional, e.g., "ms", "dB", "Hz")
    ui_control: "slider" | "checkbox" | "select" | "number" (default "slider")
    applies_to_stems: list of stem names (e.g., ["toms"])
    yaml_paths:
        global: ["onset_detection", "<id>"]  (default; overridable)
        per_stem: {<stem>: [<stem>, <id>], ...}  (default; overridable)
    reason_template: language-agnostic template with {value},
        {threshold}, {field} placeholders
    value_format: "int" | "float1" | "float2" (default "int")

The script:
1. Validates the spec
2. Adds the entry to stems_to_midi/filter_registry.json
3. Adds the threshold key to midiconfig.yaml under each stem
   in applies_to_stems AND under onset_detection (global)
4. Adds the resolved entry to webui/api/projects.py
   (get_project_tuning_config)
5. Adds a Python wrapper to stems_to_midi/pga_event_builder.py
6. Adds a JavaScript function to webui/static/js/threshold-tuning.js
7. Adds test scaffolding to:
   - stems_to_midi/tests/test_pga_event_builder.py
   - webui/test_snap_delta_mask.py
"""
import argparse
import json
import re
import sys
from pathlib import Path

# Repo root (the script lives at .github/skills/add-filter/scripts/,
# so the repo root is 4 levels up: scripts → add-filter → skills → .github → REPO)
REPO_ROOT = Path(__file__).resolve().parents[4]
REGISTRY_PATH = REPO_ROOT / 'stems_to_midi' / 'filter_registry.json'
MIDICONFIG_PATH = REPO_ROOT / 'midiconfig.yaml'
PROJECTS_API_PATH = REPO_ROOT / 'webui' / 'api' / 'projects.py'
PGA_EVENT_BUILDER_PATH = REPO_ROOT / 'stems_to_midi' / 'pga_event_builder.py'
THRESHOLD_TUNING_JS_PATH = REPO_ROOT / 'webui' / 'static' / 'js' / 'threshold-tuning.js'
PGA_TESTS_PATH = REPO_ROOT / 'stems_to_midi' / 'tests' / 'test_pga_event_builder.py'
WEBUI_TESTS_PATH = REPO_ROOT / 'webui' / 'test_snap_delta_mask.py'


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

VALID_KINDS = {
    'min_value', 'max_value', 'nonzero_when_enabled',
    'and', 'or', 'not',
}
VALID_VALUE_FORMATS = {'int', 'float1', 'float2'}
VALID_UI_CONTROLS = {'slider', 'checkbox', 'select', 'number', 'text'}
VALID_STEMS = {'kick', 'snare', 'toms', 'hihat', 'cymbals'}
ID_PATTERN = re.compile(r'^[a-z][a-z0-9_]*$')
CAMEL_PATTERN = re.compile(r'_([a-z])')


def _camelize(snake: str) -> str:
    """snake_case → camelCase. 'pga_min_prominence' -> 'pgaMinProminence'."""
    return CAMEL_PATTERN.sub(lambda m: m.group(1).upper(), snake)


def _pascalize(snake: str) -> str:
    """snake_case → PascalCase. 'pga_min_prominence' -> 'PgaMinProminence'."""
    return snake[0].upper() + _camelize(snake)[1:]


def validate_spec(spec: dict) -> None:
    """Raise ValueError on the first invalid field. Lists all
    errors at once when multiple fields are wrong."""
    errors = []
    required = [
        'id', 'label', 'description', 'kind', 'field', 'default',
        'min', 'max', 'step', 'applies_to_stems',
        'reason_template', 'value_format',
    ]
    for key in required:
        if key not in spec:
            errors.append(f"missing required field: {key!r}")

    if 'id' in spec and not ID_PATTERN.match(spec['id']):
        errors.append(
            f"id {spec['id']!r} must match snake_case "
            f"(lowercase letters, digits, underscores; "
            f"must start with a letter)"
        )

    if 'kind' in spec and spec['kind'] not in VALID_KINDS:
        errors.append(
            f"kind {spec['kind']!r} is not in the closed enum: "
            f"{sorted(VALID_KINDS)}"
        )

    if 'value_format' in spec and spec['value_format'] not in VALID_VALUE_FORMATS:
        errors.append(
            f"value_format {spec['value_format']!r} must be one of: "
            f"{sorted(VALID_VALUE_FORMATS)}"
        )

    if 'ui_control' in spec and spec['ui_control'] not in VALID_UI_CONTROLS:
        errors.append(
            f"ui_control {spec['ui_control']!r} must be one of: "
            f"{sorted(VALID_UI_CONTROLS)}"
        )

    if 'applies_to_stems' in spec:
        stems = spec['applies_to_stems']
        if not isinstance(stems, list) or not stems:
            errors.append(
                f"applies_to_stems must be a non-empty list"
            )
        else:
            bad = [s for s in stems if s not in VALID_STEMS]
            if bad:
                errors.append(
                    f"applies_to_stems contains unknown stems: {bad}. "
                    f"Valid stems: {sorted(VALID_STEMS)}"
                )

    if 'reason_template' in spec:
        tmpl = spec['reason_template']
        for placeholder in ('{value}', '{threshold}'):
            if placeholder not in tmpl:
                errors.append(
                    f"reason_template must contain {placeholder!r}; "
                    f"got {tmpl!r}"
                )

    if errors:
        msg = "Spec validation failed:\n  - " + "\n  - ".join(errors)
        raise ValueError(msg)


def fill_defaults(spec: dict) -> dict:
    """Apply defaults for optional fields. Mutates and returns spec."""
    spec.setdefault('unit', '')
    spec.setdefault('ui_control', 'slider')
    spec.setdefault('yaml_paths', {
        'global': ['onset_detection', spec['id']],
        'per_stem': {stem: [stem, spec['id']] for stem in spec['applies_to_stems']},
    })
    return spec


def load_spec(args) -> dict:
    """Build the spec dict from CLI args (--spec file OR individual flags)."""
    if args.spec:
        with open(args.spec) as f:
            spec = json.load(f)
        return fill_defaults(spec)

    spec = {
        'id': args.id,
        'label': args.label,
        'description': args.description,
        'kind': args.kind,
        'field': args.field,
        'default': args.default,
        'min': args.min,
        'max': args.max,
        'step': args.step,
        'unit': args.unit,
        'ui_control': args.ui_control,
        'applies_to_stems': [s.strip() for s in args.stems.split(',') if s.strip()],
        'reason_template': args.reason_template,
        'value_format': args.value_format,
    }
    return fill_defaults(spec)


# ---------------------------------------------------------------------------
# File mutators (each idempotent — re-running with the same input
# is safe and produces the same output)
# ---------------------------------------------------------------------------


def update_registry(spec: dict, dry_run: bool = False) -> None:
    """Add or replace the filter entry in filter_registry.json.
    Preserves the existing file format (single-line lists) by
    doing text-level insertion rather than re-dumping the JSON."""
    text = REGISTRY_PATH.read_text()

    # Build the new entry as a JSON string. Use json.dumps with
    # custom separators to keep lists compact (matching the
    # existing file's format: ["toms"] on one line).
    new_entry_dict = {
        'id': spec['id'],
        'label': spec['label'],
        'description': spec['description'],
        'default': spec['default'],
        'min': spec['min'],
        'max': spec['max'],
        'step': spec['step'],
        'unit': spec['unit'],
        'ui_control': spec['ui_control'],
        'applies_to_stems': spec['applies_to_stems'],
        'yaml_paths': spec['yaml_paths'],
        'filter': {
            'kind': spec['kind'],
            'field': spec['field'],
            'reason_template': spec['reason_template'],
            'value_format': spec['value_format'],
        },
    }
    new_entry_str = json.dumps(
        new_entry_dict, indent=2, separators=(',', ': '),
        ensure_ascii=False,
    )

    # Check if the entry already exists (idempotent).
    entry_pattern = re.compile(
        r'\{\s*"id":\s*"' + re.escape(spec['id']) + r'",\s*\n',
    )
    if entry_pattern.search(text):
        # Replace the existing entry. Find the start and end of
        # the existing entry by counting braces (the entry is
        # a top-level object in the filters array).
        m_start = entry_pattern.search(text)
        start = m_start.start()
        # Find the matching closing brace. Count braces.
        depth = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        # Strip any trailing comma after the existing entry.
        while end < len(text) and text[end] in (',', ' ', '\n', '\r', '\t'):
            end += 1
        if dry_run:
            print(f"  ~ filter_registry.json: would update {spec['id']} entry")
            return
        text = text[:start] + new_entry_str + text[end:]
        action = 'updated'
    else:
        # Append the new entry before the closing ']' of filters.
        # Find '  ]' or ' ]' that closes the filters array.
        m = re.search(r'\n  \]\n\}', text)
        if not m:
            # Fallback: find the first ']' after 'filters'.
            m = re.search(r'"filters":\s*\[[^\]]*\]', text, re.DOTALL)
            if not m:
                print(f"  ! filter_registry.json: could not find filters array")
                return
        if dry_run:
            print(f"  ~ filter_registry.json: would add {spec['id']} entry")
            return
        # Insert before the closing ']' of the filters array.
        # The pattern matches the indentation of the array.
        new_text = text[:m.start()] + ',\n' + new_entry_str + text[m.start():]
        text = new_text
        action = 'added'

    REGISTRY_PATH.write_text(text)
    print(f"  ✓ filter_registry.json: {spec['id']} entry ({action})")


def update_midiconfig(spec: dict, dry_run: bool = False) -> None:
    """Add the threshold key to midiconfig.yaml under each stem in
    applies_to_stems and under onset_detection (global). Idempotent
    — re-running overwrites existing values."""
    text = MIDICONFIG_PATH.read_text()

    # Per-stem entries: e.g., "  min_decay_col_min_db: -80.0"
    for stem in spec['applies_to_stems']:
        yaml_path = spec['yaml_paths']['per_stem'].get(stem, [stem, spec['id']])
        key = yaml_path[-1]
        text = _upsert_yaml_key_in_stem_section(
            text, stem, key, spec['default'],
            comment=_build_per_stem_comment(spec),
        )

    # Global entry: e.g., "  pga_min_prominence: 3000"
    global_path = spec['yaml_paths'].get('global', ['onset_detection', spec['id']])
    key = global_path[-1]
    text = _upsert_yaml_key_in_onset_detection(text, key, spec['default'],
                                             comment=_build_global_comment(spec))

    if dry_run:
        print(f"  ~ midiconfig.yaml: would add {spec['id']} under "
              f"{', '.join(spec['applies_to_stems'])} and onset_detection")
        return

    MIDICONFIG_PATH.write_text(text)
    print(f"  ✓ midiconfig.yaml: {spec['id']} under "
          f"{', '.join(spec['applies_to_stems'])} and onset_detection")


def _build_per_stem_comment(spec: dict) -> str:
    """Build the YAML comment block for the per-stem entry."""
    return (
        f"# 2026-06-17: per-stem override of the global\n"
        f"# onset_detection.{spec['id']}. See\n"
        f"# stems_to_midi/filter_registry.json for the full spec.\n"
        f"# Default: {spec['default']}. See filter_kinds.md for the\n"
        f"# kind={spec['kind']} semantics."
    )


def _build_global_comment(spec: dict) -> str:
    """Build the YAML comment block for the global entry."""
    return (
        f"# 2026-06-17: global {spec['id']}. Per-stem overrides\n"
        f"# (e.g., toms.{spec['id']}) win over this value. See\n"
        f"# stems_to_midi/filter_registry.json for the full spec."
    )


def _upsert_yaml_key_in_stem_section(text, stem, key, value, comment):
    """Insert or replace a key in a stem section. The comment is
    placed ABOVE the key (YAML convention)."""
    # Find the stem section.
    pattern = re.compile(
        rf"^{re.escape(stem)}:\s*$",
        re.MULTILINE,
    )
    m = pattern.search(text)
    if not m:
        return text  # stem section doesn't exist; skip
    section_start = m.end()

    # Find the next "^{other_stem}:" or end of file.
    next_section = re.search(
        r"^[a-z]+:\s*$",
        text[section_start:],
        re.MULTILINE,
    )
    section_end = (section_start + next_section.start()) if next_section else len(text)

    section_text = text[section_start:section_end]

    # Look for the key in the section.
    key_pattern = re.compile(
        rf"^(\s*){re.escape(key)}:\s*[^\n]*$",
        re.MULTILINE,
    )
    km = key_pattern.search(section_text)
    if km:
        # Replace the existing key.
        new_section = key_pattern.sub(
            rf"\g<1>{key}: {_yaml_value(value)}", section_text, count=1,
        )
    else:
        # Insert at the end of the section with the comment.
        # Trim trailing whitespace from the section.
        new_section = section_text.rstrip() + "\n\n"
        new_section += f"{comment}\n  {key}: {_yaml_value(value)}\n"

    return text[:section_start] + new_section + text[section_end:]


def _upsert_yaml_key_in_onset_detection(text, key, value, comment):
    """Insert or replace a key in the onset_detection: section."""
    pattern = re.compile(r"^onset_detection:\s*$", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        return text
    section_start = m.end()
    next_section = re.search(
        r"^[a-z_]+:\s*$",
        text[section_start:],
        re.MULTILINE,
    )
    section_end = (section_start + next_section.start()) if next_section else len(text)

    section_text = text[section_start:section_end]
    key_pattern = re.compile(
        rf"^(\s*){re.escape(key)}:\s*[^\n]*$",
        re.MULTILINE,
    )
    km = key_pattern.search(section_text)
    if km:
        new_section = key_pattern.sub(
            rf"\g<1>{key}: {_yaml_value(value)}", section_text, count=1,
        )
    else:
        new_section = section_text.rstrip() + "\n"
        new_section += f"  {comment}\n  {key}: {_yaml_value(value)}\n"

    return text[:section_start] + new_section + text[section_end:]


def _yaml_value(value) -> str:
    """Format a value for YAML. Numbers as-is, strings quoted."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return 'null'
    # String — quote with double quotes.
    return f'"{value}"'


def update_projects_api(spec: dict, dry_run: bool = False) -> None:
    """Add the resolved entry to webui/api/projects.py
    get_project_tuning_config. Idempotent — re-running overwrites
    the existing entry."""
    text = PROJECTS_API_PATH.read_text()

    # Build the Python dict literal that resolves the threshold.
    # Pattern: '<id>': (\n                stem_cfg.get('<id>')\n                if stem_cfg.get('<id>') is not None\n                else onset_cfg.get('<id>', <default>)\n            ),
    new_entry = (
        f"            '{spec['id']}': (\n"
        f"                stem_cfg.get('{spec['id']}')\n"
        f"                if stem_cfg.get('{spec['id']}') is not None\n"
        f"                else onset_cfg.get('{spec['id']}', {repr(spec['default'])})\n"
        f"            ),"
    )

    # Strategy: find the existing pga-style entry in
    # get_project_tuning_config (e.g., 'pga_min_prominence'
    # or 'min_decay_col_min_db') and add the new entry right
    # after it. The known toms keys at time of writing are
    # 'pga_min_prominence' and 'min_decay_col_min_db'.
    anchor_keys = ['pga_min_prominence', 'min_decay_col_min_db',
                   'attack_rise_max_ms']

    # Check if the entry already exists (idempotent).
    if f"'{spec['id']}'" in text:
        entry_pattern = re.compile(
            r"^\s+'" + re.escape(spec['id'])
            + r"': \(\n.*?\),",
            re.MULTILINE | re.DOTALL,
        )
        em = entry_pattern.search(text)
        if em:
            new_text = text[:em.start()] + new_entry + text[em.end():]
        else:
            new_text = text
    else:
        # Find an anchor — the LAST existing toms key in the
        # tuning config (the most recently added one).
        anchor = None
        for key in anchor_keys:
            if f"'{key}'" in text:
                anchor = key
        if anchor is None:
            # Fallback: add at the end of the resolved dict
            # (just before the `return jsonify(resolved), 200`).
            m = re.search(
                r"^(            'band_max_ratio_max'.*?\),)\n",
                text,
                re.MULTILINE | re.DOTALL,
            )
            if not m:
                print(f"  ! webui/api/projects.py: could not find "
                      f"anchor for '{spec['id']}'. Add it manually.")
                return
            new_text = text[:m.end()] + "\n" + new_entry + text[m.end():]
        else:
            # Find the closing `),` of the anchor's entry.
            entry_pattern = re.compile(
                r"^\s+'" + re.escape(anchor) + r"': \(\n.*?\),",
                re.MULTILINE | re.DOTALL,
            )
            m = entry_pattern.search(text)
            if not m:
                print(f"  ! webui/api/projects.py: could not find "
                      f"the anchor entry for '{anchor}'. Add the "
                      f"new entry manually.")
                return
            new_text = text[:m.end()] + "\n" + new_entry + text[m.end():]

    if dry_run:
        print(f"  ~ webui/api/projects.py: would add {spec['id']} to "
              f"get_project_tuning_config")
        return

    PROJECTS_API_PATH.write_text(new_text)
    print(f"  ✓ webui/api/projects.py: get_project_tuning_config "
          f"includes {spec['id']}")


def update_python_wrapper(spec: dict, dry_run: bool = False) -> None:
    """Add the Python wrapper to pga_event_builder.py. The wrapper
    mirrors apply_pga_prominence_filter / apply_pga_decay_col_min_filter
    / apply_attack_rise_max_filter."""
    text = PGA_EVENT_BUILDER_PATH.read_text()

    fn_name = f"apply_{spec['id']}"
    if f"def {fn_name}(" in text:
        # Already exists — replace the docstring + body (idempotent).
        # Build the regex without raw f-string (avoids 3.11
        # f-string `\}` escape limitation).
        pattern = re.compile(
            r"def " + re.escape(fn_name)
            + r"\(.*?\n(?=\ndef |\nclass |\Z)",
            re.DOTALL,
        )
        new_fn = _build_python_wrapper(spec)
        text = pattern.sub(new_fn, text, count=1)
        action = "updated"
    else:
        # Insert before `def build_pga_events(`.
        anchor = "def build_pga_events("
        new_fn = _build_python_wrapper(spec)
        text = text.replace(anchor, new_fn + "\n\n" + anchor, 1)
        action = "added"

    # Update __all__.
    text = _upsert_all_entry(text, fn_name)

    if dry_run:
        print(f"  ~ pga_event_builder.py: would {action} {fn_name}")
        return

    PGA_EVENT_BUILDER_PATH.write_text(text)
    print(f"  ✓ pga_event_builder.py: {fn_name} ({action})")


def _build_python_wrapper(spec: dict) -> str:
    """Build the Python wrapper function source."""
    return (
        f"def apply_{spec['id']}(\n"
        f"    events: List[Dict[str, Any]],\n"
        f"    threshold: float,\n"
        f"    disabled_ids: Optional[Set[Any]] = None,\n"
        f") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:\n"
        f"    \"\"\"Re-tag PGA events with status='FILTERED' based on\n"
        f"    the {spec['field']} diagnostic (2026-06-17).\n"
        f"\n"
        f"    Registry-driven wrapper (filter kind={spec['kind']}).\n"
        f"    Reads the filter spec from\n"
        f"    ``stems_to_midi/filter_registry.json`` under the\n"
        f"    ``{spec['id']}`` entry; the predicate is evaluated by\n"
        f"    the shared :func:`evaluate_filter` in\n"
        f"    :mod:`stems_to_midi.filter_kinds`. Mirrors the\n"
        f"    pattern of :func:`apply_pga_prominence_filter`,\n"
        f"    :func:`apply_pga_decay_col_min_filter`, and\n"
        f"    :func:`apply_attack_rise_max_filter` — same\n"
        f"    `_apply_pga_filter` helper.\n"
        f"\n"
        f"    Returns ``(kept, filtered)``. Layered composition:\n"
        f"    pass the events that PASSED the previous filter,\n"
        f"    not the full events list — otherwise this filter\n"
        f"    overwrites the previous filter's FILTERED status\n"
        f"    with KEPT (the 2026-06-17 composition bug).\n"
        f"    \"\"\"\n"
        f"    return _apply_pga_filter(\n"
        f"        events, find_filter('{spec['id']}'), threshold, disabled_ids,\n"
        f"    )\n\n\n"
    )


def _upsert_all_entry(text, fn_name):
    """Add or replace the entry in __all__."""
    pattern = re.compile(
        rf"^(\s+)'{re.escape(fn_name)}',?\s*$",
        re.MULTILINE,
    )
    if pattern.search(text):
        return text
    # Insert before the closing ']' of __all__. Find the line
    # that ends with ']' and is the __all__ list.
    m = re.search(r"^__all__\s*=\s*\[(.*?)\n\]", text, re.DOTALL | re.MULTILINE)
    if not m:
        return text
    inner = m.group(1).rstrip()
    if not inner.endswith(','):
        inner += ','
    new_inner = inner + f"\n    '{fn_name}',"
    return text[:m.start()] + f"__all__ = [{new_inner}\n]" + text[m.end():]


def update_js_wrapper(spec: dict, dry_run: bool = False) -> None:
    """Add the JavaScript function to threshold-tuning.js."""
    text = THRESHOLD_TUNING_JS_PATH.read_text()

    fn_name = f"apply{_pascalize(spec['id'])}"
    if f"function {fn_name}(" in text:
        # Build the regex without raw f-string (avoids the
        # 3.11 f-string `\}` escape limitation).
        pattern = re.compile(
            r"function " + re.escape(fn_name) + r"\(.*?\n\}\n",
            re.DOTALL,
        )
        new_fn = _build_js_function(spec)
        text = pattern.sub(new_fn.rstrip() + "\n", text, count=1)
        action = "updated"
    else:
        # Insert before the applyTuningFilter() definition.
        anchor = "function applyTuningFilter()"
        new_fn = _build_js_function(spec)
        text = text.replace(anchor, new_fn + "\n" + anchor, 1)
        action = "added"

    if dry_run:
        print(f"  ~ threshold-tuning.js: would {action} {fn_name}")
        return

    THRESHOLD_TUNING_JS_PATH.write_text(text)
    print(f"  ✓ threshold-tuning.js: {fn_name} ({action})")


def _build_js_function(spec: dict) -> str:
    """Build the JavaScript filter function source."""
    fn_name = f"apply{_pascalize(spec['id'])}"
    return (
        f"/**\n"
        f" * {fn_name} — added by .github/skills/add-filter\n"
        f" * ({spec['id']}, {spec['kind']}).\n"
        f" *\n"
        f" * Registry-driven wrapper: reads the filter spec from\n"
        f" * the loaded registry and calls the shared\n"
        f" * `evaluateFilter` from filter_kinds.js. The hard-coded\n"
        f" * fallback below mirrors the old behavior so the panel\n"
        f" * still works when the registry API is down.\n"
        f" *\n"
        f" * Composition: pass the events that PASSED the\n"
        f" * previous filter (NOT tuningBaseEvents). Otherwise this\n"
        f" * filter overwrites the previous filter's FILTERED\n"
        f" * status with KEPT (the 2026-06-17 composition bug).\n"
        f" *\n"
        f" * Returns [kept, filtered] (mirrors the Python\n"
        f" * `_apply_pga_filter`).\n"
        f" */\n"
        f"function {fn_name}(events, threshold, disabledIds) {{\n"
        f"    const registry = _filterRegistryCache;\n"
        f"    const spec = registry\n"
        f"        ? findFilter(registry, '{spec['id']}')\n"
        f"        : null;\n"
        f"    const disabled = disabledIds || new Set();\n"
        f"    const kept = [];\n"
        f"    const filtered = [];\n"
        f"    for (const ev of events) {{\n"
        f"        const evId = ev.id != null ? ev.id : ev.time;\n"
        f"        const isDisabled = disabled.has(evId);\n"
        f"\n"
        f"        if (isDisabled) {{\n"
        f"            ev.status = 'FILTERED';\n"
        f"            ev.filter_reason = 'manually disabled via WebUI';\n"
        f"            filtered.push(ev);\n"
        f"        }} else if (spec) {{\n"
        f"            // Registry-driven evaluation.\n"
        f"            const result = evaluateFilter(spec, ev, threshold);\n"
        f"            if (result === false) {{\n"
        f"                ev.status = 'FILTERED';\n"
        f"                ev.filter_reason = buildFilterReason(spec, ev, threshold);\n"
        f"                filtered.push(ev);\n"
        f"            }} else {{\n"
        f"                ev.status = 'KEPT';\n"
        f"                delete ev.filter_reason;\n"
        f"                kept.push(ev);\n"
        f"            }}\n"
        f"        }} else {{\n"
        f"            // Fallback: registry not loaded.\n"
        f"            const value = ev.{spec['field']};\n"
        f"            // The exact predicate depends on the kind.\n"
        f"            // Update the predicate here for the kind.\n"
        f"            ev.status = 'KEPT';\n"
        f"            delete ev.filter_reason;\n"
        f"            kept.push(ev);\n"
        f"        }}\n"
        f"        // Update pga_filter_config so the tooltip shows the live threshold.\n"
        f"        if (ev.pga_filter_config) {{\n"
        f"            ev.pga_filter_config.{spec['id']} = threshold;\n"
        f"        }}\n"
        f"    }}\n"
        f"    return [kept, filtered];\n"
        f"}}\n\n"
    )


def update_test_scaffolding(spec: dict, dry_run: bool = False) -> None:
    """Add test scaffolding to test_pga_event_builder.py and
    test_snap_delta_mask.py. Generates skeleton tests the user
    should fill in with real test cases."""
    class_name = f"Test{_pascalize(spec['id'])}"

    # Python test scaffolding.
    py_text = PGA_TESTS_PATH.read_text()
    py_import_marker = "from stems_to_midi.pga_event_builder import"
    if f"apply_{spec['id']}" not in py_text:
        # Add the import.
        py_text = py_text.replace(
            py_import_marker,
            f"{py_import_marker}  # noqa: E402\n    apply_{spec['id']},",
            1,
        )
    if f"class {class_name}:" not in py_text:
        py_text += _build_python_test_skeleton(spec, class_name)
        if not dry_run:
            PGA_TESTS_PATH.write_text(py_text)
        print(f"  {'~' if dry_run else '✓'} "
              f"test_pga_event_builder.py: {class_name} (skeleton)")
    else:
        print(f"  ~ test_pga_event_builder.py: {class_name} already exists")

    # JS test scaffolding.
    js_text = WEBUI_TESTS_PATH.read_text()
    if f"class {class_name}:" not in js_text:
        js_text += _build_js_test_skeleton(spec, class_name)
        if not dry_run:
            WEBUI_TESTS_PATH.write_text(js_text)
        print(f"  {'~' if dry_run else '✓'} "
              f"test_snap_delta_mask.py: {class_name} (skeleton)")
    else:
        print(f"  ~ test_snap_delta_mask.py: {class_name} already exists")


def _build_python_test_skeleton(spec: dict, class_name: str) -> str:
    """Generate a Python test class skeleton."""
    fn_name = f"apply_{spec['id']}"
    return (
        f"\n\nclass {class_name}:\n"
        f"    \"\"\"``{fn_name}`` — added by .github/skills/add-filter.\n"
        f"\n"
        f"    Skeleton. Fill in real test cases — the boilerplate\n"
        f"    below mirrors TestDecayColMinFilter and\n"
        f"    TestAttackRiseMaxFilter.\n"
        f"    \"\"\"\n"
        f"\n"
        f"    def test_drops_qualifying_events(self):\n"
        f"        \"\"\"An event matching the kind's predicate (e.g.,\n"
        f"        below threshold for min_value) is FILTERED; an\n"
        f"        event not matching is KEPT.\"\"\"\n"
        f"        events = [\n"
        f"            {{'time': 0.5, '{spec['field']}: ...}},  # TODO\n"
        f"        ]\n"
        f"        kept, filtered = {fn_name}(events, {repr(spec['default'])})\n"
        f"        # TODO: assert kept and filtered counts and reasons.\n"
        f"\n"
        f"    def test_skips_none_values(self):\n"
        f"        \"\"\"An event with no ``{spec['field']}`` field is\n"
        f"        KEPT (the filter can't act on it).\"\"\"\n"
        f"        events = [\n"
        f"            {{'time': 1.0}},  # TODO: no {spec['field']} field\n"
        f"        ]\n"
        f"        # TODO: assert kept count and that filter_reason\n"
        f"        # was not set.\n"
        f"\n"
        f"    def test_threshold_resolution_in_build_pga_events_with_filter(self):\n"
        f"        \"\"\"_build_pga_events_with_filter must include\n"
        f"        {spec['id']} in pga_filter_config with per-stem\n"
        f"        > global > default precedence.\"\"\"\n"
        f"        # TODO: build a fake config dict and assert the\n"
        f"        # resolved value on a sample event's\n"
        f"        # pga_filter_config.\n"
    )


def _build_js_test_skeleton(spec: dict, class_name: str) -> str:
    """Generate a JS test class skeleton."""
    fn_name = f"apply{_pascalize(spec['id'])}"
    # Pre-format the expected applyTuningFilter regex (avoid f-string
    # brace escape issues by formatting outside the raw string).
    func_regex = (
        r"function\s*applyTuningFilter\s*\(\s*\)\s*\{(.*?)\n\}\n"
    )
    return (
        f"\n\nclass {class_name}:\n"
        f"    \"\"\"{fn_name} — added by .github/skills/add-filter.\n"
        f"\n"
        f"    Skeleton. Fill in real test cases — the boilerplate\n"
        f"    below mirrors TestPgaFilterFunctions and\n"
        f"    TestAttackRiseFilter.\n"
        f"    \"\"\"\n"
        f"\n"
        f"    def test_function_exists(self, threshold_tuning_js_text):\n"
        f"        m = re.search(\n"
        f"            r\"function\\s+{re.escape(fn_name)}\\s*\\(\\s*events\\s*,\\s*threshold\",\n"
        f"            threshold_tuning_js_text,\n"
        f"        )\n"
        f"        assert m is not None, (\n"
        f"            f\"expected `function {fn_name}(events, threshold, \"\n"
        f"            f\"disabledIds)` in threshold-tuning.js\"\n"
        f"        )\n"
        f"\n"
        f"    def test_filter_wired_into_apply_tuning_filter(self, threshold_tuning_js_text):\n"
        f"        \"\"\"{fn_name} must be called from applyTuningFilter\n"
        f"        for the stems in applies_to_stems.\"\"\"\n"
        f"        m = re.search(\n"
        f"            {func_regex!r},\n"
        f"            threshold_tuning_js_text,\n"
        f"            re.DOTALL,\n"
        f"        )\n"
        f"        assert m is not None\n"
        f"        body = m.group(1)\n"
        f"        assert '{fn_name}' in body, (\n"
        f"            f\"applyTuningFilter must call {fn_name} for \"\n"
        f"            f\"the stems in applies_to_stems ({spec['applies_to_stems']}).\"\n"
        f"        )\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--spec',
        help='Path to a JSON spec file (recommended).',
    )
    parser.add_argument('--id')
    parser.add_argument('--label')
    parser.add_argument('--description')
    parser.add_argument('--kind')
    parser.add_argument('--field')
    parser.add_argument('--default', type=lambda s: json.loads(s))
    parser.add_argument('--min', dest='min', type=lambda s: json.loads(s))
    parser.add_argument('--max', dest='max', type=lambda s: json.loads(s))
    parser.add_argument('--step', type=lambda s: json.loads(s))
    parser.add_argument('--unit', default='')
    parser.add_argument('--ui-control', default='slider')
    parser.add_argument('--stems', help='Comma-separated, e.g. "toms"')
    parser.add_argument('--reason-template')
    parser.add_argument('--value-format', default='int')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be done without modifying any files.',
    )

    args = parser.parse_args()

    if not args.spec:
        # Validate that all required CLI flags are present.
        required = ('id', 'label', 'description', 'kind', 'field',
                    'default', 'min', 'max', 'step', 'stems',
                    'reason_template')
        missing = [r for r in required if getattr(args, r) is None]
        if missing:
            parser.error(
                f"missing required flags when --spec is not used: "
                f"{missing}. Either pass --spec <file.json> or "
                f"all of: {', '.join('--' + r.replace('_', '-') for r in required)}"
            )

    spec = load_spec(args)
    validate_spec(spec)

    if args.dry_run:
        print(f"[DRY RUN] Would add filter {spec['id']!r} "
              f"({spec['kind']}, stems={spec['applies_to_stems']}, "
              f"default={spec['default']}):")

    else:
        print(f"Adding filter {spec['id']!r} ({spec['kind']}, "
              f"stems={spec['applies_to_stems']}, default={spec['default']}):")

    update_registry(spec, dry_run=args.dry_run)
    update_midiconfig(spec, dry_run=args.dry_run)
    update_projects_api(spec, dry_run=args.dry_run)
    update_python_wrapper(spec, dry_run=args.dry_run)
    update_js_wrapper(spec, dry_run=args.dry_run)
    update_test_scaffolding(spec, dry_run=args.dry_run)
    if args.dry_run:
        print()
        print("[DRY RUN] No files were modified. Re-run without "
              "--dry-run to apply the changes.")
    else:
        print()
        print(f"Done. Next step: wire {spec['id']} into the apply chain.")
        print(f"See .github/skills/add-filter/SKILL.md Step 3 for the wiring steps.")


if __name__ == '__main__':
    main()