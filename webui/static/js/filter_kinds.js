/**
 * filter_kinds.js — JS-side filter registry evaluator (2026-06-15).
 *
 * Single source of truth for per-event filters shared between
 * Python (stems_to_midi.filter_kinds) and the WebUI. The
 * registry lives in stems_to_midi/filter_registry.json. Both
 * languages read the same file — Python loads it via
 * `json.load`, the WebUI fetches it via `/api/filters/schema`.
 * Adding a new filter is a JSON entry; the parity problem is
 * gone.
 *
 * This module is the JS mirror of `stems_to_midi.filter_kinds`.
 * The two evaluators sit side-by-side and implement the SAME
 * closed enum of filter kinds (see `kinds` in the JSON):
 *
 *   - min_value: KEPT if event[field] >= threshold. Returns
 *     null if field is missing. Covers PGA prominence,
 *     decay_col_min, geomean, sustain, strength,
 *     attack_sharpness.
 *
 *   - max_value: KEPT if event[field] <= threshold. Returns
 *     null if field is missing. Covers band_max_ratio_max.
 *
 *   - nonzero_when_enabled: KEPT if not enabled, OR if
 *     enabled and event[field] > 0. Returns false if enabled
 *     and field is null or <= 0. Covers show_only_snap_events.
 *
 *   - and: KEPT if all children are KEPT. Empty children = true.
 *
 *   - or: KEPT if any child is KEPT. All-null children = null.
 *
 *   - not: inverts a single child. null child = null.
 *
 * Public API:
 *   - loadFilterRegistry() — fetch /api/filters/schema and
 *     cache the result. Returns a Promise<registry>.
 *   - findFilter(registry, filterId) — find a filter entry
 *     by id, or null if not found.
 *   - listFiltersForStem(registry, stemType) — return the
 *     filters whose applies_to_stems contains the stem.
 *   - evaluateFilter(filterSpec, event, threshold,
 *       enabled=true) — evaluate a single event. Returns
 *     true / false / null (null = cannot evaluate).
 *   - buildFilterReason(filterSpec, event, threshold) —
 *     format the reason_template with the event's value
 *     and threshold.
 *   - resolveThreshold(filterSpec, stemType, config) —
 *     per_stem > global > default precedence. Returns the
 *     resolved value (number, bool, or null).
 *
 * The Python and JS evaluators are designed to be
 * parity-checkable. If you change one, change the other,
 * and update the test suite for both.
 */

// ---------------------------------------------------------------------------
// Registry loading
// ---------------------------------------------------------------------------

let _registryCache = null;

async function loadFilterRegistry(force = false) {
    if (_registryCache && !force) {
        return _registryCache;
    }
    const resp = await fetch('/api/filters/schema');
    if (!resp.ok) {
        throw new Error(
            `Failed to load filter registry: ${resp.status} ${resp.statusText}`
        );
    }
    _registryCache = await resp.json();
    return _registryCache;
}

/** Synchronous access to the cached registry. Throws if
 * loadFilterRegistry hasn't been called yet. */
function getCachedRegistry() {
    if (!_registryCache) {
        throw new Error(
            'Filter registry not loaded — call loadFilterRegistry() first'
        );
    }
    return _registryCache;
}

function findFilter(registry, filterId) {
    if (!registry || !registry.filters) return null;
    return registry.filters.find(f => f.id === filterId) || null;
}

function listFiltersForStem(registry, stemType) {
    if (!registry || !registry.filters) return [];
    return registry.filters.filter(
        f => Array.isArray(f.applies_to_stems)
            && f.applies_to_stems.includes(stemType)
    );
}

// ---------------------------------------------------------------------------
// Value formatting (mirrors _format_value in filter_kinds.py)
// ---------------------------------------------------------------------------

function _formatValue(value, valueFormat) {
    if (value === null || value === undefined) return 'N/A';
    switch (valueFormat) {
        case 'int':
            return String(Math.round(Number(value)));
        case 'float1':
            return Number(value).toFixed(1);
        case 'float2':
            return Number(value).toFixed(2);
        default:
            return String(value);
    }
}

// ---------------------------------------------------------------------------
// AST evaluation (mirrors _evaluate_node in filter_kinds.py)
// ---------------------------------------------------------------------------

function _evaluateNode(filterNode, event, threshold, enabled) {
    const kind = filterNode.kind;

    if (kind === 'min_value') {
        const value = event[filterNode.field];
        if (value === null || value === undefined) return null;
        return value >= threshold;
    }

    if (kind === 'max_value') {
        const value = event[filterNode.field];
        if (value === null || value === undefined) return null;
        return value <= threshold;
    }

    if (kind === 'nonzero_when_enabled') {
        if (!enabled) return true;
        const value = event[filterNode.field];
        if (value === null || value === undefined || value <= 0) return false;
        return true;
    }

    if (kind === 'and') {
        const children = filterNode.filters || [];
        if (children.length === 0) return true;
        for (const child of children) {
            if (_evaluateNode(child, event, threshold, enabled) === false) {
                return false;
            }
        }
        return true;
    }

    if (kind === 'or') {
        const children = filterNode.filters || [];
        if (children.length === 0) return true;
        let sawNone = false;
        for (const child of children) {
            const result = _evaluateNode(child, event, threshold, enabled);
            if (result === true) return true;
            if (result === null) sawNone = true;
        }
        return sawNone ? null : false;
    }

    if (kind === 'not') {
        const child = filterNode.filter;
        if (!child) return true;
        const result = _evaluateNode(child, event, threshold, enabled);
        if (result === null) return null;
        return !result;
    }

    throw new Error(`Unknown filter kind: ${kind}`);
}

function evaluateFilter(filterSpec, event, threshold, enabled = true) {
    return _evaluateNode(filterSpec.filter, event, threshold, enabled);
}

function buildFilterReason(filterSpec, event, threshold) {
    const inner = filterSpec.filter || {};
    const template = inner.reason_template || '';
    const valueFormat = inner.value_format || 'float2';
    const field = inner.field || '?';
    const kind = inner.kind;

    let value = null;
    if (kind === 'min_value' || kind === 'max_value') {
        value = event[field];
    }

    return template
        .replace('{value}', _formatValue(value, valueFormat))
        .replace('{threshold}', _formatValue(threshold, valueFormat))
        .replace('{field}', field);
}

// ---------------------------------------------------------------------------
// Threshold resolution (per-stem > global > default)
//
// Walks the config object along the yaml_path. Returns the value
// at the path, or undefined if any key is missing.
// ---------------------------------------------------------------------------

function _lookupYamlPath(config, path) {
    let current = config;
    for (const key of path) {
        if (
            current === null
            || current === undefined
            || typeof current !== 'object'
            || !(key in current)
        ) {
            return undefined;
        }
        current = current[key];
    }
    return current;
}

function resolveThreshold(filterSpec, stemType, config) {
    const yamlPaths = filterSpec.yaml_paths || {};
    const perStem = yamlPaths.per_stem || {};
    const globalPath = yamlPaths.global;

    // Try per-stem first
    if (stemType in perStem) {
        const val = _lookupYamlPath(config, perStem[stemType]);
        if (val !== null && val !== undefined) return val;
    }

    // Then global
    if (globalPath) {
        const val = _lookupYamlPath(config, globalPath);
        if (val !== null && val !== undefined) return val;
    }

    // Fall back to the JSON default
    return filterSpec.default;
}

// ---------------------------------------------------------------------------
// WebUI integration helper: build a STEM_SLIDER_CONFIGS-shaped
// object from the registry. This is the bridge that lets
// threshold-tuning.js render the toms sliders from the
// registry (replacing the hard-coded STEM_SLIDER_CONFIGS.toms
// entry).
//
// Shape: { [stemType]: [{ key, label, min, max, step,
//                          fallback, unit, yamlPath }] }
// ---------------------------------------------------------------------------

function buildSliderConfigsForStem(registry, stemType) {
    return listFiltersForStem(registry, stemType).map(f => ({
        key: f.id,
        label: f.label,
        min: f.min,
        max: f.max,
        step: f.step,
        fallback: f.default,
        unit: f.unit || '',
        // The WebUI's save path expects yamlPath as a list.
        // Use the per-stem path if available, else the global.
        yamlPath:
            (f.yaml_paths && f.yaml_paths.per_stem
                && f.yaml_paths.per_stem[stemType])
            || (f.yaml_paths && f.yaml_paths.global)
            || null,
    }));
}
