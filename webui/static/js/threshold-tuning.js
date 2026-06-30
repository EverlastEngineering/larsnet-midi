/**
 * Threshold Tuning Module
 *
 * Provides interactive client-side sliders that re-filter events_configured
 * data in the browser. No server round-trip needed — filtering is just
 * comparisons on the pre-computed spectral features.
 *
 * Integrates with waveform.js via:
 *   - waveformTuningEvents: when non-null, drawWaveform() renders these
 *   - waveformTuningActive: boolean flag for visual indicator
 *   - drawWaveform(): called after each re-filter
 */

// ─── Filter registry (2026-06-15) ────────────────────────────────────────
//
// The STEM_SLIDER_CONFIGS below is the hard-coded FALLBACK. The
// real metadata comes from the filter registry JSON (same file
// the Python side reads). After the registry loads (see
// _ensureFilterRegistryLoaded), the toms entry is OVERRIDDEN
// with the registry-derived version. This is the "single
// source of truth" hook: the JSON is the metadata, the
// hard-coded entry is the offline fallback.

let _filterRegistryCache = null;
let _filterRegistryLoadInFlight = null;

async function _ensureFilterRegistryLoaded() {
    if (_filterRegistryCache) return _filterRegistryCache;
    if (_filterRegistryLoadInFlight) return _filterRegistryLoadInFlight;
    _filterRegistryLoadInFlight = (async () => {
        try {
            _filterRegistryCache = await loadFilterRegistry();

            // 2026-06-19: cache the classification sliders
            // (open_geomean_min, open_sustain_ms,
            // openness_score_threshold,
            // expected_clusters, cluster_feature) BEFORE the
            // registry override — they live in the hard-coded
            // STEM_SLIDER_CONFIGS because they don't fit the
            // filter-registry shape (they re-label KEPT events,
            // not filter them). The
            // registry override below replaces the entire array
            // for PGA-pipeline stems, so without this cache
            // + restore, the WebUI would silently lose all
            // classification sliders.
            const _classificationSlidersByStem = {};
            for (const stem of Object.keys(STEM_SLIDER_CONFIGS)) {
                _classificationSlidersByStem[stem] = (
                    STEM_SLIDER_CONFIGS[stem] || []
                ).filter(s => s.classification === true);
            }

            // All stems with PGA pipeline (2026-06-19): REPLACE
            // the hard-coded entry with the registry-derived one.
            // As of 2026-06-19 the only WebUI-exposed filter is
            // pga_min_prominence — min_decay_col_min_db and
            // attack_rise_max_ms were removed from the WebUI
            // because they did not perform well in practice.
            // The Python pipeline still applies all three
            // filters as a layered chain; this loop only manages
            // the slider metadata for the WebUI panel.
            for (const stem of ['toms', 'snare', 'hihat', 'kick', 'cymbals']) {
                const fromRegistry = buildSliderConfigsForStem(
                    _filterRegistryCache, stem
                );
                if (Array.isArray(fromRegistry) && fromRegistry.length > 0) {
                    // Replace the filter sliders, then re-append
                    // the cached classification sliders so the
                    // WebUI shows both groups in their original
                    // order (filters first, classification
                    // toggles below).
                    STEM_SLIDER_CONFIGS[stem] = [
                        ...fromRegistry,
                        ...(_classificationSlidersByStem[stem] || []),
                    ];
                }
            }
        } catch (err) {
            // Soft failure — fall back to the hard-coded STEM_SLIDER_CONFIGS.
            console.warn('Failed to load filter registry:', err.message);
        }
        _filterRegistryLoadInFlight = null;
        return _filterRegistryCache;
    })();
    return _filterRegistryLoadInFlight;
}

// ─── State ───────────────────────────────────────────────────────────────

/** Whether the tuning panel is open */
let tuningPanelOpen = false;

/** Whether hihat open/closed classification is enabled (per stem) */
let hihatClassificationEnabled = {};

/** Current slider values per stem (persisted across tab switches) */
let tuningSliderValues = {};

/**
 * Live midiconfig.yaml values for the active stem, fetched from
 * /api/projects/<n>/tuning-config/<stem_type> on panel open / stem
 * switch / save completion. Replaces the analysis.json `logic` block
 * as the source of slider defaults (2026-06-15). yaml is the single
 * source of truth for config; the sidecar is output only.
 *
 * Format: { stemType: { geomean_threshold, pga_min_prominence, ... } }
 */
let tuningConfig = {};

/** Animation frame ID for debounced re-filtering */
let tuningRafId = null;

/**
 * Cached deep-copy of events_configured for the active stem.
 * Created once when entering tuning mode (or switching stems).
 * applyTuningFilter modifies status in-place on these — no re-copying.
 */
let tuningBaseEvents = null;

// ─── Per-Stem Configuration ──────────────────────────────────────────────

/**
 * Slider definitions per stem type.
 * Each slider: { key, label, min, max, step, defaultValue, unit }
 *
 * The defaults here are fallbacks. Actual defaults come from the
 * analysis.json logic block at runtime.
 */
const STEM_SLIDER_CONFIGS = {
    // 2026-06-19: All five stems (toms, snare, hihat, kick, cymbals)
    // now expose a single WebUI filter slider — pga_min_prominence
    // — loaded from the filter registry at runtime. The previous
    // energy-based filters (geomean_threshold, min_sustain_ms,
    // min_strength_threshold, reverb_continuation_attack_threshold)
    // and the other two PGA filters (min_decay_col_min_db,
    // attack_rise_max_ms) were removed from the WebUI because they
    // did not perform well in practice. The Python pipeline still
    // applies all three PGA filters as a layered chain; only the
    // WebUI exposure was simplified.
    //
    // Per-stem classification controls (open_geomean_min,
    // open_sustain_ms, expected_clusters) are NOT filters — they
    // re-label KEPT events (e.g. "open hihat" vs "closed hihat",
    // "crash" vs "ride"). They're kept here as hard-coded entries
    // because they don't fit the filter-registry shape and have
    // stem-specific UI labels (🔓 emoji, 🎵 emoji).
    kick: [
        // No classification controls for kick. The pga_min_prominence
        // slider is added at runtime by _ensureFilterRegistryLoaded.
    ],
    snare: [
        { key: 'pga_min_prominence', label: 'PGA Min Prominence', min: 0, max: 10000, step: 100, fallback: 1000, unit: '', yamlPath: ['snare', 'pga_min_prominence'] }
    ],
    toms: [
        { key: 'pga_min_prominence', label: 'PGA Min Prominence', min: 0, max: 10000, step: 100, fallback: 1000, unit: '', yamlPath: ['toms', 'pga_min_prominence'] }
    ],
    hihat: [
        // 2026-06-29: openness-score classifier (REPLACES the
        // decay-slope rule that lived here through 2026-06-28).
        // ``hihat_openness_score`` is stamped on every detected
        // hihat event at detect time by
        // note_classification_core.stamp_hihat_openness_score
        // (score = 0.7 × normalized mid-band tail energy + 0.3
        // × normalized decay gradient, clamped to [0, 1]).
        // Events with score >= threshold are classified open.
        // Default threshold 0.8 was calibrated empirically on
        // the Taylor Swift (project 6) reference where
        // KMeans-ground-truth open-rate climbs from ~11% in
        // score 0.4-0.6 to ~33% at score 0.8+. Sliding the
        // threshold toward 0 lowers the bar for open (more
        // events become open); sliding toward 1 raises it.
        //
        // The legacy ``open_decay_slope_max`` slider is
        // REMOVED from the WebUI — its server-side rule is
        // retained as a fallback in classify_hihat_notes for
        // older sidecars that don't carry hihat_openness_score.
        //
        // 2026-06-19: open_geomean_min and open_sustain_ms were
        // removed from the WebUI. They are obsolete — the slope
        // rule is the only hihat open/closed classifier on
        // current sidecars. The legacy geomean+sustain rule in
        // classify_hihat_notes is a defensive fallback that
        // only fires when decay_slope_db is missing (older
        // sidecars from before 2026-06-19), so users never need
        // to tune it.
        { key: 'openness_score_threshold', label: '🔓 Open/Closed: Openness Score', min: 0, max: 1, step: 0.05, decimals: 2, fallback: 0.8, unit: 'score', classification: true }
    ],
    cymbals: [
        // 2026-06-20: expected_clusters slider removed (Phase 3
        // deleted the schema entry — cymbals clustering is now
        // auto-derived). The pga_min_prominence slider is added
        // at runtime by _ensureFilterRegistryLoaded.
    ]
};

/**
 * Filter mode per stem — determines how geomean and sustain thresholds
 * combine. Matches analysis_core.py get_spectral_config_for_stem().
 *
 * 2026-06-19: no longer used by the WebUI (the geomean/sustain
 * filter sliders were removed from hihat / kick / cymbals). Kept
 * for backward compat with any future legacy-style filter that
 * might be re-added. The PGA-only slideout does not need it.
 */
const STEM_FILTER_MODES = {
    kick: 'geomean_only',
    snare: 'geomean_only',
    toms: 'geomean_only',
    hihat: 'geomean_only',  // min_sustain applied at end (after reverb)
    cymbals: 'require_both'
};

// ─── Public API ──────────────────────────────────────────────────────────

/**
 * Fetch the live midiconfig.yaml values for a stem and cache them.
 * Called on panel open, on stem switch, and after Save & Reconvert
 * commits. The cache is the source of truth for slider defaults —
 * the analysis.json `logic` block is no longer consulted (2026-06-15).
 *
 * On fetch failure, logs a warning and leaves any cached value in
 * place (fallback). New sliders built before the fetch returns will
 * fall back to the slider config's `fallback` value.
 */
async function loadTuningConfig(stemType) {
    if (!currentProject || !stemType) return;
    // 2026-06-15: ensure the filter registry is loaded so the
    // STEM_SLIDER_CONFIGS can be derived from it. Awaited so
    // the first slider render uses the registry values, not
    // the hard-coded fallback. Soft-fails on API error —
    // _ensureFilterRegistryLoaded leaves the fallback in
    // place if the API is down.
    await _ensureFilterRegistryLoaded();
    try {
        const cfg = await api.getTuningConfig(currentProject.number, stemType);
        if (cfg && typeof cfg === 'object') {
            tuningConfig[stemType] = cfg;
        }
    } catch (err) {
        // Soft failure — don't block the panel from opening.
        // Build sliders from the static `fallback` defaults.
        console.warn(`loadTuningConfig(${stemType}) failed:`, err.message);
    }
}

/**
 * Toggle the tuning panel visibility.
 * Called from the "Tune" button in the analysis section.
 */
async function toggleTuningPanel() {
    tuningPanelOpen = !tuningPanelOpen;
    const panel = document.getElementById('tuning-panel');
    const btn = document.getElementById('tuning-toggle-btn');

    if (!panel) return;

    if (tuningPanelOpen) {
        panel.classList.remove('hidden');
        if (btn) btn.classList.add('tuning-btn-active');

        // Load the live yaml config for the active stem BEFORE
        // building sliders — slider defaults are sourced from the
        // yaml, not the sidecar. The fetch is awaited so the first
        // render shows correct values (not a flash of stale fallback
        // defaults). 2026-06-15.
        //
        // Do NOT call applyTuningFilter() here. The user wants
        // the Kept count to stay at the sidecar's value until they
        // actually drag a slider — not jump to the live-tuned count
        // the moment Tune opens. The first slider input handler
        // kicks the filter pass off
        // naturally. 2026-06-15.
        if (waveformActiveStem) {
            await loadTuningConfig(waveformActiveStem);
            buildSlidersForStem(waveformActiveStem);
            initTuningBaseEvents(waveformActiveStem);
        }
    } else {
        panel.classList.add('hidden');
        if (btn) btn.classList.remove('tuning-btn-active');

        // Cancel any pending RAF
        if (tuningRafId) cancelAnimationFrame(tuningRafId);

        // Clear cluster UI and caches
        hideClusterCards();
        tuningBaseEvents = null;

        // Clear tuning overlay — revert to configured display
        waveformTuningEvents = null;
        waveformTuningActive = false;
        drawWaveform();
    }

    // Update collapsible section height (delay to let DOM settle)
    requestAnimationFrame(() => {
        if (typeof updateCollapsibleHeights === 'function') {
            updateCollapsibleHeights();
        }
    });
}

/**
 * Called when the active stem changes (from selectStem in waveform.js).
 * Updates sliders if the tuning panel is open.
 */
async function onTuningStemChanged(stemType) {
    if (!tuningPanelOpen) return;
    // Cancel any pending RAF from the previous stem
    if (tuningRafId) cancelAnimationFrame(tuningRafId);
    hideClusterCards();
    tuningBaseEvents = null;
    // Fetch the new stem's live yaml config before rebuilding
    // sliders. Same rationale as toggleTuningPanel: avoid a flash
    // of fallback defaults between stem switches. 2026-06-15.
    //
    // Like on open, don't call applyTuningFilter() here — the
    // Kept count should hold at the sidecar's value until the
    // user moves a slider on the new stem. 2026-06-15.
    await loadTuningConfig(stemType);
    buildSlidersForStem(stemType);
    initTuningBaseEvents(stemType);
}

/**
 * Reset slider values to the configured defaults from analysis.json.
 */
function resetTuningSliders() {
    if (!waveformActiveStem || !waveformAnalysisData) return;

    // Cancel any pending RAF
    if (tuningRafId) cancelAnimationFrame(tuningRafId);

    // Clear stored values so buildSlidersForStem reads from logic block
    delete tuningSliderValues[waveformActiveStem];
    delete clusterNoteOverrides[waveformActiveStem];
    tuningBaseEvents = null;
    hideClusterCards();
    buildSlidersForStem(waveformActiveStem);
    initTuningBaseEvents(waveformActiveStem);
    applyTuningFilter();
    reapplyClientSideClassification(waveformActiveStem);
    updateTuningSaveButton();
}

// ─── Slider UI ───────────────────────────────────────────────────────────

/**
 * Build slider controls for the given stem type.
 */
function buildSlidersForStem(stemType) {
    const container = document.getElementById('tuning-sliders');
    if (!container) return;

    const sliderConfigs = STEM_SLIDER_CONFIGS[stemType];
    if (!sliderConfigs) {
        container.innerHTML = '<p class="text-xs text-gray-500">No tunable parameters for this stem.</p>';
        return;
    }

    // Get defaults from live midiconfig.yaml (2026-06-15). The
    // analysis.json `logic` block is no longer consulted — yaml is
    // the single source of truth for config. tuningConfig is loaded
    // by loadTuningConfig() on panel open / stem switch / save
    // completion. Falls back to an empty dict while the fetch is in
    // flight (slider config's `fallback` then wins).
    const stemData = waveformAnalysisData?.stems?.[stemType];
    const logic = tuningConfig[stemType] || {};

    // Compute the dataset's actual band_max_ratio max (2026-06-10).
    // The ratio slider in the sidecar exposes the full range so
    // the user can distinguish band_max_ratio 18.99 from 459.12
    // (a difference the old clamp-to-1.0 strength field masked).
    // We walk events_spectral for the active stem and pick the
    // max observed value; the slider uses that as `max` and
    // max/1000 as `step` so the user gets full resolution.
    // Falls back to the slider's static config when no events are
    // available (first-load state).
    let dataMaxBandMaxRatio = null;
    const spectralEventsForMax = stemData?.events_spectral || [];
    for (const ev of spectralEventsForMax) {
        const r = ev?.band_max_ratio;
        if (typeof r === 'number' && Number.isFinite(r) && r >= 1) {
            if (dataMaxBandMaxRatio == null || r > dataMaxBandMaxRatio) {
                dataMaxBandMaxRatio = r;
            }
        }
    }

    // 2026-06-26: pga_min_combined_score slider gets its min/max
    // from the sidecar's events_pga. The registry's static
    // defaults of -10000/10000 are fallbacks for the no-data
    // first-load state; with data, we use the actual min/max
    // of combined_score across all events so the slider's full
    // resolution is usable within the dataset's range (most
    // songs are well within ±10000 — hihat is ±8000 — so the
    // default range makes the slider unusable). Step is
    // range/2000 for ~2000 increments across the data range.
    let dataMinCombinedScore = null;
    let dataMaxCombinedScore = null;
    const pgaEventsForRange = stemData?.events_pga || [];
    for (const ev of pgaEventsForRange) {
        const cs = ev?.combined_score;
        if (typeof cs === 'number' && Number.isFinite(cs)) {
            if (dataMinCombinedScore == null || cs < dataMinCombinedScore) {
                dataMinCombinedScore = cs;
            }
            if (dataMaxCombinedScore == null || cs > dataMaxCombinedScore) {
                dataMaxCombinedScore = cs;
            }
        }
    }

    // Get stored values or use defaults
    const stored = tuningSliderValues[stemType] || {};

    container.innerHTML = sliderConfigs.map(slider => {
        // Priority: stored value > logic block value > fallback
        const logicValue = logic[slider.key];
        const defaultVal = logicValue != null ? logicValue : slider.fallback;
        const currentVal = stored[slider.key] != null ? stored[slider.key] : defaultVal;

        // Store the initial value
        if (!tuningSliderValues[stemType]) tuningSliderValues[stemType] = {};
        tuningSliderValues[stemType][slider.key] = currentVal;

        // Toggle control (e.g. snap_mask_enabled). Rendered as a
        // switch; when off, dependent rows (dependsOn=this key) are
        // hidden via CSS class + JS visibility update below. The
        // help text under the label is taken from slider.help when
        // provided; falls back to a generic description so legacy
        // toggle entries without help text still render cleanly.
        if (slider.type === 'toggle') {
            const checked = currentVal ? 'checked' : '';
            const help = slider.help || 'Toggle to enable or disable this filter';
            return `
                <div class="tuning-slider-row" data-slider-key="${slider.key}">
                    <div class="flex items-center justify-between">
                        <div class="flex-1">
                            <label class="text-xs text-gray-300">${slider.label}</label>
                            <p class="text-[10px] text-gray-500">${help}</p>
                        </div>
                        <label class="relative inline-flex items-center cursor-pointer flex-shrink-0 ml-2">
                            <input type="checkbox" id="tuning-slider-${slider.key}"
                                   class="sr-only peer"
                                   ${checked}
                                   data-key="${slider.key}"
                                   data-toggle="true">
                            <div class="w-9 h-5 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-larsnet-primary"></div>
                        </label>
                    </div>
                </div>`;
        }

        // Pre-compute disabled / class state for sliders and
        // text inputs that depend on a toggle. Both branches below
        // need these. (Hoisted to here to avoid the temporal dead
        // zone when a text input references them.)
        let hidden = '';
        let disabledAttr = '';
        let rowClass = 'tuning-slider-row';

        // Text input control (e.g. advanced_snap_ring_direction).
        // Free-form text — used for string enum-style settings
        // where the user types a value (we don't render a select
        // here to avoid re-encoding the allowed_values in the JS).
        if (slider.type === 'text') {
            const isDisabled = disabledAttr !== '';
            return `
                <div class="${rowClass}" data-slider-key="${slider.key}" data-depends-on="${slider.dependsOn || ''}"${hidden}>
                    <div class="flex items-center justify-between mb-1">
                        <label class="text-xs text-gray-300">${slider.label}</label>
                        <span class="text-xs text-gray-500 font-mono" id="tuning-val-${slider.key}">${String(currentVal)}</span>
                    </div>
                    <input type="text"
                           id="tuning-slider-${slider.key}"
                           class="w-full text-xs bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text-gray-200"${isDisabled ? ' disabled' : ''}
                           value="${String(currentVal)}"
                           data-key="${slider.key}"
                           data-text="true">
                </div>`;
        }

        // Determine visibility: hidden if this slider depends on a
        // toggle that's currently off. (The state vars were
        // declared above so the text-input branch could see them
        // without hitting a TDZ error.)
        if (slider.dependsOn) {
            const depVal = tuningSliderValues[stemType]?.[slider.dependsOn];
            // Treat null/undefined as 'off' for a fresh project that
            // has no stored toggle value (schema default for
            // snap_mask_enabled is false, so OFF is the safe default).
            // Exception: for `onset_events_enabled`, the default is
            // true, so missing-bool = ON — that means the filter
            // sliders are ACTIVE by default. We handle that here by
            // using the slider's `fallback` (which is the schema
            // default) as the assumed state when the toggle value is
            // null/undefined.
            if (depVal === undefined || depVal === null) {
                // Look up the toggle's fallback from the configs
                // (a small scan — there are at most a couple of
                // toggles per stem).
                const depToggle = sliderConfigs.find(s => s.key === slider.dependsOn);
                const depOn = depToggle?.fallback === true;
                if (!depOn) hidden = ' style="display: none"';
                else disabledAttr = '';
            } else if (depVal === false) {
                // Toggle is explicitly off. The snap-mask slider
                // should hide entirely (its slider value is
                // meaningless without the toggle on). The
                // onset-filter sliders should be VISIBLE but
                // disabled (the user can see the value, just not
                // adjust it).
                if (slider.dependsOn === 'snap_mask_enabled') {
                    hidden = ' style="display: none"';
                } else {
                    disabledAttr = ' disabled';
                    rowClass += ' opacity-50 pointer-events-none';
                }
            }
        }

        const unitLabel = slider.unit ? ` <span class="text-gray-500">${slider.unit}</span>` : '';
        const defaultLabel = logicValue != null
            ? `<span class="text-gray-600 text-xs ml-1">(configured: ${logicValue})</span>`
            : '';

        // Dynamic range for the band_max_ratio slider (2026-06-10).
        // The slider's static `max`/`step` (1000 / 0.1) are
        // fallbacks for the no-data first-load case. When we have
        // spectral events, we substitute the dataset's actual max
        // (rounded up to the next "nice" number for a clean
        // UI) and a step that gives ~1000 increments so the user
        // can express the full range without losing precision.
        // 0 remains the "Off / Disabled" sentinel — that special
        // case is handled in the filter logic, not the range.
        let sliderMin = slider.min;
        let sliderMax = slider.max;
        let sliderStep = slider.step;
        if (slider.key === 'band_max_ratio_max' && dataMaxBandMaxRatio != null) {
            sliderMax = niceCeil(dataMaxBandMaxRatio);
            // 1000 increments across the full range; minimum
            // step of 0.01 so the user can express very small
            // values when the dataset max is small.
            sliderStep = Math.max(0.01, sliderMax / 1000);
        }
        // 2026-06-26: pga_min_combined_score slider uses the
        // sidecar's combined_score min/max. This is the same
        // pattern as band_max_ratio_max above — the registry
        // defaults are -10000/10000 (step 50) which is far
        // wider than the actual data range. Using the data
        // range gives the slider's full resolution to the
        // user. Step is fixed at 1 so 0 is always on a step
        // boundary (data-derived steps like dataRange/2000
        // can produce non-integer steps where 0 falls between
        // grid points and gets snapped to a nonzero value by
        // the browser — see commit history for the 5.654 bug).
        if (slider.key === 'pga_min_combined_score'
            && dataMinCombinedScore != null
            && dataMaxCombinedScore != null) {
            sliderMin = Math.floor(dataMinCombinedScore);
            sliderMax = Math.ceil(dataMaxCombinedScore);
            // Step = 1 keeps 0 (and all integers) on the step
            // boundary from any min. The data range might be
            // tens of thousands of units (e.g. -44268 to 9494
            // for hihat = 53k increments), but integer step
            // is the most useful for a warble filter whose
            // primary decision boundary is value=0.
            sliderStep = 1;
        }

        return `
            <div class="${rowClass}" data-slider-key="${slider.key}" data-depends-on="${slider.dependsOn || ''}"${hidden}>
                <div class="flex items-center justify-between mb-1">
                    <label class="text-xs text-gray-300">${slider.label}${defaultLabel}</label>
                    <span class="text-xs text-larsnet-primary font-mono" id="tuning-val-${slider.key}">${formatSliderValue(currentVal, slider.decimals)}${unitLabel}</span>
                </div>
                <input type="range"
                       id="tuning-slider-${slider.key}"
                       class="tuning-range w-full"
                       min="${sliderMin}"
                       max="${sliderMax}"
                       step="${sliderStep}"
                       value="${currentVal}"${disabledAttr}
                       data-key="${slider.key}"
                       data-unit="${slider.unit || ''}"
                       data-classification="${slider.classification ? 'true' : 'false'}"
                       data-decimals="${slider.decimals != null ? slider.decimals : ''}">
            </div>`;
    }).join('');

    // 2026-06-20: cluster-feature dropdown removed. The Python
    // pipeline auto-derives k-means labels for snare/cymbals;
    // there's no per-user feature choice to expose. (See
    // STEM_FEATURE_CHOICES doc above for context.)

    // Add hihat open/closed classification toggle (only for hihat stem)
    if (stemType === 'hihat') {
        const enabled = hihatClassificationEnabled[stemType] !== false; // default true
        
        container.insertAdjacentHTML('beforeend', `
            <div class="tuning-slider-row mt-3 pt-2 border-t border-gray-700">
                <div class="flex items-center justify-between">
                    <div class="flex-1">
                        <label class="text-xs text-gray-300">🔔 Open/Closed Classification</label>
                        <p class="text-[10px] text-gray-500">Distinguish open vs closed hi-hat</p>
                    </div>
                    <label class="relative inline-flex items-center cursor-pointer flex-shrink-0">
                        <input type="checkbox" id="tuning-hihat-classify" class="sr-only peer" ${enabled ? 'checked' : ''}>
                        <div class="w-9 h-5 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-larsnet-primary"></div>
                    </label>
                </div>
            </div>`);

        document.getElementById('tuning-hihat-classify')?.addEventListener('change', onHihatClassificationToggle);

        // Initialize slider visibility based on toggle state
        const sliderKeys = ['openness_score_threshold'];
        const isEnabled = hihatClassificationEnabled[stemType] !== false;
        sliderKeys.forEach(key => {
            const sliderRow = document.querySelector(`[data-slider-key="${key}"]`);
            if (sliderRow) {
                sliderRow.style.display = isEnabled ? '' : 'none';
            }
        });
    }

    // Attach input listeners
    container.querySelectorAll('input[type=range]').forEach(input => {
        input.addEventListener('input', onSliderInput);
    });

    // Attach change listeners for toggle controls (e.g. snap_mask_enabled)
    container.querySelectorAll('input[type=checkbox][data-toggle="true"]').forEach(input => {
        input.addEventListener('change', onToggleInput);
    });

    // Attach input listeners for text controls (e.g. advanced_snap_ring_direction)
    container.querySelectorAll('input[type=text][data-text="true"]').forEach(input => {
        input.addEventListener('change', onTextInput);
    });

    // Update save button visibility for this stem
    updateTuningSaveButton();
}

/**
 * Format a slider value for display.
 */
function formatSliderValue(val, decimals) {
    if (val == null) return '—';
    // 2026-06-10: the band_max_ratio_max slider uses 0 as the
    // "Off / Disabled" sentinel (the filter is a no-op at 0).
    // Show the user an explicit "Off" label at that position
    // so they can confirm the filter is inactive — important
    // because the slider's visible value alone doesn't reveal
    // whether the filter is on or off.
    if (val === 0) return 'Off';
    // 2026-06-22: if the caller passes an explicit `decimals`
    // (from the slider config), honor it. The hihat
    // openness_score_threshold slider passes `decimals: 2` so
    // the UI preview shows e.g. "0.80" instead of "0.8".
    if (Number.isInteger(decimals) && decimals >= 0) {
        return val.toFixed(decimals);
    }
    if (Number.isInteger(val) || val >= 10) return Math.round(val).toString();
    if (val >= 1) return val.toFixed(1);
    return val.toFixed(2);
}

/**
 * Round a positive number UP to the next "nice" round number for
 * a UI slider max. E.g. 459.12 → 500, 18.99 → 20, 1.05 → 2.
 * Used to compute the band_max_ratio slider's max from the
 * dataset's actual max ratio (2026-06-10). 1.0 itself is the
 * floor (band_max_ratio is always >= 1 by construction — see
 * spectral_transient_core._band_max_from_powers).
 */
function niceCeil(v) {
    if (!Number.isFinite(v) || v <= 0) return 1;
    if (v <= 1) return 1;
    if (v <= 2) return 2;
    if (v <= 5) return 5;
    if (v <= 10) return 10;
    if (v <= 20) return 20;
    if (v <= 50) return 50;
    if (v <= 100) return 100;
    if (v <= 200) return 200;
    if (v <= 500) return 500;
    if (v <= 1000) return 1000;
    if (v <= 2000) return 2000;
    if (v <= 5000) return 5000;
    if (v <= 10000) return 10000;
    // For very large ratios, round to the next 10k.
    return Math.ceil(v / 10000) * 10000;
}

/**
 * Convert tuning slider values for a stem into a flat dotted-path
 * config_overrides dict suitable for /api/rebuild-midi.
 *
 * Maps the per-stem slider keys to their YAML nested paths:
 *   geomean_threshold           → <stem>.geomean_threshold
 *   min_sustain_ms              → <stem>.min_sustain_ms
 *   min_strength_threshold      → <stem>.min_strength_threshold
 *   open_geomean_min            → hihat.open_geomean_min
 *   open_sustain_ms             → hihat.open_sustain_ms
 *   expected_clusters           → <stem>.expected_clusters
 *   cluster_feature             → <stem>.cluster_feature
 *   reverb_continuation_attack_threshold (global) → filtering.reverb_continuation_attack_threshold
 *
 * Returns an empty object when no sliders have been moved (i.e. the
 * stored values are identical to the logic block defaults — the
 * rebuild would do nothing useful anyway).
 */
function _buildConfigOverrides(stemType, stored) {
    const overrides = {};
    if (!stored) return overrides;

    // Per-stem keys (each stem has its own section in the YAML).
    // Note: the 2026-06-10 replacement set removed snap_mask_* and
    // advanced_filter_* in favor of `show_only_snap_events` and
    // `band_max_ratio_max` (which read the raw band_max_ratio
    // instead of the lossy strength clamp). The keys below are
    // the only ones that propagate to the YAML midiconfig.yaml on
    // Save & Reconvert.
    for (const key of [
        'geomean_threshold', 'min_sustain_ms', 'min_strength_threshold',
        // 2026-06-19: open_geomean_min and open_sustain_ms removed
        // from the override writeback — the slope rule is the
        // only hihat open/closed classifier on current sidecars.
        // 2026-06-29: openness_score_threshold replaces the slope
        // rule on the production path. open_decay_slope_max is
        // kept here as a fallback path for older sidecars — if the
        // user has it set explicitly, it still gets written to YAML.
        'openness_score_threshold', 'open_decay_slope_max',
        'expected_clusters',
        'cluster_feature', 'onset_events_enabled',
        'show_only_snap_events', 'band_max_ratio_max',
    ]) {
        if (stored[key] != null) {
            overrides[`${stemType}.${key}`] = stored[key];
        }
    }

    // Global filtering key (lives in [filtering] not per-stem).
    if (stored.reverb_continuation_attack_threshold != null) {
        overrides['filtering.reverb_continuation_attack_threshold'] =
            stored.reverb_continuation_attack_threshold;
    }

    // yamlPath-bearing slider keys (2026-06-13): some sliders are
    // exposed on a per-stem panel but live at a non-default path in
    // the YAML — e.g. the toms pga_min_prominence slider writes to
    // `onset_detection.pga_min_prominence`, not to a per-stem key.
    // The rebuild path consumes dotted paths, so join yamlPath with
    // '.' to match the same format consumed by
    // stems_to_midi_cli._apply_cli_overrides_to_config. This keeps
    // the UI and the saved MIDI rebuild consistent when the user
    // clicks Save & Reconvert — even though step 5 will eventually
    // trigger a full PGA re-detection, this keeps the override
    // plumbing correct today.
    const sliderConfigs = STEM_SLIDER_CONFIGS[stemType] || [];
    for (const slider of sliderConfigs) {
        if (!Array.isArray(slider.yamlPath) || slider.yamlPath.length === 0) continue;
        const key = slider.key;
        if (stored[key] == null) continue;
        overrides[slider.yamlPath.join('.')] = stored[key];
    }

    return overrides;
}

/**
 * Handle slider input events — debounced via requestAnimationFrame.
 * Classification sliders (marked with data-classification="true") trigger
 * a server-side reclassify call instead of local filtering.
 */
function onSliderInput(e) {
    const key = e.target.dataset.key;
    const unit = e.target.dataset.unit || '';
    const val = parseFloat(e.target.value);
    const isClassification = e.target.dataset.classification === 'true';
    // 2026-06-22: per-slider decimal precision (e.g. hihat
    // openness_score_threshold uses 2 decimals so 0.80
    // displays as "0.80", not "0.8"). Falls back to undefined
    // when the slider config didn't set one — formatSliderValue's
    // default branches then apply.
    const decimalsAttr = e.target.dataset.decimals;
    const decimals = decimalsAttr === '' || decimalsAttr == null
        ? undefined
        : Number(decimalsAttr);

    // Update stored value
    if (waveformActiveStem) {
        if (!tuningSliderValues[waveformActiveStem]) tuningSliderValues[waveformActiveStem] = {};
        tuningSliderValues[waveformActiveStem][key] = val;
    }

    // Update numeric display
    const unitLabel = unit ? ` <span class="text-gray-500">${unit}</span>` : '';
    const display = document.getElementById(`tuning-val-${key}`);
    if (display) display.innerHTML = `${formatSliderValue(val, decimals)}${unitLabel}`;

    // Update Save button visibility
    updateTuningSaveButton();

    if (isClassification) {
        // 2026-06-22: classification sliders re-classify entirely
        // client-side — no /api/reclassify round-trip. The sidecar
        // already carries per-event data (e.g. `hihat_openness_score`
        // for hihat) needed to relabel KEPT events against the new
        // threshold. applyTuningFilter() runs first to refresh
        // waveformTuningEvents (it rebuilds it from tuningBaseEvents
        // and re-applies filter passes), then the new classifier
        // mutates hihat_state / note in place, and the legend +
        // event-count update + redraw pick up the new labels.
        if (tuningRafId) cancelAnimationFrame(tuningRafId);
        tuningRafId = requestAnimationFrame(() => {
            applyTuningFilter();
            tuningRafId = null;
            reapplyClientSideClassification(waveformActiveStem);
            if (waveformAnalysisData) {
                const stemData = waveformAnalysisData.stems?.[waveformActiveStem];
                if (stemData) updateEventCounts(stemData);
            }
            drawWaveform();
        });
    } else {
        // Filtering slider — local filter + (if the stem has
        // classification data) re-apply classification. No server
        // round-trip in either branch as of 2026-06-22.
        if (tuningRafId) cancelAnimationFrame(tuningRafId);
        tuningRafId = requestAnimationFrame(() => {
            applyTuningFilter();
            tuningRafId = null;
            reapplyClientSideClassification(waveformActiveStem);
            if (waveformAnalysisData) {
                const stemData = waveformAnalysisData.stems?.[waveformActiveStem];
                if (stemData) updateEventCounts(stemData);
            }
            drawWaveform();
        });
    }
}

/**
 * Handle text input change (e.g. advanced_snap_ring_direction).
 * Stores the string value in tuningSliderValues and refreshes the
 * Save button dirty check. Unlike onSliderInput / onToggleInput,
 * text inputs don't trigger a local filter pass — the value is
 * used in the rebuild path (after Save), not in the live
 * applyTuningFilter() call.
 */
function onTextInput(e) {
    const key = e.target.dataset.key;
    const val = String(e.target.value || '');

    if (waveformActiveStem) {
        if (!tuningSliderValues[waveformActiveStem]) tuningSliderValues[waveformActiveStem] = {};
        tuningSliderValues[waveformActiveStem][key] = val;
    }

    // Refresh the value display span.
    const display = document.getElementById(`tuning-val-${key}`);
    if (display) display.textContent = val;

    updateTuningSaveButton();
}

/**
 * Handle toggle control change (e.g. snap_mask_enabled).
 *
 * Stores the bool in tuningSliderValues, shows/hides dependent
 * slider rows (those with `data-depends-on="<this key>"`), and
 * re-runs the local filter. The threshold-tuning.js filter
 * (`applyTuningFilter` → `applySnapDeltaMask`) only applies the
 * mask when the toggle is on, so the user can disable the mask
 * mid-session to recover filtered events client-side.
 */
function onToggleInput(e) {
    const key = e.target.dataset.key;
    const enabled = e.target.checked;

    if (waveformActiveStem) {
        if (!tuningSliderValues[waveformActiveStem]) tuningSliderValues[waveformActiveStem] = {};
        tuningSliderValues[waveformActiveStem][key] = enabled;
    }

    // Show/hide dependent slider rows. Two behaviors:
    //   - snap_mask_enabled: toggle hides/shows the threshold
    //     slider row entirely (its value is meaningless without
    //     the mask enabled).
    //   - onset_events_enabled: toggle disables+grays the filter
    //     sliders but keeps them visible — the user can see what
    //     the current values are, just not adjust them. This is
    //     the "I can read the slider but it's grayed" UX.
    // The two cases are distinguished by which key the row
    // depends on (the row's data-depends-on attribute).
    document.querySelectorAll(`[data-depends-on="${key}"]`).forEach(row => {
        if (key === 'onset_events_enabled') {
            // Disable the input and gray the row.
            const input = row.querySelector('input[type=range]');
            if (input) input.disabled = !enabled;
            row.classList.toggle('opacity-50', !enabled);
            row.classList.toggle('pointer-events-none', !enabled);
        } else {
            // snap_mask_enabled or any other future toggle: hide.
            row.style.display = enabled ? '' : 'none';
        }
    });

    updateTuningSaveButton();

    // Re-run the local filter so the user sees the snap-mask
    // effect change live (mask on → red bars appear; mask off →
    // red bars recover). Uses the existing RAF debouncer so
    // rapid toggling doesn't thrash.
    if (tuningRafId) cancelAnimationFrame(tuningRafId);
    tuningRafId = requestAnimationFrame(() => {
        applyTuningFilter();
        tuningRafId = null;
        reapplyClientSideClassification(waveformActiveStem);
        if (waveformAnalysisData) {
            const stemData = waveformAnalysisData.stems?.[waveformActiveStem];
            if (stemData) updateEventCounts(stemData);
        }
        drawWaveform();
    });
}

/**
 * Handle cluster feature dropdown change — stores override and reclassifies.
 */
/**
 * 2026-06-20: onClusterFeatureChange was removed. The cluster-feature
 * dropdown (which re-ran k-means over a user-chosen per-event feature)
 * was tied to the dead `*_cluster_feature` schema entries (snare/cymbals),
 * which Phase 3 removed. The Python pipeline auto-derives k-means
 * labels for snare/cymbals now — no user choice needed.
 */

/**
 * Handle hihat open/closed classification toggle.
 */
function onHihatClassificationToggle(e) {
    const enabled = e.target.checked;
    const stemType = waveformActiveStem || 'hihat';

    hihatClassificationEnabled[stemType] = enabled;

    // Show/hide the open/closed classification sliders
    const sliderKeys = ['openness_score_threshold'];
    sliderKeys.forEach(key => {
        const sliderRow = document.querySelector(`[data-slider-key="${key}"]`);
        if (sliderRow) {
            sliderRow.style.display = enabled ? '' : 'none';
        }
    });

    // 2026-06-19: notify the waveform (and any other consumer
    // of the per-event classification color) that the toggle
    // state changed. The waveform listens for this event and
    // re-renders so the per-classification color overlay
    // appears/disappears in lockstep with the toggle.
    // The event is dispatched on the window so other modules
    // (advanced-midi.js, future export pipelines) can listen
    // without depending on the threshold-tuning module.
    window.dispatchEvent(new CustomEvent('larsnet:classification-toggle', {
        detail: { stem: stemType, enabled },
    }));

    // 2026-06-22: re-run classification client-side. The toggle
    // just controls legend visibility — but if any KEPT events
    // were relabeled by a previous slider drag, the legend
    // counts need a re-render to reflect the current
    // hihat_state. Mirrors the new RAF pipeline in onSliderInput.
    if (tuningRafId) cancelAnimationFrame(tuningRafId);
    tuningRafId = requestAnimationFrame(() => {
        applyTuningFilter();
        tuningRafId = null;
        reapplyClientSideClassification(stemType);
        if (waveformAnalysisData) {
            const stemData = waveformAnalysisData.stems?.[stemType];
            if (stemData) updateEventCounts(stemData);
        }
        drawWaveform();
    });
}

/**
 * Per-stem note assignment overrides from cluster dropdowns.
 * Format: { stemType: { classificationIndex: noteNumber } }
 */
let clusterNoteOverrides = {};

/**
 * 2026-06-20: STEM_FEATURE_CHOICES was removed in the PGA-universal
 * cleanup. The cluster-feature dropdown was a UI surface for the
 * legacy `*_cluster_feature` schema entries (snare/cymbals), which
 * Phase 3 removed from the schema. The Python pipeline falls back
 * to auto-derived labels via k-means — no user choice is needed.
 */

/**
 * Available MIDI note choices per stem for the cluster note dropdowns.
 * Hihat uses the standard General MIDI mapping; users with custom
 * drum maps can pick from the same choices the other stems offer.
 */
const STEM_NOTE_CHOICES = {
    snare: [
        { note: 38, label: 'Snare' },
        { note: 37, label: 'Rimshot' },
        { note: 39, label: 'Clap' },
    ],
    toms: [
        { note: 45, label: 'Low Tom' },
        { note: 47, label: 'Mid Tom' },
        { note: 50, label: 'High Tom' },
    ],
    cymbals: [
        { note: 49, label: 'Crash' },
        { note: 51, label: 'Ride' },
        { note: 52, label: 'Chinese' },
    ],
    hihat: [
        { note: 42, label: 'Closed' },
        { note: 46, label: 'Open' },
    ],
};

/**
 * 2026-06-22: scheduleReclassify and doReclassify were removed. The
 * server-side `/api/reclassify` round-trip is gone; classification
 * sliders re-classify in place via reapplyClientSideClassification
 * (no debounce, no network call). Save & Reconvert still calls the
 * server (rebuild-midi) so the new threshold is persisted to YAML
 * and stamped into the sidecar.
 */

/**
 * Render cluster info cards in the tuning panel.
 * Each card shows: cluster description, event count, distinguishing feature stats,
 * and a note dropdown for reassignment.
 */
function renderClusterCards(stemType, clusterInfo) {
    const container = document.getElementById('tuning-clusters');
    if (!container) return;

    const noteChoices = STEM_NOTE_CHOICES[stemType] || [];
    const noteOverrides = clusterNoteOverrides[stemType] || {};

    const html = clusterInfo.map(cluster => {
        const currentNote = noteOverrides[cluster.classification] != null
            ? noteOverrides[cluster.classification]
            : cluster.note;
        const dotColor = CLASSIFICATION_COLORS[cluster.classification] || '#9ca3af';

        // Build feature stats display
        const feat = cluster.distinguishing_feature;
        const featStats = cluster.features?.[feat];
        const featLabel = cluster.distinguishing_label || feat;
        let statsHtml = '';
        if (featStats) {
            statsHtml = `<span class="text-gray-500">${featLabel}: ${featStats.mean.toFixed(3)} (${featStats.min.toFixed(3)}–${featStats.max.toFixed(3)})</span>`;
        }

        // Build note dropdown options
        const options = noteChoices.map(nc => {
            const selected = nc.note === currentNote ? 'selected' : '';
            return `<option value="${nc.note}" ${selected}>${nc.label}</option>`;
        }).join('');

        return `
            <div class="flex items-center gap-2 py-1.5 px-2 rounded bg-gray-750 border border-gray-600" data-classification="${cluster.classification}">
                <span class="w-2.5 h-2.5 rounded-full flex-shrink-0" style="background: ${dotColor}"></span>
                <div class="flex-1 min-w-0">
                    <div class="flex items-center gap-1.5">
                        <span class="text-xs text-gray-200 font-medium">${cluster.description}</span>
                        <span class="text-xs text-gray-500">(${cluster.count})</span>
                    </div>
                    <div class="text-[10px] leading-tight mt-0.5">${statsHtml}</div>
                </div>
                <select class="cluster-note-select text-xs bg-gray-700 border border-gray-600 rounded px-1.5 py-0.5 text-gray-200 flex-shrink-0"
                        data-classification="${cluster.classification}"
                        data-stem="${stemType}">
                    ${options}
                </select>
            </div>`;
    }).join('');

    container.innerHTML = `
        <div class="text-[10px] text-gray-500 mb-1.5">Cluster assignments (${clusterInfo[0]?.distinguishing_label || 'feature'}-based)</div>
        <div class="space-y-1">${html}</div>`;
    container.classList.remove('hidden');

    // Attach change listeners to dropdowns
    container.querySelectorAll('.cluster-note-select').forEach(select => {
        select.addEventListener('change', onClusterNoteChange);
    });
}

/**
 * Hide cluster cards (e.g. when switching to 1 cluster or non-clusterable stem).
 */
function hideClusterCards() {
    const container = document.getElementById('tuning-clusters');
    if (container) {
        container.innerHTML = '';
        container.classList.add('hidden');
    }
}

/**
 * Handle cluster note dropdown change — remap events client-side immediately.
 */
function onClusterNoteChange(e) {
    const stemType = e.target.dataset.stem;
    const classification = parseInt(e.target.dataset.classification, 10);
    const newNote = parseInt(e.target.value, 10);

    // Store the override
    if (!clusterNoteOverrides[stemType]) clusterNoteOverrides[stemType] = {};
    clusterNoteOverrides[stemType][classification] = newNote;

    // Re-map displayed events immediately (no server round-trip)
    const displayEvents = waveformTuningEvents || getEventsForStem(waveformAnalysisData?.stems?.[stemType]);
    if (displayEvents) {
        for (const event of displayEvents) {
            if (event.status !== 'KEPT') continue;
            if (event.classification === classification) {
                event.note = newNote;
            }
        }
        drawWaveform();
    }

    updateTuningSaveButton();
}

// ─── Save & Reconvert ────────────────────────────────────────────────────

/**
 * Check whether current slider values differ from the configured values.
 * Shows/hides the Save & Reconvert button accordingly.
 */
function updateTuningSaveButton() {
    const btn = document.getElementById('tuning-save-btn');
    if (!btn || !waveformActiveStem) return;

    const stemType = waveformActiveStem;
    // Get defaults from live midiconfig.yaml (2026-06-15). The
    // analysis.json `logic` block is no longer consulted — yaml is
    // the single source of truth for config. tuningConfig is loaded
    // by loadTuningConfig() on panel open / stem switch / save
    // completion. Falls back to an empty dict while the fetch is in
    // flight (slider config's `fallback` then wins).
    const logic = tuningConfig[stemType] || {};
    const sliderConfigs = STEM_SLIDER_CONFIGS[stemType];
    const stored = tuningSliderValues[stemType] || {};

    if (!sliderConfigs) {
        btn.classList.add('hidden');
        return;
    }

    // Check if any value differs from the configured value
    let hasChanges = false;
    for (const slider of sliderConfigs) {
        const configuredVal = logic[slider.key] != null ? logic[slider.key] : slider.fallback;
        const currentVal = stored[slider.key];
        if (_sliderValueChanged(slider, currentVal, configuredVal)) {
            hasChanges = true;
            break;
        }
    }

    // Check if cluster note overrides have been set
    if (!hasChanges) {
        const noteOverrides = clusterNoteOverrides[stemType];
        if (noteOverrides && Object.keys(noteOverrides).length > 0) {
            hasChanges = true;
        }
    }

    // 2026-06-20: cluster-feature change-detection removed
    // (clusterFeatureOverrides no longer exists).

    btn.classList.toggle('hidden', !hasChanges);
}

/**
 * Keys that live in the global [filtering] section rather than per-stem.
 */
const GLOBAL_FILTERING_KEYS = new Set(['reverb_continuation_attack_threshold']);

/**
 * Compare a stored slider/toggle value against its configured value
 * to decide whether the Save button should light up.
 *
 * Toggles (slider.type === 'toggle'): exact boolean equality.
 * Sliders: existing tolerance (slider.step * 0.01) so floating-point
 * drift doesn't trigger spurious Save indicators.
 *
 * Returns true if the value is meaningfully different (Save is dirty),
 * false otherwise. Returns false when currentVal is null/undefined
 * (we don't treat "no value" as a change — only an explicit value
 * difference counts).
 */
function _sliderValueChanged(slider, currentVal, configuredVal) {
    if (currentVal == null) return false;
    if (slider.type === 'toggle') {
        // Booleans: treat as different only on an explicit mismatch.
        // null/undefined currentVal is filtered above.
        return currentVal !== configuredVal;
    }
    if (slider.type === 'text') {
        // Strings: exact equality. Whitespace differences would
        // also count as a change, which is the right user signal
        // for a direction enum.
        return currentVal !== configuredVal;
    }
    if (slider.step == null) return currentVal !== configuredVal;
    return Math.abs(currentVal - configuredVal) > slider.step * 0.01;
}

/**
 * Build config update paths for the current stem's tuned values.
 * Maps slider keys to their YAML paths: per-stem keys go to
 * [stemType, key], global filtering keys go to ["filtering", key],
 * and any slider with an explicit `yamlPath` field (e.g. the toms
 * `pga_min_prominence` slider, which lives at
 * `onset_detection.pga_min_prominence` in midiconfig.yaml) uses
 * that path verbatim.
 */
function buildConfigUpdates(stemType) {
    const sliderConfigs = STEM_SLIDER_CONFIGS[stemType];
    const stored = tuningSliderValues[stemType] || {};
    // Get defaults from live midiconfig.yaml (2026-06-15). The
    // analysis.json `logic` block is no longer consulted — yaml is
    // the single source of truth for config. tuningConfig is loaded
    // by loadTuningConfig() on panel open / stem switch / save
    // completion. Falls back to an empty dict while the fetch is in
    // flight (slider config's `fallback` then wins).
    const logic = tuningConfig[stemType] || {};
    const updates = [];

    if (!sliderConfigs) return updates;

    for (const slider of sliderConfigs) {
        const configuredVal = logic[slider.key] != null ? logic[slider.key] : slider.fallback;
        const currentVal = stored[slider.key];
        if (!_sliderValueChanged(slider, currentVal, configuredVal)) continue;
        // Route to correct YAML section. Order of precedence:
        //   1. slider.yamlPath (explicit override) — used when a
        //      slider key lives outside the per-stem section (e.g.
        //      pga_min_prominence lives at onset_detection.* even
        //      though it's exposed on the toms tuning panel).
        //   2. Global filtering key (lives in [filtering]).
        //   3. Default: per-stem section ([stemType, key]).
        let path;
        if (Array.isArray(slider.yamlPath) && slider.yamlPath.length > 0) {
            path = slider.yamlPath;
        } else if (GLOBAL_FILTERING_KEYS.has(slider.key)) {
            path = ['filtering', slider.key];
        } else {
            path = [stemType, slider.key];
        }
        updates.push({ path, value: currentVal });
    }

    // Include cluster note overrides if any
    const noteOverrides = clusterNoteOverrides[stemType];
    if (noteOverrides && Object.keys(noteOverrides).length > 0) {
        updates.push({
            path: [stemType, 'cluster_note_map'],
            value: noteOverrides,
        });
    }

    // 2026-06-20: cluster-feature override removed from
    // config_overrides (the dropped STEM_FEATURE_CHOICES dropdown
    // was the only producer of cluster_feature overrides).

    return updates;
}

/**
 * Save tuned thresholds to config YAML and rebuild MIDI from cached analysis.
 *
 * Uses the fast rebuild endpoint (sub-second, no audio re-detection).
 * Falls back to full pipeline if no analysis data is cached.
 */
async function saveTuningAndReconvert() {
    if (!currentProject || !waveformActiveStem) return;

    const btn = document.getElementById('tuning-save-btn');
    const stemType = waveformActiveStem;
    const updates = buildConfigUpdates(stemType);

    if (updates.length === 0) {
        showToast('No changes to save', 'info');
        return;
    }

    // Disable button during save
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-1"></i>Saving…';
    }

    try {
        // Step 1: Save config changes
        await api.updateConfig(currentProject.number, 'midiconfig', updates);
        // Re-fetch the live yaml so tuningConfig reflects the
        // committed values (the Save button reset below clears
        // tuningSliderValues, so the next panel open will read
        // straight from tuningConfig). 2026-06-15.
        await loadTuningConfig(stemType);
        showToast(`Saved ${updates.length} threshold${updates.length > 1 ? 's' : ''} for ${stemType}`, 'success');

        // Step 2: Try fast rebuild from cached analysis
        if (btn) btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-1"></i>Rebuilding…';

        try {
            const result = await api.rebuildMidi(currentProject.number, {
                honor_overrides: true,
            });

            if (result.success) {
                // Update analysis data in place — no page refresh needed
                if (result.analysis_data) {
                    waveformAnalysisData = result.analysis_data;
                }

                const totalEvents = Object.values(result.events_by_stem || {})
                    .reduce((sum, events) => sum + events.length, 0);
                showToast(
                    `Rebuilt ${totalEvents} events across ${result.stems_rebuilt.length} stems in ${result.elapsed_ms}ms`,
                    'success'
                );

                // Bug C: surface any data-integrity warnings from the loader
                // (e.g. events_configured has events not in events_sensitive).
                // Logged to console only — toasts were too noisy on every
                // rebuild and the warnings are diagnostic, not blocking.
                if (Array.isArray(result.data_integrity_warnings) &&
                    result.data_integrity_warnings.length > 0) {
                    for (const warning of result.data_integrity_warnings) {
                        console.warn('Data integrity warning:', warning);
                    }
                }

                // Re-render waveform with updated analysis data
                const stemData = waveformAnalysisData?.stems?.[stemType];
                if (stemData) {
                    updateEventCounts(stemData);
                }
                // Reset tuning state since changes are now committed
                waveformTuningEvents = null;
                waveformTuningActive = false;
                tuningBaseEvents = null;
                delete tuningSliderValues[stemType];
                delete clusterNoteOverrides[stemType];
                hideClusterCards();

                // Rebuild sliders from fresh logic block if panel is still open
                if (tuningPanelOpen) {
                    buildSlidersForStem(stemType);
                    // Re-enter tuning mode so the display stays in tuning view
                    initTuningBaseEvents(stemType);
                    applyTuningFilter();
                } else {
                    drawWaveform();
                }
                return;
            }
        } catch (rebuildErr) {
            // If rebuild returns 409 (needs full pipeline) or fails, fall back
            console.warn('Rebuild failed, falling back to full pipeline:', rebuildErr.message);
        }

        // Step 3: Fall back to full pipeline
        // Clear all tuning state before full reconvert - new analysis will be loaded when job completes
        waveformTuningEvents = null;
        waveformTuningActive = false;
        tuningBaseEvents = null;
        if (typeof hideClusterCards === 'function') hideClusterCards();

        if (btn) btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-1"></i>Reconverting…';
        const result = await api.stemsToMidi(currentProject.number);
        showToast('Full MIDI reconversion started (no cached analysis)', 'info');
        monitorJob(result.job_id, 'stems-to-midi');
        toggleTuningPanel();

    } catch (err) {
        console.error('Save & reconvert failed:', err);
        showToast(`Failed: ${err.message}`, 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-save mr-1"></i>Save &amp; Reconvert';
        }
    }
}

// ─── Client-Side Filtering ───────────────────────────────────────────────

/**
 * Create the cached base events for tuning from the stem's primary onset list.
 * Called once when entering tuning mode or switching stems.
 *
 * Uses events_configured (not events_sensitive) so the tune UI operates
 * on the exact same onset set that the pipeline produces. Sensitive events
 * have different onset start points and can contain events that don't exist
 * in the configured set, making them unreliable for previewing reconvert
 * results.
 *
 * For PGA-only stems (toms, 2026-06-15), events_configured is absent;
 * events_pga is the sole source and is used as the fallback.
 */
function initTuningBaseEvents(stemType) {
    const stemData = waveformAnalysisData?.stems?.[stemType];
    // 2026-06-30: prefer events_pga over events_configured. The
    // PGA-detected events_pga is the canonical source for any
    // stem that uses the PGA detector. For legacy sidecars that
    // ALSO carry events_configured (an energy-detected subset
    // from before the 2026-06-15 PGA-only refactor), using
    // events_configured caused the waveform to render in green
    // after a slider drag (because the events_configured set
    // has no method='percentile_gated' marker, so getEventColor
    // fell through to the markerKept green fallback). Picking
    // events_pga keeps the data source consistent with the
    // non-tuning path.
    //
    // events_configured: energy-detected onsets (legacy).
    // events_pga: PGA-only onsets (canonical, 2026-06-15 refactor).
    const configuredEvents = stemData?.events_pga || stemData?.events_configured;
    if (!configuredEvents || configuredEvents.length === 0) {
        tuningBaseEvents = null;
        return;
    }
    // Deep-copy once — these are reused across filter passes
    tuningBaseEvents = configuredEvents.map(e => ({ ...e }));
}

/**
 * Apply PGA prominence filter to tuning events.
 * Mirrors apply_pga_prominence_filter() in pga_event_builder.py.
 *
 * Re-tags events with status='FILTERED' when prominence < threshold.
 * Disabled-id path (manual WebUI toggle-off) is handled separately
 * by the override system — this function applies only the
 * threshold gate.
 *
 * @param {Array}  events      - Event dicts (mutated in place)
 * @param {number} threshold  - Minimum prominence for KEPT status
 * @param {Set}    disabledIds - Optional set of event ids to force-FILTER
 */
function applyPgaProminenceFilter(events, threshold, disabledIds) {
    // 2026-06-15: thin wrapper around the filter registry. The
    // filter logic lives in stems_to_midi/filter_registry.json
    // and is evaluated by the shared `evaluateFilter` in
    // filter_kinds.js — same JSON, same evaluator as the
    // Python side. This is the JS mirror of the Python
    // apply_pga_prominence_filter; both call the same
    // registry. Adding a new filter is a JSON entry.
    //
    // 2026-06-17 bug fix: this function now returns
    // [kept, filtered] (instead of mutating only). The caller
    // chains the result with applyPgaDecayColMinFilter so
    // the second filter only sees events that PASSED the
    // first. Without this, the second filter would
    // overwrite the first's FILTERED status with KEPT
    // (events that were filtered by prominence but pass
    // decay_col_min would incorrectly light up). Mirrors
    // the Python `_apply_pga_filter` (stems_to_midi/
    // pga_event_builder.py) which returns (kept, filtered).
    //
    // The disabled_ids check stays in the wrapper (it's a
    // WebUI-specific concept, not a registry concept). The
    // pga_filter_config.pga_min_prominence update also
    // stays here (it's a sidecar-format concern).
    const registry = _filterRegistryCache;
    const spec = registry ? findFilter(registry, 'pga_min_prominence') : null;
    const disabled = disabledIds || new Set();
    const kept = [];
    const filtered = [];
    for (const ev of events) {
        const evId = ev.id != null ? ev.id : ev.time;
        const isDisabled = disabled.has(evId);

        if (isDisabled) {
            ev.status = 'FILTERED';
            ev.filter_reason = 'manually disabled via WebUI';
            filtered.push(ev);
        } else if (spec) {
            // Registry-driven evaluation.
            const result = evaluateFilter(spec, ev, threshold);
            if (result === false) {
                ev.status = 'FILTERED';
                ev.filter_reason = buildFilterReason(spec, ev, threshold);
                filtered.push(ev);
            } else {
                ev.status = 'KEPT';
                delete ev.filter_reason;
                kept.push(ev);
            }
        } else {
            // Fallback: registry not loaded (e.g., API down).
            // Mirror the old hard-coded behavior so the
            // panel still works offline.
            const prom = ev.prominence;
            if (prom != null && prom < threshold) {
                ev.status = 'FILTERED';
                ev.filter_reason = (
                    `below pga_min_prominence (${prom.toFixed(0)} < ${threshold.toFixed(0)})`
                );
                filtered.push(ev);
            } else {
                ev.status = 'KEPT';
                delete ev.filter_reason;
                kept.push(ev);
            }
        }
        // Update pga_filter_config so the tooltip shows the live threshold
        if (ev.pga_filter_config) {
            ev.pga_filter_config.pga_min_prominence = threshold;
        }
    }
    return [kept, filtered];
}

/**
 * 2026-06-22: Apply client-side classification to the active stem's
 * tuning events. Mirrors the per-stem branches in
 * applyHihatOpennessScoreClassification (hihat) and any future
 * classification sliders — each slider is a single value-compare on
 * a per-event field that's already in the sidecar. No network call.
 *
 * Called after every slider drag (in onSliderInput's RAF) and from
 * onHihatClassificationToggle when the open/closed overlay is
 * toggled. Safe to call when no classification slider is configured
 * for the stem (no-op for kick/cymbals).
 */
function reapplyClientSideClassification(stemType) {
    if (!stemType || !waveformTuningEvents) return;
    const stored = tuningSliderValues[stemType] || {};
    if (stemType === 'hihat') {
        const scoreThreshold = stored.openness_score_threshold;
        if (scoreThreshold != null) {
            applyHihatOpennessScoreClassification(waveformTuningEvents, scoreThreshold);
        }
    }
}

/**
 * 2026-06-29: Apply hihat open/closed classification to KEPT events.
 * Mirrors classify_hihat_notes in
 * stems_to_midi/note_classification_core.py — score >= threshold → open,
 * score < threshold → closed. Operates entirely client-side: the sidecar
 * already carries `hihat_openness_score` on every hihat event (written by
 * _serialize_pga_events in stems_to_midi/midi.py via the dynamic
 * passthrough), so no server round-trip is needed. The Save & Reconvert
 * path still re-applies classification server-side and stamps the result
 * into the sidecar.
 *
 * Events without a numeric `hihat_openness_score` are left as-is
 * (defensive — older sidecars from before 2026-06-29 may lack the
 * field; the server-side classify_hihat_notes falls back to the
 * decay_slope / geomean+sustain path for those).
 *
 * Mutates the events array in place. Updates `hihat_state` and `note`
 * (GM 42 = closed, 46 = open) so legend counts and bar colors reflect
 * the new threshold without waiting for a reclassify round-trip.
 */
function applyHihatOpennessScoreClassification(events, scoreThreshold) {
    for (const ev of events) {
        if (ev.status !== 'KEPT') continue;
        const score = ev.hihat_openness_score;
        if (score == null) continue;
        ev.hihat_state = score >= scoreThreshold ? 'open' : 'closed';
        ev.note = ev.hihat_state === 'open' ? 46 : 42;
    }
}

/**
 * applyPgaDecayColMinFilter — 2026-06-15 bug fix, 2026-06-17
 * composition fix.
 *
 * Sister function to applyPgaProminenceFilter. Same registry-
 * driven pattern, but for the decay_col_min_median_db field
 * (the high-res STFT ring quality check — see
 * docs/stems_toms_prominence_and_decay_col_min.md). The
 * threshold is in dB; events whose decay_col_min_median_db
 * is below the threshold are tagged FILTERED. Default -80 dB
 * (the cut between real strikes and noise pops).
 *
 * Composition (2026-06-17): must be called on the events
 * that PASSED the prominence filter, NOT on the full
 * events list. If the full list is passed, this function
 * overwrites the prominence FILTERED status back to KEPT
 * for events that pass decay_col_min. The caller
 * (applyTuningFilter) chains the two: pass the kept
 * list from the prominence filter to this one.
 *
 * Returns [kept, filtered] (mirrors the Python
 * `_apply_pga_filter`). Adding a new filter is a JSON
 * entry.
 */
function applyPgaDecayColMinFilter(events, threshold, disabledIds) {
    const registry = _filterRegistryCache;
    const spec = registry
        ? findFilter(registry, 'min_decay_col_min_db')
        : null;
    const disabled = disabledIds || new Set();
    const kept = [];
    const filtered = [];
    for (const ev of events) {
        const evId = ev.id != null ? ev.id : ev.time;
        const isDisabled = disabled.has(evId);

        if (isDisabled) {
            ev.status = 'FILTERED';
            ev.filter_reason = 'manually disabled via WebUI';
            filtered.push(ev);
        } else if (spec) {
            // Registry-driven evaluation.
            const result = evaluateFilter(spec, ev, threshold);
            if (result === false) {
                ev.status = 'FILTERED';
                ev.filter_reason = buildFilterReason(spec, ev, threshold);
                filtered.push(ev);
            } else {
                ev.status = 'KEPT';
                delete ev.filter_reason;
                kept.push(ev);
            }
        } else {
            // Fallback: registry not loaded (e.g., API down).
            // Mirror the hard-coded behavior so the panel
            // still works offline.
            const colMin = ev.decay_col_min_median_db;
            if (colMin != null && colMin < threshold) {
                ev.status = 'FILTERED';
                ev.filter_reason = (
                    `below min_decay_col_min_db `
                    + `(${colMin.toFixed(1)}dB < ${threshold.toFixed(1)}dB)`
                );
                filtered.push(ev);
            } else {
                ev.status = 'KEPT';
                delete ev.filter_reason;
                kept.push(ev);
            }
        }
        // Update pga_filter_config so the tooltip shows the live threshold
        if (ev.pga_filter_config) {
            ev.pga_filter_config.min_decay_col_min_db = threshold;
        }
    }
    return [kept, filtered];
}

/**
 * applyAttackRiseMaxFilter — 2026-06-17.
 *
 * Third PGA pass. Filters events whose attack_rise_ms is
 * above the threshold. Catches wire-tail / step-back FPs
 * that pass prominence + decay_col_min but have an
 * unusually long 10-90% rise time on the high-res STFT
 * envelope (these FPs 'step back' to a previous attack
 * before rising to their own peak). User observation on
 * project 6 (Taylor Swift toms): all real strikes have
 * attack_rise < 20ms; FPs cluster at 100-500ms. Default
 * 20 ms is the empirical cut.
 *
 * Composition: must be called on the events that PASSED
 * both prominence and decay_col_min. Passing the full
 * events list would overwrite the prior filters'
 * FILTERED status — the same bug as decay_col_min (see
 * the 2026-06-17 fix in applyPgaDecayColMinFilter).
 *
 * Returns [kept, filtered] (mirrors the Python
 * `_apply_pga_filter`).
 */
function applyAttackRiseMaxFilter(events, threshold, disabledIds) {
    const registry = _filterRegistryCache;
    const spec = registry
        ? findFilter(registry, 'attack_rise_max_ms')
        : null;
    const disabled = disabledIds || new Set();
    const kept = [];
    const filtered = [];
    for (const ev of events) {
        const evId = ev.id != null ? ev.id : ev.time;
        const isDisabled = disabled.has(evId);

        if (isDisabled) {
            ev.status = 'FILTERED';
            ev.filter_reason = 'manually disabled via WebUI';
            filtered.push(ev);
        } else if (spec) {
            // Registry-driven evaluation (max_value: KEPT if
            // value <= threshold).
            const result = evaluateFilter(spec, ev, threshold);
            if (result === false) {
                ev.status = 'FILTERED';
                ev.filter_reason = buildFilterReason(spec, ev, threshold);
                filtered.push(ev);
            } else {
                ev.status = 'KEPT';
                delete ev.filter_reason;
                kept.push(ev);
            }
        } else {
            // Fallback: registry not loaded (e.g., API down).
            // Mirror the hard-coded behavior so the panel
            // still works offline.
            const ar = ev.attack_rise_ms;
            if (ar != null && ar > threshold) {
                ev.status = 'FILTERED';
                ev.filter_reason = (
                    `above attack_rise_max_ms `
                    + `(${ar.toFixed(1)}ms > ${threshold.toFixed(1)}ms)`
                );
                filtered.push(ev);
            } else {
                ev.status = 'KEPT';
                delete ev.filter_reason;
                kept.push(ev);
            }
        }
        // Update pga_filter_config so the tooltip shows the live threshold
        if (ev.pga_filter_config) {
            ev.pga_filter_config.attack_rise_max_ms = threshold;
        }
    }
    return [kept, filtered];
}

/**
 * Apply the current slider values to the cached tuning events and update
 * the waveform display. This replicates the server-side filter passes
 * from analysis_core.py.
 *
 * Uses tuningBaseEvents (created once) rather than deep-copying every call.
 * After filtering, re-applies cached classification so note colors persist
 * across slider drags.
 */
/**
 * applyPgaMinEnvelopeValue — added by .github/skills/add-filter
 * (pga_min_envelope_value, min_value).
 *
 * Registry-driven wrapper: reads the filter spec from
 * the loaded registry and calls the shared
 * `evaluateFilter` from filter_kinds.js. The hard-coded
 * fallback below mirrors the old behavior so the panel
 * still works when the registry API is down.
 *
 * Composition: pass the events that PASSED the
 * previous filter (NOT tuningBaseEvents). Otherwise this
 * filter overwrites the previous filter's FILTERED
 * status with KEPT (the 2026-06-17 composition bug).
 *
 * Returns [kept, filtered] (mirrors the Python
 * `_apply_pga_filter`).
 */
/**
 * applyPgaMinEnvelopeValue — added by .github/skills/add-filter
 * (pga_min_envelope_value, min_value).
 *
 * Registry-driven wrapper: reads the filter spec from
 * the loaded registry and calls the shared
 * `evaluateFilter` from filter_kinds.js. The hard-coded
 * fallback below mirrors the old behavior so the panel
 * still works when the registry API is down.
 *
 * Composition: pass the events that PASSED the
 * previous filter (NOT tuningBaseEvents). Otherwise this
 * filter overwrites the previous filter's FILTERED
 * status with KEPT (the 2026-06-17 composition bug).
 *
 * Returns [kept, filtered] (mirrors the Python
 * `_apply_pga_filter`).
 */
function applyPgaMinCombinedScore(events, threshold, disabledIds) {
    // 2026-06-26: warble filter. combined_score = prominence ×
    // delta5_stability (sign-bearing). Sister wrapper to
    // applyPgaProminenceFilter / applyPgaMinEnvelopeValue —
    // same registry-driven pattern. The slider's default value
    // 0.0 is a perfect precision separator on the hihat data
    // (528 FPs with cs ≤ 0, 225 real hits with cs > 0).
    const registry = _filterRegistryCache;
    const spec = registry
        ? findFilter(registry, 'pga_min_combined_score')
        : null;
    const disabled = disabledIds || new Set();
    const kept = [];
    const filtered = [];
    for (const ev of events) {
        const evId = ev.id != null ? ev.id : ev.time;
        const isDisabled = disabled.has(evId);

        if (isDisabled) {
            ev.status = 'FILTERED';
            ev.filter_reason = 'manually disabled via WebUI';
            filtered.push(ev);
        } else if (spec) {
            // Registry-driven evaluation.
            const result = evaluateFilter(spec, ev, threshold);
            if (result === false) {
                ev.status = 'FILTERED';
                ev.filter_reason = buildFilterReason(spec, ev, threshold);
                filtered.push(ev);
            } else {
                ev.status = 'KEPT';
                delete ev.filter_reason;
                kept.push(ev);
            }
        } else {
            // Fallback: registry not loaded.
            ev.status = 'KEPT';
            delete ev.filter_reason;
            kept.push(ev);
        }
        // Update pga_filter_config so the tooltip shows the live threshold.
        if (ev.pga_filter_config) {
            ev.pga_filter_config.pga_min_combined_score = threshold;
        }
    }
    return [kept, filtered];
}


function applyPgaMinEnvelopeValue(events, threshold, disabledIds) {
    const registry = _filterRegistryCache;
    const spec = registry
        ? findFilter(registry, 'pga_min_envelope_value')
        : null;
    const disabled = disabledIds || new Set();
    const kept = [];
    const filtered = [];
    for (const ev of events) {
        const evId = ev.id != null ? ev.id : ev.time;
        const isDisabled = disabled.has(evId);

        if (isDisabled) {
            ev.status = 'FILTERED';
            ev.filter_reason = 'manually disabled via WebUI';
            filtered.push(ev);
        } else if (spec) {
            // Registry-driven evaluation.
            const result = evaluateFilter(spec, ev, threshold);
            if (result === false) {
                ev.status = 'FILTERED';
                ev.filter_reason = buildFilterReason(spec, ev, threshold);
                filtered.push(ev);
            } else {
                ev.status = 'KEPT';
                delete ev.filter_reason;
                kept.push(ev);
            }
        } else {
            // Fallback: registry not loaded.
            const value = ev.envelope_value;
            // The exact predicate depends on the kind.
            // Update the predicate here for the kind.
            ev.status = 'KEPT';
            delete ev.filter_reason;
            kept.push(ev);
        }
        // Update pga_filter_config so the tooltip shows the live threshold.
        if (ev.pga_filter_config) {
            ev.pga_filter_config.pga_min_envelope_value = threshold;
        }
    }
    return [kept, filtered];
}


function applyTuningFilter() {
    if (!waveformActiveStem || !waveformAnalysisData) return;

    const stemType = waveformActiveStem;
    const stemData = waveformAnalysisData.stems[stemType];
    if (!stemData) return;

    if (!tuningBaseEvents || tuningBaseEvents.length === 0) {
        waveformTuningEvents = null;
        waveformTuningActive = false;
        updateEventCounts(stemData);
        drawWaveform();
        return;
    }

    const params = tuningSliderValues[stemType] || {};
    const filterMode = STEM_FILTER_MODES[stemType] || 'geomean_only';

    // PGA-only stems (2026-06-19): every stem in the WebUI
    // slideout is now PGA-only (toms, snare, hihat, kick,
    // cymbals). The filter registry exposes only
    // pga_min_prominence for the WebUI; the energy-derived
    // filters (Pass 1 / 2) are no longer in the WebUI panel.
    // The Python pipeline still applies the energy-derived
    // filters for stems that aren't on the PGA path yet
    // (defensive skip here mirrors the 2026-06-15 toms
    // branch's rationale: applying Pass 1 to PGA-only events
    // would reset every event to KEPT and wipe out the PGA
    // filter's FILTERED decisions).
    const isPgaOnlyStem = (
        stemType === 'toms' || stemType === 'snare' ||
        stemType === 'hihat' || stemType === 'kick' ||
        stemType === 'cymbals'
    );

    // Master onset-filter gate (2026-06-10). When OFF (explicit
    // Onset events visibility gate (2026-06-10 round 2). When
    // the user has turned the toms onset_events toggle OFF, the
    // energy-detected events are DROPPED from the tuning view
    // entirely. Spectral events are unaffected. The
    // geomean/sustain/strength filter and the reverb continuation
    // filter still run on the remaining (spectral + remaining
    // energy) events before the drop, so the user sees a
    // consistent picture: filter rules applied → drop energy →
    // display. The snap-mask pass is independent and still runs.
    //
    // The drop here is local to the tuning view (the displayed
    // waveform). The sidecar-level drop (which removes them from
    // events_configured on Save) happens on the server in
    // rebuild_events_from_analysis. So the user can preview the
    // spectral-only view by toggling here without committing the
    // drop to the saved MIDI.
    const onsetEventsEnabled = params.onset_events_enabled !== false;

    // Pass 0: PGA prominence filter for PGA-only stems
    // (toms 2026-06-15; snare 2026-06-18). Runs before any
    // spectral-style filter. PGA events (method='percentile_gated')
    // are skipped by applySpectralFilter so this is the only filter
    // that touches them here.
    //
    // 2026-06-17 bug fix: the two PGA filters are now
    // CHAINED, not independent. The decay_col_min filter
    // runs on the KEPT events from the prominence filter,
    // not on the full events list. Without this, an event
    // that was FILTERED by prominence but passes
    // decay_col_min would incorrectly get its status
    // overwritten to KEPT — the user reported that the
    // live preview lit up events that should have stayed
    // faded. Mirrors the Python rebuild_core._refilter_stem_pga
    // layering exactly.
    if (isPgaOnlyStem) {
        let pgaKept = tuningBaseEvents;
        let pgaFiltered = [];
        // Pass 0.4: pga_min_envelope_value filter
        // (2026-06-22, sister to pga_min_prominence). Drops
        // events whose linear envelope_value is below the
        // threshold. Sister to prominence: envelope_value
        // measures the absolute height of the peak in the
        // broadband contrast envelope, prominence measures
        // the peak's vertical distance to the local
        // contour. Pass BEFORE prominence in the chain so
        // low-energy noise events are dropped first, then
        // prominence culls the relative-low ones. The WebUI
        // panel order also shows envelope_value above
        // prominence (new filter first), matching this
        // chain order.
        const envelopeValueThreshold = params.pga_min_envelope_value;
        if (envelopeValueThreshold != null) {
            const [kept0, filtered0] = applyPgaMinEnvelopeValue(
                tuningBaseEvents, envelopeValueThreshold
            );
            pgaKept = kept0;
            pgaFiltered = filtered0;
        }
        const pgaThreshold = params.pga_min_prominence;
        if (pgaThreshold != null) {
            const [kept1, filtered1] = applyPgaProminenceFilter(
                pgaKept, pgaThreshold
            );
            pgaKept = kept1;
            pgaFiltered = pgaFiltered.concat(filtered1);
        }
        // Pass 0.5: decay_col_min filter on the KEPT
        // events from Pass 0. Layered on top of the
        // prominence filter, not independent. Matches the
        // Python rebuild_core._refilter_stem_pga ordering
        // exactly.
        const decayColMinThreshold = params.min_decay_col_min_db;
        if (decayColMinThreshold != null) {
            const [kept2, filtered2] = applyPgaDecayColMinFilter(
                pgaKept, decayColMinThreshold
            );
            pgaKept = kept2;
            pgaFiltered = pgaFiltered.concat(filtered2);
        }
        // Pass 0.7: attack_rise filter (2026-06-17). Third
        // PGA pass. Catches wire-tail / step-back FPs that
        // pass prominence + decay_col_min but have an
        // unusually long 10-90% rise time. Layered on top
        // of the previous filters; events passing all three
        // are KEPT.
        const attackRiseThreshold = params.attack_rise_max_ms;
        if (attackRiseThreshold != null) {
            const [kept3, filtered3] = applyAttackRiseMaxFilter(
                pgaKept, attackRiseThreshold
            );
            pgaKept = kept3;
            pgaFiltered = pgaFiltered.concat(filtered3);
        }
        // 2026-06-26: warble filter (last in the PGA chain).
        // combined_score is sign-bearing: positive = real sustained
        // strike, negative = warble spike from stem-splitter demuxing.
        // Applied last so its filter_reason is the one that ends
        // up in the tooltip when the event is actually filtered
        // by this rule. Mirrors the Python pipeline order in
        // pga_event_builder._build_pga_events_with_filter and in
        // rebuild_core._refilter_stem_pga.
        const combinedScoreThreshold = params.pga_min_combined_score;
        if (combinedScoreThreshold != null) {
            const [kept4, filtered4] = applyPgaMinCombinedScore(
                pgaKept, combinedScoreThreshold
            );
            pgaKept = kept4;
            pgaFiltered = pgaFiltered.concat(filtered4);
        }
    }

    // Run the energy-derived filters (Pass 1 and Pass 2) so
    // their statuses are consistent with the saved sidecar.
    // For PGA-only stems (toms + snare as of 2026-06-18),
    // events are all method='percentile_gated' and have no
    // geomean / sustain / strength / attack_sharpness fields —
    // applySpectralFilter would reset them all to KEPT and the
    // geomean/sustain/strength checks would silently no-op on
    // null values, wiping out the PGA filter's KEPT/FILTERED
    // decisions above. Skip both passes for PGA-only stems.
    // 2026-06-15 (toms), 2026-06-18 (snare).
    if (!isPgaOnlyStem) {
        // Pass 1: Spectral filter (geomean + sustain + strength)
        applySpectralFilter(tuningBaseEvents, params, filterMode);

        // Pass 2: Reverb continuation filter
        const attackThreshold = params.reverb_continuation_attack_threshold;
        if (attackThreshold != null) {
            applyReverbContinuationFilter(tuningBaseEvents, attackThreshold);
        }
    }

    // If the onset events gate is off, drop energy events from
    // the tuning view (in-place so subsequent passes operate on
    // the spectral-only set). This is purely a display drop —
    // the sidecar is unchanged until the user Saves with the
    // toggle off. After this filter, downstream passes (snap
    // mask, advanced filter) only see spectral events.
    if (!onsetEventsEnabled) {
        // Mutate in place: replace contents of tuningBaseEvents
        // with the spectral-only subset. We use a manual
        // splice because Array.prototype.filter creates a new
        // array and the rest of the function expects the same
        // reference.
        let writeIdx = 0;
        for (let readIdx = 0; readIdx < tuningBaseEvents.length; readIdx++) {
            if (tuningBaseEvents[readIdx].method === 'spectral') {
                if (writeIdx !== readIdx) {
                    tuningBaseEvents[writeIdx] = tuningBaseEvents[readIdx];
                }
                writeIdx++;
            }
        }
        tuningBaseEvents.length = writeIdx;
    }

    // Pass 3: Final min_sustain filter for hihat (after reverb filtering)
    // This catches events with very short sustain that got through earlier filters
    const minSustainMs = params.min_sustain_ms;
    if (minSustainMs != null && stemType === 'hihat') {
        applyMinSustainFilter(tuningBaseEvents, minSustainMs);
    }

    // Pass 4: "Show Only Snap Events" toggle (2026-06-10, opt-in).
    // Replaces the 2026-06-09 snap-mask chain. When on, drop any
    // spectral event whose snap_delta is zero or null — the
    // classic wire-tail / decay signature. Idempotent: turning
    // it off restores any previously-filtered snap-zero events
    // (the old snap-mask was effectively a one-way ratchet).
    if (params.show_only_snap_events === true && stemType === 'toms') {
        applyShowOnlySnapEvents(tuningBaseEvents);
    }

    // Pass 5: "Filter Events with Top/2nd Ratio Greater Than"
    // ceiling (2026-06-10, opt-in). Replaces the lossy
    // `advanced_filter_high_strength` Stage 3 (which operated on
    // the clamp-to-1.0 strength field). When the slider is > 0,
    // drop any spectral event whose band_max_ratio exceeds the
    // threshold. When 0 (the default), the filter is a no-op —
    // the slider in the sidecar shows "Off" at this position
    // so the user can confirm it's inactive.
    //
    // 2026-06-18: still toms-only. Snare PGA events don't
    // carry band_max_ratio (it's a spectral-detector field),
    // so the pass would no-op on snare — keeping the gate to
    // toms avoids exposing a slider that can't affect anything.
    if (stemType === 'toms') {
        const ratioMax = params.band_max_ratio_max;
        if (ratioMax != null && ratioMax > 0) {
            applyBandMaxRatioMax(tuningBaseEvents, ratioMax);
        }
    }

    // Re-apply classification after the filter chain. As of
    // 2026-06-22 this is client-side only — no /api/reclassify
    // round-trip. Re-stamps hihat_state / note from the live slider
    // values onto every KEPT event. The pass is a no-op for stems
    // that have no classification slider (kick, cymbals, etc.).
    reapplyClientSideClassification(stemType);

    // Set tuning state for waveform.js
    waveformTuningEvents = tuningBaseEvents;
    waveformTuningActive = true;

    updateEventCounts(stemData);
    drawWaveform();
}

/**
 * Pass 1: Spectral filter — replicates filter_onsets_by_spectral() logic.
 *
 * Note (2026-06-09, bug fix): spectral events (method='spectral') are
 * EXEMPT from the geomean / sustain / strength filters. Those signals
 * are properties of the energy-detector output (filter_onsets_by_spectral
 * computes geomean, sustain_ms, strength from the per-band energies).
 * Spectral-transient events have none of those fields — they carry
 * band_powers / band_max_ratio / band_delta / snap_delta instead. The
 * previous implementation filtered them out whenever geomean was null
 * (which it always is for spectral events), which silently destroyed
 * all magenta events when the user dragged the geomean slider.
 */
function applySpectralFilter(events, params, filterMode) {
    const geomeanThreshold = params.geomean_threshold;
    const minSustainMs = params.min_sustain_ms;
    const minStrength = params.min_strength_threshold;

    for (const event of events) {
        // Start as KEPT (re-evaluate from scratch each pass)
        event.status = 'KEPT';

        // Spectral events are not subject to the energy-derived
        // filters (geomean / sustain / strength). They have their
        // own quality signal (band_max_ratio) which the server-side
        // band-ratio quality floor already enforces. Keep them
        // unconditionally here; the snap_mask pass handles the
        // low-snap-delta FPs.
        if (event.method === 'spectral') continue;

        // Strength gate (applies first, all modes)
        if (minStrength != null && event.strength != null) {
            if (event.strength < minStrength) {
                event.status = 'FILTERED';
                continue;
            }
        }

        // If no geomean/sustain thresholds, keep everything
        if (geomeanThreshold == null && minSustainMs == null) continue;

        if (filterMode === 'require_both') {
            // Cymbals: must pass BOTH thresholds (if both are set)
            if (geomeanThreshold != null && minSustainMs != null) {
                const passGeomean = event.geomean != null && event.geomean > geomeanThreshold;
                const passSustain = event.sustain_ms != null && event.sustain_ms >= minSustainMs;
                if (!passGeomean || !passSustain) {
                    event.status = 'FILTERED';
                }
            } else if (minSustainMs != null) {
                if (event.sustain_ms == null || event.sustain_ms < minSustainMs) {
                    event.status = 'FILTERED';
                }
            } else if (geomeanThreshold != null) {
                if (event.geomean == null || event.geomean <= geomeanThreshold) {
                    event.status = 'FILTERED';
                }
            }
        } else {
            // geomean_only: only geomean matters
            if (geomeanThreshold != null) {
                if (event.geomean == null || event.geomean <= geomeanThreshold) {
                    event.status = 'FILTERED';
                }
            }
        }
    }
}

/**
 * Pass 2: Reverb continuation filter — replicates mark_reverb_continuations().
 *
 * Processes KEPT events in time order. An event is marked REVERB_CONTINUATION
 * when all three conditions are met:
 *   1. Adjacent: gap between prev end and curr start ≤ 5ms
 *   2. Amplitude continuous: |curr.amplitude_at_start - prev.amplitude_at_end| ≤ 0.001
 *   3. Smooth envelope: curr.attack_sharpness < attackThreshold
 */
function applyReverbContinuationFilter(events, attackThreshold) {
    const TIME_MARGIN_SEC = 0.005; // 5ms
    const AMP_MARGIN = 0.001;

    // Sort by time (stable)
    events.sort((a, b) => (a.time || 0) - (b.time || 0));

    for (let i = 1; i < events.length; i++) {
        const curr = events[i];
        const prev = events[i - 1];

        // Skip if current not KEPT, or prev not KEPT/REVERB_CONTINUATION
        if (curr.status !== 'KEPT') continue;
        if (prev.status !== 'KEPT' && prev.status !== 'REVERB_CONTINUATION') continue;

        // Need required fields
        if (prev.duration_sec == null || prev.amplitude_at_end == null ||
            curr.amplitude_at_start == null) continue;

        // Timing: current starts right when previous ends
        const prevEndTime = prev.time + prev.duration_sec;
        const gap = curr.time - prevEndTime;
        const isAdjacent = Math.abs(gap) <= TIME_MARGIN_SEC;

        // Amplitude continuity
        const ampDiff = Math.abs(curr.amplitude_at_start - prev.amplitude_at_end);
        const isAmplitudeContinuous = ampDiff <= AMP_MARGIN;

        // Attack sharpness
        const isSmooth = curr.attack_sharpness != null &&
                         curr.attack_sharpness < attackThreshold;

        if (isAdjacent && isAmplitudeContinuous && isSmooth) {
            curr.status = 'REVERB_CONTINUATION';
        }
    }
}

/**
 * Pass 3: Temporal filter for hihat - applied after reverb filtering.
 * This catches events that are too close to the previous kept event (likely bleed/double-triggers).
 * Uses minSustainMs as the minimum time gap in milliseconds.
 */
function applyMinSustainFilter(events, minSustainMs) {
    if (!minSustainMs || minSustainMs <= 0) return events;
    
    // Sort events by time
    events.sort((a, b) => a.time - b.time);
    
    // Temporal filter: keep event only if gap from previous kept event >= minSustainMs
    const keptTimes = [];
    for (const event of events) {
        if (event.status !== 'KEPT') continue;
        
        // Check gap from last kept event
        let canKeep = true;
        if (keptTimes.length > 0) {
            const gapMs = (event.time - keptTimes[keptTimes.length - 1]) * 1000;
            if (gapMs < minSustainMs) {
                canKeep = false;
            }
        }
        
        if (canKeep) {
            keptTimes.push(event.time);
        } else {
            event.status = 'FILTERED';
        }
    }

    return events;
}

/**
 * Pass 4: "Show Only Snap Events" filter (2026-06-10).
 *
 * Mark any KEPT spectral event as FILTERED if its snap_delta is
 * zero or null. snap_delta is the per-frame minimum per-bin-mean
 * linear power across the snap_bands (see
 * spectral_transient_core.py). At the event's peak frame, this is
 * the diagnostic value the WebUI tooltip shows as
 * "Snap Δ (min of snap_bands)".
 *
 * Interpretation:
 *   - snap_delta == 0 : the RING signal fired but the SNAP signal
 *                       did not. Typical of wire-tail / decay
 *                       events where the broadband attack
 *                       component had already decayed.
 *   - snap_delta > 0  : both RING and SNAP signals fired with a
 *                       real broadband percussive attack in the
 *                       snap bands. The events the user wants to
 *                       keep.
 *
 * Energy events (no snap_delta) and overridden events are
 * untouched. Idempotent across rebuilds — turning the toggle off
 * restores any previously-filtered snap-zero events, unlike the
 * old snap-mask which was effectively a one-way ratchet.
 *
 * Mirrors the server-side pass in
 * ``rebuild_core._apply_show_only_snap_events`` so the tuning
 * panel matches the saved MIDI.
 */
function applyShowOnlySnapEvents(events) {
    for (const event of events) {
        if (event._overridden) continue;
        if (event.method !== 'spectral') continue;
        if (event.status !== 'KEPT') continue;
        const sd = event.snap_delta;
        if (sd == null || sd <= 0) {
            event.status = 'FILTERED';
        }
    }
}

/**
 * Pass 5: "Filter Events with Top/2nd Ratio Greater Than"
 * ceiling (2026-06-10).
 *
 * Mark any KEPT spectral event as FILTERED if its band_max_ratio
 * is strictly greater than the threshold. band_max_ratio is the
 * ratio of the loudest of the 5 frequency bands to the
 * second-loudest band at the event's peak frame (top / 1e-20 if
 * all bands are zero). The user identified this as the
 * "extreme dominance" FP signature — a real tom hit is typically
 * <20×; their calibration case had FPs at 459×.
 *
 * Replaces the lossy Stage 3 of the old advanced filter, which
 * operated on the clamp-to-1.0 `strength` field and therefore
 * could not distinguish a band_max_ratio of 11 from 459. This
 * filter reads the RAW band_max_ratio directly.
 *
 * The threshold is 0 (the default) when the slider is at the
 * "Off" position. The UI labels this as "Off / Disabled" so the
 * user always knows the filter is inactive.
 *
 * Energy events (no band_max_ratio) and overridden events are
 * untouched. Mirrors the server-side pass in
 * ``rebuild_core._apply_band_max_ratio_max``.
 */
function applyBandMaxRatioMax(events, threshold) {
    for (const event of events) {
        if (event._overridden) continue;
        if (event.method !== 'spectral') continue;
        if (event.status !== 'KEPT') continue;
        const ratio = event.band_max_ratio;
        if (ratio == null) continue;
        if (ratio > threshold) {
            event.status = 'FILTERED';
        }
    }
}

// ─── UI Updates ──────────────────────────────────────────────────────────

/**
 * Update the event count display in the tuning panel.
 */
function updateEventCounts(stemData) {
    const countEl = document.getElementById('tuning-event-counts');
    if (!countEl) return;

    const sensitiveTotal = (stemData.events_sensitive || stemData.events_pga || []).length;
    const configuredKept = getEventsForStem(stemData).filter(e => e.status === 'KEPT').length;

    if (waveformTuningEvents) {
        const tuningKept = waveformTuningEvents.filter(e => e.status === 'KEPT').length;
        const tuningFiltered = waveformTuningEvents.filter(e => e.status === 'FILTERED').length;
        const tuningReverb = waveformTuningEvents.filter(e => e.status === 'REVERB_CONTINUATION').length;

        const diff = tuningKept - configuredKept;
        const diffStr = diff > 0 ? `+${diff}` : `${diff}`;
        const diffColor = diff > 0 ? 'text-yellow-400' : diff < 0 ? 'text-red-400' : 'text-gray-400';

        countEl.innerHTML = `
            <span class="text-green-400">${tuningKept} kept</span>
            <span class="text-gray-600">·</span>
            <span class="text-red-400">${tuningFiltered} filtered</span>
            ${tuningReverb > 0 ? `<span class="text-gray-600">·</span><span class="text-orange-400">${tuningReverb} reverb</span>` : ''}
            <span class="text-gray-600">·</span>
            <span class="${diffColor}">${diffStr} vs configured</span>
            <span class="text-gray-600">·</span>
            <span class="text-gray-500">${sensitiveTotal} total sensitive</span>`;
    } else {
        countEl.innerHTML = `
            <span class="text-green-400">${configuredKept} kept (configured)</span>
            <span class="text-gray-600">·</span>
            <span class="text-gray-500">${sensitiveTotal} total sensitive</span>`;
    }
}
