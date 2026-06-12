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

// ─── State ───────────────────────────────────────────────────────────────

/** Whether the tuning panel is open */
let tuningPanelOpen = false;

/** Whether hihat open/closed classification is enabled (per stem) */
let hihatClassificationEnabled = {};

/** Current slider values per stem (persisted across tab switches) */
let tuningSliderValues = {};

/** Animation frame ID for debounced re-filtering */
let tuningRafId = null;

/** Timeout ID for debounced reclassify API calls */
let reclassifyTimeoutId = null;

/** Whether a reclassify request is in flight */
let reclassifyInFlight = false;

/**
 * Cached deep-copy of events_configured for the active stem.
 * Created once when entering tuning mode (or switching stems).
 * applyTuningFilter modifies status in-place on these — no re-copying.
 */
let tuningBaseEvents = null;

/**
 * Cached classification results from the last reclassify call, keyed by time.
 * Format: { timeKey: { classification, note, hihat_state } }
 * Re-applied after each filter pass to avoid losing note colors.
 */
let lastClassification = null;

// ─── Per-Stem Configuration ──────────────────────────────────────────────

/**
 * Slider definitions per stem type.
 * Each slider: { key, label, min, max, step, defaultValue, unit }
 *
 * The defaults here are fallbacks. Actual defaults come from the
 * analysis.json logic block at runtime.
 */
const STEM_SLIDER_CONFIGS = {
    kick: [
        { key: 'geomean_threshold', label: 'Geomean Threshold', min: 0, max: 3000, step: 10, fallback: 800, unit: '' },
        { key: 'reverb_continuation_attack_threshold', label: 'Reverb Attack Threshold', min: 0, max: 1.0, step: 0.01, fallback: 0.4, unit: '' }
    ],
    snare: [
        { key: 'geomean_threshold', label: 'Geomean Threshold', min: 0, max: 500, step: 1, fallback: 40, unit: '' },
        { key: 'reverb_continuation_attack_threshold', label: 'Reverb Attack Threshold', min: 0, max: 1.0, step: 0.01, fallback: 0.4, unit: '' },
        { key: 'expected_clusters', label: '🥁 Sound Types', min: 1, max: 3, step: 1, fallback: 2, unit: '', classification: true }
    ],
    // Toms (2026-06-12): the toms pipeline is PGA-only (see
    // processing_shell._build_events_configured and
    // percentile_gated_detector.py). The previous energy + spectral
    // filters (geomean / reverb attack / sound types / snap mask /
    // band-max-ratio ceiling / onset-events gate) operated on events
    // that are no longer in events_configured for toms — they were
    // removed because toms detection now uses the percentile-gated
    // broad-attack (PGA) detector exclusively. The toms Threshold
    // Tuning slideout is therefore empty until a PGA-specific slider
    // (e.g. pga_min_prominence) is wired in. Keeping the array
    // non-empty here so the build path still produces the
    // "No tunable parameters for this stem." fallback cleanly.
    toms: [],
    hihat: [
        { key: 'geomean_threshold', label: 'Geomean Threshold', min: 0, max: 200, step: 0.5, fallback: 8, unit: '' },
        { key: 'min_sustain_ms', label: 'Min Sustain', min: 0, max: 500, step: 5, fallback: 25, unit: 'ms' },
        { key: 'min_strength_threshold', label: 'Min Strength', min: 0, max: 1.0, step: 0.01, fallback: 0.02, unit: '' },
        { key: 'reverb_continuation_attack_threshold', label: 'Reverb Attack Threshold', min: 0, max: 1.0, step: 0.01, fallback: 0.4, unit: '' },
        { key: 'open_geomean_min', label: '🔓 Open/Closed: GeoMean', min: 50, max: 1000, step: 10, fallback: 262, unit: '', classification: true },
        { key: 'open_sustain_ms', label: '🔓 Open/Closed: Sustain', min: 20, max: 500, step: 5, fallback: 150, unit: 'ms', classification: true }
    ],
    cymbals: [
        { key: 'geomean_threshold', label: 'Geomean Threshold', min: 0, max: 1000, step: 5, fallback: 100, unit: '' },
        { key: 'min_sustain_ms', label: 'Min Sustain', min: 0, max: 500, step: 5, fallback: 150, unit: 'ms' },
        { key: 'min_strength_threshold', label: 'Min Strength', min: 0, max: 1.0, step: 0.01, fallback: 0.1, unit: '' },
        { key: 'reverb_continuation_attack_threshold', label: 'Reverb Attack Threshold', min: 0, max: 1.0, step: 0.01, fallback: 0.4, unit: '' },
        { key: 'expected_clusters', label: '🎵 Sound Types', min: 1, max: 4, step: 1, fallback: 2, unit: '', classification: true }
    ]
};

/**
 * Filter mode per stem — determines how geomean and sustain thresholds
 * combine. Matches analysis_core.py get_spectral_config_for_stem().
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
 * Toggle the tuning panel visibility.
 * Called from the "Tune" button in the analysis section.
 */
function toggleTuningPanel() {
    tuningPanelOpen = !tuningPanelOpen;
    const panel = document.getElementById('tuning-panel');
    const btn = document.getElementById('tuning-toggle-btn');

    if (!panel) return;

    if (tuningPanelOpen) {
        panel.classList.remove('hidden');
        if (btn) btn.classList.add('tuning-btn-active');

        // Initialize sliders for the active stem
        if (waveformActiveStem) {
            buildSlidersForStem(waveformActiveStem);
            initTuningBaseEvents(waveformActiveStem);
            applyTuningFilter();
            // Immediately reclassify so events get colored on open
            scheduleReclassify();
        }
    } else {
        panel.classList.add('hidden');
        if (btn) btn.classList.remove('tuning-btn-active');

        // Cancel any pending reclassify
        if (reclassifyTimeoutId) { clearTimeout(reclassifyTimeoutId); reclassifyTimeoutId = null; }

        // Clear cluster UI and caches
        hideClusterCards();
        tuningBaseEvents = null;
        lastClassification = null;

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
function onTuningStemChanged(stemType) {
    if (!tuningPanelOpen) return;
    // Cancel any pending reclassify from the previous stem
    if (reclassifyTimeoutId) { clearTimeout(reclassifyTimeoutId); reclassifyTimeoutId = null; }
    hideClusterCards();
    tuningBaseEvents = null;
    lastClassification = null;
    buildSlidersForStem(stemType);
    initTuningBaseEvents(stemType);
    applyTuningFilter();
    scheduleReclassify();
}

/**
 * Reset slider values to the configured defaults from analysis.json.
 */
function resetTuningSliders() {
    if (!waveformActiveStem || !waveformAnalysisData) return;

    // Cancel any pending reclassify
    if (reclassifyTimeoutId) { clearTimeout(reclassifyTimeoutId); reclassifyTimeoutId = null; }

    // Clear stored values so buildSlidersForStem reads from logic block
    delete tuningSliderValues[waveformActiveStem];
    delete clusterNoteOverrides[waveformActiveStem];
    delete clusterFeatureOverrides[waveformActiveStem];
    lastClassification = null;
    tuningBaseEvents = null;
    hideClusterCards();
    buildSlidersForStem(waveformActiveStem);
    initTuningBaseEvents(waveformActiveStem);
    applyTuningFilter();
    scheduleReclassify();
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

    // Get defaults from analysis.json logic block
    const stemData = waveformAnalysisData?.stems?.[stemType];
    const logic = stemData?.logic || {};

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

        return `
            <div class="${rowClass}" data-slider-key="${slider.key}" data-depends-on="${slider.dependsOn || ''}"${hidden}>
                <div class="flex items-center justify-between mb-1">
                    <label class="text-xs text-gray-300">${slider.label}${defaultLabel}</label>
                    <span class="text-xs text-larsnet-primary font-mono" id="tuning-val-${slider.key}">${formatSliderValue(currentVal)}${unitLabel}</span>
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
                       data-classification="${slider.classification ? 'true' : 'false'}">
            </div>`;
    }).join('');

    // Add cluster feature dropdown if this stem supports it
    const featureChoices = STEM_FEATURE_CHOICES[stemType];
    if (featureChoices) {
        const configuredFeature = logic.cluster_feature || 'auto';
        const currentFeature = clusterFeatureOverrides[stemType] || configuredFeature;

        const options = featureChoices.map(fc => {
            const selected = fc.value === currentFeature ? 'selected' : '';
            return `<option value="${fc.value}" ${selected}>${fc.label}</option>`;
        }).join('');

        const configuredLabel = configuredFeature !== 'auto'
            ? `<span class="text-gray-600 text-xs ml-1">(configured: ${configuredFeature})</span>`
            : '';

        container.insertAdjacentHTML('beforeend', `
            <div class="tuning-slider-row">
                <div class="flex items-center justify-between mb-1">
                    <label class="text-xs text-gray-300">🔬 Cluster By${configuredLabel}</label>
                </div>
                <select id="tuning-cluster-feature"
                        class="w-full text-xs bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text-gray-200"
                        data-stem="${stemType}">
                    ${options}
                </select>
            </div>`);

        document.getElementById('tuning-cluster-feature')?.addEventListener('change', onClusterFeatureChange);
    }

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
        const sliderKeys = ['open_geomean_min', 'open_sustain_ms'];
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
function formatSliderValue(val) {
    if (val == null) return '—';
    // 2026-06-10: the band_max_ratio_max slider uses 0 as the
    // "Off / Disabled" sentinel (the filter is a no-op at 0).
    // Show the user an explicit "Off" label at that position
    // so they can confirm the filter is inactive — important
    // because the slider's visible value alone doesn't reveal
    // whether the filter is on or off.
    if (val === 0) return 'Off';
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
        'open_geomean_min', 'open_sustain_ms', 'expected_clusters',
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

    // Update stored value
    if (waveformActiveStem) {
        if (!tuningSliderValues[waveformActiveStem]) tuningSliderValues[waveformActiveStem] = {};
        tuningSliderValues[waveformActiveStem][key] = val;
    }

    // Update numeric display
    const unitLabel = unit ? ` <span class="text-gray-500">${unit}</span>` : '';
    const display = document.getElementById(`tuning-val-${key}`);
    if (display) display.innerHTML = `${formatSliderValue(val)}${unitLabel}`;

    // Update Save button visibility
    updateTuningSaveButton();

    if (isClassification) {
        // Classification slider — only needs server reclassify, no local filtering
        scheduleReclassify();
    } else {
        // Filtering slider — local filter first, then reclassify for note colors
        if (tuningRafId) cancelAnimationFrame(tuningRafId);
        tuningRafId = requestAnimationFrame(() => {
            applyTuningFilter();
            tuningRafId = null;
            // After filtering, reclassify to update note assignments on new KEPT set
            scheduleReclassify();
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
        scheduleReclassify();
    });
}

/**
 * Handle cluster feature dropdown change — stores override and reclassifies.
 */
function onClusterFeatureChange(e) {
    const stemType = e.target.dataset.stem;
    const feature = e.target.value;

    if (feature === 'auto') {
        delete clusterFeatureOverrides[stemType];
    } else {
        clusterFeatureOverrides[stemType] = feature;
    }

    // Clear existing cluster note overrides since clusters will change
    delete clusterNoteOverrides[stemType];

    // Store in slider values so it gets sent as config override
    if (!tuningSliderValues[stemType]) tuningSliderValues[stemType] = {};
    tuningSliderValues[stemType]['cluster_feature'] = feature;

    updateTuningSaveButton();
    scheduleReclassify();
}

/**
 * Handle hihat open/closed classification toggle.
 */
function onHihatClassificationToggle(e) {
    const enabled = e.target.checked;
    const stemType = waveformActiveStem || 'hihat';
    
    hihatClassificationEnabled[stemType] = enabled;
    
    // Show/hide the open/closed classification sliders
    const sliderKeys = ['open_geomean_min', 'open_sustain_ms'];
    sliderKeys.forEach(key => {
        const sliderRow = document.querySelector(`[data-slider-key="${key}"]`);
        if (sliderRow) {
            sliderRow.style.display = enabled ? '' : 'none';
        }
    });
    
    // Re-run classification
    scheduleReclassify();
}

/**
 * Keys that are classification parameters (sent as config_overrides to reclassify).
 */
const CLASSIFICATION_KEYS = new Set(['open_geomean_min', 'open_sustain_ms', 'expected_clusters', 'cluster_feature']);

/**
 * Per-stem note assignment overrides from cluster dropdowns.
 * Format: { stemType: { classificationIndex: noteNumber } }
 */
let clusterNoteOverrides = {};

/**
 * Per-stem cluster feature override from dropdown.
 * Format: { stemType: featureName }
 */
let clusterFeatureOverrides = {};

/**
 * Available clustering features per stem for the feature dropdown.
 */
const STEM_FEATURE_CHOICES = {
    snare: [
        { value: 'auto', label: 'Auto' },
        { value: 'stereo_width', label: 'Stereo Width' },
        { value: 'pan_confidence', label: 'Pan Position' },
        { value: 'spectral_centroid_hz', label: 'Brightness' },
        { value: 'pitch_hz', label: 'Pitch' },
    ],
    // Toms (2026-06-12): the toms pipeline is PGA-only, so the
    // cluster-feature dropdown (which re-runs k-means over a chosen
    // per-event feature) is no longer exposed in the toms slideout.
    // The cluster cards below (Low/Mid/High pitch note assignment)
    // still render because they're useful for mapping the auto-
    // derived k-means labels onto MIDI notes — that part doesn't
    // change. The feature dropdown is what the user removed.
    cymbals: [
        { value: 'auto', label: 'Auto' },
        { value: 'spectral_centroid_hz', label: 'Brightness' },
        { value: 'stereo_width', label: 'Stereo Width' },
        { value: 'pan_confidence', label: 'Pan Position' },
        { value: 'pitch_hz', label: 'Pitch' },
    ],
};

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
 * Schedule a debounced reclassify API call (500ms).
 * Collects classification slider overrides and calls the server.
 */
function scheduleReclassify() {
    if (reclassifyTimeoutId) clearTimeout(reclassifyTimeoutId);
    reclassifyTimeoutId = setTimeout(() => {
        reclassifyTimeoutId = null;
        doReclassify();
    }, 500);
}

/**
 * Call the reclassify API and merge results into displayed events.
 * Renders cluster info cards when the API returns cluster metadata.
 */
async function doReclassify() {
    if (!currentProject || !waveformActiveStem || !waveformAnalysisData) return;
    if (reclassifyInFlight) return; // Skip if already in flight

    const stemType = waveformActiveStem;
    const stored = tuningSliderValues[stemType] || {};

    // Build config overrides from classification slider values
    const configOverrides = {};
    for (const key of CLASSIFICATION_KEYS) {
        if (stored[key] != null) {
            configOverrides[key] = stored[key];
        }
    }

    reclassifyInFlight = true;
    try {
        const result = await api.reclassify(currentProject.number, stemType, configOverrides);
        if (!result || !result.events) return;

        // Build a time-keyed lookup and cache for reapplication after filter passes
        const classificationByTime = {};
        const noteOverrides = clusterNoteOverrides[stemType] || {};

        for (const ev of result.events) {
            if (ev.time != null) {
                const timeKey = ev.time.toFixed(4);
                // Apply note override if set in dropdown
                const overrideNote = ev.classification != null ? noteOverrides[ev.classification] : undefined;
                classificationByTime[timeKey] = {
                    classification: ev.classification,
                    note: overrideNote != null ? overrideNote : ev.note,
                    hihat_state: ev.hihat_state,
                };
            }
        }

        // Cache for reapplication after future filter passes
        lastClassification = classificationByTime;

        // Merge into the displayed events (tuning or configured)
        const displayEvents = waveformTuningEvents || getEventsForStem(waveformAnalysisData.stems[stemType]);
        for (const event of displayEvents) {
            if (event.status !== 'KEPT' || event.time == null) continue;
            const timeKey = event.time.toFixed(4);
            const cls = classificationByTime[timeKey];
            if (cls) {
                if (cls.hihat_state != null) event.hihat_state = cls.hihat_state;
                if (cls.classification != null) event.classification = cls.classification;
                if (cls.note != null) event.note = cls.note;
            }
        }

        // Render cluster info cards if available
        if (result.cluster_info && result.cluster_info.length > 1) {
            renderClusterCards(stemType, result.cluster_info);
        } else {
            hideClusterCards();
        }

        // Update collapsible section height after content changes
        requestAnimationFrame(() => {
            if (typeof updateCollapsibleHeights === 'function') {
                updateCollapsibleHeights();
            }
        });

        // Re-render with updated colors
        drawWaveform();
    } catch (err) {
        console.warn('Reclassify failed:', err.message);
    } finally {
        reclassifyInFlight = false;
    }
}

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

    // Update the classification cache so filter re-application preserves the override
    if (lastClassification) {
        for (const entry of Object.values(lastClassification)) {
            if (entry.classification === classification) {
                entry.note = newNote;
            }
        }
    }

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
    const stemData = waveformAnalysisData?.stems?.[stemType];
    const logic = stemData?.logic || {};
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

    // Check if cluster feature has been changed
    if (!hasChanges && clusterFeatureOverrides[stemType]) {
        const configuredFeature = logic.cluster_feature || 'auto';
        if (clusterFeatureOverrides[stemType] !== configuredFeature) {
            hasChanges = true;
        }
    }

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
 * Maps slider keys to their YAML paths: per-stem keys go to [stemType, key],
 * global filtering keys go to ["filtering", key].
 */
function buildConfigUpdates(stemType) {
    const sliderConfigs = STEM_SLIDER_CONFIGS[stemType];
    const stored = tuningSliderValues[stemType] || {};
    const stemData = waveformAnalysisData?.stems?.[stemType];
    const logic = stemData?.logic || {};
    const updates = [];

    if (!sliderConfigs) return updates;

    for (const slider of sliderConfigs) {
        const configuredVal = logic[slider.key] != null ? logic[slider.key] : slider.fallback;
        const currentVal = stored[slider.key];
        if (!_sliderValueChanged(slider, currentVal, configuredVal)) continue;
        // Route to correct YAML section
        const path = GLOBAL_FILTERING_KEYS.has(slider.key)
            ? ['filtering', slider.key]
            : [stemType, slider.key];
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

    // Include cluster feature override if changed
    const featureOverride = clusterFeatureOverrides[stemType];
    if (featureOverride) {
        const configuredFeature = logic.cluster_feature || 'auto';
        if (featureOverride !== configuredFeature) {
            updates.push({
                path: [stemType, 'cluster_feature'],
                value: featureOverride,
            });
        }
    }

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
        showToast(`Saved ${updates.length} threshold${updates.length > 1 ? 's' : ''} for ${stemType}`, 'success');

        // Step 2: Try fast rebuild from cached analysis
        if (btn) btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-1"></i>Rebuilding…';

        try {
            // Bug D: build config_overrides from the current slider values
            // so the server-side rebuild uses the same thresholds the user
            // sees in the tuning panel. Without this, the UI filter and
            // the actual saved MIDI filter would diverge for any slider
            // (especially reverb_continuation_attack_threshold).
            const stored = tuningSliderValues[stemType] || {};
            const configOverrides = _buildConfigOverrides(stemType, stored);

            const result = await api.rebuildMidi(currentProject.number, {
                honor_overrides: true,
                config_overrides: configOverrides,
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
                lastClassification = null;
                delete tuningSliderValues[stemType];
                delete clusterNoteOverrides[stemType];
                delete clusterFeatureOverrides[stemType];
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
        lastClassification = null;
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
 * Create the cached base events for tuning from events_configured.
 * Called once when entering tuning mode or switching stems.
 *
 * Uses events_configured (not events_sensitive) so the tune UI operates
 * on the exact same onset set that the pipeline produces. Sensitive events
 * have different onset start points and can contain events that don't exist
 * in the configured set, making them unreliable for previewing reconvert
 * results.
 */
function initTuningBaseEvents(stemType) {
    const stemData = waveformAnalysisData?.stems?.[stemType];
    const configuredEvents = stemData?.events_configured;
    if (!configuredEvents || configuredEvents.length === 0) {
        tuningBaseEvents = null;
        return;
    }
    // Deep-copy once — these are reused across filter passes
    tuningBaseEvents = configuredEvents.map(e => ({ ...e }));
}

/**
 * Re-apply cached classification results to the current tuning events.
 * Called after each filter pass to preserve note colors across slider drags.
 */
function reapplyClassification(events) {
    if (!lastClassification) return;
    for (const event of events) {
        if (event.status !== 'KEPT' || event.time == null) continue;
        const timeKey = event.time.toFixed(4);
        const cls = lastClassification[timeKey];
        if (cls) {
            if (cls.classification != null) event.classification = cls.classification;
            if (cls.note != null) event.note = cls.note;
            if (cls.hihat_state != null) event.hihat_state = cls.hihat_state;
        }
    }
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

    // Run the energy-derived filters (Pass 1 and Pass 2) so
    // their statuses are consistent with the saved sidecar.
    // Pass 1: Spectral filter (geomean + sustain + strength)
    applySpectralFilter(tuningBaseEvents, params, filterMode);

    // Pass 2: Reverb continuation filter
    const attackThreshold = params.reverb_continuation_attack_threshold;
    if (attackThreshold != null) {
        applyReverbContinuationFilter(tuningBaseEvents, attackThreshold);
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
    if (stemType === 'toms') {
        const ratioMax = params.band_max_ratio_max;
        if (ratioMax != null && ratioMax > 0) {
            applyBandMaxRatioMax(tuningBaseEvents, ratioMax);
        }
    }

    // Re-apply any cached classification data (note colors, types)
    reapplyClassification(tuningBaseEvents);

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

    const sensitiveTotal = (stemData.events_sensitive || []).length;
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
