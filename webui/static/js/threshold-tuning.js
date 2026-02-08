/**
 * Threshold Tuning Module
 *
 * Provides interactive client-side sliders that re-filter events_sensitive
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

/** Current slider values per stem (persisted across tab switches) */
let tuningSliderValues = {};

/** Animation frame ID for debounced re-filtering */
let tuningRafId = null;

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
        { key: 'reverb_continuation_attack_threshold', label: 'Reverb Attack Threshold', min: 0, max: 1.0, step: 0.01, fallback: 0.4, unit: '' }
    ],
    toms: [
        { key: 'geomean_threshold', label: 'Geomean Threshold', min: 0, max: 500, step: 1, fallback: 80, unit: '' },
        { key: 'reverb_continuation_attack_threshold', label: 'Reverb Attack Threshold', min: 0, max: 1.0, step: 0.01, fallback: 0.4, unit: '' }
    ],
    hihat: [
        { key: 'geomean_threshold', label: 'Geomean Threshold', min: 0, max: 200, step: 0.5, fallback: 8, unit: '' },
        { key: 'min_sustain_ms', label: 'Min Sustain', min: 0, max: 500, step: 5, fallback: 25, unit: 'ms' },
        { key: 'min_strength_threshold', label: 'Min Strength', min: 0, max: 1.0, step: 0.01, fallback: 0.02, unit: '' },
        { key: 'reverb_continuation_attack_threshold', label: 'Reverb Attack Threshold', min: 0, max: 1.0, step: 0.01, fallback: 0.4, unit: '' }
    ],
    cymbals: [
        { key: 'geomean_threshold', label: 'Geomean Threshold', min: 0, max: 1000, step: 5, fallback: 100, unit: '' },
        { key: 'min_sustain_ms', label: 'Min Sustain', min: 0, max: 500, step: 5, fallback: 150, unit: 'ms' },
        { key: 'min_strength_threshold', label: 'Min Strength', min: 0, max: 1.0, step: 0.01, fallback: 0.1, unit: '' },
        { key: 'reverb_continuation_attack_threshold', label: 'Reverb Attack Threshold', min: 0, max: 1.0, step: 0.01, fallback: 0.4, unit: '' }
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
    hihat: 'geomean_only',
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
            applyTuningFilter();
        }
    } else {
        panel.classList.add('hidden');
        if (btn) btn.classList.remove('tuning-btn-active');

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
    buildSlidersForStem(stemType);
    applyTuningFilter();
}

/**
 * Reset slider values to the configured defaults from analysis.json.
 */
function resetTuningSliders() {
    if (!waveformActiveStem || !waveformAnalysisData) return;

    // Clear stored values so buildSlidersForStem reads from logic block
    delete tuningSliderValues[waveformActiveStem];
    buildSlidersForStem(waveformActiveStem);
    applyTuningFilter();
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

        const unitLabel = slider.unit ? ` <span class="text-gray-500">${slider.unit}</span>` : '';
        const defaultLabel = logicValue != null
            ? `<span class="text-gray-600 text-xs ml-1">(configured: ${logicValue})</span>`
            : '';

        return `
            <div class="tuning-slider-row">
                <div class="flex items-center justify-between mb-1">
                    <label class="text-xs text-gray-300">${slider.label}${defaultLabel}</label>
                    <span class="text-xs text-larsnet-primary font-mono" id="tuning-val-${slider.key}">${formatSliderValue(currentVal)}${unitLabel}</span>
                </div>
                <input type="range"
                       id="tuning-slider-${slider.key}"
                       class="tuning-range w-full"
                       min="${slider.min}"
                       max="${slider.max}"
                       step="${slider.step}"
                       value="${currentVal}"
                       data-key="${slider.key}"
                       data-unit="${slider.unit || ''}">
            </div>`;
    }).join('');

    // Attach input listeners
    container.querySelectorAll('input[type=range]').forEach(input => {
        input.addEventListener('input', onSliderInput);
    });
}

/**
 * Format a slider value for display.
 */
function formatSliderValue(val) {
    if (val == null) return '—';
    if (Number.isInteger(val) || val >= 10) return Math.round(val).toString();
    if (val >= 1) return val.toFixed(1);
    return val.toFixed(2);
}

/**
 * Handle slider input events — debounced via requestAnimationFrame.
 */
function onSliderInput(e) {
    const key = e.target.dataset.key;
    const unit = e.target.dataset.unit || '';
    const val = parseFloat(e.target.value);

    // Update stored value
    if (waveformActiveStem) {
        if (!tuningSliderValues[waveformActiveStem]) tuningSliderValues[waveformActiveStem] = {};
        tuningSliderValues[waveformActiveStem][key] = val;
    }

    // Update numeric display
    const unitLabel = unit ? ` <span class="text-gray-500">${unit}</span>` : '';
    const display = document.getElementById(`tuning-val-${key}`);
    if (display) display.innerHTML = `${formatSliderValue(val)}${unitLabel}`;

    // Debounced re-filter
    if (tuningRafId) cancelAnimationFrame(tuningRafId);
    tuningRafId = requestAnimationFrame(() => {
        applyTuningFilter();
        tuningRafId = null;
    });
}

// ─── Client-Side Filtering ───────────────────────────────────────────────

/**
 * Apply the current slider values to events_sensitive and update the
 * waveform display. This is the core re-filtering logic that replicates
 * the server-side filter passes from analysis_core.py.
 */
function applyTuningFilter() {
    if (!waveformActiveStem || !waveformAnalysisData) return;

    const stemType = waveformActiveStem;
    const stemData = waveformAnalysisData.stems[stemType];
    if (!stemData) return;

    const sensitiveEvents = stemData.events_sensitive;
    if (!sensitiveEvents || sensitiveEvents.length === 0) {
        waveformTuningEvents = null;
        waveformTuningActive = false;
        updateEventCounts(stemData);
        drawWaveform();
        return;
    }

    const params = tuningSliderValues[stemType] || {};
    const filterMode = STEM_FILTER_MODES[stemType] || 'geomean_only';

    // Deep-copy events so we don't mutate the original data
    const events = sensitiveEvents.map(e => ({ ...e }));

    // Pass 1: Spectral filter (geomean + sustain + strength)
    applySpectralFilter(events, params, filterMode);

    // Pass 2: Reverb continuation filter
    const attackThreshold = params.reverb_continuation_attack_threshold;
    if (attackThreshold != null) {
        applyReverbContinuationFilter(events, attackThreshold);
    }

    // Set tuning state for waveform.js
    waveformTuningEvents = events;
    waveformTuningActive = true;

    updateEventCounts(stemData);
    drawWaveform();
}

/**
 * Pass 1: Spectral filter — replicates filter_onsets_by_spectral() logic.
 */
function applySpectralFilter(events, params, filterMode) {
    const geomeanThreshold = params.geomean_threshold;
    const minSustainMs = params.min_sustain_ms;
    const minStrength = params.min_strength_threshold;

    for (const event of events) {
        // Start as KEPT (sensitive events are all pre-KEPT)
        event.status = 'KEPT';

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
