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

/** Timeout ID for debounced reclassify API calls */
let reclassifyTimeoutId = null;

/** Whether a reclassify request is in flight */
let reclassifyInFlight = false;

/**
 * Cached deep-copy of events_sensitive for the active stem.
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
        { key: 'expected_clusters', label: '🥁 Sound Types', min: 1, max: 4, step: 1, fallback: 2, unit: '', classification: true }
    ],
    toms: [
        { key: 'geomean_threshold', label: 'Geomean Threshold', min: 0, max: 500, step: 1, fallback: 80, unit: '' },
        { key: 'reverb_continuation_attack_threshold', label: 'Reverb Attack Threshold', min: 0, max: 1.0, step: 0.01, fallback: 0.4, unit: '' },
        { key: 'expected_clusters', label: '🥁 Sound Types', min: 1, max: 4, step: 1, fallback: 3, unit: '', classification: true }
    ],
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

    // Attach input listeners
    container.querySelectorAll('input[type=range]').forEach(input => {
        input.addEventListener('input', onSliderInput);
    });

    // Update save button visibility for this stem
    updateTuningSaveButton();
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
        { value: 'spectral_centroid_hz', label: 'Pitch / Brightness' },
    ],
    toms: [
        { value: 'auto', label: 'Auto' },
        { value: 'spectral_centroid_hz', label: 'Pitch / Brightness' },
        { value: 'stereo_width', label: 'Stereo Width' },
        { value: 'pan_confidence', label: 'Pan Position' },
    ],
    cymbals: [
        { value: 'auto', label: 'Auto' },
        { value: 'spectral_centroid_hz', label: 'Pitch / Brightness' },
        { value: 'stereo_width', label: 'Stereo Width' },
        { value: 'pan_confidence', label: 'Pan Position' },
    ],
};

/**
 * Available MIDI note choices per stem for the cluster note dropdowns.
 */
const STEM_NOTE_CHOICES = {
    snare: [
        { note: 38, label: 'Snare' },
        { note: 37, label: 'Rimshot' },
        { note: 39, label: 'Clap' },
        { note: 40, label: 'Clap+Snare' },
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
        if (currentVal != null && Math.abs(currentVal - configuredVal) > slider.step * 0.01) {
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
        if (currentVal != null && Math.abs(currentVal - configuredVal) > slider.step * 0.01) {
            // Route to correct YAML section
            const path = GLOBAL_FILTERING_KEYS.has(slider.key)
                ? ['filtering', slider.key]
                : [stemType, slider.key];
            updates.push({ path, value: currentVal });
        }
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
                }
                drawWaveform();
                return;
            }
        } catch (rebuildErr) {
            // If rebuild returns 409 (needs full pipeline) or fails, fall back
            console.warn('Rebuild failed, falling back to full pipeline:', rebuildErr.message);
        }

        // Step 3: Fall back to full pipeline
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
 * Create the cached base events for tuning from events_sensitive.
 * Called once when entering tuning mode or switching stems.
 */
function initTuningBaseEvents(stemType) {
    const stemData = waveformAnalysisData?.stems?.[stemType];
    const sensitiveEvents = stemData?.events_sensitive;
    if (!sensitiveEvents || sensitiveEvents.length === 0) {
        tuningBaseEvents = null;
        return;
    }
    // Deep-copy once — these are reused across filter passes
    tuningBaseEvents = sensitiveEvents.map(e => ({ ...e }));
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

    // Pass 1: Spectral filter (geomean + sustain + strength)
    applySpectralFilter(tuningBaseEvents, params, filterMode);

    // Pass 2: Reverb continuation filter
    const attackThreshold = params.reverb_continuation_attack_threshold;
    if (attackThreshold != null) {
        applyReverbContinuationFilter(tuningBaseEvents, attackThreshold);
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
