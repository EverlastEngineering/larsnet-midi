/**
 * Waveform Viewer Component — Dual-Panel Layout
 *
 * Two vertically stacked canvas panels with synchronized zoom/pan:
 *   Top panel:  Energy envelope (mirrored DAW-style L/R waveform)
 *   Bottom panel: Event markers as amplitude bars (height = velocity)
 *
 * Features:
 *   - 60dB dynamic range log scaling for envelope
 *   - Bar-graph event markers (height proportional to velocity/amplitude)
 *   - Synchronized zoom (mouse wheel) and pan (click-drag)
 *   - Vertical crosshair cursor spanning both panels
 *   - Legend and tuning indicator in dedicated bar (outside plot area)
 *   - Hover tooltip on events
 */

// ─── Constants ───────────────────────────────────────────────────────────

const WAVEFORM_COLORS = {
    background: '#111827',
    axisLine: '#374151',
    axisText: '#9ca3af',
    envelopeLeft: 'rgba(59, 130, 246, 0.8)',   // blue
    envelopeRight: 'rgba(139, 92, 246, 0.6)',   // purple
    envelopeFillLeft: 'rgba(59, 130, 246, 0.35)',
    envelopeFillRight: 'rgba(139, 92, 246, 0.25)',
    thresholdLine: 'rgba(251, 191, 36, 0.7)',   // amber dashed
    markerKept: '#10b981',       // green  (energy-detected events)
    markerPga: '#8b5cf6',        // violet (percentile-gated broad-attack, method='percentile_gated')
    markerFiltered: '#ef4444',   // red
    markerReverbCont: '#f59e0b', // orange/amber
    markerSensitive: 'rgba(156, 163, 175, 0.3)',
    markerUnknown: '#6b7280',    // gray
    tooltipBg: 'rgba(17, 24, 39, 0.95)',
    tooltipBorder: '#4b5563',
    tooltipText: '#e5e7eb',
    crosshair: 'rgba(255, 255, 255, 0.3)',
    playbackLine: '#22d3ee',          // cyan playback position indicator
};

const STEM_COLORS = {
    kick:    { accent: '#3b82f6', label: 'Kick' },
    snare:   { accent: '#8b5cf6', label: 'Snare' },
    hihat:   { accent: '#10b981', label: 'Hi-Hat' },
    cymbals: { accent: '#f59e0b', label: 'Cymbals' },
    toms:    { accent: '#ef4444', label: 'Toms' },
};

/**
 * Classification colors for event types, indexed by classification (0-3).
 * Standard across all stems for visual consistency.
 *   0 = green, 1 = purple, 2 = cyan, 3 = yellow
 * Red (#ef4444) = disabled/filtered (tuning only)
 * Orange (#f59e0b) = reverb continuation (tuning only)
 */
const CLASSIFICATION_COLORS = [
    '#10b981',   // 0 — green  (primary / default)
    '#a855f7',   // 1 — purple
    '#22d3ee',   // 2 — cyan
    '#eab308',   // 3 — yellow
];

// Hihat open/closed classification colors
const HIHAT_OPEN_COLOR = '#f97316';   // Orange - open hi-hat
const HIHAT_CLOSED_COLOR = '#06b6d4';  // Cyan - closed hi-hat

// 2026-06-19: per-stem classification color map. The key is
// the classification label as it appears on each event:
// - hihat: 'open' / 'closed' (set by hihat_state)
// - toms, snare, cymbals: integer string like '0', '1', '2'
//   (set by event.classification, from k-means cluster id)
// The map is the single source of truth for "what color is
// which classification". getEventColor() consults it after
// checking that the per-stem classification toggle is on.
const STEM_CLASSIFICATION_COLORS = {
    hihat: {
        open: HIHAT_OPEN_COLOR,
        closed: HIHAT_CLOSED_COLOR,
    },
    // Other stems fall back to the legacy CLASSIFICATION_COLORS
    // palette (0..3) when their classification toggle is on.
    // They don't get a stem-specific override here because the
    // hihat stem is the only one with semantic labels
    // ('open' / 'closed'); the others use generic cluster ids.
};

const STEM_ORDER = ['kick', 'snare', 'toms', 'hihat', 'cymbals'];

// Padding for each canvas panel (in CSS pixels)
const ENV_PAD = { top: 6, bottom: 6, left: 45, right: 12 };
const EVT_PAD = { top: 6, bottom: 28, left: 45, right: 12 };

// ─── State ───────────────────────────────────────────────────────────────

let waveformAnalysisData = null;
let waveformEnvelopeCache = {};
let waveformActiveStem = null;
let waveformHoverEvent = null;
// 2026-06-22: repurposed from "Show sensitive" (the gray energy-detector
// overlay layer is gone — the user removed the feature). The checkbox
// now controls visibility of FILTERED events on the waveform and is
// persisted across page reloads via localStorage. Default is unchecked
// so the canvas shows only KEPT events on first load (matches the
// pre-refactor behavior when the Tune panel was closed).
const WAVEFORM_SHOW_FILTERED_KEY = 'larsnet:waveform:showFiltered';
function readShowFilteredFromStorage() {
    try {
        return localStorage.getItem(WAVEFORM_SHOW_FILTERED_KEY) === 'true';
    } catch (e) {
        // localStorage can throw in private-browsing mode in some browsers.
        return false;
    }
}
function writeShowFilteredToStorage(value) {
    try {
        localStorage.setItem(WAVEFORM_SHOW_FILTERED_KEY, value ? 'true' : 'false');
    } catch (e) { /* private mode etc. — silently ignore */ }
}
let waveformShowFiltered = readShowFilteredFromStorage();
let waveformTuningEvents = null;
let waveformTuningActive = false;

// 2026-06-19: per-stem classification toggle state. Mirrors
// hihatClassificationEnabled in threshold-tuning.js. The
// threshold-tuning module dispatches a
// 'larsnet:classification-toggle' CustomEvent on the window
// when the user flips the toggle; this map captures that
// state and is consulted by getEventColor() to decide
// whether to apply the per-classification color overlay
// (orange/cyan for hihat open/closed, the legacy cluster
// palette for other stems). Defaults to true (classification
// is on) — matches the threshold-tuning default.
let classificationEnabledByStem = {};

// Dual-canvas references
let envelopeCanvas = null;
let envelopeCtx = null;
let eventsCanvas = null;
let eventsCtx = null;

// Zoom/pan state
let waveformZoom = 1;        // 1 = full view, higher = zoomed in
let waveformPanOffset = 0;   // 0..1, fraction of total time range scrolled
let waveformIsDragging = false;
let waveformDragStartX = 0;
let waveformDragStartPan = 0;

// Crosshair state
let waveformMouseX = null;   // CSS-pixel X relative to canvas parent

// Backward-compatible aliases (threshold-tuning.js references these)
let waveformCanvas = null;
let waveformCtx = null;

// Audio playback state (Web Audio API)
let audioCtx = null;
let audioBufferCache = {};   // stemType -> AudioBuffer
let audioSource = null;      // currently playing AudioBufferSourceNode
let audioIsPlaying = false;
let audioPlaybackTime = null; // time (in song seconds) where playback started
let audioStartContextTime = null; // audioCtx.currentTime when playback started
let playbackAnimFrameId = null;   // requestAnimationFrame ID for playback indicator

// Event override state: { stemType: { "<frame>": { status, [classification] } } }
// Each override record carries at minimum a `status` ("KEPT"|"FILTERED").
// An optional `classification` is set when the user has cycled past the
// first "on" state via cycleEventOverride. The classification drives the
// per-event note via the standard classify_notes path on rebuild.
//
// 2026-06-30: keys are now frame integers (str(frame)) instead of
// time strings (time.toFixed(4)). Frame is the canonical per-event
// identifier and avoids the float-precision mismatch the user hit
// when the file was written with "2.954" but the lookup used
// "2.9540". See _eventOverrideKey below.
let eventOverrides = {};
// in-memory ≠ JSON (cleared by the debounced save)
let eventOverridesDirty = false;
// user has unsaved changes waiting for Save & Reconvert
// (cleared only by saveTuningAndReconvert's server sync)
let sessionOverridesDirty = false;
let eventOverridesSaveTimer = null;

// ─── Loading Indicator ───────────────────────────────────────────────────

function showWaveformLoading(text = 'Loading…', pct = 0) {
    const overlay = document.getElementById('waveform-loading-overlay');
    const bar = document.getElementById('waveform-loading-bar');
    const label = document.getElementById('waveform-loading-text');
    if (!overlay) return;
    overlay.classList.remove('hidden');
    overlay.style.display = 'flex';
    if (bar) bar.style.width = Math.round(pct) + '%';
    if (label) label.textContent = text;
}

function updateWaveformLoading(text, pct) {
    const bar = document.getElementById('waveform-loading-bar');
    const label = document.getElementById('waveform-loading-text');
    if (bar) bar.style.width = Math.round(pct) + '%';
    if (label) label.textContent = text;
}

function hideWaveformLoading() {
    const overlay = document.getElementById('waveform-loading-overlay');
    if (!overlay) return;
    overlay.classList.add('hidden');
    overlay.style.display = 'none';
}

// 2026-06-19: listen for the per-stem classification toggle.
// threshold-tuning.js dispatches this event on the window when
// the user flips the "Open/Closed Classification" toggle (or
// any future classification toggle). We update the local map
// and re-render so the per-classification color overlay
// (orange/cyan for hihat, the cluster palette for other stems)
// appears or disappears in lockstep with the toggle. The
// window-level dispatch is a single channel — threshold-tuning
// is the producer, waveform is one of many potential
// consumers (advanced-midi.js, future export pipelines, etc.).
window.addEventListener('larsnet:classification-toggle', (e) => {
    const { stem, enabled } = e.detail || {};
    if (!stem) return;
    classificationEnabledByStem[stem] = !!enabled;
    // Re-render so the per-event color updates immediately.
    // The waveform panel is the only canvas that reads the
    // classification color today; re-rendering it is cheap
    // (one redraw of the events layer) and keeps the legend
    // in sync with the visible event colors.
    if (waveformActiveStem === stem && typeof renderWaveform === 'function') {
        renderWaveform();
    }
});

// ─── Public API ──────────────────────────────────────────────────────────

async function initWaveformViewer(project) {
    const section = document.getElementById('analysis-section');
    if (!section) return;

    // Reset state
    waveformAnalysisData = null;
    waveformEnvelopeCache = {};
    waveformActiveStem = null;
    waveformHoverEvent = null;
    waveformTuningEvents = null;
    waveformTuningActive = false;
    waveformZoom = 1;
    waveformPanOffset = 0;
    stopAudioPlayback();
    audioBufferCache = {};
    eventOverrides = {};
    eventOverridesDirty = false;
    sessionOverridesDirty = false;
    // Clear all tuning state so fresh project loads with logic-block defaults
    tuningSliderValues = {};
    clusterNoteOverrides = {};
    clusterFeatureOverrides = {};
    tuningBaseEvents = null;
    lastClassification = null;

    if (!project.has_analysis) {
        section.classList.add('hidden');
        return;
    }

    section.classList.remove('hidden');
    showWaveformLoading('Loading analysis data…', 10);

    try {
        waveformAnalysisData = await api.getProjectAnalysis(project.number);
    } catch (err) {
        console.error('Failed to load analysis data:', err);
        hideWaveformLoading();
        section.classList.add('hidden');
        return;
    }

    if (!waveformAnalysisData || !waveformAnalysisData.stems) {
        hideWaveformLoading();
        section.classList.add('hidden');
        return;
    }

    // Bug C: surface any data-integrity warnings from the loader so
    // the user is told when events_configured contains events not in
    // events_sensitive. Loaded fresh on every project open, so this
    // also catches warnings for projects opened without going through
    // the rebuild path. Logged to console only — toasts were too noisy
    // on every project open and the warnings are diagnostic, not
    // blocking.
    if (Array.isArray(waveformAnalysisData.data_integrity_warnings) &&
        waveformAnalysisData.data_integrity_warnings.length > 0) {
        for (const warning of waveformAnalysisData.data_integrity_warnings) {
            console.warn('Data integrity warning:', warning);
        }
    }

    updateWaveformLoading('Loading overrides…', 40);
    await loadEventOverrides();

    updateWaveformLoading('Preparing stems…', 50);
    const availableStems = Object.keys(waveformAnalysisData.stems);
    renderStemTabs(availableStems);

    // Set up dual canvases
    envelopeCanvas = document.getElementById('envelope-canvas');
    eventsCanvas = document.getElementById('events-canvas');
    if (!envelopeCanvas || !eventsCanvas) return;
    envelopeCtx = envelopeCanvas.getContext('2d');
    eventsCtx = eventsCanvas.getContext('2d');

    // Backward-compatible alias
    waveformCanvas = eventsCanvas;
    waveformCtx = eventsCtx;

    // Mouse interaction — both canvases
    setupCanvasInteraction(envelopeCanvas);
    setupCanvasInteraction(eventsCanvas);

    // 2026-06-22: Show Filtered toggle (repurposed from "Show
    // sensitive"). Visibility of FILTERED events on the waveform
    // is now purely user-controlled and independent of the Tune
    // panel state. Persisted in localStorage so the preference
    // sticks across page reloads.
    const filteredToggle = document.getElementById('waveform-filtered-toggle');
    if (filteredToggle) {
        filteredToggle.checked = waveformShowFiltered;
        filteredToggle.onchange = () => {
            waveformShowFiltered = filteredToggle.checked;
            writeShowFilteredToStorage(waveformShowFiltered);
            drawWaveform();
        };
    }

    // Tune button visibility — show if any stem has tuning-source data.
    // events_sensitive: energy-detected onsets (most stems).
    // events_pga: PGA-only onsets (toms, 2026-06-15 refactor).
    const tuneBtn = document.getElementById('tuning-toggle-btn');
    if (tuneBtn) {
        const hasAnySensitive = availableStems.some(s => {
            const sd = waveformAnalysisData.stems[s];
            return (sd.events_sensitive && sd.events_sensitive.length > 0) ||
                   (sd.events_pga && sd.events_pga.length > 0);
        });
        tuneBtn.classList.toggle('hidden', !hasAnySensitive);
    }

    // Close tuning panel on new project
    const tuningPanel = document.getElementById('tuning-panel');
    if (tuningPanel && !tuningPanel.classList.contains('hidden')) {
        tuningPanelOpen = false;
        tuningPanel.classList.add('hidden');
        if (tuneBtn) tuneBtn.classList.remove('tuning-btn-active');
    }

    // Select first stem
    const firstStem = STEM_ORDER.find(s => availableStems.includes(s)) || availableStems[0];
    if (firstStem) selectStem(firstStem);
}

async function selectStem(stemType) {
    if (!waveformAnalysisData || !waveformAnalysisData.stems[stemType]) return;

    waveformActiveStem = stemType;
    waveformHoverEvent = null;
    waveformTuningEvents = null;
    waveformTuningActive = false;
    waveformZoom = 1;
    waveformPanOffset = 0;

    document.querySelectorAll('.waveform-stem-tab').forEach(tab => {
        const isActive = tab.dataset.stem === stemType;
        tab.classList.toggle('waveform-tab-active', isActive);
        tab.classList.toggle('waveform-tab-inactive', !isActive);
    });

    // 2026-06-22: Show the "Show Filtered" container whenever the
    // active stem has any events that could be hidden (i.e. any
    // events_configured or events_pga entry — every entry has a
    // status of KEPT or FILTERED). The container is purely
    // cosmetic; the checkbox works regardless of the stem. We
    // used to gate on events_sensitive, which the new feature
    // ignores.
    const filteredContainer = document.getElementById('waveform-filtered-container');
    if (filteredContainer) {
        const stemData = waveformAnalysisData.stems[stemType];
        const hasEvents = (stemData.events_configured && stemData.events_configured.length > 0) ||
                          (stemData.events_pga && stemData.events_pga.length > 0);
        filteredContainer.classList.toggle('hidden', !hasEvents);
    }

    // Load envelope data
    if (!waveformEnvelopeCache[stemType]) {
        showWaveformLoading('Loading waveform for ' + stemType + '…', 30);
        try {
            const envelope = await api.getProjectEnvelope(currentProject.number, stemType);
            waveformEnvelopeCache[stemType] = envelope;
            updateWaveformLoading('Rendering…', 90);
        } catch {
            waveformEnvelopeCache[stemType] = null;
        }
    }

    hideWaveformLoading();
    drawWaveform();

    // Pre-fetch audio buffer in background for click-to-play
    ensureAudioBuffer(stemType);

    if (typeof onTuningStemChanged === 'function') {
        await onTuningStemChanged(stemType);
    }
}

// ─── Tab Rendering ───────────────────────────────────────────────────────

function renderStemTabs(availableStems) {
    const container = document.getElementById('waveform-stem-tabs');
    if (!container) return;

    const ordered = STEM_ORDER.filter(s => availableStems.includes(s));
    availableStems.forEach(s => { if (!ordered.includes(s)) ordered.push(s); });

    container.innerHTML = ordered.map(stem => {
        const info = STEM_COLORS[stem] || { accent: '#6b7280', label: stem };
        return `<button class="waveform-stem-tab waveform-tab-inactive px-3 py-1.5 rounded text-xs font-medium transition-smooth"
                        data-stem="${stem}"
                        style="--tab-accent: ${info.accent}"
                        onclick="selectStem('${stem}')">
                    ${info.label}
                </button>`;
    }).join('');
}

// ─── Zoom / Pan Helpers ──────────────────────────────────────────────────

/** Compute the visible time window based on zoom and pan. */
function computeVisibleRange(tMinFull, tMaxFull) {
    const fullSpan = tMaxFull - tMinFull;
    const visibleSpan = fullSpan / waveformZoom;
    const maxOffset = fullSpan - visibleSpan;
    const offset = waveformPanOffset * maxOffset;
    return { tMin: tMinFull + offset, tMax: tMinFull + offset + visibleSpan };
}

function clampPan() {
    waveformPanOffset = Math.max(0, Math.min(1, waveformPanOffset));
}

// ─── Main Draw ───────────────────────────────────────────────────────────

function drawWaveform() {
    if (!envelopeCanvas || !eventsCanvas || !waveformActiveStem) return;

    const stemData = waveformAnalysisData.stems[waveformActiveStem];
    const configuredEvents = getEventsForStem(stemData);
    const sensitiveEvents = getSensitiveEventsForStem(stemData);
    // 2026-06-15: in tuning mode, render the PGA layer from the
    // live tuning events (with slider-driven KEPT/FILTERED status)
    // instead of the unmodified sidecar. This is the fix for the
    // "faded lines should show up as kept" symptom: the PGA bar
    // layer now follows the slider drag in real time, matching the
    // event count display (which already reads from
    // waveformTuningEvents).
    const pgaEvents = (waveformTuningActive && waveformTuningEvents)
        ? waveformTuningEvents
        : getPgaEventsForStem(stemData);
    const envelope = waveformEnvelopeCache[waveformActiveStem];

    const displayEvents = (waveformTuningActive && waveformTuningEvents)
        ? waveformTuningEvents
        : configuredEvents;

    // Full time range (for zoom reference). PGA events are
    // included in the range so the auto-zoom covers them when the
    // user has no other events to anchor the timeline. Without this,
    // a project that hasn't generated any configured events (a
    // broken pipeline) would still zoom to the PGA events.
    const { tMin: tMinFull, tMax: tMaxFull } = computeTimeRange(
        configuredEvents, sensitiveEvents, envelope, pgaEvents,
    );
    if (tMaxFull <= tMinFull) return;

    // Visible time range (affected by zoom/pan)
    const { tMin, tMax } = computeVisibleRange(tMinFull, tMaxFull);

    // Draw envelope panel
    drawEnvelopePanel(envelope, tMin, tMax, stemData, configuredEvents, sensitiveEvents);

    // Draw events panel
    drawEventsPanel(displayEvents, sensitiveEvents, configuredEvents, pgaEvents, tMin, tMax, stemData);

    // Update legend bar (HTML, outside canvas)
    updateLegendBar(stemData, displayEvents, pgaEvents);

    // Draw crosshair on both panels
    if (waveformMouseX != null) {
        drawCrosshair(envelopeCtx, envelopeCanvas);
        drawCrosshair(eventsCtx, eventsCanvas);
    }

    // Draw tooltip on events panel
    if (waveformHoverEvent) {
        const rect = eventsCanvas.parentElement.getBoundingClientRect();
        drawTooltip(waveformHoverEvent, rect.width, rect.height);
    } else {
        hideTooltip();
    }
}

// ─── Envelope Panel ──────────────────────────────────────────────────────

function drawEnvelopePanel(envelope, tMin, tMax, stemData, configuredEvents, sensitiveEvents) {
    const canvas = envelopeCanvas;
    const ctx = envelopeCtx;
    const dpr = window.devicePixelRatio || 1;

    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = rect.height;
    const PAD = ENV_PAD;
    const plotW = W - PAD.left - PAD.right;
    const plotH = H - PAD.top - PAD.bottom;

    // Background
    ctx.fillStyle = WAVEFORM_COLORS.background;
    ctx.fillRect(0, 0, W, H);

    const timeToX = t => PAD.left + ((t - tMin) / (tMax - tMin)) * plotW;

    // Envelope
    if (envelope && envelope.times) {
        const envelopeMax = computeEnvelopeMax(envelope);
        drawEnvelope(ctx, envelope, timeToX, PAD, plotH, envelopeMax);
    }

    // Threshold line (on geomean scale). Priority:
    //   1. Live slider value (tuning mode) — what the user just moved
    //   2. Live midiconfig.yaml (tuningConfig) — the committed value
    //      (2026-06-15: the analysis.json `logic` block is no longer
    //      read; yaml is the single source of truth)
    //   3. Hidden (null) — no threshold configured for this stem
    const geomeanMax = computeMaxGeomean(configuredEvents, sensitiveEvents);
    const tuningGeomean = waveformTuningActive && tuningSliderValues?.[waveformActiveStem]?.geomean_threshold;
    const liveYamlGeomean = (typeof tuningConfig !== 'undefined' && tuningConfig[waveformActiveStem])
        ? tuningConfig[waveformActiveStem].geomean_threshold
        : null;
    const thresholdVal = tuningGeomean != null ? tuningGeomean : liveYamlGeomean;
    if (thresholdVal != null && geomeanMax > 0) {
        const geomeanToY = v => PAD.top + plotH - (v / (geomeanMax * 1.2 || 1)) * plotH;
        drawThresholdLine(ctx, thresholdVal, geomeanToY, PAD, plotW);
    }

    // dB scale labels on left axis
    drawEnvelopeAxis(ctx, PAD, plotH);
}

function drawEnvelopeAxis(ctx, PAD, plotH) {
    ctx.fillStyle = WAVEFORM_COLORS.axisText;
    ctx.font = '9px system-ui, sans-serif';
    ctx.textAlign = 'right';

    const centerY = PAD.top + plotH / 2;

    const labels = [0, -12, -24, -48];
    for (const dB of labels) {
        const frac = dB === 0 ? 1 : Math.max(0, 1 + dB / 60);
        const yUp = centerY - frac * (plotH / 2);

        if (dB === 0) {
            ctx.fillText('0dB', PAD.left - 4, yUp + 3);
        } else if (yUp >= PAD.top - 2) {
            ctx.fillText(`${dB}`, PAD.left - 4, yUp + 3);
        }
    }

    // Center line indicator
    ctx.fillStyle = 'rgba(107, 114, 128, 0.5)';
    ctx.fillText('—', PAD.left - 4, centerY + 3);
}

// ─── Events Panel ────────────────────────────────────────────────────────

function drawEventsPanel(displayEvents, sensitiveEvents, configuredEvents, pgaEvents, tMin, tMax, stemData) {
    const canvas = eventsCanvas;
    const ctx = eventsCtx;
    const dpr = window.devicePixelRatio || 1;

    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = rect.height;
    const PAD = EVT_PAD;
    const plotW = W - PAD.left - PAD.right;
    const plotH = H - PAD.top - PAD.bottom;

    // Background
    ctx.fillStyle = WAVEFORM_COLORS.background;
    ctx.fillRect(0, 0, W, H);

    const timeToX = t => PAD.left + ((t - tMin) / (tMax - tMin)) * plotW;

    // Time axis (at bottom)
    drawTimeAxis(ctx, W, H, PAD, plotW, plotH, tMin, tMax, timeToX);

    // Velocity scale labels on left axis
    drawVelocityAxis(ctx, PAD, plotW, plotH);

    // 2026-06-22: gray "Sensitive" overlay layer removed. The
    // checkbox is repurposed for FILTERED visibility (handled
    // by the eventsToRender filter below), and the gray
    // energy-detector onsets layer is dead UI.

    // 2026-06-30: UNIFIED render path. The events panel
    // previously had TWO draw calls — drawEventBars (for
    // non-PGA events, color via getEventColor with alpha=0.9
    // for KEPT) and drawPgaEventBars (for PGA events, hardcoded
    // violet with the faded-red alpha=0.35 for FILTERED). The
    // two paths produced inconsistent colors (e.g. kick in
    // project 6 rendered violet before a slider touch and
    // green after, because the data source switched from
    // events_pga to events_configured). With getEventsForStem
    // now preferring events_pga and the unified drawPgaEventBars
    // below handling any event type via getEventColor, a single
    // call renders the whole layer consistently.
    //
    // 2026-06-22: FILTERED visibility is controlled by the
    // "Show Filtered" checkbox (waveformShowFiltered), NOT by
    // the Tune panel. When the checkbox is checked, all events
    // (KEPT + FILTERED + REVERB_CONTINUATION) are drawn; when
    // unchecked, only KEPT events are drawn. The user can
    // toggle this independently of the panel.
    const eventsToRender = waveformShowFiltered
        ? displayEvents
        : displayEvents.filter(e => e.status === 'KEPT');
    drawPgaEventBars(ctx, eventsToRender, timeToX, PAD, plotW, plotH);
}

function drawVelocityAxis(ctx, PAD, plotW, plotH) {
    ctx.fillStyle = WAVEFORM_COLORS.axisText;
    ctx.font = '9px system-ui, sans-serif';
    ctx.textAlign = 'right';

    const ticks = [127, 96, 64, 32];
    for (const v of ticks) {
        const y = PAD.top + plotH - (v / 127) * plotH;
        ctx.fillText(v, PAD.left - 4, y + 3);

        // Subtle grid line
        ctx.strokeStyle = 'rgba(55, 65, 81, 0.4)';
        ctx.lineWidth = 0.5;
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(PAD.left, y);
        ctx.lineTo(PAD.left + plotW, y);
        ctx.stroke();
        ctx.setLineDash([]);
    }
}

/**
 * Draw event markers as amplitude bars.
 * Height is proportional to velocity (0-127). Color-coded by status.
 */
function drawEventBars(ctx, events, timeToX, PAD, plotW, plotH, isSensitiveLayer) {
    const barWidth = isSensitiveLayer ? 1.5 : 3;

    for (const event of events) {
        if (event.time == null) continue;

        // Toms PGA cleanup (2026-06-12). PGA events
        // (method='percentile_gated') are rendered exclusively
        // by drawPgaEventBars() at the bottom of this panel;
        // skipping them here prevents a second violet bar
        // being drawn for the same event (which was visible
        // as a "narrow bar on top of a wide bar" stack before
        // the cleanup). The sensitive background layer
        // (isSensitiveLayer=true) is still allowed to draw
        // PGA events — that's the max-sensitivity layer used
        // for tuning, and it can include any method.
        if (!isSensitiveLayer && event.method === 'percentile_gated') continue;

        const x = timeToX(event.time);
        if (x < PAD.left - barWidth || x > PAD.left + plotW + barWidth) continue;

        const color = isSensitiveLayer
            ? WAVEFORM_COLORS.markerSensitive
            : getEventColor(event);

        // Bar height from velocity (0-127)
        // When velocity is missing (sensitive/tuning events),
        // estimate it from the available quality signal:
        //   - energy events:   event.strength is the energy
        //                       detector's [0, 1] normalized
        //                       onset strength (already 0-1).
        //   - spectral events: the lossy clamp-to-1.0
        //                       `strength` field is gone (replaced
        //                       by raw band_max_ratio). Use
        //                       min(1, band_max_ratio / 10) as a
        //                       soft saturation — the same
        //                       intent as the old formula but
        //                       without collapsing everything
        //                       above 10× to the same value.
        //   - missing both:    fall back to 64 (the typical
        //                       mid-velocity default).
        let velocity;
        if (event.velocity != null) {
            velocity = event.velocity;
        } else if (event.method === 'spectral' && event.band_max_ratio != null) {
            const v = Math.min(1, Math.max(0, event.band_max_ratio / 10));
            velocity = Math.round(40 + v * (127 - 40));
            velocity = Math.max(1, Math.min(127, velocity));
        } else if (event.strength != null) {
            velocity = Math.round(40 + event.strength * (127 - 40));
            velocity = Math.max(1, Math.min(127, velocity));
        } else {
            velocity = 64;
        }
        const barH = Math.max(2, (velocity / 127) * plotH);
        const barTop = PAD.top + plotH - barH;

        ctx.globalAlpha = isSensitiveLayer ? 0.4 : 0.9;
        ctx.fillStyle = color;
        ctx.fillRect(x - barWidth / 2, barTop, barWidth, barH);

        if (!isSensitiveLayer) {
            ctx.strokeStyle = color;
            ctx.lineWidth = 0.5;
            ctx.globalAlpha = 0.5;
            ctx.strokeRect(x - barWidth / 2, barTop, barWidth, barH);

            // Override indicator: small white diamond at top of bar
            if (event._overridden) {
                ctx.globalAlpha = 1.0;
                ctx.fillStyle = '#ffffff';
                const dSize = 3;
                ctx.beginPath();
                ctx.moveTo(x, barTop - dSize);
                ctx.lineTo(x + dSize, barTop);
                ctx.lineTo(x, barTop + dSize);
                ctx.lineTo(x - dSize, barTop);
                ctx.closePath();
                ctx.fill();
            }
        }

        ctx.globalAlpha = 1.0;
    }
}

// 2026-06-30: unified event-bar renderer (formerly named
// drawPgaEventBars). Replaces the old two-path render:
// drawEventBars (for non-PGA events, color via getEventColor
// with the alpha=0.9 / full-strength convention) and
// drawPgaEventBars (for PGA events, hardcoded violet with the
// faded-red convention). The two paths produced inconsistent
// colors (e.g. kick in project 6 rendered violet before a
// slider touch and green after — user-reported bug). This
// unified function handles all event types via getEventColor,
// uses the faded-red alpha convention for FILTERED events
// (0.35) and a slightly faded full strength for KEPT (0.85),
// and supports the sensitive-layer case (tuning background)
// via the isSensitiveLayer flag.
//
// Color resolution (via getEventColor, in priority order):
//   1. FILTERED → red (#ef4444), alpha 0.35 (faded)
//   2. REVERB_CONTINUATION → orange (#f59e0b)
//   3. hihat open/closed → orange/cyan (when classification toggle on)
//   4. method === 'percentile_gated' → violet (#8b5cf6)
//   5. classification index → classification palette
//   6. else → green (markerKept)
function drawPgaEventBars(ctx, events, timeToX, PAD, plotW, plotH, isSensitiveLayer) {
    // 2026-06-30: was 2. The unified render path needs to feel
    // like the events panel's bars (3-px wide) so the "faded red
    // = filtered" convention is visually consistent. The
    // sensitive-layer case uses a thinner 1.5-px bar so it
    // reads as a faded background, not a foreground event.
    const barWidth = isSensitiveLayer ? 1.5 : 2.5;
    // Toms PGA cleanup (2026-06-12). The previous top-aligned
    // draw ("grow down from markerTop") made PGA bars hard to
    // compare against the other stems' velocity bars, which are
    // bottom-anchored (velocity 0 at the top, 127 at the bottom).
    // Switched to bottom-anchored growth so velocity 127 fills
    // the full plotH and velocity 80 fills 80/127 of plotH,
    // matching the green velocity-bar convention used elsewhere.
    const maxBarH = plotH;

    for (const event of events) {
        if (event.time == null) continue;
        const isFiltered = event.status === 'FILTERED';
        // 2026-06-30: the "Show Filtered" toggle gates
        // FILTERED events on the main layer only. The
        // sensitive-layer case (tuning background) shows
        // filtered events for context — the user always
        // wants to see them when investigating the filter.
        if (!isSensitiveLayer && isFiltered && !waveformShowFiltered) {
            continue;
        }
        const x = timeToX(event.time);
        if (x < PAD.left - barWidth || x > PAD.left + plotW + barWidth) continue;

        // Bar height = midi_velocity / 127, mapped to full plotH.
        // Falls back to a sensible default if velocity is missing
        // (older sidecars).
        let velocity = event.midi_velocity;
        if (velocity == null) {
            velocity = isFiltered ? 60 : 100;
        }
        const barH = Math.max(4, (velocity / 127) * maxBarH);
        // Bottom-anchored: the bar grows UP from the bottom of
        // the events panel (barTop = PAD.top + plotH - barH).
        const barTop = PAD.top + plotH - barH;

        // 2026-06-30: use getEventColor (which already encodes
        // FILTERED → red, hihat open/closed, classification
        // palette, REVERB_CONTINUATION → orange) for the main
        // layer. The sensitive layer uses markerSensitive gray
        // for all events (it's a faded context background,
        // not a status indicator).
        const barColor = isSensitiveLayer
            ? WAVEFORM_COLORS.markerSensitive
            : getEventColor(event);

        // Faded-red convention (user feedback 2026-06-30):
        //   - main layer, KEPT       → alpha 0.85
        //   - main layer, FILTERED   → alpha 0.35 (faded red)
        //   - sensitive layer (any)   → alpha 0.40
        let alpha;
        if (isSensitiveLayer) {
            alpha = 0.40;
        } else if (isFiltered) {
            alpha = 0.35;
        } else {
            alpha = 0.85;
        }
        ctx.globalAlpha = alpha;
        ctx.fillStyle = barColor;
        ctx.fillRect(x - barWidth / 2, barTop, barWidth, barH);

        // Outline for crispness at zoom levels
        ctx.globalAlpha = isFiltered
            ? 0.40
            : (isSensitiveLayer ? 0.50 : 1.0);
        ctx.strokeStyle = barColor;
        ctx.lineWidth = 0.5;
        ctx.strokeRect(x - barWidth / 2, barTop, barWidth, barH);
    }
    ctx.globalAlpha = 1.0;
}

// ─── Data Helpers ────────────────────────────────────────────────────────

function getEventsForStem(stemData) {
    // 2026-06-30: prefer events_pga over events_configured. The
    // PGA-detected events_pga is the canonical source for any
    // stem that uses the PGA detector (kick/snare/toms/hihat/
    // cymbals on the 2026-06-15 PGA-only refactor). For legacy
    // sidecars that ALSO carry events_configured (an energy-
    // detected subset from before the refactor), preferring
    // events_configured led to a data-source mismatch: the
    // events panel rendered events_configured (no method,
    // no classification → green) while the PGA overlay rendered
    // events_pga (PGA method → violet) at the same X positions.
    // After touching a slider, the tuning path swapped the
    // source to waveformTuningEvents (initialized from
    // events_configured) so both panels rendered the same
    // 190-event subset in green — the bug the user reported.
    // events_pga is the only consistent source.
    if (stemData.events_pga) return stemData.events_pga;
    if (stemData.events_configured) return stemData.events_configured;
    if (stemData.events) return stemData.events;
    return [];
}

function getSensitiveEventsForStem(stemData) {
    return stemData.events_sensitive || [];
}

// PGA events live in their own sidecar field (events_pga) —
// the third complementary detector. With the 2026-06-30
// unified-rendering refactor, the events panel and the PGA
// overlay both pull from the same list (events_pga, see
// getEventsForStem). This helper is kept as an alias so
// existing callers don't break — the previous KEPT-only filter
// is no longer needed because the unified draw function
// handles status filtering itself.
function getPgaEventsForStem(stemData) {
    return getEventsForStem(stemData);
}

function computeTimeRange(events, sensitiveEvents, envelope, pgaEvents) {
    let tMin = Infinity, tMax = -Infinity;

    for (const e of events) {
        if (e.time != null) { tMin = Math.min(tMin, e.time); tMax = Math.max(tMax, e.time); }
    }
    for (const e of sensitiveEvents) {
        if (e.time != null) { tMin = Math.min(tMin, e.time); tMax = Math.max(tMax, e.time); }
    }
    // PGA events contribute to the zoom range so the auto-zoom
    // covers them when the project has no other events to anchor
    // the timeline. The optional-arg pattern is intentional —
    // pgaEvents may be undefined on older sidecars that predate
    // the PGA detector (2026-06-10).
    if (pgaEvents) {
        for (const e of pgaEvents) {
            if (e.time != null) { tMin = Math.min(tMin, e.time); tMax = Math.max(tMax, e.time); }
        }
    }
    if (envelope && envelope.times && envelope.times.length > 0) {
        tMin = Math.min(tMin, envelope.times[0]);
        tMax = Math.max(tMax, envelope.times[envelope.times.length - 1]);
    }

    const span = tMax - tMin || 1;
    return { tMin: tMin - span * 0.02, tMax: tMax + span * 0.02 };
}

function computeEnvelopeMax(envelope) {
    let maxVal = 0;
    if (envelope) {
        if (envelope.left) for (const v of envelope.left) maxVal = Math.max(maxVal, v);
        if (envelope.right) for (const v of envelope.right) maxVal = Math.max(maxVal, v);
    }
    return maxVal;
}

function computeMaxGeomean(events, sensitiveEvents) {
    let maxVal = 0;
    for (const e of events) {
        if (e.geomean != null) maxVal = Math.max(maxVal, e.geomean);
    }
    for (const e of sensitiveEvents) {
        if (e.geomean != null) maxVal = Math.max(maxVal, e.geomean);
    }
    return maxVal;
}

// ─── Drawing Subroutines ─────────────────────────────────────────────────

function drawTimeAxis(ctx, W, H, PAD, plotW, plotH, tMin, tMax, timeToX) {
    ctx.strokeStyle = WAVEFORM_COLORS.axisLine;
    ctx.lineWidth = 1;

    // Bottom axis line
    ctx.beginPath();
    ctx.moveTo(PAD.left, PAD.top + plotH);
    ctx.lineTo(PAD.left + plotW, PAD.top + plotH);
    ctx.stroke();

    // Time ticks
    const duration = tMax - tMin;
    const tickInterval = computeTickInterval(duration);
    const firstTick = Math.ceil(tMin / tickInterval) * tickInterval;

    ctx.fillStyle = WAVEFORM_COLORS.axisText;
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'center';

    for (let t = firstTick; t <= tMax; t += tickInterval) {
        const x = timeToX(t);
        if (x < PAD.left || x > PAD.left + plotW) continue;

        ctx.beginPath();
        ctx.moveTo(x, PAD.top + plotH);
        ctx.lineTo(x, PAD.top + plotH + 4);
        ctx.stroke();

        ctx.fillText(formatTime(t), x, PAD.top + plotH + 16);

        // Subtle vertical grid line
        ctx.strokeStyle = 'rgba(55, 65, 81, 0.3)';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(x, PAD.top);
        ctx.lineTo(x, PAD.top + plotH);
        ctx.stroke();

        ctx.strokeStyle = WAVEFORM_COLORS.axisLine;
        ctx.lineWidth = 1;
    }
}

function computeTickInterval(duration) {
    if (duration > 300) return 60;
    if (duration > 120) return 30;
    if (duration > 60) return 10;
    if (duration > 30) return 5;
    if (duration > 10) return 2;
    if (duration > 4) return 1;
    if (duration > 1) return 0.5;
    return 0.25;
}

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    if (m > 0) {
        return `${m}:${String(Math.floor(s)).padStart(2, '0')}`;
    }
    if (seconds < 10 && seconds % 1 !== 0) {
        return `${s.toFixed(1)}s`;
    }
    return `${Math.floor(s)}s`;
}

// High-precision tooltip time (2026-06-10). Used for the
// tooltip's "Time" line ONLY — the axis labels still use the
// compact formatTime() above so they don't get cluttered with
// 3-decimal digits. Format: "M:SS.mmm" when >= 60s, else
// "S.mmm s" when < 10s, else "S.mmm s". Always 3 decimal
// places so the user can A/B-compare PGA / spectral / energy
// event times against their known ground truth at sub-frame
// precision (~0.3ms per 0.001 increment at sr=44100).
function formatTimePrecise(seconds) {
    if (seconds == null || !isFinite(seconds)) return '?';
    const m = Math.floor(seconds / 60);
    const s = seconds - m * 60;
    if (m > 0) {
        return `${m}:${s.toFixed(3).padStart(6, '0')}`;
    }
    return `${s.toFixed(3)}s`;
}

/**
 * Draw energy envelope as a mirrored DAW-style waveform.
 * Left channel extends upward from center, right channel extends downward.
 */
function drawEnvelope(ctx, envelope, timeToX, PAD, plotH, maxAmp) {
    if (!envelope.times || envelope.times.length === 0) return;
    if (maxAmp <= 0) return;

    const times = envelope.times;
    const centerY = PAD.top + plotH / 2;
    const halfH = plotH / 2;

    const plotW = timeToX(times[times.length - 1]) - timeToX(times[0]);
    const step = Math.max(1, Math.floor(times.length / (Math.max(plotW, 1) * 2)));

    const hasLeft = envelope.left && envelope.left.length > 0;
    const hasRight = envelope.right && envelope.right.length > 0;

    if (hasLeft) {
        drawEnvelopeHalf(ctx, times, envelope.left, timeToX, centerY, -halfH, maxAmp, step,
            WAVEFORM_COLORS.envelopeLeft, WAVEFORM_COLORS.envelopeFillLeft);
    }
    if (hasRight) {
        drawEnvelopeHalf(ctx, times, envelope.right, timeToX, centerY, halfH, maxAmp, step,
            WAVEFORM_COLORS.envelopeRight, WAVEFORM_COLORS.envelopeFillRight);
    }
    if (hasLeft && !hasRight) {
        drawEnvelopeHalf(ctx, times, envelope.left, timeToX, centerY, halfH, maxAmp, step,
            WAVEFORM_COLORS.envelopeRight, WAVEFORM_COLORS.envelopeFillRight);
    }

    // Center line
    ctx.strokeStyle = 'rgba(107, 114, 128, 0.3)';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(PAD.left, centerY);
    const endX = timeToX(times[times.length - 1]);
    ctx.lineTo(endX, centerY);
    ctx.stroke();
}

function drawEnvelopeHalf(ctx, times, values, timeToX, centerY, direction, maxAmp, step, strokeColor, fillColor) {
    if (!values || values.length === 0) return;

    const sign = direction < 0 ? -1 : 1;
    const halfH = Math.abs(direction);

    // 60dB dynamic range log scaling
    const dynamicRange = 60;
    const noiseFloor = maxAmp * Math.pow(10, -dynamicRange / 20);
    const normalize = v => {
        if (v <= noiseFloor) return 0;
        const dB = 20 * Math.log10(v / maxAmp);
        return Math.max(0, (1 + dB / dynamicRange)) * halfH;
    };

    // Filled area
    ctx.beginPath();
    ctx.moveTo(timeToX(times[0]), centerY);
    for (let i = 0; i < times.length; i += step) {
        let maxV = values[i];
        for (let j = 1; j < step && i + j < times.length; j++) {
            maxV = Math.max(maxV, values[i + j]);
        }
        ctx.lineTo(timeToX(times[i]), centerY + sign * normalize(maxV));
    }
    const lastI = times.length - 1;
    ctx.lineTo(timeToX(times[lastI]), centerY + sign * normalize(values[lastI]));
    ctx.lineTo(timeToX(times[lastI]), centerY);
    ctx.closePath();
    ctx.fillStyle = fillColor;
    ctx.fill();

    // Stroke
    ctx.beginPath();
    for (let i = 0; i < times.length; i += step) {
        let maxV = values[i];
        for (let j = 1; j < step && i + j < times.length; j++) {
            maxV = Math.max(maxV, values[i + j]);
        }
        const x = timeToX(times[i]);
        const y = centerY + sign * normalize(maxV);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.lineTo(timeToX(times[lastI]), centerY + sign * normalize(values[lastI]));
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 1;
    ctx.stroke();
}

function drawThresholdLine(ctx, threshold, geomeanToY, PAD, plotW) {
    const y = geomeanToY(threshold);
    if (y < PAD.top - 5 || y > PAD.top + 500) return;

    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = WAVEFORM_COLORS.thresholdLine;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(PAD.left, y);
    ctx.lineTo(PAD.left + plotW, y);
    ctx.stroke();
    ctx.setLineDash([]);

    // Label in left axis margin (not overlapping data)
    ctx.fillStyle = WAVEFORM_COLORS.thresholdLine;
    ctx.font = '8px system-ui, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText('thr', PAD.left - 4, y + 3);
}

function getMarkerColor(status, classification, hihatState = null) {
    switch (status) {
        case 'KEPT':
            // Check for hihat open/closed classification first
            if (hihatState === 'open') {
                return HIHAT_OPEN_COLOR;
            }
            if (hihatState === 'closed') {
                return HIHAT_CLOSED_COLOR;
            }
            // Fall back to classification index colors
            if (classification != null && CLASSIFICATION_COLORS[classification]) {
                return CLASSIFICATION_COLORS[classification];
            }
            return WAVEFORM_COLORS.markerKept;
        case 'FILTERED': return WAVEFORM_COLORS.markerFiltered;
        case 'REVERB_CONTINUATION': return WAVEFORM_COLORS.markerReverbCont;
        default: return WAVEFORM_COLORS.markerUnknown;
    }
}

/**
 * Resolve the waveform bar color for a single event.
 *
 * Precedence (highest first):
 *   1. FILTERED → red
 *   2. REVERB_CONTINUATION → orange
 *   3. hihat open/closed → dedicated colors (when classification
 *      overlay is enabled)
 *   4. method === 'percentile_gated' → violet (PGA)
 *   5. classification index → classification palette (when
 *      classification overlay is enabled)
 *   6. else → markerKept (green)
 *
 * Steps 1-3 are intentional: filter status and hihat open/closed are
 * visual identities that the user has learned to look for, so they
 * win over the classification distinction.
 *
 * Args:
 *   event: A single event dict (the same shape as items in
 *          ``events_configured``). Must have a ``status`` field
 *          and may have ``method``, ``classification``,
 *          ``hihat_state``, etc.
 *
 * Returns:
 *   A CSS color string. Always present, never null.
 */
function getEventColor(event) {
    if (!event) return WAVEFORM_COLORS.markerUnknown;
    const status = event.status;
    // Non-KEPT statuses win immediately so the user can always spot
    // a filtered or reverb-continuation event in the noise.
    if (status === 'FILTERED') return WAVEFORM_COLORS.markerFiltered;
    if (status === 'REVERB_CONTINUATION') return WAVEFORM_COLORS.markerReverbCont;
    if (status !== 'KEPT') return WAVEFORM_COLORS.markerUnknown;
    // 2026-06-19: per-stem classification color overlay. Gated
    // on the per-stem toggle (default true). When the user
    // flips the "Open/Closed Classification" toggle off, every
    // KEPT event falls through to the default green (or its
    // method-based color) so the per-classification color
    // disappears. The toggle is dispatched by
    // threshold-tuning.js via a 'larsnet:classification-toggle'
    // window event; classificationEnabledByStem is updated by
    // the listener above. The default true (== 'on') matches
    // the threshold-tuning default.
    const classEnabled = classificationEnabledByStem[waveformActiveStem] !== false;
    if (classEnabled) {
        // Hihat open/closed is a hit-type identity, but only relevant for
        // the hihat stem. Kept as a per-stem override.
        if (event.hihat_state === 'open') return HIHAT_OPEN_COLOR;
        if (event.hihat_state === 'closed') return HIHAT_CLOSED_COLOR;
    }
    // 2026-06-30: classification wins over the PGA-method marker.
    // Previously the PGA check came first, so every PGA event
    // rendered violet regardless of classification — that hid
    // the 3 different snare classes (cls 0, 1, 2 → notes 38,
    // 37, 39) under a single violet color. Swapping the order
    // means: a PGA event with a classification index renders
    // in the classification palette (snare 0 = green, 1 =
    // purple, 2 = cyan), a PGA event WITHOUT a classification
    // (e.g. kick, toms) falls through to the violet marker.
    // The classification palette is the visual identity the
    // user has learned to look for on the tuning panel; the
    // violet marker is a fallback for events that have no
    // classification metadata.
    if (classEnabled
        && event.classification != null
        && CLASSIFICATION_COLORS[event.classification]) {
        return CLASSIFICATION_COLORS[event.classification];
    }
    // Percentile-gated broad-attack events get the dedicated violet
    // (2026-06-10). PGA is a THIRD complementary detector; it runs
    // alongside energy + spectral but isn't part of the
    // energy-vs-spectral A/B comparison. The violet is the
    // fallback for PGA events that DON'T have a classification
    // (i.e. stems with k-means disabled or single-class output).
    if (event.method === 'percentile_gated') {
        return WAVEFORM_COLORS.markerPga;
    }
    return WAVEFORM_COLORS.markerKept;
}

// ─── Legend Bar (HTML, outside canvas) ────────────────────────────────────

function updateLegendBar(stemData, displayEvents, pgaEvents) {
    // Tuning indicator
    const tuningLabel = document.getElementById('waveform-tuning-label');
    if (tuningLabel) {
        tuningLabel.classList.toggle('hidden', !waveformTuningActive);
    }

    // Legend items
    const container = document.getElementById('waveform-legend-items');
    if (!container) return;

    const events = (waveformTuningActive && waveformTuningEvents)
        ? waveformTuningEvents
        : getEventsForStem(stemData);
    const keptEvents = events.filter(e => e.status === 'KEPT');
    const filteredCount = events.filter(e => e.status === 'FILTERED').length;
    const reverbCount = events.filter(e => e.status === 'REVERB_CONTINUATION').length;
    // 2026-06-22: sensitiveCount removed (the gray "Sensitive (N)"
    // legend entry and the gray overlay layer are both gone).

    const items = [];

    // Group KEPT events by classification index for color-coded legend
    // For hihat stem, also group by hihat_state (open/closed)
    const classGroups = {};
    const hihatOpenGroups = { open: 0, closed: 0 };
    const isHihat = waveformActiveStem === 'hihat';

    for (const e of keptEvents) {
        // PGA events (method='percentile_gated') are surfaced as a
        // single "PGA (N)" legend entry below; skip them in the
        // per-classification grouping so the user's "Type 1/2/3"
        // grouping (which is a per-stem k-means classification
        // label, not a method tag) doesn't show phantom Type entries
        // for toms. Toms is PGA-only (2026-06-12) — its
        // events_configured list contains only method=
        // 'percentile_gated' events, and the per-classification
        // grouping is meaningless for that single-method view.
        // 2026-06-19: PGA events for hihat ARE the
        // per-classification events (hihats use a per-event
        // hihat_state label, not a k-means cluster label).
        // Falling through to the hihat branch lets them count
        // toward hihatOpenGroups. For other stems the PGA
        // events should still skip — their per-classification
        // grouping uses k-means (cluster_id) and the "PGA (N)"
        // entry below already covers them.
        if (e.method === 'percentile_gated' && !isHihat) continue;
        // Check for hihat open/closed classification first
        if (isHihat && e.hihat_state) {
            hihatOpenGroups[e.hihat_state]++;
        } else {
            const cls = e.classification != null ? e.classification : 0;
            if (!classGroups[cls]) classGroups[cls] = 0;
            classGroups[cls]++;
        }
    }

    // 2026-06-19: gate the per-classification legend on the
    // toggle. When the toggle is off, the per-classification
    // legend entries (open/closed for hihat, Type N for other
    // stems) are not surfaced — instead a single "Kept (N)"
    // entry appears so the user can still see the event count
    // but doesn't see colors that aren't actually being
    // applied to the events on the canvas. Same default-on
    // behavior as getEventColor() above.
    const classEnabled = classificationEnabledByStem[waveformActiveStem] !== false;
    if (classEnabled && isHihat && (hihatOpenGroups.open > 0 || hihatOpenGroups.closed > 0)) {
        if (hihatOpenGroups.open > 0) {
            items.push({ color: HIHAT_OPEN_COLOR, label: `🔓 Open (${hihatOpenGroups.open})` });
        }
        if (hihatOpenGroups.closed > 0) {
            items.push({ color: HIHAT_CLOSED_COLOR, label: `🔒 Closed (${hihatOpenGroups.closed})` });
        }
    } else {
        const classKeys = Object.keys(classGroups).map(Number).sort();
        if (classKeys.length <= 1 || !classEnabled) {
            // Single classification (or no data) — show simple "Kept (N)".
            // 2026-06-19: also reached when the classification
            // toggle is off; the user sees event count but not
            // a per-classification breakdown.
            if (keptEvents.length > 0) {
                const cls = classKeys.length === 1 ? classKeys[0] : 0;
                items.push({
                    color: CLASSIFICATION_COLORS[cls] || WAVEFORM_COLORS.markerKept,
                    label: `Kept (${keptEvents.length})`
                });
            }
        } else {
            // Multiple classifications — show each with its color
            for (const cls of classKeys) {
                const color = CLASSIFICATION_COLORS[cls] || WAVEFORM_COLORS.markerKept;
                items.push({ color, label: `Type ${cls + 1} (${classGroups[cls]})` });
            }
        }
    }

    // PGA legend entry (2026-06-10). The third complementary
    // detector — always shown (not gated on an overlay flag)
    // because PGA is its own signal, not an A/B candidate.
    // Sourced from the sidecar's events_pga list (not from the
    // configured/sensitive lists above) so the count reflects
    // what the user is actually seeing as violet bars on the
    // canvas. Falls back to 0 when the sidecar predates the PGA
    // detector (pgaEvents undefined on older analyses).
    const pgaCount = pgaEvents ? pgaEvents.length : 0;
    if (pgaCount > 0) {
        items.push({
            color: WAVEFORM_COLORS.markerPga,
            label: `PGA (${pgaCount})`,
            title: "Violet = percentile-gated broad-attack (method='percentile_gated'); broadband percussive onset, fires independently of the energy/RING signal",
        });
    }

    // 2026-06-22: "Filtered (N)" legend entry is now gated on
    // the user-controlled "Show Filtered" checkbox, NOT on the
    // Tune panel. The count only makes sense when the red
    // filtered bars are actually drawn — otherwise the legend
    // would dangle and confuse ("where are the red bars?").
    // "Reverb cont. (N)" stays unconditionally visible (those
    // events are always drawn faded, regardless of the panel
    // or the filtered toggle). The "Sensitive (N)" gray entry
    // is removed entirely — the gray overlay layer is dead UI.
    if (filteredCount > 0 && waveformShowFiltered) {
        items.push({ color: WAVEFORM_COLORS.markerFiltered, label: `Filtered (${filteredCount})` });
    }
    if (reverbCount > 0) {
        items.push({ color: WAVEFORM_COLORS.markerReverbCont, label: `Reverb cont. (${reverbCount})` });
    }

    container.innerHTML = items.map(item =>
        `<span class="flex items-center gap-1"${item.title ? ` title="${item.title}"` : ''}>
            <span class="inline-block w-2 h-2 rounded-full" style="background:${item.color}"></span>
            <span class="text-gray-300">${item.label}</span>
        </span>`
    ).join('');
}

// ─── Crosshair ───────────────────────────────────────────────────────────

function drawCrosshair(ctx, canvas) {
    if (waveformMouseX == null) return;

    const rect = canvas.parentElement.getBoundingClientRect();
    const H = rect.height;

    ctx.save();
    ctx.strokeStyle = WAVEFORM_COLORS.crosshair;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(waveformMouseX, 0);
    ctx.lineTo(waveformMouseX, H);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
}

/**
 * Draw a solid playback position indicator line on a canvas.
 * Visually distinct from the dotted crosshair: solid, bright cyan, with glow.
 */
function drawPlaybackIndicator(ctx, canvas, x) {
    const rect = canvas.parentElement.getBoundingClientRect();
    const H = rect.height;

    ctx.save();
    // Glow effect
    ctx.shadowColor = WAVEFORM_COLORS.playbackLine;
    ctx.shadowBlur = 6;
    ctx.strokeStyle = WAVEFORM_COLORS.playbackLine;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
    ctx.restore();
}

/**
 * Get the current playback song time based on AudioContext timing.
 */
function getCurrentPlaybackTime() {
    if (!audioIsPlaying || audioPlaybackTime == null || audioStartContextTime == null || !audioCtx) {
        return null;
    }
    const elapsed = audioCtx.currentTime - audioStartContextTime;
    return audioPlaybackTime + elapsed;
}

/**
 * Animation loop that redraws the playback indicator at ~60fps.
 */
function animatePlaybackIndicator() {
    if (!audioIsPlaying) {
        playbackAnimFrameId = null;
        drawWaveform(); // Final redraw to clear the indicator
        return;
    }

    const currentTime = getCurrentPlaybackTime();
    if (currentTime != null) {
        const x = timeToCanvasX(currentTime);
        // Redraw waveform (clears old indicator) then overlay the indicator
        drawWaveform();
        if (x != null) {
            drawPlaybackIndicator(envelopeCtx, envelopeCanvas, x);
            drawPlaybackIndicator(eventsCtx, eventsCanvas, x);
        }
    }

    playbackAnimFrameId = requestAnimationFrame(animatePlaybackIndicator);
}

// ─── Tooltip ─────────────────────────────────────────────────────────────

// Tooltip is a DOM div (#waveform-tooltip) inside the
// .waveform-panels-container, positioned absolutely with z-index 30
// so it sits above both the envelope and events canvases. The
// previous canvas-drawing version was clipped to the events canvas
// (120px tall), which cut off the top of tall spectral tooltips
// (192px) when the cursor was near the top of the events canvas.
let _tooltipEl = null;
function getTooltipEl() {
    if (!_tooltipEl) _tooltipEl = document.getElementById('waveform-tooltip');
    return _tooltipEl;
}

function hideTooltip() {
    const el = getTooltipEl();
    if (el) el.style.display = 'none';
}

function drawTooltip(event, W, H) {
    const el = getTooltipEl();
    if (!el) return;

    // Build the same diagnostic content the old canvas-draw
    // version had. Lines are kept as an array so we can compute
    // the tooltip height accurately (used for vertical centering
    // within the panels container).
    const lines = [];
    lines.push(`Time: ${formatTimePrecise(event.time)}`);
    lines.push(`Status: ${event.status}`);
    // Detection method: 'spectral' = spectral-transient detector,
    // everything else (rms, peak_hold, spectral_flux, etc.) is the
    // energy detector. The two map to magenta / green on the bars;
    // surfacing it here makes the color choice explainable.
    if (event.method != null) {
        const methodLabel = event.method === 'spectral'
            ? 'spectral (magenta)'
            : `${event.method} (energy)`;
        lines.push(`Method: ${methodLabel}`);
    }
    if (event.velocity != null) lines.push(`Velocity: ${event.velocity}`);
    if (event.note != null) {
        lines.push(`Note: ${event.note}`);
    }
    if (event.classification != null) {
        lines.push(`Type: ${event.classification + 1}`);
    }
    if (event.hihat_state != null) lines.push(`Hi-hat: ${event.hihat_state}`);
    // For spectral events, show the full per-band profile so the
    // user can troubleshoot detection issues at a glance. The 5
    // user-specified bands are: 60-200Hz (low/kick), 200-600Hz
    // (toms/snare body), 600-1200Hz (snare/hi-hat fund.),
    // 1200-2400Hz (snare wire/hi-hat edge/cymbal edge),
    // 2400-8000Hz (hi-hat sizzle/cymbal body).
    // band_max_ratio = top band / second-highest band (clear
    // band-dominance signature of a strike; ~1.0 means
    // broadband/decay). Strength is the same value normalized to
    // [0, 1] by min(1, ratio/10).
    if (event.method === 'spectral') {
        if (Array.isArray(event.band_powers) && event.band_powers.length === 5) {
            const bandLabels = [
                'B0 60-200Hz',
                'B1 200-600Hz',
                'B2 600-1200Hz',
                'B3 1200-2400Hz',
                'B4 2400-8000Hz',
            ];
            for (let i = 0; i < 5; i++) {
                const marker = i === event.band_max_idx ? ' *' : '  ';
                lines.push(`${bandLabels[i]}${marker}: ${event.band_powers[i].toExponential(2)}`);
            }
        }
        if (event.band_max_idx != null) {
            lines.push(`Top band: B${event.band_max_idx}`);
        }
        if (event.band_max_ratio != null) {
            lines.push(`Top/2nd ratio: ${event.band_max_ratio.toFixed(2)} (higher = clearer strike)`);
        }
        if (event.band_delta != null) {
            lines.push(`Ring Δ (max-median, all bands): ${event.band_delta.toFixed(2)}`);
        }
        if (event.snap_delta != null) {
            lines.push(`Snap Δ (min of snap_bands): ${event.snap_delta.toFixed(4)}`);
        }
        // Derived ratios (2026-06-10). Both are diagnostic — they
        // tell the user WHY an event fired or why it was filtered.
        // - snap_to_ring_ratio: snap/band_delta. Low values mean the
        //   broadband attack (snap) is much weaker than the sustained
        //   ring — typical of wire-tail / decay events. The user's
        //   calibration case (ring=665, snap=0.01) gives ~0.000015.
        // - snap_to_top_ratio: snap/band_max_ratio. How the snap
        //   compares to the top-band dominance metric. Close to 1.0
        //   means the snap is roughly as strong as the band peak
        //   (real hit); low means the band-dominance is in a
        //   non-snap band (sustained ring without attack).
        if (event.snap_to_ring_ratio != null) {
            lines.push(`Snap/Ring ratio: ${event.snap_to_ring_ratio.toExponential(2)} (lower = weaker attack than sustain)`);
        }
        if (event.snap_to_top_ratio != null) {
            lines.push(`Snap/Top ratio: ${event.snap_to_top_ratio.toExponential(2)} (closer to 1 = real hit)`);
        }
        // 2026-06-10: the lossy "Strength (ratio/10)" line was
        // removed — the "Top/2nd ratio" line directly above shows
        // the raw band_max_ratio, which is what the user actually
        // wants to see (the clamp-to-1.0 strength field masked
        // everything >= 10). For energy events, fall through to
        // the strength display below — that strength is the
        // energy detector's [0, 1] normalized onset strength, not
        // the spectral one, so it's still meaningful.
    } else if (event.strength != null) {
        lines.push(`Strength: ${event.strength}`);
    }
    // PGA (percentile-gated broad attack) diagnostic fields
    // (2026-06-10). The PGA detector measures broadband change
    // above a per-bin noise floor and surfaces its key signals
    // here so the user can see WHY a violet bar appeared (or
    // didn't) and A/B-compare with the energy/spectral signals
    // at the same time point. All four fields are diagnostic
    // and don't gate the configured pipeline.
    if (event.method === 'percentile_gated') {
        if (event.frame != null) lines.push(`Frame: ${event.frame}`);
        if (event.envelope_value != null) lines.push(`Envelope value: ${event.envelope_value.toFixed(2)} (contrast sum, 600-8000 Hz)`);
        if (event.prominence != null) lines.push(`Prominence: ${event.prominence.toFixed(2)} (find_peaks)`);
        if (event.iqr_threshold != null) lines.push(`IQR threshold: ${event.iqr_threshold.toFixed(2)} (q3 + 2.5*IQR of envelope)`);
        if (event.envelope_value != null && event.iqr_threshold != null) {
            const ratio = event.envelope_value / event.iqr_threshold;
            lines.push(`Envelope / IQR thr: ${ratio.toFixed(2)}× (higher = more confident strike)`);
        }
        // Per-event classification features (2026-06-10).
        // These are the inputs the eventual classifier
        // will consume. For now they're displayed so the
        // user can see WHY a strike is or isn't a real hit.
        // Classifier ranges (typical, not absolute):
        //   duration_ms:  click<30, hihat 30-100, snare 80-300,
        //                 kick 200-800, toms 300-1500, cymbals 500+
        //   attack_rise_ms:  click 1-3, stick 3-10, mallet 10-30
        //   pitch_hz:       kick 40-80, toms 80-200, snare 150-300
        //   decay_t60_ms:  closed hihat 30-80, open hihat 200-400,
        //                  snare 80-250, toms 300-800, cymbals 800-2000
        //   spectral_centroid_hz:  kick/toms 200-1500, snare 800-3000,
        //                         cymbals/hihat 4000-8000
        //   spectral_flatness:  real strikes tend low (tonal:
        //                       fundamental + harmonics); broadband
        //                       "pop" / "click" artifacts tend high
        //                       (noise-like). Diagnostic only —
        //                       not used as a filter.
        if (event.duration_ms != null) lines.push(`Duration: ${event.duration_ms.toFixed(1)} ms (ring time, slope-of-decline)`);
        if (event.attack_rise_ms != null) lines.push(`Attack rise: ${event.attack_rise_ms.toFixed(1)} ms (10-90% of peak)`);
        if (event.inter_onset_ms != null) lines.push(`Inter-onset: ${event.inter_onset_ms.toFixed(1)} ms (time to next event)`);
        if (event.pitch_hz != null) {
            const confStr = event.pitch_confidence != null ? ` (conf ${event.pitch_confidence.toFixed(2)})` : '';
            lines.push(`Root pitch: ${event.pitch_hz.toFixed(1)} Hz${confStr} (pYIN on body)`);
        }
        if (event.decay_t60_ms != null) lines.push(`Decay T60: ${event.decay_t60_ms.toFixed(0)} ms (200-4000Hz band)`);
        if (event.spectral_centroid_hz != null) lines.push(`Centroid: ${event.spectral_centroid_hz.toFixed(0)} Hz (brightness)`);
        if (event.spectral_flatness != null) lines.push(`Flatness: ${event.spectral_flatness.toFixed(4)} (600-3000Hz attack region; 0=tonal, 1=noise-like)`);
        // High-res attack+decay signature (2026-06-11).
        // Different STFT (n_fft=128, hop=4) than the rest
        // of the pipeline. Used to distinguish real strikes
        // (sustained decaying ring) from pop/gap artifacts
        // (no ring). Diagnostic only — not a filter.
        //   hr_peak_offset_ms:        how late the high-res
        //                             peak is vs the PGA
        //                             report (5-11ms typical
        //                             for real strikes; FPs
        //                             can be anywhere)
        //   decay_envelope_energy:    ring energy in 15ms
        //                             post-peak (FPs <60K,
        //                             real >60K on project 4)
        //   decay_col_min_median_db:  broadband level in the
        //                             decay window (FPs -84
        //                             to -90 dB; real -60 to
        //                             -84 dB)
        if (event.hr_peak_offset_ms != null) lines.push(`HR peak offset: ${event.hr_peak_offset_ms >= 0 ? '+' : ''}${event.hr_peak_offset_ms.toFixed(1)} ms (n_fft=128/hop=4 peak vs PGA time)`);
        if (event.decay_envelope_energy != null) lines.push(`Decay envelope: ${event.decay_envelope_energy.toFixed(0)} (15ms post-peak ring energy; FPs <60K, real >60K)`);
        if (event.decay_col_min_median_db != null) lines.push(`Decay col_min: ${event.decay_col_min_median_db.toFixed(1)} dB (15ms post-peak broadband floor; FPs -84 to -90, real -60 to -84)`);
        // Toms cleanup (2026-06-11): filter status, midi
        // velocity (the value that lands in the MIDI file),
        // and the filter reason. Filtered events are kept
        // in the sidecar so the user can see them faded
        // when the analysis panel is open.
        if (event.status != null) {
            // 2026-06-22: FILTERED visibility is controlled by
            // the "Show Filtered" checkbox, not the panel.
            // Reflect current visibility in the tooltip.
            const statusTag = event.status === 'FILTERED'
                ? (waveformShowFiltered
                    ? ' (faded — toggle "Show Filtered" to hide)'
                    : ' (hidden — toggle "Show Filtered" to reveal)')
                : '';
            lines.push(`Status: ${event.status}${statusTag}`);
        }
        if (event.midi_velocity != null) lines.push(`MIDI velocity: ${event.midi_velocity} (PGA envelope → [min, max] from settings)`);
        if (event.filter_reason != null) lines.push(`Filter reason: ${event.filter_reason}`);
        if (event.pga_filter_config != null) {
            const cfg = event.pga_filter_config;
            lines.push(`Active filter: pga_min_prominence=${cfg.pga_min_prominence}, velocity=[${cfg.min_velocity}, ${cfg.max_velocity}]`);
        }
    }
    if (event.geomean != null) lines.push(`Geomean: ${event.geomean}`);
    if (event.amplitude != null) lines.push(`Amplitude: ${event.amplitude}`);
    if (event.total_energy != null) lines.push(`Total energy: ${event.total_energy}`);
    if (event.sustain_ms != null) lines.push(`Sustain: ${event.sustain_ms}ms`);
    if (event.stereo_width != null) lines.push(`Stereo width: ${event.stereo_width.toFixed(3)}`);
    if (event.pan_confidence != null) lines.push(`Pan: ${event.pan_confidence.toFixed(3)}`);

    // Render as <div> children — each line on its own row. Using a
    // <div> per line (instead of <br> in one innerHTML) keeps
    // future styling hooks (e.g. coloring the band_max_idx line)
    // trivial.
    el.innerHTML = '';
    for (const line of lines) {
        const div = document.createElement('div');
        div.textContent = line;
        el.appendChild(div);
    }
    el.style.display = 'block';

    // Position. _mouseX / _mouseY are in the events-canvas
    // coordinate space; translate to the panels-container space by
    // adding the events canvas's offset within the container.
    // (The events canvas is the bottom panel, so its top is
    // envelope height below the container top.)
    const lineH = 16;
    const pad = 8;
    const containerRect = el.parentElement.getBoundingClientRect();
    const eventsRect = eventsCanvas.getBoundingClientRect();
    const offsetX = eventsRect.left - containerRect.left;
    const offsetY = eventsRect.top - containerRect.top;

    const tooltipW = el.offsetWidth;
    const tooltipH = lines.length * lineH + pad * 2;

    // Convert from events-canvas-relative mouse coords to
    // container-relative tooltip coords.
    let tx = offsetX + event._mouseX + 12;
    let ty = offsetY + event._mouseY - tooltipH / 2;
    // Keep on screen within the container, allowing the tooltip to
    // extend ABOVE the events panel (into the envelope area) when
    // needed — this is the whole reason we moved off canvas.
    if (tx + tooltipW > containerRect.width) tx = offsetX + event._mouseX - tooltipW - 12;
    if (ty < 0) ty = 4;
    if (ty + tooltipH > containerRect.height) ty = containerRect.height - tooltipH - 4;

    el.style.left = tx + 'px';
    el.style.top = ty + 'px';
}

function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.arcTo(x + w, y, x + w, y + r, r);
    ctx.lineTo(x + w, y + h - r);
    ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
    ctx.lineTo(x + r, y + h);
    ctx.arcTo(x, y + h, x, y + h - r, r);
    ctx.lineTo(x, y + r);
    ctx.arcTo(x, y, x + r, y, r);
    ctx.closePath();
}

// ─── Mouse Interaction ───────────────────────────────────────────────────

function setupCanvasInteraction(canvas) {
    canvas.addEventListener('mousemove', onCanvasMouseMove);
    canvas.addEventListener('mouseleave', onCanvasMouseLeave);
    canvas.addEventListener('wheel', onCanvasWheel, { passive: false });
    canvas.addEventListener('mousedown', onCanvasDragStart);
}

function onCanvasMouseMove(e) {
    if (!waveformActiveStem || !waveformAnalysisData) return;

    const canvasRect = e.target.parentElement.getBoundingClientRect();
    const mouseX = e.clientX - canvasRect.left;
    const mouseY = e.clientY - canvasRect.top;

    // Crosshair X (same for both canvases since they share width)
    waveformMouseX = mouseX;

    // Handle drag (pan)
    if (waveformIsDragging) {
        const plotW = canvasRect.width - EVT_PAD.left - EVT_PAD.right;
        const dx = (mouseX - waveformDragStartX) / plotW;
        waveformPanOffset = waveformDragStartPan - dx;
        clampPan();
        drawWaveform();
        return;
    }

    // Event hit testing (on events panel only)
    const isEventsPanel = e.target === eventsCanvas;
    if (isEventsPanel) {
    const stemData = waveformAnalysisData.stems[waveformActiveStem];
    const configuredEvents = getEventsForStem(stemData);
    const sensitiveEvents = getSensitiveEventsForStem(stemData);
    // 2026-06-15: in tuning mode, hit-test against the live tuning
    // events so the user can hover over faded bars that are about
    // to become KEPT as they drag the slider. See drawWaveform for
    // the same logic in the render path.
    const pgaEvents = (typeof waveformTuningActive !== 'undefined' && waveformTuningActive && waveformTuningEvents)
        ? waveformTuningEvents
        : getPgaEventsForStem(stemData);
    const envelope = waveformEnvelopeCache[waveformActiveStem];

        const { tMin: tMinFull, tMax: tMaxFull } = computeTimeRange(configuredEvents, sensitiveEvents, envelope, pgaEvents);
        const { tMin, tMax } = computeVisibleRange(tMinFull, tMaxFull);

        const PAD = EVT_PAD;
        const plotW = canvasRect.width - PAD.left - PAD.right;
        const xToTime = x => tMin + ((x - PAD.left) / plotW) * (tMax - tMin);
        const mouseTime = xToTime(mouseX);
        const hitRadius = (tMax - tMin) / plotW * 5;

        const displayEvents = (waveformTuningActive && waveformTuningEvents)
            ? waveformTuningEvents
            : configuredEvents;
        // PGA events participate in hover hit-testing (2026-06-10)
        // — clicking near a violet marker surfaces its info in
        // 2026-06-22: the sensitive-events pool is no longer
        // included in the hover hit-test — the gray overlay
        // that drew them is gone, so hovering an empty area
        // would yield confusing tooltips. PGA events stay
        // (they're always drawn as violet bars).
        const allEvents = displayEvents.concat(pgaEvents);

        let closest = null;
        let closestDist = Infinity;
        for (const evt of allEvents) {
            if (evt.time == null) continue;
            const dist = Math.abs(evt.time - mouseTime);
            if (dist < hitRadius && dist < closestDist) {
                closestDist = dist;
                closest = evt;
            }
        }

        if (closest) {
            waveformHoverEvent = { ...closest, _mouseX: mouseX, _mouseY: mouseY };
        } else {
            waveformHoverEvent = null;
        }
    } else {
        waveformHoverEvent = null;
    }

    // Set cursor
    const cursorStyle = waveformIsDragging ? 'grabbing' : (waveformZoom > 1 ? 'grab' : 'crosshair');
    if (envelopeCanvas) envelopeCanvas.style.cursor = cursorStyle;
    if (eventsCanvas) eventsCanvas.style.cursor = waveformHoverEvent ? 'crosshair' : cursorStyle;

    drawWaveform();
}

function onCanvasMouseLeave() {
    waveformMouseX = null;
    waveformHoverEvent = null;
    if (envelopeCanvas) envelopeCanvas.style.cursor = 'default';
    if (eventsCanvas) eventsCanvas.style.cursor = 'default';
    drawWaveform();
}

function onCanvasWheel(e) {
    e.preventDefault();
    if (!waveformActiveStem) return;

    const canvasRect = e.target.parentElement.getBoundingClientRect();
    const mouseX = e.clientX - canvasRect.left;
    const PAD = EVT_PAD;
    const plotW = canvasRect.width - PAD.left - PAD.right;

    // Mouse position as fraction of visible plot
    const mouseFrac = Math.max(0, Math.min(1, (mouseX - PAD.left) / plotW));

    const oldZoom = waveformZoom;
    const zoomFactor = e.deltaY < 0 ? 1.25 : 1 / 1.25;
    waveformZoom = Math.max(1, Math.min(100, waveformZoom * zoomFactor));

    // Adjust pan so the time under the mouse stays in place
    if (waveformZoom > 1) {
        const oldVisibleFrac = 1 / oldZoom;
        const newVisibleFrac = 1 / waveformZoom;
        const maxOldStart = 1 - oldVisibleFrac;
        const oldStart = maxOldStart > 0 ? waveformPanOffset * maxOldStart : 0;
        const timeAtMouse = oldStart + mouseFrac * oldVisibleFrac;
        const newStart = timeAtMouse - mouseFrac * newVisibleFrac;
        const maxNewStart = 1 - newVisibleFrac;
        waveformPanOffset = maxNewStart > 0 ? newStart / maxNewStart : 0;
    } else {
        waveformPanOffset = 0;
    }

    clampPan();
    drawWaveform();
}

function onCanvasDragStart(e) {
    const canvasRect = e.target.parentElement.getBoundingClientRect();
    const mouseX = e.clientX - canvasRect.left;
    const startX = mouseX;
    let hasMoved = false;

    // When zoomed in: distinguish between hold (audio playback) and drag (pan)
    // Audio starts immediately on mousedown; if user drags, we stop audio and pan instead.
    if (waveformZoom > 1) {
        waveformIsDragging = false;
        waveformDragStartX = mouseX;
        waveformDragStartPan = waveformPanOffset;
        let startedAudio = false;

        // Check for event bar toggle first (instant, no hold needed)
        const isEventsPanel = e.target === eventsCanvas;
        if (isEventsPanel && waveformActiveStem) {
            const hitEvent = hitTestEvent(mouseX);
            if (hitEvent) {
                cycleEventOverride(waveformActiveStem, hitEvent);
                return;
            }
        }

        // Start audio playback immediately on mousedown
        if (waveformActiveStem) {
            const clickTime = canvasXToTime(mouseX);
            if (clickTime != null && clickTime >= 0) {
                ensureAudioBuffer(waveformActiveStem).then(buffer => {
                    if (hasMoved) return; // User started dragging before buffer loaded
                    if (!buffer) return;
                    startAudioPlayback(waveformActiveStem, clickTime);
                    startedAudio = true;
                });
            }
        }

        const onMove = (me) => {
            const rect = e.target.parentElement.getBoundingClientRect();
            const mx = me.clientX - rect.left;
            
            // If moved more than 3 pixels, switch to drag mode
            if (Math.abs(mx - startX) > 3) {
                if (!hasMoved) {
                    // First time crossing threshold: stop audio if it started
                    hasMoved = true;
                    if (startedAudio) {
                        stopAudioPlayback();
                        startedAudio = false;
                    }
                }
                waveformIsDragging = true;
                const plotW = rect.width - EVT_PAD.left - EVT_PAD.right;
                const dx = (mx - waveformDragStartX) / plotW;
                waveformPanOffset = waveformDragStartPan - dx;
                clampPan();
                waveformMouseX = me.clientX - rect.left;
                drawWaveform();
            }
        };

        const onUp = () => {
            waveformIsDragging = false;
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
            const cursorStyle = waveformZoom > 1 ? 'grab' : 'crosshair';
            if (envelopeCanvas) envelopeCanvas.style.cursor = cursorStyle;
            if (eventsCanvas) eventsCanvas.style.cursor = cursorStyle;

            // Stop audio on mouse release
            if (startedAudio) {
                stopAudioPlayback();
            }
        };

        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);

        if (envelopeCanvas) envelopeCanvas.style.cursor = 'grabbing';
        if (eventsCanvas) eventsCanvas.style.cursor = 'grabbing';
        return;
    }

    // When not zoomed: check for event click (toggle) or audio playback
    if (!waveformActiveStem) return;

    // On events canvas: check if clicking on an event bar to toggle override
    const isEventsPanel = e.target === eventsCanvas;
    if (isEventsPanel) {
        const hitEvent = hitTestEvent(mouseX);
        if (hitEvent) {
            cycleEventOverride(waveformActiveStem, hitEvent);
            return;
        }
    }

    // Click-and-hold on empty area: play audio from cursor position
    const clickTime = canvasXToTime(mouseX);
    if (clickTime == null || clickTime < 0) return;

    console.log('Click-to-play at time:', clickTime.toFixed(2), 'seconds');
    ensureAudioBuffer(waveformActiveStem).then(buffer => {
        if (!buffer) {
            console.warn('No audio buffer available for playback');
            return;
        }
        startAudioPlayback(waveformActiveStem, clickTime);
    });

    const onUp = () => {
        stopAudioPlayback();
        document.removeEventListener('mouseup', onUp);
    };
    document.addEventListener('mouseup', onUp);
}

// ─── Audio Playback (Click-and-Hold) ─────────────────────────────────────

/**
 * Fetch and decode stem audio into an AudioBuffer (cached per stem).
 */
async function ensureAudioBuffer(stemType) {
    if (audioBufferCache[stemType]) {
        console.log('Using cached audio buffer for', stemType);
        return audioBufferCache[stemType];
    }
    
    if (!currentProject) {
        console.error('No currentProject available for audio buffer');
        return null;
    }
    
    if (!currentProject.files) {
        console.error('No files in currentProject:', currentProject);
        return null;
    }

    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        console.log('Created audio context, state:', audioCtx.state);
    }

    // Find the stem filename from project files (e.g. "SongName-kick.wav" or "SongName.kick.wav")
    // Try cleaned folder first, then stems folder (matches the CLI in
    // stems_to_midi_cli.py — both prefer the processed/cleaned audio
    // over the raw separated output when available).
    const stemFiles = currentProject.files.stems || [];
    const cleanedFiles = currentProject.files.cleaned || [];
    console.log('Available stem files:', stemFiles);
    console.log('Available cleaned files:', cleanedFiles);

    // Match both dash and dot patterns: "file-kick.wav" or "file.kick.wav"
    let stemFile = cleanedFiles.find(f => f.includes(`-${stemType}.`) || f.includes(`.${stemType}.`));
    let fileType = 'cleaned';

    if (!stemFile) {
        stemFile = stemFiles.find(f => f.includes(`-${stemType}.`) || f.includes(`.${stemType}.`));
        fileType = 'stems';
    }
    
    if (!stemFile) {
        console.error('No stem file found for type:', stemType, 'in stems or cleaned folders');
        return null;
    }
    console.log('Found stem file:', stemFile, 'for type:', stemType, 'in', fileType, 'folder');

    const url = `/api/projects/${currentProject.number}/download/${fileType}/${stemFile}`;
    console.log('Loading audio buffer for', stemType, 'from', url);
    try {
        const response = await fetch(url);
        if (!response.ok) {
            console.warn('Audio fetch failed:', response.status, response.statusText);
            return null;
        }
        const arrayBuf = await response.arrayBuffer();
        console.log('Decoding audio buffer, size:', arrayBuf.byteLength, 'bytes');
        const audioBuf = await audioCtx.decodeAudioData(arrayBuf);
        audioBufferCache[stemType] = audioBuf;
        console.log('Audio buffer ready, duration:', audioBuf.duration.toFixed(2), 'seconds');
        return audioBuf;
    } catch (err) {
        console.error('Audio buffer load failed for', stemType, err);
        return null;
    }
}

/**
 * Start audio playback from a given time (in seconds within the song).
 */
function startAudioPlayback(stemType, startTime) {
    stopAudioPlayback();

    const buffer = audioBufferCache[stemType];
    if (!buffer || !audioCtx) return;

    // Resume context if suspended (autoplay policy)
    if (audioCtx.state === 'suspended') {
        console.log('Resuming suspended audio context');
        audioCtx.resume();
    }

    audioSource = audioCtx.createBufferSource();
    audioSource.buffer = buffer;
    audioSource.connect(audioCtx.destination);
    audioSource.onended = () => {
        // Auto-stop indicator when audio reaches end of buffer
        audioIsPlaying = false;
        audioPlaybackTime = null;
        audioStartContextTime = null;
    };

    const offset = Math.max(0, Math.min(startTime, buffer.duration - 0.01));
    console.log('Starting audio playback at offset:', offset.toFixed(2), 'seconds');
    audioSource.start(0, offset);
    audioIsPlaying = true;
    audioPlaybackTime = startTime;
    audioStartContextTime = audioCtx.currentTime;

    // Start playback indicator animation
    if (playbackAnimFrameId) cancelAnimationFrame(playbackAnimFrameId);
    playbackAnimFrameId = requestAnimationFrame(animatePlaybackIndicator);
}

function stopAudioPlayback() {
    if (audioSource) {
        try { audioSource.stop(); } catch { /* already stopped */ }
        audioSource.disconnect();
        audioSource = null;
    }
    audioIsPlaying = false;
    audioPlaybackTime = null;
    audioStartContextTime = null;

    // Stop playback indicator animation
    if (playbackAnimFrameId) {
        cancelAnimationFrame(playbackAnimFrameId);
        playbackAnimFrameId = null;
    }
}

/**
 * Convert a canvas mouse X position to a song time (seconds).
 */
function canvasXToTime(mouseX) {
    if (!waveformActiveStem || !waveformAnalysisData) return null;

    const stemData = waveformAnalysisData.stems[waveformActiveStem];
    const configuredEvents = getEventsForStem(stemData);
    const sensitiveEvents = getSensitiveEventsForStem(stemData);
    const envelope = waveformEnvelopeCache[waveformActiveStem];

    const { tMin: tMinFull, tMax: tMaxFull } = computeTimeRange(configuredEvents, sensitiveEvents, envelope);
    const { tMin, tMax } = computeVisibleRange(tMinFull, tMaxFull);

    const PAD = EVT_PAD;
    const plotW = (eventsCanvas ? eventsCanvas.parentElement.getBoundingClientRect().width : 800) - PAD.left - PAD.right;
    return tMin + ((mouseX - PAD.left) / plotW) * (tMax - tMin);
}

/**
 * Convert a song time (seconds) to a canvas X position (CSS pixels).
 * Returns null if the time is outside the visible range.
 */
function timeToCanvasX(songTime) {
    if (!waveformActiveStem || !waveformAnalysisData) return null;

    const stemData = waveformAnalysisData.stems[waveformActiveStem];
    const configuredEvents = getEventsForStem(stemData);
    const sensitiveEvents = getSensitiveEventsForStem(stemData);
    const envelope = waveformEnvelopeCache[waveformActiveStem];

    const { tMin: tMinFull, tMax: tMaxFull } = computeTimeRange(configuredEvents, sensitiveEvents, envelope);
    const { tMin, tMax } = computeVisibleRange(tMinFull, tMaxFull);

    if (songTime < tMin || songTime > tMax) return null;

    const PAD = EVT_PAD;
    const plotW = (eventsCanvas ? eventsCanvas.parentElement.getBoundingClientRect().width : 800) - PAD.left - PAD.right;
    return PAD.left + ((songTime - tMin) / (tMax - tMin)) * plotW;
}

// ─── Event Overrides (Click-to-Toggle) ───────────────────────────────────

/**
 * Load event overrides from server for the current project.
 */
async function loadEventOverrides() {
    if (!currentProject) return;
    try {
        const data = await api.getEventOverrides(currentProject.number);
        eventOverrides = data.overrides || {};
    } catch {
        eventOverrides = {};
    }
    // 2026-06-30: refresh the window reference. The export
    // `window.eventOverrides = eventOverrides` runs at module
    // load time — but `eventOverrides` is a `let` that gets
    // reassigned here (and on every cycle click). The window
    // reference becomes stale the moment we load a project's
    // overrides, so the cross-module `hasOverrides` check in
    // threshold-tuning.js's `saveTuningAndReconvert` was always
    // looking at the initial empty object. The user-reported
    // "no changes to save" bug is this stale reference.
    window.eventOverrides = eventOverrides;
    applyOverridesToEvents();
    // After loading, the in-memory `eventOverrides` matches
    // what's on disk — not dirty. The session-dirty flag is
    // also cleared (we just committed the previous session by
    // loading). Tuning changes are tracked separately by
    // threshold-tuning.js.
    eventOverridesDirty = false;
    sessionOverridesDirty = false;
    if (typeof window.updateSessionSaveButton === 'function') {
        window.updateSessionSaveButton();
    }
}

/**
 * Apply stored overrides to in-memory event data.
 *
 * Overrides are keyed by stem type and event time (4-decimal string).
 * Each value is a record: { status: "KEPT"|"FILTERED",
 * [classification]: int }. The classification override is applied
 * to the event when present — this is the "click to set class N"
 * feature the user asked for.
 *
 * 2026-06-30 (Bug 1 fix): previously this only iterated
 * events_configured || events_sensitive. For PGA-only stems
 * (the entire post-2026-06-15 refactor world — kick, snare,
 * toms, hihat, cymbals on project 6), the sidecar carries
 * events_pga, not events_configured. The override would be
 * silently ignored on initial load. Now we iterate events_pga
 * for PGA-only stems (the only data source the sidecar
 * actually has). Legacy projects with non-empty
 * events_configured still get the override applied to that
 * list for back-compat with the 2026-06-15 refactor.
 */
function applyOverridesToEvents() {
    if (!waveformAnalysisData || !waveformAnalysisData.stems) return;

    for (const [stemType, stemData] of Object.entries(waveformAnalysisData.stems)) {
        const stemOverrides = eventOverrides[stemType];
        if (!stemOverrides) continue;

        // Apply to events_pga (the canonical post-2026-06-15
        // source) AND to events_configured / events_sensitive
        // (legacy). Same frame key matches all three lists.
        const allEvents = [
            ...(stemData.events_pga || []),
            ...(stemData.events_configured || []),
            ...(stemData.events_sensitive || []),
        ];
        for (const event of allEvents) {
            // 2026-06-30: key on `event.frame` (integer) instead
            // of `event.time.toFixed(4)` (string). The previous
            // time-based key had a dangerous mismatch: the JSON
            // could have keys like "2.954" (3-decimal) that didn't
            // match the 4-decimal format produced by toFixed(4).
            // Frame is an integer, no rounding issues, and is
            // stable across precision changes. Falls back to
            // time-string key for legacy data that doesn't
            // have a frame.
            const key = _eventOverrideKey(event);
            const override = stemOverrides[key];
            if (!override) continue;
            event.status = override.status;
            if (override.classification != null) {
                event.classification = override.classification;
            }
            event._overridden = true;
        }
    }
}

/**
 * The override key for an event. Uses `event.frame` when
 * available (the canonical integer frame index from the
 * detector); falls back to `event.time.toFixed(4)` for legacy
 * data without a frame field.
 *
 * 2026-06-30: switched from time-string to frame-integer to fix
 * the user-reported "time: 2.954 vs '2.9540' mismatch" — a file
 * with non-4-decimal time keys would never match the lookup.
 * Frame is always an integer (no float precision issues) and
 * is the canonical per-event identifier.
 */
function _eventOverrideKey(event) {
    if (event.frame != null) return String(event.frame);
    return event.time.toFixed(4);
}

/**
 * Collect the unique classification values that exist in the
 * sidecar for the given stem's events_pga. Returns a sorted
 * array of integers (e.g. [0, 1, 2] for snare with 3
 * classes) or [] for stems with no classification data.
 *
 * Used by cycleEventOverride to step through the classes in
 * order (off → cls[0] → cls[1] → ... → off).
 */
function collectClassesForStem(stemType) {
    const stemData = waveformAnalysisData?.stems?.[stemType];
    if (!stemData) return [];
    const pga = stemData.events_pga || [];
    const classes = new Set();
    for (const ev of pga) {
        if (ev.classification != null) classes.add(ev.classification);
    }
    return Array.from(classes).sort((a, b) => a - b);
}

/**
 * Cycle an event's status + classification on click.
 *
 * Cycle logic (user's spec, 2026-06-30):
 *   - If currently FILTERED (or no override): next click →
 *     KEPT, classification = smallest available class (or null
 *     if no classes — single-class / no-class stems).
 *   - If currently KEPT and at the highest class index:
 *     next click → FILTERED (cycle off).
 *   - Otherwise: advance to the next class.
 *
 * Hihat: when the event has a hihat_state (open/closed), the
 * cycle first alternates hihat_state (open ↔ closed), then
 * status. Hihat classification is server-side, so the
 * per-event classification override is moot for hihat (use
 * the cluster card in the Tune panel for that).
 */
function cycleEventOverride(stemType, event) {
    if (!event || event.time == null) return;

    // 2026-06-30: key on frame (integer) instead of time.toFixed(4)
    // (string). See _eventOverrideKey for the rationale.
    const key = _eventOverrideKey(event);
    if (!eventOverrides[stemType]) eventOverrides[stemType] = {};
    const existing = eventOverrides[stemType][key] || {};
    const currentStatus = existing.status || event.status;
    const currentClass = existing.classification ?? event.classification ?? null;
    const currentHihatState =
        existing.hihat_state || event.hihat_state || 'open';

    // Hihat: 3-state cycle through FILTERED → open → closed →
    // FILTERED. (The hihat stem has no per-event classification
    // override — its identity is the open/closed state, and
    // the per-event override is the (status, hihat_state) pair.)
    if (stemType === 'hihat' && event.hihat_state) {
        // Determine the next state from the current state. The
        // current "status" is the override's status (or the
        // sidecar's natural state if no override). The current
        // "hihat_state" is the override's hihat_state (or the
        // event's current hihat_state).
        let nextStatus;
        let nextHihatState;
        if (currentStatus !== 'KEPT') {
            // Currently FILTERED (or no override). Turn on with
            // hihat_state='open' as the default.
            nextStatus = 'KEPT';
            nextHihatState = 'open';
        } else if (currentHihatState === 'open') {
            // Currently KEPT + open. Advance to closed.
            nextStatus = 'KEPT';
            nextHihatState = 'closed';
        } else {
            // Currently KEPT + closed. Cycle off.
            nextStatus = 'FILTERED';
            nextHihatState = event.hihat_state;  // preserve
        }
        event.status = nextStatus;
        event.hihat_state = nextHihatState;
        event._overridden = true;
        eventOverrides[stemType][key] = {
            status: nextStatus,
            hihat_state: nextHihatState,
        };
        scheduleOverrideSave();
        drawWaveform();
        return;
    }

    // Collect the stem's available classes (from events_pga).
    const classes = collectClassesForStem(stemType);
    const hasClasses = classes.length > 0;

    let nextStatus, nextClass;

    if (currentStatus !== 'KEPT') {
        // Currently FILTERED (or no override). Turn on, default
        // to the smallest class.
        nextStatus = 'KEPT';
        nextClass = hasClasses ? classes[0] : null;
    } else if (currentClass == null || !hasClasses) {
        // KEPT but no class. Cycle off.
        nextStatus = 'FILTERED';
        nextClass = null;
    } else {
        // KEPT with a class. Find position in classes.
        const idx = classes.indexOf(currentClass);
        if (idx === -1) {
            // Override class isn't in the sidecar's class set
            // (slider changed under us). Default to the lowest.
            nextStatus = 'KEPT';
            nextClass = classes[0];
        } else if (idx === classes.length - 1) {
            // At the highest class. Cycle off.
            nextStatus = 'FILTERED';
            nextClass = null;
        } else {
            // Advance to the next class.
            nextStatus = 'KEPT';
            nextClass = classes[idx + 1];
        }
    }

    // Update in-memory event. Mark classification only if the
    // override set one — leave the sidecar's natural
    // classification alone when the override is null
    // (status-only toggle for kick/single-class toms).
    event.status = nextStatus;
    if (nextClass != null) {
        event.classification = nextClass;
    }
    event._overridden = true;

    // Build the override record. Drop the classification key
    // when null (cleaner JSON, no spurious nulls).
    const overrideRecord = { status: nextStatus };
    if (nextClass != null) {
        overrideRecord.classification = nextClass;
    }
    eventOverrides[stemType][key] = overrideRecord;

    scheduleOverrideSave();
    drawWaveform();

    // 2026-06-30: light up the session-dirty Save button at the
    // top of the analysis section. The debounced save clears
    // eventOverridesDirty 500ms later (in-memory ↔ JSON in
    // sync), but sessionOverridesDirty stays set until the user
    // commits via Save & Reconvert. This is the UX fix the
    // user asked for: the Save button stays visible until they
    // actually click it.
    if (typeof window.updateSessionSaveButton === 'function') {
        window.updateSessionSaveButton();
    }
}

function scheduleOverrideSave() {
    eventOverridesDirty = true;
    // 2026-06-30: sessionOverridesDirty is set here (and in
    // cycleEventOverride above) so the Save button lights up
    // immediately. It is NOT cleared by the debounced save —
    // only by saveTuningAndReconvert (which syncs the in-memory
    // state from the server's cleaned dict). This way the
    // button stays visible until the user actually commits.
    sessionOverridesDirty = true;
    clearTimeout(eventOverridesSaveTimer);
    eventOverridesSaveTimer = setTimeout(saveEventOverrides, 500);

    if (typeof window.updateSessionSaveButton === 'function') {
        window.updateSessionSaveButton();
    }
}

/**
 * Persist overrides to server.
 */
async function saveEventOverrides() {
    if (!currentProject || !eventOverridesDirty) return;
    try {
        await api.saveEventOverrides(currentProject.number, eventOverrides);
        // Clear the in-memory ≠ JSON flag. The session-dirty flag
        // stays set so the Save button remains visible — the
        // user still needs to click Save & Reconvert for the
        // override to reach the MIDI.
        eventOverridesDirty = false;
        if (typeof window.updateSessionSaveButton === 'function') {
            window.updateSessionSaveButton();
        }
    } catch (err) {
        console.warn('Failed to save event overrides:', err);
    }
}

/**
 * Sync the in-memory `eventOverrides` with the server-cleaned
 * version (e.g. after a Save & Reconvert that auto-pruned
 * redundant entries). Exported on `window` so other modules
 * (threshold-tuning's saveTuningAndReconvert flow) can call it
 * after a rebuild round-trip.
 */
function syncEventOverridesFromServer(cleaned) {
    if (cleaned && typeof cleaned === 'object') {
        eventOverrides = cleaned;
        // 2026-06-30: also rewrite the window-scoped reference.
        // `window.eventOverrides = eventOverrides` at module
        // init captured the OLD dict by reference; reassigning
        // the local `eventOverrides` binding doesn't update
        // window. Update both so callers reading
        // `window.eventOverrides.stem[key]` after a rebuild
        // see the cleaned state.
        window.eventOverrides = eventOverrides;
        eventOverridesDirty = false;
        // 2026-06-30: sync from the server means the rebuild
        // has run. The in-memory state now matches the server's
        // cleaned dict. The session is no longer dirty — the
        // user's changes have been committed to the MIDI.
        sessionOverridesDirty = false;
        // Re-evaluate the Save button. The button's hidden
        // state is sticky (toggled by updateSessionSaveButton),
        // so without this call the button would stay visible
        // after the sync even though sessionOverridesDirty is
        // now false.
        if (typeof window.updateSessionSaveButton === 'function') {
            window.updateSessionSaveButton();
        }
    }
}
window.syncEventOverridesFromServer = syncEventOverridesFromServer;
window.eventOverrides = eventOverrides;  // For saveTuningAndReconvert
                                       // to detect "user has overrides to
                                       // commit even when no config
                                       // updates".
window.eventOverridesDirty = () => eventOverridesDirty;
window.sessionOverridesDirty = () => sessionOverridesDirty;
window.cycleEventOverride = cycleEventOverride;
window.collectClassesForStem = collectClassesForStem;
// Exposed for tests; lets a Playwright spec read the current
// sidecar data (e.g. to find a KEPT event to override) without
// having to maintain a parallel reflection in window.
window.waveformAnalysisData = () => waveformAnalysisData;

/**
 * Hit-test: find the event nearest to a canvas click, within a small radius.
 */
function hitTestEvent(mouseX) {
    if (!waveformActiveStem || !waveformAnalysisData) return null;

    const stemData = waveformAnalysisData.stems[waveformActiveStem];
    const configuredEvents = getEventsForStem(stemData);
    const sensitiveEvents = getSensitiveEventsForStem(stemData);
    const envelope = waveformEnvelopeCache[waveformActiveStem];

    const { tMin: tMinFull, tMax: tMaxFull } = computeTimeRange(configuredEvents, sensitiveEvents, envelope);
    const { tMin, tMax } = computeVisibleRange(tMinFull, tMaxFull);

    const PAD = EVT_PAD;
    const rect = eventsCanvas ? eventsCanvas.parentElement.getBoundingClientRect() : null;
    if (!rect) return null;
    const plotW = rect.width - PAD.left - PAD.right;
    const xToTime = x => tMin + ((x - PAD.left) / plotW) * (tMax - tMin);
    const mouseTime = xToTime(mouseX);
    const hitRadius = (tMax - tMin) / plotW * 6; // ~6px hit radius

    const displayEvents = (waveformTuningActive && waveformTuningEvents)
        ? waveformTuningEvents
        : configuredEvents;

    let closest = null;
    let closestDist = Infinity;
    for (const evt of displayEvents) {
        if (evt.time == null) continue;
        const dist = Math.abs(evt.time - mouseTime);
        if (dist < hitRadius && dist < closestDist) {
            closestDist = dist;
            closest = evt;
        }
    }
    return closest;
}

// ─── Resize Handler ──────────────────────────────────────────────────────

let waveformResizeTimer = null;

function onWaveformResize() {
    clearTimeout(waveformResizeTimer);
    waveformResizeTimer = setTimeout(() => {
        if (waveformActiveStem) drawWaveform();
    }, 100);
}

window.addEventListener('resize', onWaveformResize);
