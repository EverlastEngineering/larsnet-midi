/**
 * Waveform Viewer Component
 *
 * Canvas-based energy envelope visualization with color-coded onset markers.
 * Supports analysis.json v2 (events) and v3 (events_configured / events_sensitive).
 *
 * Visual layers (bottom to top):
 *   1. Background with time axis
 *   2. Energy envelope (L/R as filled area)
 *   3. Geomean threshold line (dashed)
 *   4. Onset markers (vertical lines, color-coded by status)
 *   5. Hover tooltip (event details)
 */

// ─── Constants ───────────────────────────────────────────────────────────

const WAVEFORM_COLORS = {
    background: '#111827',
    axisLine: '#374151',
    axisText: '#9ca3af',
    envelopeLeft: 'rgba(59, 130, 246, 0.5)',   // blue
    envelopeRight: 'rgba(139, 92, 246, 0.35)',  // purple
    envelopeFillLeft: 'rgba(59, 130, 246, 0.15)',
    envelopeFillRight: 'rgba(139, 92, 246, 0.10)',
    thresholdLine: 'rgba(251, 191, 36, 0.7)',   // amber dashed
    markerKept: '#10b981',       // green
    markerFiltered: '#ef4444',   // red
    markerReverbCont: '#f59e0b', // orange/amber
    markerSensitive: 'rgba(156, 163, 175, 0.3)', // gray (background sensitive events)
    markerUnknown: '#6b7280',    // gray
    tooltipBg: 'rgba(17, 24, 39, 0.95)',
    tooltipBorder: '#4b5563',
    tooltipText: '#e5e7eb',
};

const STEM_COLORS = {
    kick:    { accent: '#3b82f6', label: 'Kick' },
    snare:   { accent: '#8b5cf6', label: 'Snare' },
    hihat:   { accent: '#10b981', label: 'Hi-Hat' },
    cymbals: { accent: '#f59e0b', label: 'Cymbals' },
    toms:    { accent: '#ef4444', label: 'Toms' },
};

const STEM_ORDER = ['kick', 'snare', 'toms', 'hihat', 'cymbals'];

// ─── State ───────────────────────────────────────────────────────────────

let waveformAnalysisData = null;   // Full analysis.json response
let waveformEnvelopeCache = {};    // {stemType: envelopeData}
let waveformActiveStem = null;     // Currently displayed stem
let waveformCanvas = null;         // Canvas element
let waveformCtx = null;            // Canvas 2D context
let waveformHoverEvent = null;     // Event under mouse cursor
let waveformShowSensitive = false; // Toggle for sensitive events layer
let waveformTuningEvents = null;   // Filtered events from threshold tuning (or null)
let waveformTuningActive = false;  // Whether tuning mode is visually active

// ─── Public API ──────────────────────────────────────────────────────────

/**
 * Initialize waveform viewer for a project.
 * Called from selectProject() in projects.js.
 *
 * @param {object} project - The currentProject object
 */
async function initWaveformViewer(project) {
    const section = document.getElementById('analysis-section');
    if (!section) return;

    waveformAnalysisData = null;
    waveformEnvelopeCache = {};
    waveformActiveStem = null;
    waveformHoverEvent = null;
    waveformTuningEvents = null;
    waveformTuningActive = false;

    // Check if project has analysis data
    if (!project.has_analysis) {
        section.classList.add('hidden');
        return;
    }

    section.classList.remove('hidden');

    try {
        waveformAnalysisData = await api.getProjectAnalysis(project.number);
    } catch (err) {
        console.error('Failed to load analysis data:', err);
        section.classList.add('hidden');
        return;
    }

    if (!waveformAnalysisData || !waveformAnalysisData.stems) {
        section.classList.add('hidden');
        return;
    }

    // Build stem tabs
    const availableStems = Object.keys(waveformAnalysisData.stems);
    renderStemTabs(availableStems);

    // Set up canvas
    waveformCanvas = document.getElementById('waveform-canvas');
    if (!waveformCanvas) return;
    waveformCtx = waveformCanvas.getContext('2d');

    // Mouse interaction
    waveformCanvas.addEventListener('mousemove', onCanvasMouseMove);
    waveformCanvas.addEventListener('mouseleave', onCanvasMouseLeave);

    // Toggle sensitive events checkbox
    const sensitiveToggle = document.getElementById('waveform-sensitive-toggle');
    if (sensitiveToggle) {
        sensitiveToggle.checked = waveformShowSensitive;
        sensitiveToggle.onchange = () => {
            waveformShowSensitive = sensitiveToggle.checked;
            drawWaveform();
        };
    }

    // Show/hide Tune button based on whether any stem has sensitive events
    const tuneBtn = document.getElementById('tuning-toggle-btn');
    if (tuneBtn) {
        const hasAnySensitive = availableStems.some(s => {
            const sd = waveformAnalysisData.stems[s];
            return sd.events_sensitive && sd.events_sensitive.length > 0;
        });
        tuneBtn.classList.toggle('hidden', !hasAnySensitive);
    }

    // Close tuning panel when loading a new project
    const tuningPanel = document.getElementById('tuning-panel');
    if (tuningPanel && !tuningPanel.classList.contains('hidden')) {
        tuningPanelOpen = false;
        tuningPanel.classList.add('hidden');
        if (tuneBtn) tuneBtn.classList.remove('tuning-btn-active');
    }

    // Select first available stem in kit order
    const firstStem = STEM_ORDER.find(s => availableStems.includes(s)) || availableStems[0];
    if (firstStem) {
        selectStem(firstStem);
    }
}

/**
 * Select a stem and render its waveform.
 */
async function selectStem(stemType) {
    if (!waveformAnalysisData || !waveformAnalysisData.stems[stemType]) return;

    waveformActiveStem = stemType;
    waveformHoverEvent = null;
    waveformTuningEvents = null;
    waveformTuningActive = false;

    // Update tab UI
    document.querySelectorAll('.waveform-stem-tab').forEach(tab => {
        const isActive = tab.dataset.stem === stemType;
        tab.classList.toggle('waveform-tab-active', isActive);
        tab.classList.toggle('waveform-tab-inactive', !isActive);
    });

    // Update the sensitive toggle visibility (only show for v3 with sensitive events)
    const sensitiveContainer = document.getElementById('waveform-sensitive-container');
    if (sensitiveContainer) {
        const stemData = waveformAnalysisData.stems[stemType];
        const hasSensitive = stemData.events_sensitive && stemData.events_sensitive.length > 0;
        sensitiveContainer.classList.toggle('hidden', !hasSensitive);
    }

    // Try to load envelope data (may not exist for older projects)
    if (!waveformEnvelopeCache[stemType]) {
        try {
            const envelope = await api.getProjectEnvelope(currentProject.number, stemType);
            waveformEnvelopeCache[stemType] = envelope;
        } catch {
            // Envelope data not available — that's OK, we'll draw without it
            waveformEnvelopeCache[stemType] = null;
        }
    }

    drawWaveform();

    // Notify threshold tuning module (if panel is open, rebuild sliders)
    if (typeof onTuningStemChanged === 'function') {
        onTuningStemChanged(stemType);
    }
}

// ─── Tab Rendering ───────────────────────────────────────────────────────

function renderStemTabs(availableStems) {
    const container = document.getElementById('waveform-stem-tabs');
    if (!container) return;

    // Sort stems in kit order
    const ordered = STEM_ORDER.filter(s => availableStems.includes(s));
    // Add any stems not in STEM_ORDER
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

// ─── Canvas Drawing ──────────────────────────────────────────────────────

function drawWaveform() {
    if (!waveformCanvas || !waveformCtx || !waveformActiveStem) return;

    const canvas = waveformCanvas;
    const ctx = waveformCtx;
    const dpr = window.devicePixelRatio || 1;

    // Size canvas to container
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = rect.height;
    const PADDING = { top: 10, bottom: 28, left: 45, right: 15 };
    const plotW = W - PADDING.left - PADDING.right;
    const plotH = H - PADDING.top - PADDING.bottom;

    // Clear
    ctx.fillStyle = WAVEFORM_COLORS.background;
    ctx.fillRect(0, 0, W, H);

    // Get data
    const stemData = waveformAnalysisData.stems[waveformActiveStem];
    const configuredEvents = getEventsForStem(stemData);
    const sensitiveEvents = getSensitiveEventsForStem(stemData);
    const envelope = waveformEnvelopeCache[waveformActiveStem];

    // In tuning mode, use tuning-filtered events as the primary layer
    const displayEvents = (waveformTuningActive && waveformTuningEvents)
        ? waveformTuningEvents
        : configuredEvents;

    // Compute time range from events (and envelope if available)
    const { tMin, tMax } = computeTimeRange(configuredEvents, sensitiveEvents, envelope);
    if (tMax <= tMin) return;

    const timeToX = t => PADDING.left + ((t - tMin) / (tMax - tMin)) * plotW;
    const maxAmplitude = computeMaxAmplitude(configuredEvents, sensitiveEvents, envelope);
    const valToY = v => PADDING.top + plotH - (v / (maxAmplitude || 1)) * plotH;

    // Layer 1: Time axis
    drawTimeAxis(ctx, W, H, PADDING, plotW, plotH, tMin, tMax, timeToX);

    // Layer 2: Envelope (if available)
    if (envelope && envelope.times) {
        drawEnvelope(ctx, envelope, timeToX, valToY, PADDING, plotH, maxAmplitude);
    }

    // Layer 3: Geomean threshold line
    // In tuning mode, show the slider value; otherwise show configured value
    const logic = stemData.logic || {};
    const tuningGeomean = waveformTuningActive && tuningSliderValues?.[waveformActiveStem]?.geomean_threshold;
    const thresholdVal = tuningGeomean != null ? tuningGeomean : logic.geomean_threshold;
    if (thresholdVal != null && maxAmplitude > 0) {
        drawThresholdLine(ctx, thresholdVal, valToY, PADDING, plotW);
    }

    // Layer 3.5: Sensitive events (background, if toggled on — but not in tuning mode)
    if (!waveformTuningActive && waveformShowSensitive && sensitiveEvents.length > 0) {
        drawOnsetMarkers(ctx, sensitiveEvents, timeToX, PADDING, plotH, true);
    }

    // Layer 4: Primary onset markers (configured or tuning-filtered)
    drawOnsetMarkers(ctx, displayEvents, timeToX, PADDING, plotH, false);

    // Layer 5: Hover tooltip
    if (waveformHoverEvent) {
        drawTooltip(ctx, waveformHoverEvent, W, H);
    }

    // Tuning mode indicator
    if (waveformTuningActive) {
        ctx.fillStyle = 'rgba(59, 130, 246, 0.8)';
        ctx.font = 'bold 9px system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('● TUNING', PADDING.left + 4, PADDING.top + 12);
    }

    // Legend
    drawLegend(ctx, W, H, stemData);
}

// ─── Data Helpers ────────────────────────────────────────────────────────

/** Get configured events, supporting v2 and v3 formats. */
function getEventsForStem(stemData) {
    // v3 format
    if (stemData.events_configured) return stemData.events_configured;
    // v2 format
    if (stemData.events) return stemData.events;
    return [];
}

/** Get sensitive events (v3 only). */
function getSensitiveEventsForStem(stemData) {
    return stemData.events_sensitive || [];
}

function computeTimeRange(events, sensitiveEvents, envelope) {
    let tMin = Infinity, tMax = -Infinity;

    for (const e of events) {
        if (e.time != null) { tMin = Math.min(tMin, e.time); tMax = Math.max(tMax, e.time); }
    }
    for (const e of sensitiveEvents) {
        if (e.time != null) { tMin = Math.min(tMin, e.time); tMax = Math.max(tMax, e.time); }
    }
    if (envelope && envelope.times && envelope.times.length > 0) {
        tMin = Math.min(tMin, envelope.times[0]);
        tMax = Math.max(tMax, envelope.times[envelope.times.length - 1]);
    }

    // Add small padding
    const span = tMax - tMin || 1;
    return { tMin: tMin - span * 0.02, tMax: tMax + span * 0.02 };
}

function computeMaxAmplitude(events, sensitiveEvents, envelope) {
    let maxVal = 0;

    // From events: use strength or amplitude
    for (const e of events) {
        if (e.strength != null) maxVal = Math.max(maxVal, e.strength);
        if (e.amplitude != null) maxVal = Math.max(maxVal, e.amplitude);
    }
    for (const e of sensitiveEvents) {
        if (e.strength != null) maxVal = Math.max(maxVal, e.strength);
        if (e.amplitude != null) maxVal = Math.max(maxVal, e.amplitude);
    }

    // From envelope
    if (envelope) {
        if (envelope.left) for (const v of envelope.left) maxVal = Math.max(maxVal, v);
        if (envelope.right) for (const v of envelope.right) maxVal = Math.max(maxVal, v);
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

        // Tick mark
        ctx.beginPath();
        ctx.moveTo(x, PAD.top + plotH);
        ctx.lineTo(x, PAD.top + plotH + 4);
        ctx.stroke();

        // Label
        ctx.fillText(formatTime(t), x, PAD.top + plotH + 16);
    }
}

function computeTickInterval(duration) {
    if (duration > 300) return 60;
    if (duration > 120) return 30;
    if (duration > 60) return 10;
    if (duration > 30) return 5;
    if (duration > 10) return 2;
    return 1;
}

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return m > 0 ? `${m}:${String(s).padStart(2, '0')}` : `${s}s`;
}

function drawEnvelope(ctx, envelope, timeToX, valToY, PAD, plotH, maxAmp) {
    if (!envelope.times || envelope.times.length === 0) return;

    const times = envelope.times;
    const baseY = PAD.top + plotH;

    // Draw left channel
    drawEnvelopeChannel(ctx, times, envelope.left, timeToX, valToY, baseY,
        WAVEFORM_COLORS.envelopeLeft, WAVEFORM_COLORS.envelopeFillLeft);

    // Draw right channel (mirrored or overlaid)
    drawEnvelopeChannel(ctx, times, envelope.right, timeToX, valToY, baseY,
        WAVEFORM_COLORS.envelopeRight, WAVEFORM_COLORS.envelopeFillRight);
}

function drawEnvelopeChannel(ctx, times, values, timeToX, valToY, baseY, strokeColor, fillColor) {
    if (!values || values.length === 0) return;

    // Filled area
    ctx.beginPath();
    ctx.moveTo(timeToX(times[0]), baseY);
    for (let i = 0; i < times.length; i++) {
        ctx.lineTo(timeToX(times[i]), valToY(values[i]));
    }
    ctx.lineTo(timeToX(times[times.length - 1]), baseY);
    ctx.closePath();
    ctx.fillStyle = fillColor;
    ctx.fill();

    // Stroke line
    ctx.beginPath();
    for (let i = 0; i < times.length; i++) {
        const x = timeToX(times[i]);
        const y = valToY(values[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 1;
    ctx.stroke();
}

function drawThresholdLine(ctx, threshold, valToY, PAD, plotW) {
    // The threshold is on the geomean scale, not directly on amplitude.
    // We draw it as a reference line — position relative to amplitude range
    // is approximate. When envelope data drives the Y axis, this is informational.
    // For now, skip drawing if threshold would be off-scale.
    const y = valToY(threshold);
    if (y < PAD.top || y > PAD.top + 400) return;

    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = WAVEFORM_COLORS.thresholdLine;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(PAD.left, y);
    ctx.lineTo(PAD.left + plotW, y);
    ctx.stroke();
    ctx.setLineDash([]);

    // Label
    ctx.fillStyle = WAVEFORM_COLORS.thresholdLine;
    ctx.font = '9px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`threshold: ${threshold}`, PAD.left + 4, y - 4);
}

function drawOnsetMarkers(ctx, events, timeToX, PAD, plotH, isSensitiveLayer) {
    for (const event of events) {
        if (event.time == null) continue;

        const x = timeToX(event.time);
        if (x < PAD.left || x > PAD.left + PAD.left + 5000) continue; // sanity

        const color = isSensitiveLayer
            ? WAVEFORM_COLORS.markerSensitive
            : getMarkerColor(event.status);

        ctx.strokeStyle = color;
        ctx.lineWidth = isSensitiveLayer ? 0.5 : 1.5;
        ctx.globalAlpha = isSensitiveLayer ? 0.4 : 1.0;

        ctx.beginPath();
        ctx.moveTo(x, PAD.top);
        ctx.lineTo(x, PAD.top + plotH);
        ctx.stroke();

        // Small triangle at top for kept events (non-sensitive only)
        if (!isSensitiveLayer && event.status === 'KEPT') {
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.moveTo(x - 3, PAD.top);
            ctx.lineTo(x + 3, PAD.top);
            ctx.lineTo(x, PAD.top + 6);
            ctx.closePath();
            ctx.fill();
        }

        ctx.globalAlpha = 1.0;
    }
}

function getMarkerColor(status) {
    switch (status) {
        case 'KEPT': return WAVEFORM_COLORS.markerKept;
        case 'FILTERED': return WAVEFORM_COLORS.markerFiltered;
        case 'REVERB_CONTINUATION': return WAVEFORM_COLORS.markerReverbCont;
        default: return WAVEFORM_COLORS.markerUnknown;
    }
}

function drawLegend(ctx, W, H, stemData) {
    // In tuning mode, use tuning events for the legend counts
    const events = (waveformTuningActive && waveformTuningEvents)
        ? waveformTuningEvents
        : getEventsForStem(stemData);
    const keptCount = events.filter(e => e.status === 'KEPT').length;
    const filteredCount = events.filter(e => e.status === 'FILTERED').length;
    const reverbCount = events.filter(e => e.status === 'REVERB_CONTINUATION').length;
    const sensitiveCount = (stemData.events_sensitive || []).length;

    const items = [];
    if (keptCount > 0) items.push({ color: WAVEFORM_COLORS.markerKept, label: `Kept (${keptCount})` });
    if (filteredCount > 0) items.push({ color: WAVEFORM_COLORS.markerFiltered, label: `Filtered (${filteredCount})` });
    if (reverbCount > 0) items.push({ color: WAVEFORM_COLORS.markerReverbCont, label: `Reverb cont. (${reverbCount})` });
    if (!waveformTuningActive && waveformShowSensitive && sensitiveCount > 0) {
        items.push({ color: '#9ca3af', label: `Sensitive (${sensitiveCount})` });
    }

    if (items.length === 0) return;

    ctx.font = '10px system-ui, sans-serif';
    let x = W - 12;

    // Draw right-to-left
    for (let i = items.length - 1; i >= 0; i--) {
        const item = items[i];
        const textW = ctx.measureText(item.label).width;
        x -= textW;
        ctx.fillStyle = item.color;
        ctx.textAlign = 'left';
        ctx.fillText(item.label, x, 18);
        x -= 14;
        // Color dot
        ctx.beginPath();
        ctx.arc(x + 4, 14, 4, 0, Math.PI * 2);
        ctx.fill();
        x -= 8;
    }
}

// ─── Tooltip ─────────────────────────────────────────────────────────────

function drawTooltip(ctx, event, W, H) {
    const lines = [];
    lines.push(`Time: ${formatTime(event.time)}`);
    lines.push(`Status: ${event.status}`);
    if (event.velocity != null) lines.push(`Velocity: ${event.velocity}`);
    if (event.note != null) lines.push(`Note: ${event.note}`);
    if (event.strength != null) lines.push(`Strength: ${event.strength}`);
    if (event.geomean != null) lines.push(`Geomean: ${event.geomean}`);
    if (event.amplitude != null) lines.push(`Amplitude: ${event.amplitude}`);
    if (event.total_energy != null) lines.push(`Total energy: ${event.total_energy}`);
    if (event.sustain_ms != null) lines.push(`Sustain: ${event.sustain_ms}ms`);

    const lineH = 16;
    const pad = 8;
    const tooltipW = 180;
    const tooltipH = lines.length * lineH + pad * 2;

    // Position near mouse but keep on screen
    let tx = event._mouseX + 12;
    let ty = event._mouseY - tooltipH / 2;
    if (tx + tooltipW > W) tx = event._mouseX - tooltipW - 12;
    if (ty < 0) ty = 4;
    if (ty + tooltipH > H) ty = H - tooltipH - 4;

    // Background
    ctx.fillStyle = WAVEFORM_COLORS.tooltipBg;
    ctx.strokeStyle = WAVEFORM_COLORS.tooltipBorder;
    ctx.lineWidth = 1;
    roundRect(ctx, tx, ty, tooltipW, tooltipH, 4);
    ctx.fill();
    ctx.stroke();

    // Text
    ctx.fillStyle = WAVEFORM_COLORS.tooltipText;
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'left';
    lines.forEach((line, i) => {
        ctx.fillText(line, tx + pad, ty + pad + (i + 1) * lineH - 3);
    });
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

function onCanvasMouseMove(e) {
    if (!waveformCanvas || !waveformActiveStem || !waveformAnalysisData) return;

    const rect = waveformCanvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const W = rect.width;
    const PADDING = { top: 10, bottom: 28, left: 45, right: 15 };
    const plotW = W - PADDING.left - PADDING.right;

    const stemData = waveformAnalysisData.stems[waveformActiveStem];
    const configuredEvents = getEventsForStem(stemData);
    // In tuning mode, hover over tuning events; otherwise use configured + optional sensitive
    const displayEvents = (waveformTuningActive && waveformTuningEvents)
        ? waveformTuningEvents
        : configuredEvents;
    const allEvents = (!waveformTuningActive && waveformShowSensitive)
        ? displayEvents.concat(getSensitiveEventsForStem(stemData))
        : displayEvents;

    const { tMin, tMax } = computeTimeRange(
        configuredEvents,
        getSensitiveEventsForStem(stemData),
        waveformEnvelopeCache[waveformActiveStem]
    );

    const xToTime = x => tMin + ((x - PADDING.left) / plotW) * (tMax - tMin);
    const mouseTime = xToTime(mouseX);
    const hitRadius = (tMax - tMin) / plotW * 5; // 5px tolerance

    // Find closest event
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
        waveformCanvas.style.cursor = 'crosshair';
    } else {
        waveformHoverEvent = null;
        waveformCanvas.style.cursor = 'default';
    }

    drawWaveform();
}

function onCanvasMouseLeave() {
    waveformHoverEvent = null;
    if (waveformCanvas) waveformCanvas.style.cursor = 'default';
    drawWaveform();
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
