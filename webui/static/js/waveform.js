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
    markerKept: '#10b981',       // green
    markerFiltered: '#ef4444',   // red
    markerReverbCont: '#f59e0b', // orange/amber
    markerSensitive: 'rgba(156, 163, 175, 0.3)',
    markerUnknown: '#6b7280',    // gray
    tooltipBg: 'rgba(17, 24, 39, 0.95)',
    tooltipBorder: '#4b5563',
    tooltipText: '#e5e7eb',
    crosshair: 'rgba(255, 255, 255, 0.3)',
};

const STEM_COLORS = {
    kick:    { accent: '#3b82f6', label: 'Kick' },
    snare:   { accent: '#8b5cf6', label: 'Snare' },
    hihat:   { accent: '#10b981', label: 'Hi-Hat' },
    cymbals: { accent: '#f59e0b', label: 'Cymbals' },
    toms:    { accent: '#ef4444', label: 'Toms' },
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
let waveformShowSensitive = false;
let waveformTuningEvents = null;
let waveformTuningActive = false;

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

    // Sensitive toggle
    const sensitiveToggle = document.getElementById('waveform-sensitive-toggle');
    if (sensitiveToggle) {
        sensitiveToggle.checked = waveformShowSensitive;
        sensitiveToggle.onchange = () => {
            waveformShowSensitive = sensitiveToggle.checked;
            drawWaveform();
        };
    }

    // Tune button visibility
    const tuneBtn = document.getElementById('tuning-toggle-btn');
    if (tuneBtn) {
        const hasAnySensitive = availableStems.some(s => {
            const sd = waveformAnalysisData.stems[s];
            return sd.events_sensitive && sd.events_sensitive.length > 0;
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

    const sensitiveContainer = document.getElementById('waveform-sensitive-container');
    if (sensitiveContainer) {
        const stemData = waveformAnalysisData.stems[stemType];
        const hasSensitive = stemData.events_sensitive && stemData.events_sensitive.length > 0;
        sensitiveContainer.classList.toggle('hidden', !hasSensitive);
    }

    // Load envelope data
    if (!waveformEnvelopeCache[stemType]) {
        try {
            const envelope = await api.getProjectEnvelope(currentProject.number, stemType);
            waveformEnvelopeCache[stemType] = envelope;
        } catch {
            waveformEnvelopeCache[stemType] = null;
        }
    }

    drawWaveform();

    if (typeof onTuningStemChanged === 'function') {
        onTuningStemChanged(stemType);
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
    const envelope = waveformEnvelopeCache[waveformActiveStem];

    const displayEvents = (waveformTuningActive && waveformTuningEvents)
        ? waveformTuningEvents
        : configuredEvents;

    // Full time range (for zoom reference)
    const { tMin: tMinFull, tMax: tMaxFull } = computeTimeRange(configuredEvents, sensitiveEvents, envelope);
    if (tMaxFull <= tMinFull) return;

    // Visible time range (affected by zoom/pan)
    const { tMin, tMax } = computeVisibleRange(tMinFull, tMaxFull);

    // Draw envelope panel
    drawEnvelopePanel(envelope, tMin, tMax, stemData, configuredEvents, sensitiveEvents);

    // Draw events panel
    drawEventsPanel(displayEvents, sensitiveEvents, configuredEvents, tMin, tMax, stemData);

    // Update legend bar (HTML, outside canvas)
    updateLegendBar(stemData, displayEvents);

    // Draw crosshair on both panels
    if (waveformMouseX != null) {
        drawCrosshair(envelopeCtx, envelopeCanvas);
        drawCrosshair(eventsCtx, eventsCanvas);
    }

    // Draw tooltip on events panel
    if (waveformHoverEvent) {
        const rect = eventsCanvas.parentElement.getBoundingClientRect();
        drawTooltip(eventsCtx, waveformHoverEvent, rect.width, rect.height);
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

    // Threshold line (on geomean scale)
    const geomeanMax = computeMaxGeomean(configuredEvents, sensitiveEvents);
    const logic = stemData.logic || {};
    const tuningGeomean = waveformTuningActive && tuningSliderValues?.[waveformActiveStem]?.geomean_threshold;
    const thresholdVal = tuningGeomean != null ? tuningGeomean : logic.geomean_threshold;
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

function drawEventsPanel(displayEvents, sensitiveEvents, configuredEvents, tMin, tMax, stemData) {
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

    // Sensitive events (background layer)
    if (!waveformTuningActive && waveformShowSensitive && sensitiveEvents.length > 0) {
        drawEventBars(ctx, sensitiveEvents, timeToX, PAD, plotW, plotH, true);
    }

    // Primary events as amplitude bars
    drawEventBars(ctx, displayEvents, timeToX, PAD, plotW, plotH, false);
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

        const x = timeToX(event.time);
        if (x < PAD.left - barWidth || x > PAD.left + plotW + barWidth) continue;

        const color = isSensitiveLayer
            ? WAVEFORM_COLORS.markerSensitive
            : getMarkerColor(event.status);

        // Bar height from velocity (0-127)
        const velocity = event.velocity != null ? event.velocity : 64;
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
        }

        ctx.globalAlpha = 1.0;
    }
}

// ─── Data Helpers ────────────────────────────────────────────────────────

function getEventsForStem(stemData) {
    if (stemData.events_configured) return stemData.events_configured;
    if (stemData.events) return stemData.events;
    return [];
}

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

function getMarkerColor(status) {
    switch (status) {
        case 'KEPT': return WAVEFORM_COLORS.markerKept;
        case 'FILTERED': return WAVEFORM_COLORS.markerFiltered;
        case 'REVERB_CONTINUATION': return WAVEFORM_COLORS.markerReverbCont;
        default: return WAVEFORM_COLORS.markerUnknown;
    }
}

// ─── Legend Bar (HTML, outside canvas) ────────────────────────────────────

function updateLegendBar(stemData, displayEvents) {
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

    container.innerHTML = items.map(item =>
        `<span class="flex items-center gap-1">
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

    let tx = event._mouseX + 12;
    let ty = event._mouseY - tooltipH / 2;
    if (tx + tooltipW > W) tx = event._mouseX - tooltipW - 12;
    if (ty < 0) ty = 4;
    if (ty + tooltipH > H) ty = H - tooltipH - 4;

    ctx.fillStyle = WAVEFORM_COLORS.tooltipBg;
    ctx.strokeStyle = WAVEFORM_COLORS.tooltipBorder;
    ctx.lineWidth = 1;
    roundRect(ctx, tx, ty, tooltipW, tooltipH, 4);
    ctx.fill();
    ctx.stroke();

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
        const envelope = waveformEnvelopeCache[waveformActiveStem];

        const { tMin: tMinFull, tMax: tMaxFull } = computeTimeRange(configuredEvents, sensitiveEvents, envelope);
        const { tMin, tMax } = computeVisibleRange(tMinFull, tMaxFull);

        const PAD = EVT_PAD;
        const plotW = canvasRect.width - PAD.left - PAD.right;
        const xToTime = x => tMin + ((x - PAD.left) / plotW) * (tMax - tMin);
        const mouseTime = xToTime(mouseX);
        const hitRadius = (tMax - tMin) / plotW * 5;

        const displayEvents = (waveformTuningActive && waveformTuningEvents)
            ? waveformTuningEvents
            : configuredEvents;
        const allEvents = (!waveformTuningActive && waveformShowSensitive)
            ? displayEvents.concat(sensitiveEvents)
            : displayEvents;

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
    if (waveformZoom <= 1) return;
    waveformIsDragging = true;
    waveformDragStartX = e.clientX - e.target.parentElement.getBoundingClientRect().left;
    waveformDragStartPan = waveformPanOffset;

    const onMove = (me) => {
        const canvasRect = e.target.parentElement.getBoundingClientRect();
        const mouseX = me.clientX - canvasRect.left;
        const plotW = canvasRect.width - EVT_PAD.left - EVT_PAD.right;
        const dx = (mouseX - waveformDragStartX) / plotW;
        waveformPanOffset = waveformDragStartPan - dx;
        clampPan();
        waveformMouseX = me.clientX - canvasRect.left;
        drawWaveform();
    };

    const onUp = () => {
        waveformIsDragging = false;
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
        const cursorStyle = waveformZoom > 1 ? 'grab' : 'crosshair';
        if (envelopeCanvas) envelopeCanvas.style.cursor = cursorStyle;
        if (eventsCanvas) eventsCanvas.style.cursor = cursorStyle;
    };

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);

    if (envelopeCanvas) envelopeCanvas.style.cursor = 'grabbing';
    if (eventsCanvas) eventsCanvas.style.cursor = 'grabbing';
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
