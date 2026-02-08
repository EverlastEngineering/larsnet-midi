# Step 5: Interactive Threshold Sliders — Plan

## Objective

Add client-side sliders for key detection/filtering parameters that re-filter the `events_sensitive` dataset in the browser with zero server round-trips. Moving a slider instantly shows events appearing/disappearing on the waveform.

## User Constraint

The existing "Advanced Per-Stem Settings" (`advanced-midi.js`) is broken and brittle. The threshold tuning UI goes in a **separate dialog** — a panel that opens from the Detection Analysis section, alongside the waveform viewer.

## Architecture

### New File: `threshold-tuning.js`

A self-contained module that:
1. Opens an inline panel below the waveform canvas (not a modal — keeps waveform visible)
2. Reads default thresholds from `waveformAnalysisData.stems[stem].logic`
3. Provides range sliders for per-stem parameters
4. Re-filters `events_sensitive` client-side on every `input` event
5. Calls `drawWaveform()` to update the canvas with the tuning preview
6. Shows a live event count comparison (configured vs. tuning preview)

### Slider Parameters (Per-Stem)

| Parameter | Applies To | Range | Step |
|---|---|---|---|
| `geomean_threshold` | All stems | 0–2000 | Varies by stem |
| `min_sustain_ms` | hihat, cymbals | 0–500 | 5 |
| `min_strength_threshold` | hihat, cymbals | 0–1.0 | 0.01 |
| `reverb_continuation_attack_threshold` | All stems | 0–1.0 | 0.01 |

### Client-Side Filtering Logic

Replicate the server-side filter passes from `analysis_core.py`:

**Pass 1 — Spectral Filter:**
- `geomean_only` mode (kick, snare, toms, hihat): if `event.geomean <= geomean_threshold` → FILTERED
- `require_both` mode (cymbals): must pass BOTH geomean AND sustain thresholds
- Strength gate: if `event.strength < min_strength_threshold` → FILTERED

**Pass 2 — Reverb Continuation:**
- Sort KEPT events by time
- For each pair: if adjacent (gap ≤ 5ms) AND amplitude-continuous (diff ≤ 0.001) AND smooth (attack_sharpness < threshold) → REVERB_CONTINUATION

**Skipped:** Decay filter (requires raw audio) and statistical filter (disabled by default, complex median computation).

### Visual Integration with waveform.js

Add a `waveformTuningEvents` state variable. When non-null, `drawWaveform()` renders these instead of the configured events, with a visual indicator that tuning mode is active.

### HTML Structure

Add inside `#analysis-container`, below the canvas:
- "Tune Thresholds" button next to the sensitive toggle
- Inline tuning panel (hidden by default) with:
  - Per-parameter sliders with numeric readout
  - Live event count: "Kept: X / Sensitive: Y (configured: Z)"
  - Reset to configured defaults button
  - Close button

## Phases

1. **Phase A**: Create threshold-tuning.js with filtering logic and slider UI generation
2. **Phase B**: Add HTML structure and CSS for the tuning panel
3. **Phase C**: Integrate with waveform.js (tuning overlay state, redraw hooks)
4. **Phase D**: Wire into app.js / projects.js / index.html
5. **Phase E**: Write tests, validate, commit

## Risks

- Performance: Re-filtering 500-2000 events on slider `input` events (60+ fps). Mitigation: debounce at 16ms (requestAnimationFrame).
- Reverb continuation filter is sequential and depends on previous events' status — must process in order.
- Different stems have different available parameters — UI must adapt per-stem.

## Success Criteria

- [ ] Sliders appear for the active stem with correct default values from `logic` block
- [ ] Moving a slider instantly re-filters events and redraws the waveform
- [ ] Event count updates in real-time
- [ ] Switching stems updates slider ranges and defaults
- [ ] Reset button restores configured defaults
- [ ] No server round-trips during slider interaction
- [ ] Works with both v2 (no sensitive events) and v3 data
- [ ] Tests pass
