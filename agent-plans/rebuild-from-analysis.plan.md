# Rebuild-from-Analysis Pipeline — Plan

**Created**: 2026-02-13  
**Depends on**: `interactive-tuning.plan.md` (Phases 1-3: naming, envelope persistence, dual-sensitivity)  
**Branch**: `daw-waveform-detection`

## Problem Statement

Four related workflow problems share a common root cause — the reconversion pipeline always re-runs the full detection-to-MIDI chain:

1. **Slow iteration**: Changing a filtering threshold triggers ~30s of redundant detection work
2. **Lost overrides**: Click-to-toggle event edits are not honored during reconversion
3. **Manual refresh**: UI requires full page reload to display updated results
4. **Limited parameters**: Only a hardcoded subset of tuning sliders are exposed

## Key Architectural Insight

The infrastructure is largely in place:

| Component | Status | Location |
|-----------|--------|----------|
| `analysis.json` v3 with full spectral data per event | ✅ Exists | `midi/{name}.analysis.json` |
| Sensitive + configured event pools | ✅ Exists (Phase 3 of prior plan to complete) | `events_sensitive`, `events_configured` in analysis.json |
| `event_overrides.json` with click-to-toggle persistence | ✅ Exists | `midi/event_overrides.json` |
| Detection / filtering separation in functional core | ✅ Exists | `analysis_core.py`: `filter_onsets_by_spectral()` |
| `create_midi_file()` accepts pre-built event lists | ✅ Exists | `midi.py` |
| Client-side filter preview from analysis.json events | ✅ Exists | `threshold-tuning.js` |
| Envelope `.npz` persistence | ✅ Exists | `midi/{name}.{stem}.envelope.npz` |

**What's missing**: A rebuild entry point that re-filters from `analysis.json` → MIDI without re-detecting.

## Architecture

### Current Flow (Full Pipeline ~30s)
```
Audio .wav
  → load_audio → detect_onsets → analyze_spectral → filter → classify → MIDI
  → analysis.json (both event pools)
  → envelope.npz (per stem)
```

### Proposed Flow (Rebuild <1s)
```
analysis.json (events_configured + events_sensitive)
  + event_overrides.json (manual include/exclude)
  + midiconfig.yaml (current filter thresholds)
  → re-filter events → re-classify → MIDI
  → updated analysis.json (event statuses only)
```

Detection params still require the full pipeline. The UI communicates which path will run.

## Phases

### Phase 1: Rebuild-from-Analysis Backend

**Goal**: Pure function that re-filters cached detection results and regenerates MIDI.

**New module**: `stems_to_midi/rebuild_core.py` (functional core, no I/O)

```python
def rebuild_events_from_analysis(
    analysis_data: dict,           # Parsed analysis.json
    overrides: dict,               # {stem: {time_key: status}}
    config: dict,                  # Parsed midiconfig.yaml
    stem_types: list[str] | None,  # None = all stems
) -> tuple[dict, list]:           # (updated_analysis, midi_events_per_stem)
```

**Logic**:
1. For each stem (or subset if `stem_types` specified):
   - Build candidate event pool from `events_configured` + `events_sensitive`
   - Deduplicate by time (configured takes precedence, within merge window)
   - Apply overrides: events with override status bypass filtering entirely
   - Apply current config filter thresholds via existing `filter_onsets_by_spectral()`
   - Re-classify (hihat state, pitch) via existing classification functions
   - Normalize velocities
2. Return updated analysis data (event statuses changed) + MIDI-ready event lists
3. Caller handles I/O (writing MIDI file, updating analysis.json)

**New module**: `stems_to_midi/rebuild_shell.py` (I/O shell)

```python
def rebuild_midi_for_project(
    project_dir: Path,
    config_updates: list[dict] | None,  # [{path, value}] to apply before rebuild
    stem_types: list[str] | None,
    honor_overrides: bool = True,
) -> dict:                              # {success, stems_rebuilt, elapsed_ms}
```

**Logic**:
1. Apply config updates to project `midiconfig.yaml` (reuse existing YAML config engine)
2. Load `analysis.json`, validate version
3. Load `event_overrides.json` if `honor_overrides`
4. Call `rebuild_events_from_analysis()` 
5. Write updated MIDI file via `create_midi_file()`
6. Update `analysis.json` event statuses
7. Return result metadata

**Tests** (in `stems_to_midi/test_rebuild_core.py`):
- Rebuild with same params as detection → identical filtered events
- Override KEPT survives strict thresholds
- Override FILTERED survives permissive thresholds
- Per-stem rebuild only modifies targeted stem, others unchanged
- Missing analysis.json returns error (not crash)
- Version mismatch returns error with fallback instruction

### Phase 2: API Endpoint + UI Auto-Refresh

**Goal**: Wire rebuild into the WebUI with instant feedback (no page refresh, no SSE polling).

**API** (in `webui/operations_bp.py` or new blueprint):
- `POST /api/rebuild-midi/<project_id>`
- Body: `{ config_updates: [{path, value}], stem_types: ["kick"], honor_overrides: true }`
- Response: `{ success, stems_rebuilt, elapsed_ms, analysis_data }` (returns full updated analysis)
- Synchronous — no job queue needed for <1s operation
- Falls back to error with `"requires_full_pipeline": true` if analysis.json missing/stale

**UI changes** (`threshold-tuning.js`):
- `saveTuningAndReconvert()` detects whether changes are detection-only, filtering-only, or mixed:
  - **Filtering-only** → calls rebuild endpoint (synchronous, fast)
  - **Detection params changed** → calls existing full pipeline with job queue + SSE
  - **Mixed** → calls full pipeline (detection changes invalidate cached analysis)
- On rebuild success:
  - Update `window.__analysisData` from response
  - Re-apply overrides to in-memory events
  - Re-render event bars (bottom canvas panel) — call existing `drawEvents()`
  - Update slider "configured" values to match new config
  - Envelope waveform (top panel) unchanged — skip re-fetch
  - Flash success indicator with timing ("Rebuilt in 0.3s")
- No page refresh needed

**Categorize parameters**: Add metadata to distinguish detection vs filtering params. This can be a simple lookup in JS or derived from config YAML comments (tag `# category: detection` vs `# category: filtering`).

### Phase 3: Event Override Integration

**Goal**: Click-to-toggle overrides participate in rebuild and appear in MIDI output.

**Backend**:
- `rebuild_events_from_analysis()` already honors overrides (Phase 1)
- Overrides in final MIDI: overridden-KEPT events get normal velocity; overridden-FILTERED events are excluded

**UI**:
- Overridden events display with visual distinction:
  - Manual-include: green bar with blue border/diamond marker
  - Manual-exclude: faded/strikethrough bar
- Rebuild response updates override visual state
- "Clear Overrides" button in tuning panel header (next to Save & Reconvert)
- When full re-detection is triggered, prompt: "Re-detection will clear manual event edits. Continue?"

**Override lifecycle**:
- Created: user clicks event bar → immediate visual update + debounced save to `event_overrides.json`
- Honored: rebuild reads overrides → events bypass filter logic → appear in MIDI
- Cleared: user clicks "Clear Overrides" or confirms before full re-detection

### Phase 4: Dynamic Parameter Exposure

**Goal**: Users can tune any filtering parameter, not just the hardcoded set.

**Parameter schema** (derived from `midiconfig.yaml` + analysis.json `logic` block):
- Each parameter has: `name`, `type`, `current_value`, `range` (from YAML comment), `category` (detection/filtering), `description` (from YAML comment first line), `stem_type`
- Serve via API: `GET /api/config/<project_id>/tuning-schema` returns categorized parameter list

**UI**:
- Default view: current slider set (filtering params for selected stem)
- "Show More" expander reveals additional filtering params
- "Advanced" toggle shows detection params with warning badge: "Changes require full re-detection (~30s)"
- Sliders auto-generated from schema — no JS hardcoding per parameter
- Detection param changes switch the Save button to "Save & Re-detect" (full pipeline)
- Filtering param changes show "Save & Rebuild" (fast path)

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Filtering logic divergence between full pipeline and rebuild | Different MIDI output for same params | Rebuild calls the same `filter_onsets_by_spectral()` function — single implementation |
| analysis.json missing (old projects, failed conversions) | Rebuild unavailable | Detect and fall back to full pipeline with user notification |
| Sensitive events lack metadata needed for classification | Incomplete rebuild | Verify sensitive detection in Phase 3 of prior plan stores all fields |
| Time-keyed overrides fragile across re-detection | User loses manual work | Overrides only apply in rebuild path; full re-detection prompts to clear |
| Synchronous rebuild endpoint blocks Flask worker | Slow for large projects | Profiling shows filtering + MIDI write is O(events), ~500ms for 2000 events. If needed, add async later |

## Success Criteria

1. **Speed**: Filter-only rebuild completes in <1 second (vs ~30s full pipeline)
2. **Correctness**: `full_pipeline(audio, config)` == `rebuild(analysis_from_same_run, same_config)` for event statuses
3. **Override persistence**: Click-toggled events survive rebuilds and appear in final MIDI
4. **No refresh**: UI updates inline after rebuild — events re-render, no page reload
5. **Parameter coverage**: Any filtering parameter from midiconfig.yaml is tunable without JS changes

## Out of Scope

- Real-time audio preview of parameter changes
- Undo/redo for parameter changes or overrides
- Incremental detection (only re-detecting part of the audio)
- Training data / velocity-1 proofing workflow (separate plan)
