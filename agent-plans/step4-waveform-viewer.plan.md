# Plan: Step 4 — WebUI Waveform Visualization

## Goal

Add a waveform viewer to the WebUI that displays per-stem energy envelopes with color-coded onset markers from analysis.json. This is the visual foundation for Step 5 (interactive threshold sliders).

## Architecture Decisions

### Rendering: HTML5 Canvas (not wavesurfer.js)

wavesurfer.js loads full audio files (potentially 100+ MB WAVs) and targets audio playback, not data visualization. The envelope data is already computed and persisted as lightweight .npz arrays (~50KB per stem). Canvas gives full control over drawing layers and is a better fit for overlaying onset markers, threshold lines, and interactive elements (Step 5).

### Data Flow

```
Browser                        Server
──────                        ──────
GET /api/projects/N/analysis → reads .analysis.json → JSON response
                              (events_configured, events_sensitive, logic)

GET /api/projects/N/envelope/STEM → reads .npz → JSON {times, left, right, sr}
                                    (numpy arrays → float32 lists)
```

### Format Compatibility

Support both v2 (single `events` array) and v3 (`events_configured` + `events_sensitive`). The viewer degrades gracefully — v2 shows events without sensitive overlay, v3 shows both layers.

## Components

### Backend (webui/api/projects.py)
- `GET /api/projects/N/analysis` — serves analysis.json as JSON
- `GET /api/projects/N/envelope/<stem_type>` — converts .npz to JSON
- Add `has_analysis` and `analysis_stems` to project detail response

### Frontend
- `webui/static/js/waveform.js` — Canvas rendering engine
- Collapsible "Detection Analysis" section in index.html
- Stem tab bar for switching between kick/snare/hihat/cymbals/toms
- Color legend: green=KEPT, red=FILTERED, orange=REVERB_CONTINUATION

### Visual Layers (bottom to top)
1. **Background**: Dark canvas with time axis
2. **Envelope**: L/R energy as filled area (blue tones)
3. **Threshold line**: Geomean threshold as horizontal dashed line
4. **Onset markers**: Vertical lines color-coded by status
5. **Hover tooltip**: Event details on marker hover (time, velocity, spectral features)

## Risks

- Large envelope arrays (300+ second songs at 86 fps ≈ 26K samples) could slow Canvas. Mitigation: downsample to canvas pixel width before drawing.
- No .npz files exist yet for projects processed before Step 2. Mitigation: envelope section shows "Re-run MIDI conversion to generate envelope data" message.
- v2 analysis.json uses old field names (primary/secondary/tertiary). Mitigation: display raw field names, don't assume naming.

## Success Criteria

- Waveform viewer renders for any project with analysis.json
- Onset markers are color-coded correctly
- Envelope overlay works when .npz data is available
- Graceful degradation when data is missing
- All existing tests pass
- New API endpoint tests pass
