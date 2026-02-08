# Plan: Interactive Detection Tuning System

Build a system where users visually tune detection and filtering thresholds on per-stem waveforms in the WebUI, then "proof" results via a velocity-1 MIDI workflow to build a community training dataset.

## Steps

### 1. Standardize naming end-to-end

Replace generic `primary`/`secondary`/`tertiary` keys with domain-specific names (`fundamental_energy`, `body_energy`, `wire_energy`, `sizzle_energy`, `brilliance_energy`, `attack_energy`) in `get_spectral_config_for_stem()`, propagate through `analyze_onset_spectral()`, `filter_onsets_by_spectral()`, `save_analysis_sidecar()`, and `midiconfig.yaml`. Remove the misleading `body_wire_geomean` variable name. Tag each field in analysis.json with a `category` of `detection`, `filtering`, or `classification`.

### 2. Persist waveform + envelope data during conversion

After `_load_and_validate_audio()` and `calculate_energy_envelope()`, save the energy envelope array and sample rate as a lightweight binary (numpy `.npz`) alongside the analysis.json. This gives the WebUI waveform data without re-running detection. Also save the raw onset list from "max sensitivity" detection (see step 3).

### 3. Dual-sensitivity detection run

During `process_stem_to_midi()`, run energy detection twice: once at max sensitivity (`threshold_db=1.0`, `min_absolute_energy=0.0001`) to get *all possible* events, and once at configured settings. Store both in analysis.json under `events_sensitive` and `events_configured`. This replaces the librosa-only learning mode with an energy-detection equivalent.

### 4. WebUI waveform visualization

Add a waveform viewer component (using wavesurfer.js or HTML5 Canvas) to the WebUI that loads the persisted envelope data and overlays onset markers from analysis.json. Display three visual layers: detection markers (vertical lines), filtering decisions (color-coded: green=KEPT, red=FILTERED, orange=REVERB_CONTINUATION), and the geomean/threshold reference line.

### 5. Interactive threshold sliders

Add client-side sliders for the key parameters (`threshold_db`, `geomean_threshold`, `reverb_continuation_attack_threshold`, `min_sustain_ms`) that re-filter the `events_sensitive` dataset in the browser (no server round-trip). Moving a slider instantly shows events appearing/disappearing on the waveform. The `events_sensitive` array has all spectral features pre-computed, so filtering is just comparisons.

### 6. Velocity-1 proofing MIDI workflow

Add a "Proofing Export" button that generates a MIDI file with configured events at normal velocity + all sensitive-only events at velocity 1. User edits in DAW (delete noise, promote good hits), re-imports. Store the proofed MIDI alongside the original analysis.json as a labeled training pair (`{stem}.proofed.mid` + `{stem}.analysis.json`). Build a simple schema for this dataset.

## Further Considerations

### Waveform library choice

wavesurfer.js gives zoom/scroll/regions for free but loads full audio. An alternative is server-rendered PNG waveforms (like SoundCloud) which are lighter but lose interactivity. Canvas-based custom rendering gives most control. Recommendation: wavesurfer.js for Phase 1, custom Canvas for Phase 2.

### Client-side vs server-side re-filtering

If the sensitive detection produces ~500–2000 events per stem with ~15 numeric fields each, that's <100KB JSON — easily filterable in the browser. No server round-trip needed for threshold preview. Only need a server call to save final settings or export MIDI.

### Training data format for future inference model

Each proofed pair (analysis.json + proofed.mid) needs a standard schema: audio hash (for dedup), stem type, config snapshot, original event count, proofed event count, per-event label (true_positive / false_positive / false_negative). This should be defined early so community contributions are consistent.
