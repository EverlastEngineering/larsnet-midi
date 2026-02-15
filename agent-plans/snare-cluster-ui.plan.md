# Snare Cluster UI Plan

## Problem
1. Snare classification defaults to `expected_clusters=1` → all events get note 38 (no differentiation)
2. Even with clusters>1, clustering uses `spectral_centroid_hz` which is a continuous distribution with no meaningful separation for snare
3. The real distinguishing feature is `stereo_width` (mono snare ~0.03 vs wide clap ~0.33)
4. UI shows a slider for cluster count but no visibility into what the clusters actually are
5. Tooltip doesn't show pan_confidence or stereo_width

## Design

### Phase 1: Tooltip Enhancement
Add `pan_confidence` and `stereo_width` to the event hover tooltip in waveform.js.

### Phase 2: Fix Snare Classification Feature
Change `classify_snare_notes` to cluster on `stereo_width` instead of `spectral_centroid_hz`.
Default snare `expected_clusters` to 2 (snare + clap).

### Phase 3: Cluster Visibility API
Enhance `/api/reclassify` to return cluster metadata alongside events:
- Per-cluster: size, distinguishing feature name + value range, centroid values
- The classifier determines which feature best separates the clusters (stereo_width, pan_confidence, spectral_centroid_hz, etc.)

### Phase 4: Cluster UI
Replace the simple slider with:
1. Cluster count selector (dropdown or small number picker, 1-4)
2. Cluster cards showing: count, distinguishing feature + range, assigned note
3. Per-cluster note dropdown (Snare/Rimshot/Clap/Clap+Snare)

### Scope
- Focus on snare for stem-specific behavior
- Tooltip enhancement applies to all stems
- Cluster visibility design should be extensible to toms/cymbals later

## Risks
- Multi-feature clustering (stereo_width + spectral_centroid) may be needed later but start with single best feature
- Note dropdown per cluster requires new config storage mechanism for cluster→note mapping
