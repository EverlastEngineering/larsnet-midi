# Threshold Learning System - Plan

## Goal
Make it easy for YOU to generate optimal defaults for each stem type by learning from labeled examples.

## Scope
- MVP: Hi-hat (open/closed) only - we have working process
- Future: Cymbals (assignment), Snare (vs claps), Toms (count/assignment)
- Architecture supports all stem types, implement one at a time

## Input/Output Pattern
```bash
optimize_thresholds.py 4 --stem hihat
# Reads: user_files/4/stems/hihat.wav + user_files/4/midiconfig.yaml
# Outputs: optimized thresholds → midiconfig.yaml
```

## Core Workflow

### 1. Extract Features
- Run detection pipeline on stem (same as normal processing)
- Capture ALL onset candidates with spectral features BEFORE filtering
- Export as CSV with columns: Time, Str, Amp, BodyE, SizzleE, Total, GeoMean, SustainMs, Status
- Include "best guess" classification from current thresholds

### 2. Label Ground Truth
- User opens: hihat.wav (audio) + hihat_features.csv (data)
- CSV has columns: Time, [...features...], CurrentGuess, GroundTruth
- User marks GroundTruth: "Open", "Closed", "No Hit" (filtered out)
- Save labeled CSV

### 3. Optimize Thresholds
- Read labeled CSV
- Run Bayesian optimization (proven best from debugging scripts)
- Find thresholds that match GroundTruth labels
- Maximize safety margin

### 4. Export Configuration
- Update project's midiconfig.yaml with new thresholds
- Add metadata: date, source, margin score
- Optionally save to presets/hihat_default.yaml for global use

## File Structure
```
stems_to_midi/
  optimization/
    __init__.py
    core.py                    # BayesianOptimizer (stem-agnostic)
    extract_features.py        # Generate CSV from stem
    interactive_labeler.py     # CLI for marking ground truth
    apply_thresholds.py        # Update midiconfig.yaml
    
    configs/                   # Stem type definitions
      hihat.yaml               # features, ranges, classification_type
      kick.yaml                # (future)
      snare.yaml               # (future)
    
    presets/                   # Learned defaults
      hihat_default.yaml
```

## Stem Type Configuration (hihat.yaml)
```yaml
stem_type: hihat
classification_type: binary
classes: [open, closed]
features:
  - name: GeoMean
    range: [200, 700]
  - name: SustainMs
    range: [60, 200]
  - name: BodyE
    range: [30, 250]
objective: maximize_margin
```

## Commands

### Step 1: Extract
```bash
python -m stems_to_midi.optimization.extract_features 4 --stem hihat
# Output: user_files/4/optimization/hihat_features.csv
```

### Step 2: Label (Interactive CLI)
```bash
python -m stems_to_midi.optimization.label 4 --stem hihat
# Prompts: "Time 1.962s - Open/Closed/NoHit? [Current: Open]"
# Plays audio snippet around timestamp
# Saves to: user_files/4/optimization/hihat_labeled.csv
```

### Step 3: Optimize
```bash
python -m stems_to_midi.optimization.optimize 4 --stem hihat
# Reads: hihat_labeled.csv + configs/hihat.yaml
# Runs Bayesian optimization
# Output: user_files/4/optimization/hihat_optimal.yaml
```

### Step 4: Apply
```bash
python -m stems_to_midi.optimization.apply 4 --stem hihat
# Updates: user_files/4/midiconfig.yaml with optimal thresholds
```

### All-in-one (with manual labeling step)
```bash
python optimize_thresholds.py 4 --stem hihat
# Runs steps 1,2,3,4 with prompts between each
```

## Key Design Principles

### Functional Core / Imperative Shell
- `core.py`: Pure optimization math (testable, no I/O)
- Other modules: I/O, user interaction, file management

### Capture BEFORE Filtering
Critical: Extract features from ALL onset candidates before velocity/strength filtering happens in detection pipeline. Otherwise we can't learn what should have been kept.

### Prefill with Current Logic
CSV includes "CurrentGuess" column populated by existing threshold logic. User corrects mistakes. Makes labeling fast.

### Metadata Tracking
```yaml
hihat:
  classification:
    geomean_threshold: 300
    sustain_threshold_ms: 80
    body_energy_threshold: 50
    # Metadata
    optimized_date: "2024-10-31"
    source_project: 4
    margin_score: 49.3
    training_samples: 160
```

## Implementation Phases

### Phase 1: Infrastructure (MVP)
- Extract features to CSV (before filtering)
- Stem config system (hihat.yaml)
- BayesianOptimizer core (generic)
- Apply thresholds to midiconfig.yaml

### Phase 2: Labeling UX
- CLI labeler with audio playback
- Show current guess, prompt for correction
- Quick keys: o=open, c=closed, x=no hit, Enter=accept guess

### Phase 3: Testing & Refinement
- Test on 2-3 projects
- Generate default hihat preset
- Validate margin scores

### Phase 4: Other Stems (Future)
- Add configs/kick.yaml, snare.yaml, etc.
- Different classification types (multiclass, assignment)
- Reuse same infrastructure

## Success Criteria
- ✓ Extract features from project 4 hihat in <5s
- ✓ Label 160 samples in <3 minutes (with prefill)
- ✓ Optimization completes in <10s
- ✓ Generated thresholds match debugging_scripts results (49.3% margin)
- ✓ Updated midiconfig.yaml produces correct MIDI output

## Dependencies
- scikit-optimize (make required, document in setup)
- soundfile or librosa (for audio playback in labeler)

## Risk Mitigation
- **Risk**: Filtering happens before feature extraction
  - **Mitigation**: Modify detection.py to optionally capture pre-filter data
- **Risk**: Different projects have different characteristics
  - **Mitigation**: Start with single project, expand after validation
- **Risk**: Labeling is tedious
  - **Mitigation**: Prefill with current logic, only correct errors

## Future Extensions (Out of Scope for MVP)
- Web UI for labeling
- Batch processing multiple projects
- Active learning (suggest which samples to label)
- Transfer learning from other projects
- Community preset sharing

---
**Status**: Planning
**Next Action**: Create Phase 1 implementation plan
