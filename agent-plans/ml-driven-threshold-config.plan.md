# ML-Driven Threshold Configuration System

## Executive Summary

Transform hi-hat (and eventually all drum) detection from hardcoded thresholds to data-driven, ML-optimized configuration. Enable users to generate training data, run optimizers, and apply results to their projects through YAML configuration.

## Problem Statement

Current detection logic uses hardcoded thresholds:
```python
if sustain_ms > 90 and body_energy > 200 and peak_amplitude < 0.15:
    return "open"
```

Issues:
- Thresholds are guesses, not validated against real data
- Different songs need different sensitivity
- No user control without code changes
- Single-feature thresholds lack safety margins
- No way to learn from mistakes

## Goals

1. **Per-project configurability**: Each song can have optimal thresholds
2. **ML-driven discovery**: Let data determine thresholds, not guesses
3. **User-friendly workflow**: Non-coders can generate/apply optimal configs
4. **Safe defaults**: Ship with robust multi-feature rules with high margins
5. **Continuous improvement**: Aggregate learning across users/songs

## Architecture

### Phase 1: YAML Configuration System

#### 1.1 Enhanced midiconfig.yaml Structure

```yaml
# Root-level detection settings
detection_version: "2.0"  # Track config format version

# Per-instrument advanced thresholds
instruments:
  hihat:
    # Traditional single thresholds (backward compatible)
    threshold: 0.1
    delta: 0.01
    geomean_threshold: 50.0
    
    # NEW: ML-optimized classification rules
    classification:
      # Which rule to use: "simple", "balanced", "robust", "custom"
      mode: "balanced"
      
      # Pre-configured rules (from ML optimization)
      rules:
        simple:
          # Single feature - easy to understand, less safe
          features: ["GeoMean"]
          thresholds:
            GeoMean: 535
          margin_score: 3.9
          
        balanced:
          # Two features - good safety, moderate complexity
          features: ["GeoMean", "SustainMs"]
          thresholds:
            GeoMean: 300
            SustainMs: 80
          margin_score: 47.0
          
        robust:
          # Three features - maximum safety, more complex
          features: ["GeoMean", "SustainMs", "BodyE"]
          thresholds:
            GeoMean: 300
            SustainMs: 80
            BodyE: 50
          margin_score: 49.3
          
        custom:
          # User can define their own after running optimizer
          features: ["GeoMean", "SustainMs"]
          thresholds:
            GeoMean: 400
            SustainMs: 100
          margin_score: null  # Unknown until validated
    
    # Open/closed classification
    open_sustain_ms: 90  # DEPRECATED - use classification rules instead
    
  snare:
    # Future: Similar ML-optimized structure
    threshold: 0.3
    
  kick:
    # Future: Similar ML-optimized structure
    threshold: 0.3
```

#### 1.2 Detection Code Integration

**File**: `stems_to_midi/detection.py`

```python
def detect_hihat_state(
    onset_times: np.ndarray,
    audio_data: np.ndarray,
    sr: int,
    config: Dict
) -> List[str]:
    """
    Classify hi-hat hits as open or closed using ML-optimized rules.
    """
    # Extract spectral features for each hit
    features = []
    for onset_time in onset_times:
        body_e, sizzle_e, sustain_ms = analyze_hihat_spectrum(...)
        geomean = np.sqrt(body_e * sizzle_e)
        
        features.append({
            'BodyE': body_e,
            'SizzleE': sizzle_e,
            'GeoMean': geomean,
            'SustainMs': sustain_ms
        })
    
    # Get classification rule from config
    rule_mode = config.get('classification', {}).get('mode', 'balanced')
    rule = config.get('classification', {}).get('rules', {}).get(rule_mode, {})
    
    # Apply multi-feature rule
    states = []
    for feat_dict in features:
        is_open = classify_hihat(feat_dict, rule)
        states.append('open' if is_open else 'closed')
    
    return states


def classify_hihat(features: Dict, rule: Dict) -> bool:
    """
    Apply multi-feature classification rule.
    
    Args:
        features: Dict with BodyE, SizzleE, GeoMean, SustainMs, etc.
        rule: Dict with 'features' list and 'thresholds' dict
    
    Returns:
        True if open hi-hat, False if closed
    """
    required_features = rule.get('features', ['GeoMean'])
    thresholds = rule.get('thresholds', {})
    
    # ALL threshold conditions must be met (AND logic)
    for feature_name in required_features:
        if feature_name not in features or feature_name not in thresholds:
            continue
            
        if features[feature_name] <= thresholds[feature_name]:
            return False  # Failed threshold, must be closed
    
    return True  # All thresholds passed, it's open
```

### Phase 2: User Workflow Tools

#### 2.1 Data Collection Helper

**File**: `debugging_scripts/generate_training_data.py`

```python
"""
Extract training data from stems_to_midi output for ML optimization.

Usage:
    python generate_training_data.py --project 4 --stem hihat --output data.csv
"""

import subprocess
import re
import pandas as pd

def extract_spectral_data(project_num: int, stem: str) -> pd.DataFrame:
    """
    Run stems_to_midi and parse spectral analysis output.
    """
    # Run stems_to_midi with output capture
    result = subprocess.run(
        ['python', 'stems_to_midi.py', str(project_num), '--stems', stem],
        capture_output=True,
        text=True
    )
    
    # Parse the spectral analysis table
    # Look for "ALL DETECTED ONSETS - SPECTRAL ANALYSIS" section
    # Extract CSV data
    
    return df

def interactive_labeling(df: pd.DataFrame, audio_path: str) -> List[float]:
    """
    Help user identify open hi-hat timestamps.
    
    Could integrate with audio playback, waveform visualization, etc.
    """
    print("Listen to the audio and note timestamps of open hi-hats.")
    print(f"Audio file: {audio_path}")
    print("\nDetected hits at these times:")
    print(df['Time'].tolist())
    
    # Future: Launch audio player, allow clicking on waveform
    
    timestamps = input("\nEnter open hi-hat times (comma-separated): ")
    return [float(t.strip()) for t in timestamps.split(',')]
```

#### 2.2 Optimization Wizard

**File**: `debugging_scripts/optimize_config.py`

```python
"""
End-to-end wizard for generating optimal midiconfig.yaml values.

Usage:
    python optimize_config.py --project 4 --instrument hihat
"""

def run_optimization_wizard(project_num: int, instrument: str):
    """
    Interactive wizard that:
    1. Generates training data
    2. Prompts for labeled examples
    3. Runs classifier for validation
    4. Runs optimizer for best thresholds
    5. Updates project midiconfig.yaml
    """
    
    print(f"=== Optimization Wizard for Project {project_num} - {instrument} ===\n")
    
    # Step 1: Generate data
    print("Step 1: Generating spectral analysis data...")
    data_file = generate_training_data(project_num, instrument)
    print(f"✓ Data saved to: {data_file}\n")
    
    # Step 2: Label examples
    print("Step 2: Identifying open hi-hats...")
    open_timestamps = interactive_labeling(data_file)
    print(f"✓ Labeled {len(open_timestamps)} open hi-hats\n")
    
    # Step 3: Run classifier
    print("Step 3: Analyzing patterns...")
    feature_importance = run_classifier(data_file, open_timestamps)
    print("✓ Feature importance:")
    for feat, importance in feature_importance.items():
        print(f"  {feat}: {importance:.1%}")
    print()
    
    # Step 4: Run optimizer
    print("Step 4: Finding optimal thresholds...")
    best_rules = run_optimizer(data_file, open_timestamps)
    print("✓ Top 3 rules found:\n")
    for i, rule in enumerate(best_rules[:3], 1):
        print(f"  {i}. {rule['name']}: {rule['description']}")
        print(f"     Margin: {rule['margin_score']:.1f}%\n")
    
    # Step 5: Choose and apply
    choice = input("Which rule to apply? (1-3, or 'custom'): ")
    selected_rule = best_rules[int(choice) - 1]
    
    print(f"\nStep 5: Updating midiconfig.yaml...")
    update_midiconfig(project_num, instrument, selected_rule)
    print("✓ Configuration updated!\n")
    
    # Step 6: Validate
    print("Step 6: Re-running stems_to_midi to validate...")
    validate_results(project_num, instrument)
    print("✓ Complete! Check your MIDI output.")
```

#### 2.3 Configuration Manager

**File**: `config_manager.py` (root level)

```python
"""
Manage midiconfig.yaml with ML-optimized rules.
"""

def update_classification_rule(
    config_path: Path,
    instrument: str,
    rule_name: str,
    features: List[str],
    thresholds: Dict[str, float],
    margin_score: float = None
):
    """
    Update or add a classification rule to midiconfig.yaml.
    
    Args:
        config_path: Path to midiconfig.yaml
        instrument: 'hihat', 'snare', etc.
        rule_name: 'simple', 'balanced', 'robust', 'custom'
        features: List of feature names
        thresholds: Dict mapping feature names to threshold values
        margin_score: Optional safety margin score
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Ensure structure exists
    if 'instruments' not in config:
        config['instruments'] = {}
    if instrument not in config['instruments']:
        config['instruments'][instrument] = {}
    if 'classification' not in config['instruments'][instrument]:
        config['instruments'][instrument]['classification'] = {}
    if 'rules' not in config['instruments'][instrument]['classification']:
        config['instruments'][instrument]['classification']['rules'] = {}
    
    # Add/update rule
    config['instruments'][instrument]['classification']['rules'][rule_name] = {
        'features': features,
        'thresholds': thresholds,
        'margin_score': margin_score
    }
    
    # Set as active if custom
    if rule_name == 'custom':
        config['instruments'][instrument]['classification']['mode'] = 'custom'
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
```

### Phase 3: User Interface

#### 3.1 Command-Line Interface

```bash
# Quick optimization for current project
python optimize_config.py --project 4 --instrument hihat

# Generate data only (for manual analysis)
python generate_training_data.py --project 4 --stem hihat --output hihat_data.csv

# Apply pre-computed rule
python config_manager.py set-rule --project 4 --instrument hihat --rule balanced

# Validate current config
python config_manager.py validate --project 4
```

#### 3.2 Future: Web UI

```
┌─────────────────────────────────────────┐
│  Project 4: Taylor Swift - Ruin...     │
│  ┌───────────────────────────────────┐ │
│  │ Hi-Hat Detection Optimizer        │ │
│  ├───────────────────────────────────┤ │
│  │ 1. Generate Training Data   [Run] │ │
│  │    ✓ 160 hits detected            │ │
│  │                                   │ │
│  │ 2. Label Open Hi-Hats             │ │
│  │    🎵 [Waveform with markers]     │ │
│  │    Click to mark open hi-hats     │ │
│  │    Labeled: 6 examples            │ │
│  │                                   │ │
│  │ 3. Optimize Thresholds      [Run] │ │
│  │    Top Rules:                     │ │
│  │    ○ Simple   (3.9% margin)       │ │
│  │    ● Balanced (47% margin) ←      │ │
│  │    ○ Robust   (49% margin)        │ │
│  │                                   │ │
│  │ 4. Apply & Test          [Apply]  │ │
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

### Phase 4: Data Aggregation & Universal Model

#### 4.1 Local Training Data Collection

**File**: `debugging_scripts/export_training_data.py`

```python
"""
Export anonymized training data for universal model building.
"""

def export_for_sharing(
    data_csv: Path,
    labeled_timestamps: List[float],
    metadata: Dict
) -> Dict:
    """
    Create shareable training data package.
    
    Args:
        data_csv: Path to local data.csv
        labeled_timestamps: User-confirmed open hi-hat times
        metadata: Song info (tempo, genre, etc.)
    
    Returns:
        Anonymized data package
    """
    df = pd.read_csv(data_csv)
    
    # Label the data
    df['OpenHH'] = df['Time'].round(3).isin(labeled_timestamps).astype(int)
    
    # Anonymize - remove identifying info
    package = {
        'version': '1.0',
        'instrument': 'hihat',
        'features': df[['BodyE', 'SizzleE', 'GeoMean', 'SustainMs', 'OpenHH']].to_dict('records'),
        'metadata': {
            'tempo': metadata.get('tempo'),
            'genre': metadata.get('genre'),
            'sample_count': len(df),
            'open_count': df['OpenHH'].sum(),
            'timestamp': datetime.now().isoformat()
        }
    }
    
    # Save locally
    export_file = Path('training_exports') / f'hihat_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    export_file.parent.mkdir(exist_ok=True)
    with open(export_file, 'w') as f:
        json.dump(package, f, indent=2)
    
    return package
```

#### 4.2 Online Aggregation Service (Future)

**Concept**: `https://api.larsnet-midi.com/training-data`

```python
# User submits training data
POST /api/v1/training-data
{
  "instrument": "hihat",
  "features": [...],
  "metadata": {...},
  "user_consent": true
}

# System aggregates across submissions
# Periodically retrains universal model

# Users download updated model
GET /api/v1/models/hihat/latest
{
  "version": "2.1.0",
  "trained_on": "10,000 songs, 1.5M samples",
  "rules": {
    "simple": {...},
    "balanced": {...},
    "robust": {...}
  },
  "performance": {
    "accuracy": 0.987,
    "recall": 0.992,
    "precision": 0.983
  }
}
```

**Privacy Considerations**:
- All data anonymized (no filenames, user info)
- Opt-in only
- Features only (no audio)
- User can review data before submission
- Open source model training (reproducible)

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Design YAML schema for classification rules
- [ ] Implement `classify_hihat()` function with multi-feature support
- [ ] Update `detect_hihat_state()` to use config-driven rules
- [ ] Add default "simple", "balanced", "robust" rules to root midiconfig.yaml
- [ ] Test with project 4 to validate

### Phase 2: User Tools (Week 3-4)
- [ ] Create `generate_training_data.py` script
- [ ] Create `optimize_config.py` wizard
- [ ] Create `config_manager.py` utilities
- [ ] Document workflow in debugging_scripts/README.md
- [ ] Create tutorial video/guide

### Phase 3: Refinement (Week 5-6)
- [ ] Add config validation (detect invalid threshold combinations)
- [ ] Add performance metrics logging (track accuracy per project)
- [ ] Create config migration tool (update old configs to new format)
- [ ] Add config presets for common genres/styles

### Phase 4: Expansion (Future)
- [ ] Extend to snare classification (rimshot vs center hit)
- [ ] Extend to cymbal classification (crash vs ride)
- [ ] Build web UI for non-technical users
- [ ] Create training data aggregation service
- [ ] Implement universal model training pipeline

## Success Criteria

1. **User can optimize in <5 minutes**: From data generation to applied config
2. **Better accuracy**: ML-driven thresholds outperform hardcoded values
3. **No code changes needed**: Everything configurable through YAML
4. **Safe defaults**: Ship robust rules that work for 80% of songs out-of-box
5. **Continuous improvement**: System learns from each new song processed

## Migration Strategy

### Backward Compatibility

Old configs still work:
```yaml
hihat:
  open_sustain_ms: 90
```

New configs enhance:
```yaml
hihat:
  classification:
    mode: "balanced"
    rules:
      balanced:
        features: ["GeoMean", "SustainMs"]
        thresholds: {GeoMean: 300, SustainMs: 80}
```

Detection code checks for new format first, falls back to old.

## Risk Mitigation

**Risk**: Overfitting to single song
- **Mitigation**: High margin scores required, validate on multiple songs

**Risk**: User confusion with complex YAML
- **Mitigation**: Wizard generates config, user just chooses simple/balanced/robust

**Risk**: ML overkill for simple problem
- **Mitigation**: Start simple (2-3 features), expand only if needed

**Risk**: Performance degradation
- **Mitigation**: Benchmark all changes, cache spectral calculations

## Future Enhancements

1. **Active Learning**: System suggests uncertain cases for user labeling
2. **Transfer Learning**: Use pre-trained model as starting point
3. **Real-time Feedback**: Live preview while adjusting thresholds
4. **A/B Testing**: Compare before/after MIDI outputs visually
5. **Community Presets**: Share/download configs for specific genres

## Documentation Needs

1. User guide: "Optimizing Hi-Hat Detection for Your Song"
2. Developer guide: "Adding ML-Optimized Classification to New Instruments"
3. Config reference: "midiconfig.yaml Classification Rules Schema"
4. API reference: "Training Data Submission Format"

## Related Files

- `/stems_to_midi/detection.py` - Core detection logic
- `/midiconfig.yaml` - Root configuration template
- `/debugging_scripts/` - All ML tools
- `/.github/instructions/` - Process documentation
