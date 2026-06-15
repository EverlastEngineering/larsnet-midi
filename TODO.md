# TODO: MIDI Conversion Improvements

---

## New June 12 2026

enable pitch estimate even if pitch_confidence is anything above 0. we don't need to KNOW the exact pitch, we just need it as a classifier.





## Strategic Direction: Minimize Time-to-Usable-MIDI

**Key Insight**: Audio-to-MIDI drum detection is a *bounded* problem. Perfect detection is impossible - even humans disagree on ghost notes, rimshots vs cross-sticks, and bleed interpretation. Chasing detection accuracy has diminishing returns.

**Reframed Goal**: Minimize **total time** from audio to usable MIDI, not maximize detection accuracy. 85% accuracy with 30-minute corrections beats 95% accuracy requiring 2 hours of tweaking.

**Two-Track Approach**:
1. **Core Tech** - Fundamental improvements to detection quality
2. **Workflow** - Streamlined human-in-the-loop correction process

---

## Track 1: Core Tech Improvements

### Stereo Channel Exploitation
**Status**: Not Started  
**Impact**: High (untapped signal)

Currently we sum stereo to mono, discarding spatial information. Drums are often panned:
- Toms: Hard left to right
- Cymbals: Spread across stereo field  
- Hi-hat: Often slightly off-center
- Kick/Snare: Usually centered

**Implementation**:
- [ ] Process each stem THREE times: left channel, right channel, mono sum
- [ ] Compare detections across all three
- [ ] Hits appearing in only L or R are likely panned instruments
- [ ] Hits in mono but not L/R might be center-panned (kick, snare)
- [ ] Use stereo position as classification feature
- [ ] Add `stereo_analysis: true` config option
- [ ] Output stereo position metadata with each hit

**Expected Benefit**: Better separation of overlapping instruments (e.g., hi-hat under crash, tom fills with cymbals).

---

### Pattern Recognition for "Obviously Missing" Hits
**Status**: Not Started  
**Impact**: High (fixes systematic detection failures)

Some missed hits are contextually obvious to humans:
- Hi-hat "missing" where crash overlaps (hi-hat shows at lower confidence)
- Consistent 8th-note hi-hat pattern with one note dropped
- Ghost snare in an otherwise regular backbeat pattern

**Implementation**:
- [ ] Build temporal pattern model from high-confidence hits
- [ ] Detect rhythmic grid (16th notes, 8ths, etc.)
- [ ] Find low-confidence hits that fit the pattern
- [ ] Promote low-confidence hits when pattern suggests they're real
- [ ] Flag pattern breaks for human review (might be intentional)
- [ ] Add `pattern_promotion_threshold` config (e.g., 0.6 → 0.85 if pattern matches)

**Algorithm Sketch**:
```
1. Extract high-confidence hits (>90%)
2. Detect dominant rhythmic interval (quantize to grid)
3. Find "holes" in the pattern
4. Check if low-confidence hit exists within tolerance of hole
5. If yes, promote to higher confidence
6. If pattern break with no low-conf hit, flag as "intentional break"
```

---

### Instrument Clustering (Limited Set Per Stem)
**Status**: Not Started  
**Impact**: Medium (reduces classification confusion)

Each stem typically has only 2-4 distinct instrument sounds:
- Cymbals: 2-3 types (crash, ride, china)
- Toms: 2-4 drums
- Snare-ish: snare center, rimshot, cross-stick, clap
- Hi-hat: open, closed, foot

**Implementation**:
- [ ] Cluster all hits by spectral fingerprint (k-means or DBSCAN)
- [ ] Present clusters to user: "I found 3 distinct sounds in this stem"
- [ ] User labels each cluster once → all hits in cluster get that label
- [ ] Outliers flagged for individual review
- [ ] Save cluster fingerprints for future songs (same kit = same clusters)

**Expected Benefit**: Instead of classifying 500 individual hits, user classifies 3-5 clusters.

---

### Confidence-Based Tiered Output
**Status**: Partially exists (learning mode has some of this)  
**Impact**: High (reduces review burden)

Every detection should output a confidence score. Processing differs by tier:
- **High confidence (>90%)**: Auto-accept, no review needed
- **Medium confidence (60-90%)**: Include in output, flag for optional review
- **Low confidence (30-60%)**: Exclude from primary output, include in "all hits" output
- **Very low (<30%)**: Likely noise, discard

**Implementation**:
- [ ] Standardize confidence score calculation across all stems
- [ ] Confidence = f(geomean, pattern_fit, stereo_consistency, velocity_curve)
- [ ] Output two MIDI files: `song_best.mid` and `song_all_hits.mid`
- [ ] Include confidence as MIDI CC or in separate metadata file
- [ ] Configurable tier thresholds per instrument

---

### Ensemble Onset Detection (Multi-Algorithm Voting)
**Status**: Not Started  
**Impact**: High (consensus = confidence)

Currently we use librosa's onset detection exclusively. Other algorithms exist:
- **librosa.onset.onset_detect** - current default
- **librosa Superflux** - designed for drums/percussive, handles vibrato
- **torchaudio** - already in environment, unused for onset detection
- **audioflux** - specialized audio analysis library
- **madmom** - neural network-based beat/onset detection
- **essentia** - comprehensive MIR library with multiple onset methods

**Key Insight**: If 4/5 algorithms agree on a hit, confidence is high. If only 1/5 detects it, it's likely noise or borderline.

**Implementation**:
- [ ] Audit current onset detection code (what's actually used?)
- [ ] Implement adapter for each algorithm with common interface
- [ ] Run all algorithms, collect onset times
- [ ] Cluster nearby onsets (within 20-30ms) as "same hit"
- [ ] Confidence = (algorithms_agreeing / total_algorithms)
- [ ] Combine with spectral features for final confidence score
- [ ] Make algorithm selection configurable (speed vs accuracy tradeoff)

**Algorithm Characteristics**:
| Algorithm | Speed | Drums | Soft Attacks | Notes |
|-----------|-------|-------|--------------|-------|
| librosa default | Fast | Good | OK | Current choice |
| Superflux | Fast | Excellent | Good | Designed for percussive |
| torchaudio | Medium | Good | Good | GPU acceleration possible |
| madmom | Slow | Excellent | Excellent | Neural network, most accurate |
| essentia | Medium | Good | Good | Many sub-methods |

**Quick Win**: Try Superflux first - it's already in librosa, specifically designed for percussive onsets, and might improve detection with zero new dependencies.

```python
# Current (probably):
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)

# Superflux alternative:
onset_env = librosa.onset.onset_strength_multi(
    y=y, sr=sr, 
    aggregate=np.median,  # Superflux uses median
    channels=[0, 32, 64, 96, 128]  # Multiple frequency bands
)
onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
```

**Unused Dependencies to Investigate**:
- `scikit-learn` - in environment.yml but seemingly unused. Could be used for clustering, classification
- `torchaudio` - in environment.yml, used for MDX separation but not onset detection

---

### Detection Output Contract (Standard Data Types)
**Status**: Not Started  
**Impact**: Critical (enables all other improvements)

Before implementing multiple algorithms or workflow improvements, define a standard output contract. All detection methods output the same data structure, enabling:
- Algorithm swapping without changing downstream code
- Ensemble voting (combine outputs from multiple detectors)
- Consistent workflow tools (analysis, comparison, visualization)
- Future algorithm improvements slot in seamlessly

**Proposed Contract** (see `agent-plans/detection-contract.plan.md` for full spec):

```python
@dataclass
class DetectedHit:
    """Single detected drum hit with metadata."""
    time_seconds: float          # Onset time in seconds
    instrument: str              # e.g., "snare", "kick", "hihat_closed"
    confidence: float            # 0.0-1.0, how certain we are
    velocity: int                # MIDI velocity 1-127
    
    # Optional metadata for analysis/debugging
    algorithm: str | None        # Which detector found this
    spectral_features: dict | None  # Raw features (geomean, frequencies, etc.)
    stereo_position: float | None   # -1.0 (left) to 1.0 (right)
    pattern_context: str | None     # "on_beat", "off_beat", "ghost", etc.

@dataclass  
class DetectionResult:
    """Output from any detection algorithm."""
    hits: list[DetectedHit]
    stem_type: str               # "snare", "kick", "toms", etc.
    audio_duration: float        # Total length in seconds
    sample_rate: int
    algorithm_name: str
    algorithm_version: str
    parameters: dict             # Thresholds/settings used
    
    # For analysis
    all_onsets: list[DetectedHit] | None  # Pre-threshold hits (for "all hits" output)
    threshold_report: ThresholdReport | None
```

**Implementation Steps**:
- [ ] Define DetectedHit and DetectionResult in `midi_types.py`
- [ ] Add ThresholdReport type for transparency output
- [ ] Refactor current detection to output DetectionResult
- [ ] All workflow tools consume DetectionResult
- [ ] Each new algorithm implements same interface
- [ ] Add serialization (JSON/pickle) for caching results

**Why This Matters**:
Once the contract exists:
- Ensemble voting = merge multiple `DetectionResult` objects
- Stereo analysis = compare L/R/mono `DetectionResult` objects  
- Pattern recognition = analyze `hits` list for rhythmic patterns
- Workflow tools = all operate on same `DetectionResult` type
- A/B testing = swap algorithms, compare outputs directly

---

### ML-Based Hit Classification (Supervised Learning)
**Status**: Not Started  
**Impact**: High (learns patterns humans can't manually encode)
**Depends On**: Detection Output Contract (need standardized feature vectors)

**Key Distinction**:
- **Threshold tuning** (Bayesian): "Given geomean, what cutoff works?" → Optimizes ONE number
- **ML classification**: "Given ALL features, is this a real hit?" → Learns PATTERNS in feature space

ML can discover nonlinear combinations of features that humans wouldn't think to check. Example: a hit might be real if (geomean > 150 AND stereo_position near center) OR (geomean > 100 AND fits_pattern AND attack_sharpness > 0.8).

**Training Data Source**:
Every user correction is labeled training data:
- User keeps hit → Label: 1 (real hit)
- User deletes hit → Label: 0 (false positive)
- User adds hit from all-hits → Label: 1 (missed real hit)
- Hit in all-hits user ignores → Label: 0 (correctly rejected)

**Feature Vector Per Onset** (input to classifier):
```python
features = {
    # Spectral features (current)
    'geomean': 185.3,
    'fundamental_energy': 0.72,
    'body_energy': 0.45,
    'attack_energy': 0.88,
    'brightness': 0.23,
    
    # Temporal features
    'onset_sharpness': 0.91,      # How sudden is the attack?
    'decay_rate': 0.15,           # How quickly does it fade?
    'time_since_last_hit': 0.125, # Seconds since previous onset
    
    # Stereo features (new)
    'stereo_position': 0.0,       # -1 to 1
    'stereo_correlation': 0.95,   # L/R similarity
    
    # Pattern features
    'distance_to_grid': 0.008,    # Seconds off from nearest 16th
    'pattern_fit_score': 0.92,    # Does it fit detected pattern?
    
    # Cross-stem features
    'kick_overlap': 0.0,          # Is there a kick at same time?
    'snare_overlap': 0.3,         # Snare proximity
}
```

**Model Options** (scikit-learn already in environment.yml):
| Model | Pros | Cons | Best For |
|-------|------|------|----------|
| Random Forest | Interpretable, handles mixed features | Needs tuning | First attempt |
| Gradient Boosting | High accuracy | Slower, less interpretable | Production |
| Logistic Regression | Very fast, very interpretable | Linear only | Baseline |
| Neural Network | Learns complex patterns | Needs more data | After 50+ songs |

**Terminology - What is this approach called?**

| Approach | What It Is | Our Use |
|----------|------------|---------|
| **Rule-based / Heuristic** | Hand-coded if/then logic | Current: `if geomean > 180` |
| **Classical ML / Traditional ML** | Learned classifier on extracted features | ← This section |
| **Deep Learning** | Neural networks on raw/minimal preprocessing | Future possibility |

This is **classical ML** or **traditional ML classification**. It IS technically "inference" (using a trained model), but:
- **Lightweight**: Model is 1-5 MB, runs instantly on CPU
- **Interpretable**: Can inspect which features matter most
- **No GPU needed**: Unlike deep learning inference
- **Fast to train**: Minutes on a laptop, not hours on cloud GPUs

**How Users Would Use Pre-trained Models (.pkl files)**:

```python
# === CURRENT APPROACH (rule-based) ===
# Hard-coded threshold on single feature
if onset.geomean > config['snare']['geomean_threshold']:  # e.g., 180
    include_hit()

# === ML APPROACH (classical ML) ===
import joblib

# Load pre-trained model once at startup (1-5 MB file)
model = joblib.load('models/snare_acoustic_v1.pkl')

# For each onset, extract features and ask model
features = extract_features(onset)  # Returns ~20-50 floats
feature_array = np.array([list(features.values())])

# Model returns probability that this is a real hit
confidence = model.predict_proba(feature_array)[0][1]

if confidence > 0.7:  # Configurable confidence threshold
    include_hit(velocity=..., confidence=confidence)
```

**User workflow to use community models**:
1. Download `snare_acoustic_v1.pkl` from releases (~2 MB)
2. Place in `models/` folder (or set path in config)
3. Config: `snare.classifier_model: models/snare_acoustic_v1.pkl`
4. Run detection as normal - code uses model instead of threshold
5. Model outputs confidence scores based on ALL features, not just geomean

**What's inside the .pkl file**:
```python
# Training (done once, by us or community):
from sklearn.ensemble import RandomForestClassifier
import joblib

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)  # X = feature vectors, y = labels (0/1)
joblib.dump(model, 'snare_acoustic_v1.pkl')

# The .pkl contains:
# - The trained decision trees (learned from corrections)
# - Feature names and their learned importance weights
# - Everything needed to classify new onsets
```

**Implementation**:
- [ ] Define feature extraction function (onset → feature vector)
- [ ] Create training data capture during correction workflow
- [ ] Store training data per-kit and globally (different kits = different models)
- [ ] Train simple Random Forest classifier as baseline
- [ ] Compare ML predictions vs threshold-based predictions
- [ ] If ML wins, use ML confidence as primary signal
- [ ] Retrain periodically as more corrections accumulate

**Training Data Schema**:
```python
@dataclass
class TrainingExample:
    """One labeled onset for ML training."""
    features: dict[str, float]
    label: int  # 1 = real hit, 0 = false positive
    
    # Metadata for analysis
    song_id: str
    stem_type: str
    time_seconds: float
    user_action: str  # "kept", "deleted", "added"
    
@dataclass
class TrainingDataset:
    """Accumulated training data."""
    examples: list[TrainingExample]
    kit_id: str | None  # None = global, else specific drum kit
    created: datetime
    last_updated: datetime
    
    def to_sklearn(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to X, y arrays for sklearn."""
        X = np.array([list(e.features.values()) for e in self.examples])
        y = np.array([e.label for e in self.examples])
        return X, y
```

**Expected Progression**:
1. Songs 1-5: Not enough data, use threshold-based detection
2. Songs 6-20: Train simple model, compare to thresholds, learn which works better
3. Songs 20+: ML model likely outperforms thresholds for this kit
4. New kit: Start over (or use global model as starting point)

**Quick Win**: Even without full ML, just *capturing* the training data is valuable. Start saving (features, label) pairs now, train models later.

---

### Community Training Data (Copyright-Safe Sharing)
**Status**: Future Vision  
**Impact**: Very High (crowdsourced improvement)
**Depends On**: ML-Based Hit Classification, Detection Output Contract

**Key Insight**: Spectral features are NOT copyrighted audio. They are:
- **Derived data** - legally distinct from source material
- **Tiny** - ~50-100 floats per onset vs millions of audio samples
- **Non-reversible** - cannot reconstruct original audio
- **Sufficient** - contain everything needed for ML classification

This is how Shazam shares fingerprints (not songs), Spotify publishes audio features (not tracks), and research datasets distribute MFCCs (not recordings).

**What Users Would Share**:
```python
@dataclass
class CommunityContribution:
    """Anonymized, copyright-safe training data."""
    # Feature vectors (no audio)
    examples: list[TrainingExample]  # features + labels only
    
    # Metadata (anonymized)
    stem_type: str           # "snare", "kick", etc.
    genre_hint: str | None   # "metal", "jazz", "pop" (optional)
    kit_type: str | None     # "acoustic", "electronic", "hybrid"
    contributor_id: str      # Anonymous hash, for tracking quality
    
    # Quality metrics
    song_count: int          # How many songs this came from
    correction_rate: float   # % of hits user modified (quality signal)
    
    # NOT included:
    # - Song name, artist, album
    # - Raw audio or waveforms
    # - Timestamps (relative timing only, offset removed)
    # - Any identifying information
```

**File Size Comparison**:
| Data Type | Size per Song | Shareable? |
|-----------|---------------|------------|
| Raw audio (WAV) | 50-500 MB | ❌ Copyright |
| Compressed audio (MP3) | 5-50 MB | ❌ Copyright |
| Spectral features (all onsets) | 50-500 KB | ✅ Derived data |
| Training examples (corrections only) | 5-50 KB | ✅ Derived data |

**Community Infrastructure** (future):
- [ ] Define contribution format (JSON/protobuf)
- [ ] Add "export training data" button to workflow
- [ ] Create simple upload endpoint (GitHub releases? S3?)
- [ ] Build aggregation pipeline (merge contributions)
- [ ] Train global models from community data
- [ ] Publish pre-trained models for common scenarios
- [ ] Quality filtering (reject low-quality contributions)
- [ ] Contributor leaderboard/attribution

**Privacy Considerations**:
- Remove absolute timestamps (shift all to start at 0)
- Hash contributor IDs (no real usernames)
- No song/artist metadata
- Optional genre/kit hints only
- Users opt-in explicitly

**Model Distribution**:
Once community data accumulates:
- Publish pre-trained models: `snare_acoustic_v1.pkl`, `kick_metal_v2.pkl`
- Users download model matching their use case
- Model improves with each release as more data arrives
- "Your corrections help everyone" → motivation to contribute

**Legal Note**: Not legal advice, but derived features (MFCCs, spectral centroids, onset envelopes) are generally considered transformative and not subject to the original copyright. Many academic audio datasets work this way. Worth consulting a lawyer if this scales significantly.

---

## Track 2: Workflow Improvements

### The Correction Flywheel
**Status**: Fragmented (pieces exist, not integrated)  
**Impact**: Very High (this is the real unlock)

The existing tools (learning mode, analysis scripts, threshold tuning) need to form a cohesive workflow where each correction improves future detections.

**Target Workflow**:
```
1. GENERATE → Best-guess MIDI + all-hits MIDI + threshold report
2. REVIEW   → User imports both into DAW, makes quick edits
3. ANALYZE  → Tool compares edited MIDI to all-hits, finds optimal thresholds
4. APPLY    → New thresholds applied, model improves
5. REPEAT   → Each song makes the next one easier
```

---

### Step 1: Generate with Transparency
**Status**: Not Started

When generating MIDI, also output:
- [ ] Threshold values that produced this result (per instrument)
- [ ] Distribution histogram of detected values (terminal ASCII + optional image)
- [ ] Suggested thresholds based on percentiles
- [ ] "All hits" MIDI file with everything above noise floor
- [ ] Metadata file linking each hit to its confidence/values

**Example Output**:
```
Snare Detection Report:
  Threshold: 180 (geomean)
  Total onsets: 847
  Passed threshold: 312 (selected)
  Just below (150-180): 45 (might be ghost notes?)
  Well below (100-150): 89 (likely bleed)
  Noise floor (<100): 401 (rejected)
  
  Distribution:
  0-100:   ████████████████████ (401)
  100-150: ████ (89)
  150-180: ██ (45) ← borderline
  180-250: ████████ (156) ✓ selected
  250+:    ████████ (156) ✓ selected
  
  Files generated:
  - snare_best.mid (312 notes)
  - snare_all.mid (847 notes, velocities = confidence)
  - snare_report.json (all values + metadata)
```

---

### Step 2: DAW Review Process
**Status**: Needs documentation

Streamlined DAW workflow:
- [ ] Document recommended DAW setup (2 tracks: best + all)
- [ ] "All hits" uses velocity = confidence (quiet = uncertain)
- [ ] User workflow: mute best, listen to all, copy missing hits to best
- [ ] Time-aligned so user can visually compare
- [ ] Color coding suggestions for different DAWs

---

### Step 3: Analyze User Corrections
**Status**: Partially exists (learning mode)

After user edits the MIDI:
- [ ] Compare edited MIDI to original "all hits"
- [ ] For each user-added note: find the closest detection in all-hits
- [ ] Calculate what threshold would have included it
- [ ] For each user-deleted note: mark as false positive
- [ ] Use Bayesian optimization to find threshold that maximizes:
  - Includes all user-kept notes
  - Excludes all user-deleted notes
  - Suggests optimal threshold + confidence interval

**Output**:
```
Correction Analysis:
  User kept: 298 of 312 best-guess notes
  User added: 23 notes from all-hits
  User deleted: 14 notes (false positives)
  
  Optimal threshold: 165 (was 180)
    - Would include 22/23 user-added notes
    - Would still exclude 12/14 false positives
    - 2 notes require manual review regardless
  
  Suggested config update:
  snare:
    geomean_threshold: 165  # was 180
```

---

### Step 4: Training Data Capture
**Status**: Not Started

Every correction is supervised training data:
- [ ] Save (audio_features, user_label) pairs from corrections
- [ ] Build per-kit, per-song, and global training sets
- [ ] Periodic model retraining from accumulated corrections
- [ ] Track improvement over time (accuracy vs song count)

---

## Existing Tools Inventory (Needs Audit)

These tools exist but may be broken/outdated:
- [ ] Learning mode (`--learning`) - needs verification
- [ ] Threshold analysis scripts - location unknown
- [ ] Hit comparison tools - referenced in old docs
- [ ] WEBUI settings interface - currently broken

**Action**: Audit each tool, document state, fix or remove.

---

## High Priority

### 1. Kick Drum - Spectral Filtering
**Status**: Not Started  
**Complexity**: Easy (copy snare logic)

Apply the same spectral filtering approach used for snare to kick drum:
- [x] Snare uses geometric mean of body (150-400Hz) + wires (2-8kHz)
- [ ] Kick should use fundamental (40-80Hz) + body/attack (80-150Hz)
- [ ] Define frequency ranges in config (kick_freq_min, kick_body_min, etc.)
- [ ] Calculate kick_geomean = sqrt(fundamental_energy * body_energy)
- [ ] Add kick.geomean_threshold to midiconfig.yaml
- [ ] Support learning mode for kick
- [ ] Test and document typical threshold values

**Rationale**: Kick has similar issues with snare bleed and artifacts. Low-frequency fundamental distinguishes real kicks.

**Config additions needed**:
```yaml
kick:
  midi_note: 36
  geomean_threshold: 150.0  # TBD from testing
  fundamental_freq_min: 40
  fundamental_freq_max: 80
  body_freq_min: 80
  body_freq_max: 150
```

---

### 2. Toms - Pitch Detection & Multi-Note Mapping
**Status**: Not Started  
**Complexity**: Medium (pitch detection + note mapping)

Toms span a wide pitch range (low/mid/high). Need to:
- [ ] Implement pitch detection (librosa.piptrack or autocorrelation)
- [ ] Detect fundamental frequency of each tom hit
- [ ] Classify into low/mid/high based on frequency ranges
- [ ] Map to different MIDI notes:
  - Low tom: 45 (A1) or 41 (F1)
  - Mid tom: 47 (B1) or 48 (C2)
  - High tom: 50 (D2) or 43 (G#1)
- [ ] Add configurable pitch ranges to config
- [ ] Support learning mode (learns pitch ranges from user edits)
- [ ] Optional: Support 2-tom, 3-tom, 4-tom setups

**Challenges**:
- Tom pitch can vary within a hit (pitch bend)
- Need to distinguish toms from kick/snare
- Some setups have 2 toms, others have 4+

**Config additions needed**:
```yaml
toms:
  # Multi-pitch detection
  detect_multiple_pitches: true
  
  # Pitch ranges (Hz) - adjust based on your drum kit
  low_tom_min: 80
  low_tom_max: 120
  mid_tom_min: 120
  mid_tom_max: 180
  high_tom_min: 180
  high_tom_max: 300
  
  # MIDI note mapping
  low_tom_note: 45    # A1
  mid_tom_note: 47    # B1
  high_tom_note: 50   # D2
  
  # Filtering
  geomean_threshold: 100.0  # Similar to snare/kick
  body_freq_min: 80
  body_freq_max: 300
  attack_freq_min: 1000
  attack_freq_max: 5000
```

---

### 3. Hi-Hat & Cymbals - Spectral Separation
**Status**: Partially Implemented (open/closed detection exists)  
**Complexity**: High (very similar spectral content)

**Current State**:
- Open/closed hi-hat detection exists (decay-based)
- No filtering for artifacts
- Cymbals use separate stem but no intelligence

**Improvements Needed**:

#### Hi-Hat:
- [ ] Apply spectral filtering (reject kicks/snare bleed)
- [ ] Improve open/closed detection (currently decay-based)
- [ ] Add foot hi-hat detection (different spectral signature)
- [ ] Add geomean_threshold for hi-hat
- [ ] Frequency ranges: focus on 5-15kHz (bright metal sounds)

#### Cymbals:
- [ ] Separate crash vs ride detection
- [ ] Pitch-based classification (ride ~400Hz, crash ~550Hz fundamentals)
- [ ] Decay analysis (crashes sustain longer)
- [ ] Map to different notes:
  - Crash 1: 49 (C#2)
  - Crash 2: 57 (A2)
  - Ride: 51 (D#2)
  - Ride bell: 53 (F2)
  - China: 52 (E2)
- [ ] Apply spectral filtering

**Challenges**:
- Hi-hat and cymbals have very similar bright/high-frequency content
- Overlapping hits (crash while hi-hat playing)
- Many cymbal types with subtle differences
- User might want to manually adjust in DAW anyway

**Config additions needed**:
```yaml
hihat:
  midi_note: 42              # Closed hi-hat
  midi_note_open: 46         # Open hi-hat
  midi_note_foot: 44         # Pedal hi-hat
  
  geomean_threshold: 80.0    # Lower than snare (lighter hits)
  
  # Spectral ranges (hi-hat is bright, 5-15kHz dominant)
  body_freq_min: 5000
  body_freq_max: 10000
  bright_freq_min: 10000
  bright_freq_max: 15000
  
  # Open/closed detection
  detect_open: true
  decay_threshold: 0.65
  open_decay_min: 0.3        # Minimum decay for open hi-hat

cymbals:
  # Cymbal type detection
  detect_cymbal_types: true
  
  # Pitch ranges for classification (fundamental frequencies)
  ride_pitch_min: 350
  ride_pitch_max: 450
  crash_pitch_min: 500
  crash_pitch_max: 650
  china_pitch_min: 400
  china_pitch_max: 550
  
  # MIDI note mapping
  crash1_note: 49
  crash2_note: 57
  ride_note: 51
  ride_bell_note: 53
  china_note: 52
  
  # Filtering
  geomean_threshold: 100.0
  body_freq_min: 400
  body_freq_max: 1000
  bright_freq_min: 5000
  bright_freq_max: 15000
```

---

## Medium Priority

### 4. User-Friendly Threshold Adjustment
**Status**: Not Started  
**Complexity**: Easy to Medium

Make it easier for users to adjust thresholds without learning mode:

- [ ] Add interactive threshold adjustment mode
- [ ] `--preview` flag: Shows all detections with thresholds in terminal
- [ ] Real-time threshold adjustment: user types new value, see results immediately
- [ ] Visual histogram of GeoMean values (ASCII art in terminal)
- [ ] Percentile-based suggestions (e.g., "90% of hits are above 250")
- [ ] Save adjusted thresholds to config

**Example workflow**:
```bash
python stems_to_midi.py --preview -i cleaned_stems/ --stems snare

# Shows:
# GeoMean distribution:
# 0-100:   ████████ (45 hits) - likely artifacts
# 100-200: ███ (18 hits) - borderline
# 200-300: ██████████████ (120 hits) - real snares
# 300-400: ████████████████ (180 hits) - real snares
# 400+:    ████████ (67 hits) - loud snares
# 
# Current threshold: 100
# Suggested: 180 (excludes bottom 10%)
# 
# Enter new threshold (or press Enter to accept): _
```

---

### 5. Multi-Stem Coordination
**Status**: Not Started  
**Complexity**: Medium

Sometimes onsets overlap across stems. Coordinate detection:

- [ ] Detect simultaneous hits across stems (within 20ms)
- [ ] Handle overlapping hits (kick + snare on same beat)
- [ ] Ensure MIDI timing is synchronized across stems
- [ ] Option to "lock" timing to kick (everything aligns to kick hits)
- [ ] Crossfade analysis (does kick have snare frequencies? Reduce both)

---

### 6. Ghost Note Detection
**Status**: Not Started  
**Complexity**: Medium

Quiet articulations between main hits:

- [ ] Detect very low amplitude hits (currently filtered out)
- [ ] Separate threshold for ghost notes
- [ ] Map to lower velocity (20-40)
- [ ] User option to include/exclude ghost notes
- [ ] Special handling for snare ghost notes

---

### 7. Flam & Drag Detection
**Status**: Not Started  
**Complexity**: High

Drum rudiments with multiple very close hits:

- [ ] Detect flams (2 hits <30ms apart)
- [ ] Detect drags (3+ hits <20ms apart)
- [ ] Option to merge into single MIDI note or keep separate
- [ ] Grace note MIDI notation
- [ ] Velocity adjustment for grace notes

---

## Low Priority / Nice to Have

### 8. Sheet Music Export
**Status**: Not Started  
**Complexity**: Medium to High

Export detected drum patterns as readable sheet music:

- [ ] Generate drum notation (standard 5-line staff)
- [ ] MusicXML export (can be opened in MuseScore, Finale, Sibelius)
- [ ] LilyPond format export (text-based music engraving)
- [ ] PDF generation with proper drum notation
- [ ] Handle drum notation conventions:
  - Kick on bottom space (F)
  - Snare on 3rd space (C)
  - Hi-hat above staff with stem up
  - Toms on appropriate lines
  - Cymbals with x noteheads
- [ ] Tempo detection and time signature
- [ ] Automatic bar lines based on detected meter
- [ ] Grace notes for flams/drags
- [ ] Articulation marks (accents, ghost notes)
- [ ] Drum key/legend on first page
- [ ] Multi-page layout for long pieces

**Libraries to Consider**:
- `music21` - Python music analysis/generation
- `abjad` - Python interface to LilyPond
- `mingus` - Music theory and notation
- Direct MusicXML generation

**Example Usage**:
```bash
python stems_to_midi.py -i cleaned_stems/ -o output/ --export-sheet-music
# Generates: output/drums_score.pdf, output/drums_score.musicxml
```

---

### 9. Export Formats
- [ ] Export to different DAW project formats (Logic, Ableton)
- [ ] Export to Superior Drummer / EZdrummer format
- [ ] Export timing data for drum replacement plugins
- [ ] Export to Hydrogen drum machine format
- [ ] Export as Reaper notation items

### 9. Performance Metrics
- [ ] Calculate timing accuracy (deviation from grid)
- [ ] Detect tempo changes
- [ ] Generate "heat map" of where most hits occur
- [ ] Export performance statistics

### 10. Advanced Detection
- [ ] Brush vs stick detection (spectral difference)
- [ ] Rim shots vs center hits
- [ ] Cross-stick detection
- [ ] Stick click detection

### 11. Machine Learning
- [ ] Train ML model on user-edited MIDIs
- [ ] Auto-classify drum types without stems
- [ ] Learn user preferences over time
- [ ] Suggest corrections based on musical context

---

## Documentation Needed

- [ ] Update README with all features
- [ ] Video tutorial for learning mode
- [ ] Troubleshooting guide for each stem type
- [ ] Comparison with other drum-to-MIDI tools
- [ ] Best practices for recording drums for conversion
- [ ] Example configs for different drum styles (jazz, metal, rock, electronic)

---

## Testing & Validation

- [ ] Unit tests for spectral analysis functions
- [ ] Integration tests for learning mode
- [ ] Test on diverse drum recordings (genres, quality, mic setups)
- [ ] Benchmark against commercial tools (Drumgizmo, etc.)
- [ ] User testing with real drummers
- [ ] Performance optimization (currently slow for long files)

---

## Bugs / Issues

- [ ] Learning mode temporary config not working correctly
- [ ] Stereo handling in all code paths
- [ ] Velocity calculation consistency across stems
- [ ] MIDI timing precision (quantization needed?)
- [ ] Memory usage for very long audio files

---

## Cleanup / Maintenance

- [ ] Remove old requirements.txt (superseded by environment.yml)
- [ ] Clean up debug MIDI folders (midi_debug*, midi_test*, etc.)
- [ ] Add .gitignore entries for debug output folders
- [ ] Consolidate documentation (multiple guides exist)
- [ ] Remove unused imports
- [ ] Code style consistency check

---

## Current Status Summary

| Feature | Kick | Snare | Toms | Hi-Hat | Cymbals |
|---------|------|-------|------|---------|---------|
| Basic detection | ✅ | ✅ | ✅ | ✅ | ✅ |
| Spectral filtering | ❌ | ✅ | ❌ | ❌ | ❌ |
| Learning mode | ❌ | ✅ | ❌ | ❌ | ❌ |
| Velocity from spectral | ❌ | ✅ | ❌ | ❌ | ❌ |
| Multi-note mapping | N/A | N/A | ❌ | ⚠️ | ❌ |
| Artifact rejection | ❌ | ✅ | ❌ | ❌ | ❌ |

Legend: ✅ Complete | ⚠️ Partial | ❌ Not Started | N/A Not Applicable

---

## Next Steps

**Immediate (This Session)**:
1. Implement kick drum spectral filtering (copy snare approach)
2. Add kick to learning mode
3. Test kick filtering on sample audio

**Short Term**:
1. Implement tom pitch detection
2. Add multi-note mapping for toms
3. Test and tune threshold values

**Long Term**:
1. Hi-hat/cymbal improvements
2. User-friendly threshold adjustment
3. Comprehensive testing and documentation
