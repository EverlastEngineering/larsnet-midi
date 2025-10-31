# ML-Driven Threshold Configuration - Results & Progress

## Current Status: Planning & Prototyping Complete ✓

**Date Started**: 2025-10-31  
**Current Phase**: Phase 0 - Discovery & Validation  
**Next Phase**: Phase 1 - Foundation Implementation

## Completed Work

### ✓ Discovery & Validation (2025-10-31)

**Problem Identified**:
- Hi-hat open/closed classification using hardcoded thresholds
- Current logic: `sustain_ms > 90 AND body_energy > 200 AND peak_amplitude < 0.15`
- Issue with project 4: Missing open hi-hats because body_energy=134 < 200

**Tools Developed**:

1. **`debugging_scripts/classifier.py`** ✓
   - Random Forest for feature importance ranking
   - Decision Tree for interpretable rules
   - Manual threshold combination testing
   - Results: GeoMean (24.7%) and Total (21.5%) most important features

2. **`debugging_scripts/threshold_optimizer.py`** ✓
   - Exhaustive grid search across threshold combinations
   - Tested 372 different rules
   - Found 141 rules with 100% accuracy
   - Ranked by safety margin score
   
3. **`debugging_scripts/data.csv`** ✓
   - 160 hi-hat hits from project 4
   - 6 labeled open hi-hats (1.962s, 7.755s, 13.944s, 19.574s, 25.960s, 31.753s)
   - Full spectral features: BodyE, SizzleE, GeoMean, SustainMs, etc.

4. **`debugging_scripts/README.md`** ✓
   - Complete documentation of tools and workflow
   - Step-by-step usage guide
   - Scientific explanations of concepts
   - Integration guidance

5. **`agent-plans/ml-driven-threshold-config.plan.md`** ✓
   - Comprehensive implementation plan
   - 4-phase rollout strategy
   - YAML schema design
   - User workflow design
   - Future aggregation service concept

**Key Findings**:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Best Rule | `GeoMean > 300 AND SustainMs > 80 AND BodyE > 50` | 3-feature combination |
| Safety Margin | 49.3% | Excellent safety buffer |
| Accuracy | 100% | Perfect on training data |
| Feature Importance | GeoMean: 24.7%, Total: 21.5%, SizzleE: 18.7% | Spectral energy matters most |

**Validation Results**:
- Single-feature rule (`GeoMean > 535`) only 3.9% margin - RISKY ⚠️
- Two-feature rule (`GeoMean > 300 AND SustainMs > 80`) 47% margin - SAFE ✓
- Three-feature rule (above) 49.3% margin - SAFEST ✓✓

**Decision**: Use multi-feature AND logic with low thresholds for best safety margins

## Decisions Made

### Architecture Decisions

1. **YAML-driven configuration** ✓
   - Store rules in midiconfig.yaml per instrument
   - Support multiple rule modes: simple, balanced, robust, custom
   - Backward compatible with existing configs

2. **Multi-feature classification** ✓
   - Use AND logic across features (all must pass)
   - Lower individual thresholds = higher combined safety
   - 2-3 features optimal (more = diminishing returns)

3. **Grid search optimization** ✓
   - Exhaustive is feasible (< 2000 combinations)
   - Better than gradient descent (interpretable thresholds)
   - Safety margin as primary ranking metric

4. **Local-first approach** ✓
   - Tools run locally, no cloud dependency
   - Privacy-preserving (features only, no audio)
   - Optional data sharing for universal model

### User Experience Decisions

1. **Wizard-based workflow** ✓
   - Single command to run full optimization
   - Interactive labeling with audio playback
   - Auto-generates config updates

2. **Pre-configured presets** ✓
   - Ship with validated simple/balanced/robust rules
   - Users choose complexity level
   - Can override with custom if needed

3. **Progressive enhancement** ✓
   - Start with hi-hat (most requested)
   - Extend to snare, cymbals later
   - Learn patterns across instruments

## Metrics & Performance

### Training Data (Project 4)
- **Total samples**: 160 hi-hat hits
- **Positive class**: 6 open hi-hats (3.75%)
- **Negative class**: 154 closed hi-hats (96.25%)
- **Class imbalance**: 25.7:1 (handled with `class_weight='balanced'`)

### Feature Statistics

**Open Hi-Hats**:
- GeoMean: 546.2 - 906.2 (mean: 738.7)
- SustainMs: 156.5 - 198.7 (mean: 182.7)
- BodyE: 108.7 - 284.1 (mean: 194.4)

**Closed Hi-Hats**:
- GeoMean: 0.6 - 523.7 (mean: 75.1)
- SustainMs: 3.0 - 199.8 (mean: 38.2)
- BodyE: 0.3 - 142.6 (mean: 20.3)

**Separation Gap**: 22.5 GeoMean units between max closed (523.7) and min open (546.2)

### Model Performance

**Random Forest** (500 trees):
- Training accuracy: 100%
- Test accuracy: 100%
- Feature importance validated

**Decision Tree** (max_depth=3):
- Training accuracy: 100%
- Rule: Single threshold on GeoMean
- Margin: Too small (3.9%)

**Grid Search Optimizer**:
- Rules tested: 372
- Perfect rules found: 141
- Best margin: 49.3%
- Runtime: < 5 seconds

## Phase Completion Status

### Phase 0: Discovery & Prototyping ✓
- [x] Identify problem with current detection
- [x] Collect training data from project 4
- [x] Build classifier for pattern discovery
- [x] Build optimizer for exhaustive search
- [x] Validate ML approach works
- [x] Document tools and workflow
- [x] Create implementation plan

### Phase 1: Foundation (Not Started)
- [ ] Design YAML schema for classification rules
- [ ] Implement `classify_hihat()` with multi-feature support
- [ ] Update `detect_hihat_state()` to use config rules
- [ ] Add default rules to root midiconfig.yaml
- [ ] Test with project 4
- [ ] Validate against more projects

### Phase 2: User Tools (Not Started)
- [ ] Create `generate_training_data.py`
- [ ] Create `optimize_config.py` wizard
- [ ] Create `config_manager.py` utilities
- [ ] Add interactive labeling support
- [ ] Create tutorial documentation

### Phase 3: Refinement (Not Started)
- [ ] Config validation
- [ ] Performance metrics logging
- [ ] Config migration tool
- [ ] Genre-specific presets

### Phase 4: Expansion (Not Started)
- [ ] Extend to snare classification
- [ ] Extend to cymbal classification
- [ ] Web UI
- [ ] Training data aggregation service
- [ ] Universal model training

## Blockers & Risks

### Current Blockers
None - ready to proceed with Phase 1

### Potential Risks

1. **Overfitting Risk**: LOW ✓
   - Mitigation: High margin scores required (>30%)
   - Validation: Test on multiple songs before production

2. **User Confusion Risk**: MEDIUM ⚠️
   - Mitigation: Wizard handles complexity, user just labels examples
   - Need: Clear documentation and tutorial video

3. **Performance Risk**: LOW ✓
   - Grid search completes in seconds
   - Spectral analysis already implemented
   - Marginal overhead from multi-feature checks

4. **Backward Compatibility Risk**: LOW ✓
   - Plan includes fallback to old threshold logic
   - Migration path documented

## Next Steps

### Immediate (This Week)
1. Implement YAML schema in root midiconfig.yaml
2. Update detection.py with multi-feature classification
3. Test on project 4 to validate improvement
4. Document changes in commit message

### Short-term (Next 2 Weeks)
1. Build optimize_config.py wizard
2. Create generate_training_data.py helper
3. Write user tutorial guide
4. Test on 3-5 different projects/genres

### Long-term (Next Month)
1. Extend to snare and cymbal classification
2. Build training data export functionality
3. Design aggregation service API
4. Plan web UI mockups

## Lessons Learned

### What Worked Well ✓
- **Grid search**: Exhaustive works great for this problem size
- **Margin scoring**: Clear metric for ranking rule safety
- **Multi-feature AND**: Lower thresholds + AND logic = higher margins
- **Visualization**: Feature importance helps understand what matters

### What Could Improve ⚠️
- **More training data**: 6 examples is minimal, need 20+ for robustness
- **Cross-validation**: Should test on held-out songs
- **Feature engineering**: Could explore more spectral bands
- **Real-time feedback**: Would help during labeling

### Surprising Findings 💡
- SustainMs less important than expected (only 13% vs GeoMean's 25%)
- Lower thresholds paradoxically safer when combined (AND logic)
- Body energy alone inadequate (134 < 200 caused false negative)
- GeoMean captures interaction better than individual energies

## Data Collection Log

| Date | Project | Instrument | Samples | Open Examples | Notes |
|------|---------|------------|---------|---------------|-------|
| 2025-10-31 | 4 | hihat | 160 | 6 | Taylor Swift - Ruin The Friendship |

## Model Versions

| Version | Date | Rules | Accuracy | Margin | Notes |
|---------|------|-------|----------|--------|-------|
| 0.1-prototype | 2025-10-31 | 3 presets | 100% | 49.3% | Initial ML-optimized rules from project 4 |

## References

- Original issue: Open hi-hat at 13.944s missed due to body_energy < 200
- Classifier output: `debugging_scripts/classifier.py` stdout
- Optimizer output: `debugging_scripts/optimal_rules.csv`
- Planning doc: `agent-plans/ml-driven-threshold-config.plan.md`

## Future Considerations

1. **Deep learning**: If >1000 songs, consider neural network
2. **Real-time optimization**: Auto-suggest rules during playback
3. **Active learning**: System identifies uncertain cases for labeling
4. **Transfer learning**: Pre-train on large dataset, fine-tune per song
5. **Multi-task learning**: Train across instruments simultaneously

---

**Last Updated**: 2025-10-31  
**Updated By**: AI Assistant + User  
**Status**: Discovery phase complete, ready for implementation
