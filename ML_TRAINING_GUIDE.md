# Machine Learning Training Guide for stems_to_midi

## Overview

The `stems_to_midi` tool can learn your preferences for drum hit detection by training a machine learning classifier on your manually edited MIDI files. This is especially useful because different drum recordings, playing styles, and genres have vastly different characteristics.

## Why Train a Custom Model?

- **Varied drum sounds**: Jazz cymbals vs metal cymbals sound completely different
- **Recording quality**: Studio recordings vs live recordings have different noise profiles
- **Playing style**: Soft brushwork vs hard hitting requires different sensitivity
- **Your preferences**: You might want to keep subtle ghost notes that others would remove

Instead of manually tuning thresholds for each song, train once on representative examples and let the ML model generalize.

## The Training Workflow

### Phase 1: Generate Initial MIDI Files (Learning Mode)

1. **Process multiple songs** with default settings to generate MIDI files:
   ```bash
   # Process 3-5 representative songs from your collection
   python stems_to_midi.py --input "separated_stems/Song1/Song1-cymbals.wav" \
                           --output "training_data/raw_midi/Song1-cymbals.mid" \
                           --stem cymbals
   
   python stems_to_midi.py --input "separated_stems/Song2/Song2-cymbals.wav" \
                           --output "training_data/raw_midi/Song2-cymbals.mid" \
                           --stem cymbals
   
   # Repeat for 3-5 songs total
   ```

2. **Review the console output** for each song to see:
   - How many hits were detected
   - The spectral analysis values (GeoMean, Sustain, etc.)
   - Any obvious false positives or missed hits

3. **Goal**: Generate 50-150 total detections across all songs
   - Too few (<30): Model won't have enough data to learn
   - Just right (50-150): Good balance for training
   - Too many (>200): Diminishing returns, takes longer to edit

### Phase 2: Edit MIDI Files in Your DAW

1. **Open each MIDI file** in Logic Pro, Ableton, Pro Tools, etc.

2. **Load the original audio** alongside the MIDI to compare timing

3. **Edit the MIDI**:
   - ✓ **Keep** hits that are real and correctly timed
   - ✗ **Delete** false positives (noise, bleed, decay retriggering)
   - ➕ **Add** any missed hits (if you want, though not required for training)

4. **Be consistent** with your decisions:
   - If you keep soft cymbal swells in Song1, keep them in Song2
   - If you remove quick double-triggers, remove them everywhere
   - The model learns YOUR pattern, so consistency matters

5. **Save edited files** with a clear naming convention:
   ```
   training_data/raw_midi/Song1-cymbals.mid          (original)
   training_data/edited_midi/Song1-cymbals_edited.mid  (your edits)
   ```

### Phase 3: Train the Model

1. **Run the training script** (to be implemented):
   ```bash
   python train_classifier.py --stem cymbals \
                              --raw-dir "training_data/raw_midi" \
                              --edited-dir "training_data/edited_midi" \
                              --output-model "models/cymbals_classifier.pkl"
   ```

2. **The trainer will**:
   - Load all raw MIDI files (all detections)
   - Load all edited MIDI files (your kept/removed decisions)
   - Re-analyze the audio to extract features for each detection
   - Build training dataset: `[Str, Amp, BodyE, BrillE, Total, GeoMean, SustainMs] → KEPT/REMOVED`
   - Train a Random Forest classifier
   - Report cross-validation accuracy
   - Show feature importance (which variables matter most)
   - Save the trained model

3. **Training output example**:
   ```
   Training Cymbal Hit Classifier
   ================================
   
   Loaded 87 total detections across 4 songs:
     - Song1-cymbals: 23 detections (15 kept, 8 removed)
     - Song2-cymbals: 31 detections (28 kept, 3 removed)
     - Song3-cymbals: 19 detections (12 kept, 7 removed)
     - Song4-cymbals: 14 detections (11 kept, 3 removed)
   
   Total training examples: 87
     - Kept: 66 (75.9%)
     - Removed: 21 (24.1%)
   
   Training Random Forest Classifier...
   
   Cross-Validation Results (5-fold):
     - Accuracy: 91.2% ± 3.4%
     - Precision: 93.1%
     - Recall: 95.5%
   
   Feature Importance:
     1. SustainMs:      28.3%  ← Most important!
     2. GeoMean:        24.1%
     3. BrillE:         15.7%
     4. BodyE:          12.9%
     5. Total:          10.2%
     6. Amp:             5.3%
     7. Str:             3.5%
   
   Model saved to: models/cymbals_classifier.pkl
   
   Recommended: Test on a new song before using in production!
   ```

### Phase 4: Use the Trained Model

1. **Process new songs** with your custom model:
   ```bash
   python stems_to_midi.py --input "new_song/drums-cymbals.wav" \
                           --output "new_song/cymbals.mid" \
                           --stem cymbals \
                           --use-ml-model "models/cymbals_classifier.pkl"
   ```

2. **The tool will**:
   - Detect all potential hits (same as before)
   - Extract features for each hit
   - Pass features to your trained model
   - Keep/remove based on model predictions (not manual thresholds)

3. **Review and refine**:
   - Check the output MIDI
   - If accuracy is good (90%+), you're done!
   - If still seeing issues, add that song to training data and retrain

### Phase 5: Iterative Improvement

1. **Add more training data** as you process new songs:
   ```bash
   # Process new song
   python stems_to_midi.py --input "new_song/cymbals.wav" --output "training_data/raw_midi/new_song.mid" --stem cymbals
   
   # Edit in DAW, save as new_song_edited.mid
   
   # Retrain with expanded dataset
   python train_classifier.py --stem cymbals \
                              --raw-dir "training_data/raw_midi" \
                              --edited-dir "training_data/edited_midi" \
                              --output-model "models/cymbals_classifier.pkl"
   ```

2. **Version your models**:
   ```
   models/cymbals_classifier_v1.pkl   (trained on 4 songs)
   models/cymbals_classifier_v2.pkl   (trained on 8 songs)
   models/cymbals_classifier_v3.pkl   (trained on 12 songs)
   ```

3. **Compare performance**:
   ```bash
   python evaluate_model.py --model "models/cymbals_classifier_v1.pkl" \
                            --test-dir "test_songs" \
                            --stem cymbals
   ```

## File Structure

```
larsnet/
├── stems_to_midi.py              # Main tool
├── train_classifier.py           # NEW: Training script
├── evaluate_model.py             # NEW: Model evaluation script
├── midiconfig.yaml               # Default thresholds (fallback if no model)
├── ML_TRAINING_GUIDE.md          # This guide
│
├── training_data/                # Your training examples
│   ├── raw_midi/                 # Generated MIDI with all detections
│   │   ├── song1-cymbals.mid
│   │   ├── song2-cymbals.mid
│   │   └── ...
│   │
│   └── edited_midi/              # Your manually edited MIDI
│       ├── song1-cymbals_edited.mid
│       ├── song2-cymbals_edited.mid
│       └── ...
│
├── models/                       # Trained ML models
│   ├── cymbals_classifier.pkl
│   ├── kick_classifier.pkl       # Train separately for each stem
│   ├── snare_classifier.pkl
│   └── ...
│
└── test_songs/                   # Hold-out test set (don't use for training!)
    ├── test1-cymbals.wav
    ├── test1-cymbals_ground_truth.mid
    └── ...
```

## Default Values (Fallback Mode)

If no trained model exists, `stems_to_midi.py` falls back to the threshold-based approach using `midiconfig.yaml`:

```yaml
cymbals:
  geomean_threshold: 10.0         # Conservative: keeps more hits
  min_sustain_ms: 100.0           # Filters out very short clicks
  onset_threshold: 0.15           # Less sensitive to avoid false triggers
  onset_delta: 0.02
  onset_wait: 10
```

**Philosophy**: Defaults err on the side of **including more hits** rather than missing real ones. It's easier to delete a few extra notes in your DAW than to add missed ones.

## Training Best Practices

### 1. Diverse Training Data

Choose songs that represent the variety you'll encounter:
- ✓ Different drummers/playing styles
- ✓ Different recording quality (studio, live, lo-fi)
- ✓ Different cymbal types (crashes, rides, splashes)
- ✓ Mix of tempos and dynamics

Don't just use 5 songs from the same album!

### 2. Consistent Editing

- Make decisions based on "is this a real cymbal hit?" not "do I like this note?"
- Be consistent about ghost notes, decay, and bleed
- If unsure, listen to the isolated stem at that exact moment

### 3. Balanced Dataset

Try to have a reasonable mix:
- Not too imbalanced (e.g., 95% kept, 5% removed)
- Aim for at least 20-30% removed hits in training data
- If your defaults are already good, you might need to temporarily lower thresholds to generate more false positives for training

### 4. Per-Stem Training

Train **separate models** for each stem type:
- Cymbals have different characteristics than kicks
- Snares have different characteristics than toms
- Don't try to train one "universal" drum classifier

### 5. Test on Unseen Data

Always keep 1-2 songs completely separate as a test set:
- Don't use them in training
- Use them to evaluate your model's real-world performance
- This tells you if the model generalizes or just memorized your training songs

## Advanced: Transfer Learning

If you're processing a specific genre extensively (e.g., jazz, metal, EDM):

1. **Train a genre-specific model** on 10-20 songs from that genre
2. **Save as `models/cymbals_classifier_jazz.pkl`**
3. **Use it for all jazz recordings**:
   ```bash
   python stems_to_midi.py --input "jazz_song.wav" \
                           --stem cymbals \
                           --use-ml-model "models/cymbals_classifier_jazz.pkl"
   ```

## Troubleshooting

### "Not enough training data"
- Need at least 50 total detections
- Process more songs or lower thresholds temporarily to generate more detections

### "Model accuracy is only 70%"
- Training data may be inconsistent (review your edits)
- May need more diverse examples
- Try different model hyperparameters (increase `n_estimators`)

### "Model keeps ALL hits" or "Model removes ALL hits"
- Dataset is too imbalanced
- Generate more false positives by lowering thresholds during training data collection
- Make sure you're actually removing some hits in your edits

### "Model works on training songs but fails on new songs"
- Overfitting: model memorized training data
- Need more diverse training examples
- Try simpler model (logistic regression instead of Random Forest)

## Implementation Roadmap

### Phase 1: Core Training Script (`train_classifier.py`)
- [ ] Load raw MIDI (all detections with timestamps)
- [ ] Load edited MIDI (user's kept hits)
- [ ] Re-analyze audio to extract features for each detection
- [ ] Build training dataset (X=features, y=kept/removed labels)
- [ ] Train Random Forest classifier
- [ ] Cross-validation evaluation
- [ ] Save model with pickle

### Phase 2: Integration (`stems_to_midi.py`)
- [ ] Add `--use-ml-model` argument
- [ ] Load trained model if provided
- [ ] Extract same features during normal processing
- [ ] Use model.predict() instead of threshold checks
- [ ] Fallback to thresholds if no model provided

### Phase 3: Evaluation Tools (`evaluate_model.py`)
- [ ] Load model and test set
- [ ] Generate predictions
- [ ] Compare to ground truth
- [ ] Report accuracy, precision, recall, F1 score
- [ ] Show confusion matrix
- [ ] Feature importance visualization

### Phase 4: Documentation & Examples
- [ ] Example training dataset (sample songs)
- [ ] Tutorial video/screenshots
- [ ] Pre-trained models for common genres
- [ ] Community model sharing (optional)

## Dependencies

Add to `requirements.txt`:
```
scikit-learn>=1.0.0    # Random Forest classifier
joblib>=1.0.0          # Model serialization
```

## Future Enhancements

- **Active learning**: Automatically identify uncertain predictions for user review
- **Confidence scores**: Show how confident the model is (0-100%) for each prediction
- **Ensemble models**: Combine multiple classifiers for better accuracy
- **Deep learning**: Use neural networks if dataset grows large enough (>1000 examples)
- **Auto-tuning**: Use Bayesian optimization to find best hyperparameters

## Questions?

- Check existing GitHub issues
- Share your trained models with the community
- Contribute training examples for different genres
- Report bugs or suggest improvements

Happy training! 🥁🎵
