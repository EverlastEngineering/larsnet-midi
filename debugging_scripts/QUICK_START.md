# Quick Start: ML-Optimized Hi-Hat Detection

**Goal**: Find the best threshold configuration for detecting open vs closed hi-hats in your song using machine learning.

**Time Required**: 5-10 minutes

## Prerequisites

- Project with separated hi-hat stem
- Python environment set up (`conda activate larsnet-midi`)
- Ability to listen to your audio file

## Workflow Overview

```
Your Audio → Generate Data → Label Examples → Run Optimizer → Update Config → Done!
   (1 min)       (2 min)          (2 min)         (1 min)         (1 min)
```

## Step-by-Step Instructions

### 1. Generate Training Data (2 minutes)

Run stems_to_midi with output capture:

```bash
cd /path/to/larsnet
conda activate larsnet-midi
python stems_to_midi.py <PROJECT_NUMBER> --stems hihat 2>&1 | tee /tmp/hihat_output.log
```

Look for the section `ALL DETECTED ONSETS - SPECTRAL ANALYSIS` in the output.

Copy the table that looks like this:
```
Time    Str    Amp    BodyE  SizzleE    Total  GeoMean  SustainMs     Status
0.267  0.959  0.044   32.1    278.2    310.3     94.5       76.7       KEPT
1.962  0.932  0.296  251.6   3264.1   3515.7    906.2      181.6       KEPT
...
```

Save it as `debugging_scripts/data.csv` (including the header row).

### 2. Label Open Hi-Hats (3 minutes)

Listen to your hi-hat stem and identify timestamps where **open hi-hats** occur.

Open your audio player, note the times like: `1.962, 7.755, 13.944`

Edit `debugging_scripts/classifier.py` line 12:
```python
open_hihat_times = [1.962, 7.755, 13.944]  # YOUR TIMES HERE
```

**Tip**: You need at least 3 examples, but 5-10 is better for accuracy.

### 3. Run the Classifier (30 seconds)

Quick validation to see if patterns make sense:

```bash
cd debugging_scripts
python classifier.py
```

Look for:
- Feature importance (GeoMean should be high)
- Simple decision rule
- Predicted times should match your labels

### 4. Run the Optimizer (30 seconds)

Find the best threshold combination. **Three options available:**

**Option A: Grid Search** (Thorough, ~10-15 seconds)
```bash
python threshold_optimizer.py
```

**Option B: Random Search** (Fast, ~3-5 seconds, good for exploration)
```bash
python random_search_optimizer.py --n-samples 10000
```

**Option C: Bayesian Optimization** (Efficient, requires scikit-optimize)
```bash
pip install scikit-optimize
python bayesian_optimizer.py --n-calls 100
```

**Option D: Compare All Methods** (Comprehensive)
```bash
python compare_optimizers.py
```

This tests hundreds of rules and shows you the top recommendations like:

```
Best BALANCED rule (2 features):
  GeoMean > 300 AND SustainMs > 80
  Margin: 47.0%

Best ROBUST rule (3 features):
  GeoMean > 300 AND SustainMs > 80 AND BodyE > 50
  Margin: 49.3%
```

Results saved to `optimal_rules.csv` for detailed review.

### 5. Choose Your Rule

Pick based on your preference:

- **Simple (1 feature)**: Easy to understand, but less safe
- **Balanced (2 features)**: Good safety margins, moderate complexity ← **Recommended**
- **Robust (3 features)**: Maximum safety, most complex

### 6. Apply to Your Project (Manual for now)

**Coming soon**: Automatic config update

**For now**, manually edit your project's `midiconfig.yaml`:

```yaml
hihat:
  # Note the values from your chosen rule
  # Example for "balanced" rule:
  open_geomean_min: 300
  open_sustain_min: 80
```

Then update `stems_to_midi/detection.py` to use these thresholds (see full implementation plan).

### 7. Validate Results

Re-run stems_to_midi and check the MIDI output:

```bash
python stems_to_midi.py <PROJECT_NUMBER> --stems hihat
```

Listen to the MIDI. Did it catch all your open hi-hats? Any false positives?

## Example Session

```bash
# Terminal 1: Generate data
$ python stems_to_midi.py 4 --stems hihat 2>&1 | tee /tmp/hihat.log
# ... copy spectral table to data.csv

# Edit classifier.py with your timestamps
$ nano debugging_scripts/classifier.py
# open_hihat_times = [1.962, 7.755, 13.944, 19.574, 25.960, 31.753]

# Run analysis
$ cd debugging_scripts
$ python classifier.py
Feature importances:
GeoMean    0.247  ← Good!
SustainMs  0.131
...

$ python threshold_optimizer.py
Testing 372 rules...
Best BALANCED rule:
  GeoMean > 300 AND SustainMs > 80
  Margin: 47.0% safety buffer

# Apply to config (manual for now)
$ cd ..
# Update your project's midiconfig.yaml

# Test
$ python stems_to_midi.py 4 --stems hihat
# ✓ All open hi-hats detected correctly!
```

## Troubleshooting

### "No rules achieve 100% accuracy"

**Cause**: Labeled examples might be inconsistent or features don't separate well.

**Solution**:
1. Double-check your timestamp labels (listen again)
2. Add more examples (try for 8-10)
3. Check if any closed hi-hats have similar features to your opens

### "All margins are very small (<10%)"

**Cause**: Your open hi-hats are very similar to your closed ones.

**Solution**:
1. This might be the nature of your recording (subtle playing)
2. Try labeling only the most obvious open hi-hats
3. Consider if some are "half-open" (ambiguous cases)

### "Optimizer is too slow"

**Cause**: Search space is large.

**Solution**: Edit `threshold_optimizer.py` line 60-62:
```python
# Use coarser steps
geomean_thresholds = np.arange(300, 600, 50)  # Was 25, now 50
sustain_thresholds = np.arange(80, 180, 20)   # Was 10, now 20
```

### "Feature importance is all equal"

**Cause**: Not enough variety in your training examples.

**Solution**:
1. Label examples from different sections of the song
2. Include soft and loud open hi-hats
3. Make sure you have at least 5 examples

## Understanding the Output

### Feature Importance
```
GeoMean    0.247  ← 24.7% of classification power
Total      0.215  ← 21.5%
SustainMs  0.131  ← 13.1%
```

**Interpretation**: Higher = more discriminative. GeoMean being highest means combined spectral energy is the best indicator of open hi-hats.

### Margin Score
```
GeoMean > 300
Margin: 246.2 units (45% below minimum)
```

**Interpretation**: Your weakest open hi-hat has GeoMean=546. The threshold is 300, which is 246 units below. That's a 45% safety buffer - very safe!

### Rules with 100% Accuracy
```
Rules with 100% recall AND 100% precision: 141
```

**Interpretation**: 141 different threshold combinations perfectly classify your data. This is good - means the pattern is robust.

## Next Steps

Once the automatic config system is implemented:

```bash
# Future one-command workflow:
$ python optimize_config.py --project 4 --instrument hihat

# Wizard will:
# 1. Generate data automatically
# 2. Ask you to label examples (with audio playback)
# 3. Run classifier and optimizer
# 4. Update your midiconfig.yaml
# 5. Re-run stems_to_midi
# 6. Show before/after comparison

# All in one command!
```

## More Information

- **Full documentation**: `README.md` in this directory
- **Implementation plan**: `../agent-plans/ml-driven-threshold-config.plan.md`
- **Data sharing info**: `../TRAINING_DATA_SHARING.md`
- **Questions**: Open an issue on GitHub

## Contributing

If your optimization works well:
1. Export your training data: `python export_training_data.py`
2. Review the anonymized JSON
3. Submit to help the community (optional)

See `TRAINING_DATA_SHARING.md` for details on privacy and benefits.

---

**Happy optimizing!** 🎵🥁

**Last Updated**: 2025-10-31
