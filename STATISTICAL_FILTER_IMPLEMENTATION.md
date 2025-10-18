# Statistical Outlier Filter Implementation

**Date**: October 17, 2025  
**Status**: ✅ Implemented  
**Stem**: Kick only (expandable to others)

## Summary

Implemented a two-pass filtering system for kick drums that catches snare bleed and artifacts that pass the basic GeoMean threshold but have statistically abnormal spectral characteristics.

## Problem

After increasing the analysis window (`peak_window_sec`) from 50ms to 100ms to capture full kick energy, we achieved better energy detection. However, this exposed a new problem: **snare bleed in the kick track** was now passing the GeoMean threshold because it had enough total energy.

Snare bleed characteristics:
- Has moderate energy in all three frequency ranges (passes GeoMean)
- But has **abnormal FundE/BodyE ratio** (lower fundamental relative to body compared to real kicks)
- Has different **total energy profile** than real kicks

## Solution: Two-Pass Filtering

### Pass 1: GeoMean Threshold (Existing)
Filters out low-energy artifacts using 3-way geometric mean:
```
GeoMean = ∛(FundE × BodyE × AttackE)
```
Rejects onsets with `GeoMean < 70.0`

### Pass 2: Statistical Outlier Detection (NEW)
Analyzes the **spectral signature** of all detected kicks and identifies outliers:

1. **Calculate population statistics** from all detected onsets:
   - Median FundE/BodyE ratio
   - Median total energy
   - Standard deviations

2. **Score each onset** with a "badness" metric (0-1):
   ```python
   ratio_deviation = (median_ratio - current_ratio) / median_ratio
   total_deviation = |median_total - current_total| / median_total
   
   badness_score = 0.7 × ratio_deviation + 0.3 × total_deviation
   ```

3. **Reject outliers** where `badness_score > 0.6`

### Why This Works

- **Real kicks** cluster together in FundE/BodyE ratio space (typically 0.3-0.8)
- **Snare bleed** has lower fundamental energy → lower ratio → higher badness score
- **Statistical comparison** is more robust than absolute thresholds

## Configuration

All settings in `midiconfig.yaml` under `kick:` section:

```yaml
kick:
  # Pass 1: GeoMean threshold (required)
  geomean_threshold: 70.0
  
  # Pass 2: Statistical outlier detection (optional)
  enable_statistical_filter: true    # Enable/disable second pass
  statistical_badness_threshold: 0.6  # Badness score threshold (0-1)
                                      # Higher = more permissive (0.6-0.7 recommended)
                                      # Lower = stricter (0.4-0.5 very strict)
  statistical_ratio_weight: 0.7       # Weight for FundE/BodyE ratio deviation
  statistical_total_weight: 0.3       # Weight for total energy deviation
```

## Implementation Details

### New Functions in `stems_to_midi/helpers.py`

#### 1. `calculate_statistical_params(onset_data_list)`
Analyzes full dataset to compute normalization parameters:
- Returns: `median_ratio`, `median_total`, `ratio_spread`, `total_spread`
- Pure function, no side effects

#### 2. `calculate_badness_score(onset_data, statistical_params, ratio_weight, total_weight)`
Computes normalized badness score for a single onset:
- Returns: Score in [0, 1] where 1.0 = maximum deviation
- Pure function, no side effects

### Modified Function

#### `filter_onsets_by_spectral()`
Added second-pass filtering after the existing geomean filter:
- Calculates statistical parameters from all onsets
- Scores each onset that passed Pass 1
- Re-filters based on badness threshold
- Stores badness scores in `onset_data['badness_score']` for debug output

### Debug Output

When `--show-all-onsets` flag is used, the output now includes:

```
      Time    Str    Amp    FundE    BodyE  AttackE    Total  GeoMean  Badness     Status
     0.906  0.405  0.002     46.2    297.8    106.8    450.8    113.7    0.234       KEPT
     1.637  0.432  0.003      0.6     12.5      8.0     21.1      3.8    0.892   REJECTED
     1.892  0.461  0.029     35.0    124.6    111.3    270.9     78.6    0.187       KEPT
```

**Badness column** shows the statistical deviation score:
- **0.0-0.4**: Very typical kick (low deviation)
- **0.4-0.6**: Borderline (at threshold)
- **0.6-1.0**: Statistical outlier (likely artifact/bleed)

### Summary Statistics

The debug output now shows both passes:

```
      FILTERING SUMMARY:
        Pass 1 - GeoMean threshold: 70.0 (adjustable in midiconfig.yaml)
        Total onsets detected: 150
        Pass 1 Kept (GeoMean > 70.0): 120
        Pass 1 Rejected (GeoMean <= 70.0): 30
        Pass 1 Kept GeoMean range: 72.3 - 456.8

        Pass 2 - Statistical Outlier Detection:
        Badness threshold: 0.6 (adjustable in midiconfig.yaml)
        Median FundE/BodyE ratio: 0.487
        Median Total energy: 342.5
        Pass 2 Kept (Badness <= 0.6): 112
        Pass 2 Rejected (Badness > 0.6): 8
        Pass 2 Kept Badness range: 0.045 - 0.589
        Pass 2 Rejected Badness range: 0.623 - 0.892
```

## Usage

### Enable Statistical Filtering
Set in `midiconfig.yaml`:
```yaml
kick:
  enable_statistical_filter: true
```

### Adjust Sensitivity
To catch more false positives (stricter):
```yaml
kick:
  statistical_badness_threshold: 0.5  # Lower threshold = stricter
```

To allow more variation (more permissive):
```yaml
kick:
  statistical_badness_threshold: 0.7  # Higher threshold = more permissive
```

### Adjust Weights
Emphasize ratio deviation more (good for snare bleed):
```yaml
kick:
  statistical_ratio_weight: 0.8
  statistical_total_weight: 0.2
```

Emphasize total energy more (good for level issues):
```yaml
kick:
  statistical_ratio_weight: 0.5
  statistical_total_weight: 0.5
```

### Disable Statistical Filtering
```yaml
kick:
  enable_statistical_filter: false  # Only use GeoMean threshold
```

## Testing

Run with debug output to see the badness scores:
```bash
python stems_to_midi.py input/song.wav --show-all-onsets
```

The output will show:
1. All detected kicks with their spectral energies
2. Badness score for each kick (if statistical filter enabled)
3. Two-pass filtering summary with statistics

## Future Enhancements

Potential improvements (not yet implemented):

1. **Adaptive thresholds**: Learn optimal badness threshold from dataset
2. **Multi-dimensional clustering**: Use more features (attack energy ratio, spectral centroid)
3. **Per-track calibration**: Calculate statistics per song rather than globally
4. **Extend to other stems**: Apply to snare/toms to catch kick bleed
5. **Temporal consistency**: Consider timing patterns (e.g., kicks on downbeats)

## Notes

- The statistical filter only runs on **kick drums**
- It's **disabled in learning mode** (to analyze all detections)
- It requires **at least one onset** to calculate statistics
- The filter is **optional** and can be disabled without affecting Pass 1

## Code Organization

All core logic is in pure functions following the functional programming pattern:
- `calculate_statistical_params()` - Dataset analysis
- `calculate_badness_score()` - Individual onset scoring
- Both are pure functions with no side effects
- Easy to test, debug, and extend

## Credits

Algorithm inspired by user-provided code for analyzing kick spectral signatures and detecting outliers using normalized deviation metrics.
