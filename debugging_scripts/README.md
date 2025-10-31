# Hi-Hat Classification Debugging Scripts

This directory contains tools for analyzing and optimizing hi-hat open/closed classification using machine learning and statistical analysis.

## 📚 Documentation Index

- **[INDEX.md](INDEX.md)** - Complete navigation guide (start here!)
- **[QUICK_START.md](QUICK_START.md)** - 5-minute getting started guide
- **[CHEAT_SHEET.md](CHEAT_SHEET.md)** - Quick command reference
- **[README.md](README.md)** (this file) - Comprehensive guide to all tools
- **[OPTIMIZATION_METHODS.md](OPTIMIZATION_METHODS.md)** - Technical deep dive into algorithms
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and data flow
- **[WHATS_NEW.md](WHATS_NEW.md)** - Recent additions and features
- **[REQUIREMENTS.txt](REQUIREMENTS.txt)** - Python dependencies

## Overview

The hi-hat detection system needs to distinguish between open and closed hi-hat hits based on spectral features extracted from audio. These scripts help you:

1. **Collect training data** from stems_to_midi processing output
2. **Analyze patterns** in open vs closed hi-hats using ML
3. **Optimize thresholds** to find the best classification rules (3 methods available!)
4. **Generate configuration values** for midiconfig.yaml

**New**: Now includes **Grid Search**, **Random Search**, and **Bayesian Optimization** approaches - choose based on your needs or compare all three!

## Workflow

```
Audio File → stems_to_midi.py → Detailed Output → data.csv
                                                      ↓
                                    classifier.py ← Manual Labeling
                                                      ↓
                                            threshold_optimizer.py
                                                      ↓
                                            optimal_rules.csv
                                                      ↓
                                            midiconfig.yaml (updated)
```

## Files in This Directory

### 1. `data.csv` - Training Data
**Purpose**: Contains spectral features for every hi-hat hit detected in a song.

**How to Generate**:
```bash
# Run stems_to_midi with verbose output on a project
conda activate larsnet-midi
cd /path/to/larsnet
python stems_to_midi.py <project_number> --stems hihat 2>&1 | tee /tmp/hihat_output.log

# Look for the "ALL DETECTED ONSETS - SPECTRAL ANALYSIS" section
# Copy the data table into data.csv
```

**Data Structure**:
```csv
Time,Str,Amp,BodyE,SizzleE,Total,GeoMean,SustainMs,Status
0.267,0.959,0.044,32.1,278.2,310.3,94.5,76.7,KEPT
1.962,0.932,0.296,251.6,3264.1,3515.7,906.2,181.6,KEPT
...
```

**Fields Explained**:
- `Time`: Timestamp in seconds when hi-hat hit occurs
- `Str`: Onset strength (0-1, how strong the transient is)
- `Amp`: Peak amplitude (0-1, normalized volume)
- `BodyE`: Body energy (500-2000Hz, mid-frequency content)
- `SizzleE`: Sizzle energy (6000-12000Hz, high-frequency shimmer)
- `Total`: Total spectral energy (BodyE + SizzleE)
- `GeoMean`: Geometric mean √(BodyE × SizzleE), combined energy measure
- `SustainMs`: Sustain duration in milliseconds (how long the hit rings)
- `Status`: KEPT or REJECTED by initial filtering

**Manual Labeling Required**:
Listen to your audio and identify timestamps where **open hi-hats** occur. Update the script:
```python
open_hihat_times = [1.962, 7.755, 13.944, 19.574, 25.960, 31.753]
```

### 2. `classifier.py` - Pattern Analysis
**Purpose**: Uses machine learning to discover which features distinguish open from closed hi-hats.

**What It Does**:
1. **Random Forest Classifier**: Tests all features to find which ones matter most
2. **Feature Importance Ranking**: Shows relative importance of each spectral feature
3. **Decision Tree Rules**: Generates human-readable classification rules
4. **Combination Testing**: Evaluates various threshold combinations

**How to Run**:
```bash
cd debugging_scripts
python classifier.py
```

**Key Output**:
```
Feature importances:
GeoMean    0.247  ← Most important!
Total      0.215
SizzleE    0.187
BodyE      0.170
SustainMs  0.131  ← Less important than expected
```

**What This Tells You**:
- Which features are most discriminative
- Whether your current thresholds are aligned with the data
- Quick validation that rules make sense

### 3. `threshold_optimizer.py` - Grid Search Optimizer
**Purpose**: Systematically tests **all possible threshold combinations** in a grid to find the optimal rule.

**What It Does**:
1. **Grid Search**: Tests every reasonable combination of thresholds
2. **Performance Metrics**: Calculates recall, precision, false positives for each rule
3. **Safety Margins**: Measures how far thresholds are from edge cases
4. **Ranking**: Sorts rules by safety + accuracy composite score

**How to Run**:
```bash
cd debugging_scripts
python threshold_optimizer.py
```

**Search Space**:
- GeoMean: 300-600 (tested every 25 units) = 12 values
- SustainMs: 80-180ms (tested every 10ms) = 10 values
- BodyE: 50-200 (tested every 10 units) = 15 values
- **Total combinations**: 12 × 10 × 15 = 1,800+ rules tested

**Key Output**:
```
Best ROBUST rule (3 features):
  GeoMean > 300 AND SustainMs > 80 AND BodyE > 50
  Margin: 49.3% safety buffer
```

### 4. `optimal_rules.csv` - Results Export
**Purpose**: Top 30 rules ranked by safety margin for further analysis.

**Columns**:
- `rule`: Human-readable threshold combination
- `n_features`: Number of features used (1-3)
- `recall`: % of open hi-hats correctly detected
- `precision`: % of detected "opens" that are truly open
- `margin_score`: Average safety margin (higher = safer)
- `tp/fp/fn`: True positives, false positives, false negatives
- Individual threshold values

**Use Cases**:
- Compare rules with different complexity levels
- Choose based on your preference for simplicity vs robustness
- Validate results before implementing in code

## Understanding the Science

### Why Machine Learning?

Traditional rule-based systems use **hardcoded thresholds**:
```python
# Old approach - guessing thresholds
if sustain_ms > 90 and body_energy > 200:
    return "open"
```

Problems:
- Thresholds are arbitrary guesses
- Don't account for feature interactions
- No validation against real data

**ML approach**:
1. Label real examples (6 open hi-hats in this case)
2. Let algorithms discover patterns in 160 total hits
3. Find thresholds that **actually separate** open from closed
4. Measure safety margins to avoid edge-case failures

### Key Concepts

**Feature Importance**: 
Which spectral characteristics actually matter? ML ranks them by discriminative power.

**Margin Score**:
Distance from failure point. Higher = safer rule that tolerates variations.
```
Example: GeoMean > 300
  Minimum open value: 546.2
  Threshold: 300
  Margin: 546.2 - 300 = 246.2 (45% safety buffer)
```

**Recall vs Precision**:
- **Recall**: Did we catch all the open hi-hats? (avoid false negatives)
- **Precision**: Are detected opens actually open? (avoid false positives)
- **Goal**: 100% on both (perfect classification)

### Why Combinations Work Better

**Single threshold**: `GeoMean > 535`
- Margin: Only 3.9% (RISKY!)
- One feature can have noise/variation

**Two thresholds**: `GeoMean > 300 AND SustainMs > 80`
- Margin: 47% (MUCH SAFER!)
- Both conditions must be true = more robust
- Each feature can be lower (safer) when combined

**Three thresholds**: `GeoMean > 300 AND SustainMs > 80 AND BodyE > 50`
- Margin: 49.3% (SAFEST!)
- Triple redundancy
- Handles edge cases across multiple dimensions

### 4. `bayesian_optimizer.py` - Intelligent Search
**Purpose**: Uses **Bayesian optimization** with Gaussian Processes to intelligently explore the parameter space.

**What It Does**:
1. **Adaptive Search**: Learns from each evaluation to guide the next search location
2. **Exploitation vs Exploration**: Balances testing promising regions vs discovering new ones
3. **Sample Efficiency**: Finds optimal solutions with fewer evaluations than grid search
4. **Uncertainty Quantification**: Provides confidence estimates for each region

**How to Run**:
```bash
# Install scikit-optimize first
pip install scikit-optimize

cd debugging_scripts
python bayesian_optimizer.py
python bayesian_optimizer.py --n-calls 200    # More thorough
python bayesian_optimizer.py --visualize      # Generate plots
```

**Advantages Over Grid Search**:
- **Faster**: ~100 evaluations vs 1,800+ for grid search
- **Smarter**: Uses Gaussian Process to model objective function
- **Continuous**: Can explore any point, not just grid intersections
- **Informative**: Shows convergence plots and objective landscape

**Key Output**:
```
Optimizing 3-feature rule...
Features: GeoMean, SustainMs, BodyE
Running 100 evaluations (20 random starts)

✓ Optimization complete!
  Best score: 149.32
  Recall: 1.00
  Precision: 1.00
  Margin: 49.32%
```

**When to Use**:
- Need to optimize quickly
- Want to explore continuous threshold space
- Interested in understanding parameter relationships
- Have expensive evaluation functions

### 5. `random_search_optimizer.py` - Exploration & Analysis
**Purpose**: Uses **random sampling** to explore the entire feature space and analyze multi-dimensional patterns.

**What It Does**:
1. **Random Sampling**: Tests thousands of random threshold combinations
2. **Distribution Analysis**: Shows statistics of successful thresholds
3. **Feature Interactions**: Reveals correlations between threshold values
4. **Robust Region Detection**: Finds parameter regions with many successful rules
5. **Multi-Dimensional Visualization**: 2D/3D plots of parameter space

**How to Run**:
```bash
cd debugging_scripts
python random_search_optimizer.py
python random_search_optimizer.py --n-samples 50000    # More thorough
python random_search_optimizer.py --visualize          # Generate plots
python random_search_optimizer.py --analyze-interactions  # Deep analysis
```

**Advantages**:
- **Fast**: Can sample millions of points per second
- **Unbiased**: No assumptions about structure of optimal regions
- **Comprehensive**: Finds unexpected solutions grid search might miss
- **Statistical**: Provides distribution analysis of successful parameters

**Key Output**:
```
Running random search with 10,000 samples...

✓ Search complete!
  Perfect rules found: 847

FEATURE DISTRIBUTION ANALYSIS
GeoMean:
  Min:    201.35
  Median: 387.24
  Max:    698.77
  Mean:   394.15 ± 112.45

ROBUST REGION ANALYSIS
Most robust rule (has 42 nearby perfect rules):
  GeoMean      > 305.2
  SustainMs    > 83.4
  BodyE        > 52.1
  Margin: 48.7%
```

**Visualizations Generated**:
- `random_2d_geomean_sustain.png` - 2D scatter plots
- `random_3d_space.png` - 3D parameter space visualization
- `feature_correlations.png` - Correlation heatmap

**When to Use**:
- Initial exploration of unknown parameter space
- Need to understand feature interactions
- Want statistical confidence in results
- Searching high-dimensional spaces (4+ features)

### 6. `compare_optimizers.py` - Method Comparison
**Purpose**: Runs all three optimizers and compares their performance, speed, and result quality.

**How to Run**:
```bash
cd debugging_scripts
python compare_optimizers.py
python compare_optimizers.py --quick          # Faster with fewer samples
python compare_optimizers.py --skip-grid      # Skip slow grid search
```

**Output**:
```
⏱️  PERFORMANCE
Method                         Time (s)    Status
------------------------------------------------------------
Grid Search                       12.45    ✓ Success
Random Search                      3.21    ✓ Success
Bayesian Optimization              8.67    ✓ Success

🎯 RESULT QUALITY
Method                         Best Margin  Perfect Rules   Top 10 Avg
--------------------------------------------------------------------
Grid Search                        49.3%             141        45.2%
Random Search                      48.7%             847        44.8%
Bayesian Optimization              49.2%              12        48.1%

💡 RECOMMENDATIONS
🏆 Best margin: Grid Search (49.3% safety margin)
⚡ Fastest: Random Search (3.21 seconds)
🔍 Most thorough: Random Search (847 perfect rules)
```

**When to Use Each Method**:

| Method | Best For | Speed | Thoroughness |
|--------|----------|-------|--------------|
| **Grid Search** | Final verification, guaranteed exhaustive search | Slow | Complete within grid |
| **Random Search** | Initial exploration, high dimensions, statistical analysis | Fast | Probabilistically complete |
| **Bayesian Opt** | Sample efficiency, continuous spaces, learning landscapes | Medium | Targeted and efficient |

### 7. `optimal_rules.csv` / `bayesian_optimal_rules.csv` / `random_search_results.csv`
**Purpose**: Results from each optimizer for comparison and analysis.

## Practical Usage Guide

### Step-by-Step: Optimize for a New Song

1. **Generate data**:
   ```bash
   python stems_to_midi.py 4 --stems hihat 2>&1 | tee /tmp/hihat.log
   ```

2. **Extract spectral analysis table** from log and save to `data.csv`

3. **Listen and label**: Identify timestamps of open hi-hats in the audio

4. **Update classifier.py**:
   ```python
   open_hihat_times = [1.962, 7.755, 13.944]  # Your timestamps
   ```

5. **Run classifier** to validate patterns:
   ```bash
   python classifier.py
   ```
   
   Check: Are feature importances reasonable? Does the simple rule make sense?

6. **Run optimizer** to find best thresholds:
   ```bash
   python threshold_optimizer.py
   ```

7. **Choose rule** from top recommendations based on your needs:
   - **Simplicity**: 2-feature rule
   - **Robustness**: 3-feature rule
   - **Conservatism**: Higher margin scores

8. **Update midiconfig.yaml** (see next section)

### Interpreting Results

**Good Signs**:
- Feature importance matches intuition (spectral energy matters more than timing)
- Multiple rules achieve 100% accuracy (robust pattern)
- High margin scores (>30%) on top rules
- GeoMean is consistently important

**Red Flags**:
- No rules with 100% accuracy (need more training examples or different features)
- All margins <10% (thresholds too close to edge cases)
- Feature importance is evenly distributed (no clear discriminator)
- Top rules require very specific threshold combinations (overfitting)

### Adding More Training Data

**To improve results**:
1. Label more open hi-hat examples from different songs
2. Include various playing styles (soft, hard, articulated)
3. Test across different recording qualities
4. Validate on held-out test songs

**Diminishing returns**: 
- 3-6 examples: Basic patterns emerge
- 10-20 examples: Good generalization
- 50+ examples: Production-ready model
- 100+ examples: Consider deep learning approaches

## Choosing the Right Optimization Approach

### Decision Tree

```
Start Here
    ↓
Are you exploring for the first time?
    YES → Use random_search_optimizer.py (10,000+ samples)
          Get statistical overview, see distribution of solutions
    NO ↓
         
Do you need guaranteed exhaustive search?
    YES → Use threshold_optimizer.py (grid search)
          Tests all combinations in defined ranges
    NO ↓

Do you have scikit-optimize installed?
    YES → Use bayesian_optimizer.py
          Most sample-efficient, learns as it goes
    NO → Use random_search_optimizer.py
         Fast and effective alternative

Unsure? → Run compare_optimizers.py to see all three!
```

### Recommended Workflow

**Phase 1: Exploration** (First Time)
```bash
python random_search_optimizer.py --n-samples 50000 --visualize --analyze-interactions
```
- Understand feature distributions
- Identify promising regions
- Detect feature correlations
- Get statistical confidence

**Phase 2: Refinement** (After Initial Results)
```bash
python bayesian_optimizer.py --n-calls 200 --visualize
```
- Efficiently explore continuous space
- Find local optima
- Generate convergence plots
- Understand objective landscape

**Phase 3: Verification** (Final Validation)
```bash
python threshold_optimizer.py
```
- Exhaustively verify best rules
- Ensure no better solutions in grid
- Get complete ranked list
- Export for implementation

**Quick Option** (Time-Constrained)
```bash
python compare_optimizers.py --quick
```
- All three methods in one go
- Automatic comparison
- Best for rapid iteration

### Performance Comparison

Typical results on 160-sample dataset:

| Method | Evaluations | Time | Perfect Rules Found | Best Margin |
|--------|-------------|------|---------------------|-------------|
| Grid Search | 1,800 | ~12s | 141 | 49.3% |
| Random (10k) | 10,000 | ~3s | 847 | 48.7% |
| Bayesian (100) | 100 | ~8s | 12 | 49.2% |

**Key Insights**:
- Random search finds more solutions but may miss the absolute best
- Grid search guarantees finding best within grid resolution
- Bayesian is most efficient per evaluation (finds good solution fastest)
- All three converge to similar optimal margins (±0.6%)

## Integration with Main System

These scripts inform the configuration values used by the main detection system:

```
debugging_scripts/                              →  midiconfig.yaml  →  stems_to_midi/detection.py
├── optimal_rules.csv                              (thresholds)        (classification logic)
├── bayesian_optimal_rules.csv
└── random_search_results.csv
```

See the planning documents for the full integration workflow.

## Common Issues

### "No rules achieve 100% accuracy"
- Your labeled examples might conflict (check timestamps carefully)
- Features might not separate well (consider adding more spectral bands)
- Need more training examples

### "All rules have same margin score"
- Data might be too homogeneous (try different songs)
- Labeled examples too similar (need variety in intensity/articulation)

### "Optimizer takes too long"
- Reduce search ranges in threshold_optimizer.py
- Use coarser steps (e.g., every 50 units instead of 25)
- Limit to 2-feature combinations only

## Next Steps

1. **Immediate**: Use these tools to optimize your current song
2. **Short-term**: Build YAML configuration system (see planning docs)
3. **Long-term**: Create user-friendly interface for this workflow
4. **Future**: Aggregate data across users for universal model

## Technical Deep Dive: Optimization Methods

### Grid Search
**Algorithm**: Exhaustive enumeration of discretized parameter space
```python
for geomean in range(300, 600, 25):
    for sustain in range(80, 180, 10):
        for body in range(50, 200, 10):
            evaluate(geomean, sustain, body)
```
**Complexity**: O(n^d) where n=steps per dimension, d=dimensions
**Guarantees**: Finds global optimum within grid resolution
**Limitations**: Exponential growth with dimensions (curse of dimensionality)

### Random Search
**Algorithm**: Uniform random sampling from continuous parameter space
```python
for i in range(n_samples):
    geomean = random.uniform(300, 600)
    sustain = random.uniform(80, 180)
    body = random.uniform(50, 200)
    evaluate(geomean, sustain, body)
```
**Complexity**: O(n) - linear in number of samples
**Guarantees**: Probabilistic coverage of parameter space
**Advantages**: Scales to high dimensions, can find solutions grid search misses

### Bayesian Optimization
**Algorithm**: Gaussian Process regression + acquisition function optimization
```python
# Build GP surrogate model from previous evaluations
gp = GaussianProcess(observations)

# Find next point that maximizes Expected Improvement
next_point = maximize(acquisition_function(gp))
evaluate(next_point)
```
**Complexity**: O(n³) for GP inference, but n can be much smaller than grid/random
**Guarantees**: Converges to optimum with high probability
**Advantages**: Sample efficient, handles noise, provides uncertainty estimates
**Key Concepts**:
- **Surrogate Model**: GP approximates expensive objective function
- **Acquisition Function**: Balances exploitation (improve best) vs exploration (reduce uncertainty)
- **Expected Improvement**: Most common acquisition function

### Why All Three?

Different strengths for different scenarios:

**Grid Search**: 
- ✓ Verifiable, reproducible
- ✓ Complete enumeration within resolution
- ✗ Misses continuous optimum between grid points
- ✗ Exponential cost in dimensions

**Random Search**:
- ✓ Fast, scalable
- ✓ Good empirical performance in practice
- ✓ Can explore entire continuous space
- ✗ No learning, treats each sample independently

**Bayesian Optimization**:
- ✓ Sample efficient (finds good solution with fewest evaluations)
- ✓ Learns from previous evaluations
- ✓ Provides uncertainty quantification
- ✗ More complex, requires additional library
- ✗ GP inference cost grows cubically

**Combined Approach** (recommended):
1. Random search for broad exploration (understand landscape)
2. Bayesian optimization for efficient refinement (find local optima)
3. Grid search for final verification (confirm no better solutions nearby)

## References

- **Random Forest**: Breiman (2001) - Ensemble learning for feature importance
- **Decision Trees**: CART algorithm for interpretable rule extraction
- **Grid Search**: Exhaustive hyperparameter optimization
- **Random Search**: Bergstra & Bengio (2012) - Often beats grid search
- **Bayesian Optimization**: Mockus (1975), Snoek et al. (2012) - GP-based optimization
- **Gaussian Processes**: Rasmussen & Williams (2006) - Probabilistic regression
- **Precision/Recall**: Standard ML evaluation metrics

## Questions?

This is a living document. As we build the system out, update this README with:
- Common issues encountered
- Best practices discovered
- New features added
- Integration points
