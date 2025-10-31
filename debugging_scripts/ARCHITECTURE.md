# Debugging Scripts Architecture

## File Organization

```
debugging_scripts/
│
├── 📊 DATA
│   ├── data.csv                          # Training data (spectral features)
│   ├── optimal_rules.csv                 # Grid search results
│   ├── bayesian_optimal_rules.csv        # Bayesian optimization results
│   └── random_search_results.csv         # Random search results
│
├── 🔧 CORE TOOLS
│   ├── classifier.py                     # ML pattern analysis
│   ├── threshold_optimizer.py            # Grid search optimizer
│   ├── bayesian_optimizer.py             # Bayesian optimization
│   └── random_search_optimizer.py        # Random search + analysis
│
├── 🎯 UTILITIES
│   ├── compare_optimizers.py             # Run & compare all methods
│   └── example_optimization_workflow.py  # Educational demo
│
└── 📚 DOCUMENTATION
    ├── README.md                         # Comprehensive guide
    ├── QUICK_START.md                    # User quick reference
    ├── OPTIMIZATION_METHODS.md           # Technical deep dive
    ├── WHATS_NEW.md                      # New features overview
    ├── ARCHITECTURE.md                   # This file
    └── REQUIREMENTS.txt                  # Python dependencies
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        AUDIO PROCESSING                          │
│                                                                  │
│  Audio File → stems_to_midi.py → Spectral Analysis Output       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ↓ (copy table)
                   ┌─────────────────┐
                   │   data.csv      │ ← Manual labeling
                   │  (160 samples)  │   (mark open hi-hats)
                   └────────┬────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ↓                  ↓                  ↓
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│  classifier.py │  │ Grid Search    │  │ Random Search  │
│                │  │                │  │                │
│ Feature        │  │ 1,800 combos   │  │ 10,000 samples │
│ Importance     │  │ ~12 seconds    │  │ ~3 seconds     │
│                │  │                │  │                │
│ Decision Rules │  │ Exhaustive     │  │ Statistical    │
└────────────────┘  └───────┬────────┘  └───────┬────────┘
                            │                    │
                            ↓                    ↓
                   ┌─────────────────┐  ┌─────────────────┐
                   │optimal_rules.csv│  │random_search.csv│
                   └────────┬────────┘  └───────┬─────────┘
                            │                    │
         ┌──────────────────┴────────────────────┘
         │
         ↓
┌─────────────────┐
│ Bayesian Opt    │
│                 │
│ 100 calls       │
│ ~9 seconds      │
│                 │
│ Learns & Refines│
└────────┬────────┘
         │
         ↓
┌─────────────────────────┐
│bayesian_optimal_rules.csv│
└────────┬────────────────┘
         │
         ↓
┌──────────────────────────────────────────────┐
│            COMPARISON & SELECTION             │
│                                               │
│  compare_optimizers.py                        │
│  ├─ Performance metrics                       │
│  ├─ Quality comparison                        │
│  └─ Recommendations                           │
└───────────────────┬──────────────────────────┘
                    │
                    ↓
            ┌───────────────┐
            │  BEST RULE    │
            │               │
            │ GeoMean > 300 │
            │ Sustain > 80  │
            │ BodyE > 50    │
            │ Margin: 49.3% │
            └───────┬───────┘
                    │
                    ↓
         ┌──────────────────────┐
         │   midiconfig.yaml    │ → stems_to_midi/detection.py
         │  (configuration)     │   (implementation)
         └──────────────────────┘
```

## Algorithm Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GRID SEARCH                                 │
│                                                                     │
│  Parameter Space:        Evaluation Strategy:                      │
│  ┌───┬───┬───┬───┐      • Test every grid point                   │
│  │ ● │ ● │ ● │ ● │      • Deterministic, exhaustive               │
│  ├───┼───┼───┼───┤      • Guarantees finding best in grid         │
│  │ ● │ ● │ ● │ ● │                                                │
│  ├───┼───┼───┼───┤      Complexity: O(n^d)                        │
│  │ ● │ ● │ ● │ ● │      Time: ~12 seconds                         │
│  ├───┼───┼───┼───┤      Results: 141 perfect rules                │
│  │ ● │ ● │ ● │ ● │                                                │
│  └───┴───┴───┴───┘                                                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       RANDOM SEARCH                                 │
│                                                                     │
│  Parameter Space:        Evaluation Strategy:                      │
│  ┌─────────────────┐    • Sample uniformly at random               │
│  │   ● ●           │    • Independent samples                      │
│  │ ●   ●  ●  ●     │    • Unbiased exploration                     │
│  │    ●    ●   ●   │                                               │
│  │ ●    ●      ●   │    Complexity: O(n)                           │
│  │   ●   ●  ●      │    Time: ~3 seconds                           │
│  │ ●      ●    ●   │    Results: 847 perfect rules                 │
│  └─────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    BAYESIAN OPTIMIZATION                            │
│                                                                     │
│  Parameter Space:        Evaluation Strategy:                      │
│  ┌─────────────────┐    • Build Gaussian Process model             │
│  │ 1 2             │    • Use acquisition function (EI)            │
│  │   3 4 5         │    • Balance exploitation vs exploration      │
│  │      6 7 8 9    │    • Learn from previous evaluations          │
│  │         ●●●     │                                               │
│  │         ●●●     │    Complexity: O(n³) per iteration            │
│  │         ●●●     │    Time: ~9 seconds                           │
│  │   (converges    │    Results: 12 perfect rules (targeted)       │
│  │    to optimum)  │                                               │
│  └─────────────────┘                                               │
│  Numbers show evaluation order → concentrates on best region       │
└─────────────────────────────────────────────────────────────────────┘
```

## Method Selection Decision Tree

```
                        Start Here
                            │
                            ↓
            ┌───────────────────────────────┐
            │ What's your primary goal?    │
            └───────┬───────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ↓           ↓           ↓
    Explore     Refine     Verify
    (First)    (Improve)   (Final)
        │           │           │
        ↓           ↓           ↓
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ Random   │ │ Bayesian │ │   Grid   │
  │ Search   │ │   Opt    │ │  Search  │
  └──────────┘ └──────────┘ └──────────┘
        │           │           │
        └───────────┼───────────┘
                    ↓
          ┌─────────────────┐
          │ compare_optimizers.py │
          │                       │
          │ Runs all three and    │
          │ provides comparison   │
          └─────────────────┘
```

## Information Flow Between Tools

```
classifier.py
    │
    ├─→ Feature Importance Rankings
    │   (Which features matter?)
    │
    └─→ Decision Tree Rules
        (Human-readable patterns)
            ↓
        Informs search ranges for:
            ↓
    ┌───────┴───────┐
    │               │
    ↓               ↓
threshold_optimizer  random_search_optimizer
    │                       │
    │                       ├─→ Distribution Analysis
    │                       │   (Parameter statistics)
    │                       │
    │                       ├─→ Robust Regions
    │                       │   (Stable parameter areas)
    │                       │
    │                       └─→ Feature Correlations
    │                           (How thresholds interact)
    │                               ↓
    └───────────┬───────────────────┘
                ↓
        Candidate Solutions
                ↓
        bayesian_optimizer.py
                │
                ├─→ Convergence Analysis
                │   (Learning progress)
                │
                ├─→ Objective Landscape
                │   (Function topology)
                │
                └─→ Uncertainty Estimates
                    (Confidence intervals)
                        ↓
                Final Recommendations
```

## Execution Patterns

### Pattern 1: Quick & Dirty
```bash
python threshold_optimizer.py
# Use grid search results directly
# Time: 12 seconds
```

### Pattern 2: Exploratory
```bash
python random_search_optimizer.py --n-samples 50000 --visualize
# Understand parameter space thoroughly
# Time: ~5 seconds + visualization
```

### Pattern 3: Efficient
```bash
pip install scikit-optimize
python bayesian_optimizer.py --n-calls 200
# Sample-efficient optimization
# Time: ~15 seconds
```

### Pattern 4: Comprehensive (Recommended)
```bash
# Phase 1: Explore
python random_search_optimizer.py --n-samples 50000

# Phase 2: Refine
python bayesian_optimizer.py --n-calls 100

# Phase 3: Verify
python threshold_optimizer.py

# Compare all results
python compare_optimizers.py
# Total time: ~35 seconds
```

### Pattern 5: One-Shot
```bash
python compare_optimizers.py
# Runs all three automatically
# Time: ~30 seconds
```

### Pattern 6: Educational
```bash
python example_optimization_workflow.py
# See all three phases in action
# Demonstrates recommended workflow
# Time: ~20 seconds
```

## Output Files

### CSV Results Structure

**optimal_rules.csv** (Grid Search):
```csv
rule,n_features,recall,precision,tp,fp,fn,margin_score,GeoMean,SustainMs,BodyE
GeoMean > 300 AND ...,3,1.0,1.0,6,0,0,49.3,300,80.0,50.0
```

**bayesian_optimal_rules.csv** (Bayesian Optimization):
```csv
rule,n_features,score,margin,recall,precision,GeoMean,SustainMs,BodyE
GeoMean > 300 AND ...,3,149.2,49.2,1.0,1.0,300.0,80.0,50.0
```

**random_search_results.csv** (Random Search):
```csv
rule,margin,GeoMean,SustainMs,BodyE
GeoMean > 305.2 AND ...,48.7,305.2,83.4,52.1
```

All three formats can be:
1. Sorted by margin to find best rules
2. Filtered by n_features for different complexity levels
3. Analyzed for parameter distributions
4. Exported to midiconfig.yaml

## Integration Points

### Current State
```
debugging_scripts/ → Manual review → midiconfig.yaml → stems_to_midi/detection.py
```

### Future State (Planned)
```
debugging_scripts/
    │
    ├─→ optimize_config.py (wizard)
    │       │
    │       ├─→ Automatic data generation
    │       ├─→ Interactive labeling
    │       ├─→ Run optimal optimizer
    │       └─→ Update midiconfig.yaml
    │
    └─→ midiconfig.yaml (enhanced schema)
            │
            └─→ stems_to_midi/detection.py (multi-feature rules)
```

See `ml-driven-threshold-config.plan.md` for full integration roadmap.

## Dependencies Graph

```
Core Dependencies (in conda env):
    pandas ─────────┐
    numpy ──────────┼─→ All scripts
    scikit-learn ───┤
    matplotlib ─────┘

Optional Dependencies:
    scikit-optimize ─→ bayesian_optimizer.py
    seaborn ─────────→ random_search_optimizer.py (enhanced viz)
```

## Performance Characteristics

```
Script                      Time    Memory   Disk Output
─────────────────────────────────────────────────────────
classifier.py               <1s     ~50MB    stdout
threshold_optimizer.py      12s     ~100MB   optimal_rules.csv (30 rows)
random_search_optimizer.py  3-5s    ~200MB   random_search_results.csv (30 rows)
                                              + PNG visualizations (3 files)
bayesian_optimizer.py       8-10s   ~150MB   bayesian_optimal_rules.csv (8 rows)
                                              + PNG plots (3-6 files)
compare_optimizers.py       25-35s  ~300MB   All CSV + comparison report
example_workflow.py         15-20s  ~100MB   stdout (educational demo)
```

## Extension Points

Future enhancements can be added:

```
debugging_scripts/
    │
    ├─→ NEW: multi_objective_optimizer.py
    │        (Optimize margin + simplicity simultaneously)
    │
    ├─→ NEW: active_learning.py
    │        (Suggest which examples to label next)
    │
    ├─→ NEW: transfer_learning.py
    │        (Use rules from other projects as prior)
    │
    └─→ NEW: ensemble_classifier.py
            (Combine multiple rules with voting)
```

## Testing Strategy

To validate optimizers:
1. Run on known dataset (`data.csv`)
2. Verify all find similar optimal margin (~49%)
3. Check consistency across runs (with fixed seed)
4. Validate against manual threshold testing
5. Test on new projects (generalization)

---

**Last Updated**: 2025-10-31
