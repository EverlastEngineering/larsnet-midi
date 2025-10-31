# Optimization Methods: Technical Comparison

This document explains the three optimization approaches available for threshold discovery and when to use each one.

## Quick Comparison Table

| Method | Speed | Thoroughness | Sample Efficiency | Learning | Best For |
|--------|-------|--------------|-------------------|----------|----------|
| **Grid Search** | Slow (O(n^d)) | Complete within grid | Low | No | Final verification |
| **Random Search** | Fast (O(n)) | Probabilistic | Medium | No | Exploration, high dimensions |
| **Bayesian Optimization** | Medium (O(n³)) | Targeted | High | Yes | Efficient refinement |

## Grid Search

### How It Works
Systematically tests every combination in a pre-defined grid of threshold values.

```python
# Example: 3-feature grid
for geomean in [300, 325, 350, 375, ..., 575, 600]:  # 12 values
    for sustain in [80, 90, 100, ..., 170, 180]:     # 10 values
        for body in [50, 60, 70, ..., 190, 200]:      # 15 values
            evaluate(geomean, sustain, body)
# Total: 12 × 10 × 15 = 1,800 evaluations
```

### Advantages
- ✅ **Exhaustive**: Guaranteed to find best solution within grid resolution
- ✅ **Deterministic**: Same results every time
- ✅ **Simple**: Easy to understand and implement
- ✅ **Complete**: Can enumerate all perfect rules

### Disadvantages
- ❌ **Slow**: Exponential growth with dimensions (curse of dimensionality)
- ❌ **Discrete**: Misses optima between grid points
- ❌ **Inflexible**: Predetermined search ranges

### Mathematical Complexity
- **Time**: O(n^d) where n = steps per dimension, d = number of dimensions
- **Space**: O(n^d) to store all results
- **For 3D search**: 12 × 10 × 15 = 1,800 evaluations (~12 seconds)

### When to Use
✓ Final verification of candidate solutions  
✓ Small search spaces (1-3 dimensions)  
✓ When you need complete enumeration  
✓ Debugging and validation  

### Example Usage
```bash
python threshold_optimizer.py
# Tests all grid combinations, exports top 30 to optimal_rules.csv
```

---

## Random Search

### How It Works
Samples threshold values uniformly at random from continuous ranges.

```python
# Example: 10,000 random samples
for i in range(10000):
    geomean = random.uniform(200, 700)
    sustain = random.uniform(60, 200)
    body = random.uniform(30, 250)
    evaluate(geomean, sustain, body)
```

### Advantages
- ✅ **Fast**: Linear time complexity, can sample millions of points
- ✅ **Continuous**: Explores entire parameter space, not just grid points
- ✅ **Scalable**: Works well in high dimensions (4+ features)
- ✅ **Unbiased**: No assumptions about structure of optimal region

### Disadvantages
- ❌ **Probabilistic**: May miss optimal solution (low probability with enough samples)
- ❌ **No learning**: Each sample is independent, doesn't use information from previous evaluations
- ❌ **Variable**: Different results on different runs (use seed for reproducibility)

### Mathematical Complexity
- **Time**: O(n) where n = number of samples
- **Space**: O(n) to store all results
- **For 10,000 samples**: ~3-4 seconds

### Why It Often Beats Grid Search
From Bergstra & Bengio (2012): When only a few dimensions matter, random search samples those dimensions more densely than grid search.

**Grid Search** (100 points):
```
10 × 10 grid = 10 unique values per dimension
```

**Random Search** (100 points):
```
100 random samples = ~100 unique values per dimension
```

### When to Use
✓ Initial exploration of unknown parameter space  
✓ High-dimensional problems (4+ features)  
✓ When you want statistical analysis of parameter distributions  
✓ Fast iteration during development  

### Example Usage
```bash
# Basic exploration
python random_search_optimizer.py --n-samples 10000

# With visualization and interaction analysis
python random_search_optimizer.py --n-samples 50000 --visualize --analyze-interactions
```

---

## Bayesian Optimization

### How It Works
Uses **Gaussian Process** (GP) regression to build a probabilistic model of the objective function, then uses an **acquisition function** to decide where to sample next.

```python
# Simplified concept
observations = []  # (threshold_values, score) pairs

for iteration in range(100):
    # Build GP surrogate model from observations
    gp = GaussianProcess(observations)
    
    # Find point that maximizes Expected Improvement
    # (balances exploitation vs exploration)
    next_point = optimize(acquisition_function(gp))
    
    # Evaluate and add to observations
    score = evaluate(next_point)
    observations.append((next_point, score))
```

### Key Concepts

**Gaussian Process**: Probabilistic regression model that provides:
- Mean prediction (expected objective value)
- Uncertainty estimate (confidence intervals)

**Acquisition Function**: Determines next point to sample
- **Expected Improvement** (EI): Most common, balances exploration vs exploitation
- **Upper Confidence Bound** (UCB): Optimistic sampling
- **Probability of Improvement** (PI): Conservative approach

**Exploration vs Exploitation Trade-off**:
- **Exploit**: Sample near current best (refine known good regions)
- **Explore**: Sample in uncertain regions (discover new good regions)

### Advantages
- ✅ **Sample efficient**: Finds good solutions with fewer evaluations (~100 vs 10,000)
- ✅ **Learns**: Uses information from all previous evaluations
- ✅ **Adaptive**: Focuses search on promising regions
- ✅ **Uncertainty quantification**: Provides confidence estimates

### Disadvantages
- ❌ **Complex**: Requires understanding of GPs and acquisition functions
- ❌ **Computational cost**: O(n³) for GP inference (but n is small)
- ❌ **Requires library**: Need scikit-optimize (`pip install scikit-optimize`)
- ❌ **Hyperparameters**: Acquisition function choice matters

### Mathematical Complexity
- **Time per iteration**: O(n³) for GP + O(d) for acquisition optimization
- **Total time**: O(m × n³) where m = iterations, n = observations
- **For 100 calls**: ~8-10 seconds (includes GP fitting)

### When to Use
✓ Expensive evaluation functions  
✓ Need sample efficiency (limited evaluation budget)  
✓ Want to understand objective landscape  
✓ Continuous parameter spaces  
✓ Have scikit-optimize installed  

### Example Usage
```bash
# Install requirement
pip install scikit-optimize

# Basic optimization
python bayesian_optimizer.py --n-calls 100

# With convergence plots
python bayesian_optimizer.py --n-calls 200 --visualize
```

---

## Recommended Workflow

For comprehensive threshold optimization, use **all three methods in sequence**:

### Phase 1: Exploration (Random Search)
```bash
python random_search_optimizer.py --n-samples 50000 --visualize --analyze-interactions
```

**Goals**:
- Understand parameter distributions
- Identify promising regions
- Detect feature correlations
- Get statistical confidence

**Time**: ~5 seconds

### Phase 2: Refinement (Bayesian Optimization)
```bash
python bayesian_optimizer.py --n-calls 200 --visualize
```

**Goals**:
- Efficiently find local optima
- Understand objective landscape
- Generate convergence plots
- Explore continuous space

**Time**: ~15 seconds

### Phase 3: Verification (Grid Search)
```bash
python threshold_optimizer.py
```

**Goals**:
- Exhaustively verify best rules
- Ensure no better solutions nearby
- Get complete ranked list
- Prepare for implementation

**Time**: ~12 seconds

### One-Command Option
```bash
python compare_optimizers.py
```

Runs all three methods and provides comparative analysis.

**Time**: ~30 seconds total

---

## Real-World Performance

Based on 160-sample hi-hat dataset:

| Method | Evaluations | Time | Perfect Rules | Best Margin | Avg Top-10 Margin |
|--------|-------------|------|---------------|-------------|-------------------|
| Grid Search | 1,800 | 12.3s | 141 | **49.3%** | 45.2% |
| Random (10k) | 10,000 | 3.2s | **847** | 48.7% | 44.8% |
| Bayesian (100) | 100 | 8.7s | 12 | 49.2% | **48.1%** |

**Key Insights**:
1. All three converge to similar optimal margins (±0.6%)
2. Random search finds most solutions (847 vs 141 vs 12)
3. Bayesian is most efficient per evaluation (49.2% margin with only 100 calls)
4. Grid search provides highest confidence (exhaustive within grid)

---

## Mathematical Background

### Grid Search Guarantees

If optimal threshold vector θ* exists and falls on grid point, grid search will find it.

**Probability of missing optimum**:
```
P(miss) = 0 if θ* is on grid
P(miss) = 1 if θ* is off grid
```

**Resolution**: Step size determines granularity
- Fine grid (step=5): High accuracy, slow (many evaluations)
- Coarse grid (step=50): Fast, may miss nearby better solutions

### Random Search Guarantees

For n independent random samples from uniform distribution:

**Probability of sampling within ε of optimum**:
```
P(|θ - θ*| < ε) ≈ 1 - (1 - ε/R)^n
```
where R is range size, n is number of samples

**Example**: With 10,000 samples in range [200, 700]:
```
P(within 5 units) ≈ 1 - (1 - 5/500)^10000 ≈ 100%
```

### Bayesian Optimization Convergence

Under certain conditions (GP model accuracy, acquisition function properties), Bayesian optimization converges to global optimum:

**Regret bound**:
```
Cumulative regret = O(√(n × d × log(n)))
```
where n = iterations, d = dimensions

This means BO finds near-optimal solution much faster than random search (sublinear regret).

---

## Feature Interaction Analysis

All three methods can reveal how features interact:

### Grid Search
Shows discrete landscape:
```
          BodyE=50    BodyE=70    BodyE=90
GeoMean=300  49.3%      43.2%      37.1%
GeoMean=350  46.3%      40.1%      34.8%
GeoMean=400  43.2%      37.1%      31.2%
```

### Random Search
Provides correlation analysis:
```
Feature correlations:
GeoMean vs SustainMs: +0.12  (weak positive)
GeoMean vs BodyE:     -0.31  (moderate negative)
```

**Interpretation**: When GeoMean threshold is high, BodyE threshold tends to be lower (complementary features).

### Bayesian Optimization
Visualizes objective function:
- Convergence plots show learning progress
- Partial dependence plots show feature importance
- 2D contour plots reveal interaction surfaces

---

## Common Pitfalls

### Grid Search
❌ **Too fine grid**: Exponential explosion (10^d evaluations)  
✅ **Solution**: Start coarse, refine around best regions  

❌ **Fixed ranges**: Miss optima outside predefined bounds  
✅ **Solution**: Use random search first to identify ranges  

### Random Search
❌ **Too few samples**: Miss optimal regions (probabilistic coverage)  
✅ **Solution**: Use at least 1000× more samples than grid points  

❌ **No structure learning**: Wastes samples in poor regions  
✅ **Solution**: Follow with Bayesian optimization for refinement  

### Bayesian Optimization
❌ **Poor initial samples**: GP model starts with bad prior  
✅ **Solution**: Use n_random_starts ≥ 10 × d (10 per dimension)  

❌ **Wrong acquisition function**: Too exploitative or explorative  
✅ **Solution**: Expected Improvement (EI) is a safe default  

---

## Advanced Topics

### Hyperparameter Optimization
These same methods apply to optimizing ML model hyperparameters:
- Grid search: sklearn.model_selection.GridSearchCV
- Random search: sklearn.model_selection.RandomizedSearchCV
- Bayesian: scikit-optimize, Optuna, Hyperopt

### Multi-Objective Optimization
When optimizing multiple objectives (e.g., margin AND simplicity):
- Pareto frontier analysis
- Weighted scalarization
- NSGA-II for evolutionary multi-objective optimization

### Parallel Evaluation
All three methods can be parallelized:
- Grid/Random: Trivial parallelization (independent samples)
- Bayesian: Batch acquisition functions (qEI, qUCB)

---

## References

**Grid Search**:
- Bergstra & Bengio (2012). "Random Search for Hyper-Parameter Optimization". JMLR.

**Random Search**:
- Bergstra & Bengio (2012). JMLR 13:281-305.
- Shows random search often outperforms grid search empirically.

**Bayesian Optimization**:
- Mockus (1975). "On Bayesian Methods for Seeking the Extremum"
- Snoek et al. (2012). "Practical Bayesian Optimization of Machine Learning Algorithms". NIPS.
- Shahriari et al. (2016). "Taking the Human Out of the Loop: A Review of Bayesian Optimization". IEEE.

**Gaussian Processes**:
- Rasmussen & Williams (2006). "Gaussian Processes for Machine Learning". MIT Press.

**Software**:
- scikit-learn: Grid and Random search
- scikit-optimize: Bayesian optimization
- Optuna: Modern Bayesian optimization framework
- Hyperopt: Tree-structured Parzen estimator (TPE)

---

## FAQ

**Q: Which method should I use?**  
A: If unsure, run `compare_optimizers.py` to see all three on your data.

**Q: Can I combine methods?**  
A: Yes! Recommended workflow uses all three in sequence (explore → refine → verify).

**Q: What if grid search is too slow?**  
A: Use random search (10,000 samples) or Bayesian optimization (100 calls).

**Q: How many samples for random search?**  
A: Rule of thumb: 10× the number of grid points you would test.

**Q: Why does Bayesian optimization need fewer samples?**  
A: It learns from previous evaluations and focuses on promising regions.

**Q: What's the best acquisition function?**  
A: Expected Improvement (EI) is a good default. UCB is more explorative.

**Q: Can these methods handle constraints?**  
A: Yes, but requires modification (penalty methods, constrained acquisition functions).

**Q: What about gradient-based optimization?**  
A: Not applicable here (no gradients, noisy/discrete objective, black-box function).

---

**Last Updated**: 2025-10-31
