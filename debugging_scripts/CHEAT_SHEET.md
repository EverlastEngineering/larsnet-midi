# Threshold Optimization Cheat Sheet

Quick reference for common tasks.

## 🚀 Quick Commands

### Run All Optimizers (Compare Results)
```bash
python compare_optimizers.py
```

### Grid Search (Exhaustive)
```bash
python threshold_optimizer.py
```

### Random Search (Fast Exploration)
```bash
python random_search_optimizer.py --n-samples 10000
```

### Bayesian Optimization (Sample Efficient)
```bash
pip install scikit-optimize  # First time only
python bayesian_optimizer.py --n-calls 100
```

### Feature Analysis (Pattern Discovery)
```bash
python classifier.py
```

---

## 📊 Which Method Should I Use?

| Situation | Command | Time | Why? |
|-----------|---------|------|------|
| "Just give me the answer" | `compare_optimizers.py` | 30s | Runs all three, picks best |
| "I want to understand my data" | `random_search_optimizer.py --visualize` | 5s | Stats + plots |
| "I need this fast" | `random_search_optimizer.py` | 3s | Fastest method |
| "I want guaranteed best" | `threshold_optimizer.py` | 12s | Exhaustive search |
| "I have limited budget" | `bayesian_optimizer.py` | 9s | Most sample-efficient |
| "First time user" | `QUICK_START.md` | 5min | Step-by-step guide |

---

## 📝 Typical Workflow

```bash
# 1. Generate training data
cd /path/to/larsnet
python stems_to_midi.py <PROJECT> --stems hihat 2>&1 | tee /tmp/hihat.log
# Copy spectral analysis table to debugging_scripts/data.csv

# 2. Label open hi-hats
# Listen to audio, note timestamps
# Edit classifier.py line 12: open_hihat_times = [1.962, 7.755, ...]

# 3. Quick validation
cd debugging_scripts
python classifier.py  # See if patterns make sense

# 4. Find optimal thresholds
python compare_optimizers.py  # OR pick individual method

# 5. Apply best rule to config (manual for now)
# Update midiconfig.yaml with threshold values

# 6. Test results
cd ..
python stems_to_midi.py <PROJECT> --stems hihat
# Verify open hi-hats are detected correctly
```

---

## 🎯 Common Tasks

### Add More Training Data
```bash
# Generate data from another song
python stems_to_midi.py <NEW_PROJECT> --stems hihat > /tmp/new_data.log

# Append to existing data.csv (keep headers once)
# Label new open hi-hats
# Re-run optimizer
```

### Visualize Parameter Space
```bash
python random_search_optimizer.py --n-samples 50000 --visualize --analyze-interactions
# Generates: random_2d_*.png, random_3d_space.png, feature_correlations.png
```

### Test Different Feature Combinations
```bash
# Edit feature_names in optimizer scripts
# Try: ['GeoMean', 'SustainMs', 'SizzleE']
# Or: ['Total', 'Amp', 'BodyE']
```

### Export Results to Spreadsheet
```bash
# All optimizers output CSV files:
# - optimal_rules.csv
# - bayesian_optimal_rules.csv
# - random_search_results.csv
# Open in Excel/Numbers/Google Sheets
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'skopt'"
```bash
pip install scikit-optimize
```

### "No rules achieve 100% accuracy"
- Check your timestamps are correct (±0.05s tolerance)
- Need more diverse training examples
- Try different features

### "All margins are very low (<10%)"
- Your open/closed hi-hats are very similar
- Label only the most obvious examples
- May need different audio features

### "Grid search too slow"
Edit `threshold_optimizer.py`:
```python
# Use coarser steps
geomean_thresholds = np.arange(300, 600, 50)  # Was 25, now 50
```

### "Random search found nothing"
- Increase samples: `--n-samples 100000`
- Check search ranges in code match your data
- Verify data.csv is correctly formatted

---

## 📈 Interpreting Results

### Feature Importance
```
GeoMean    0.247  ← 24.7% of classification power
Total      0.215  ← 21.5%
SustainMs  0.131  ← 13.1% (surprisingly low!)
```
**Higher = more discriminative**

### Margin Score
```
GeoMean > 300
Min open value: 546.2
Threshold: 300
Margin: 246.2 units = 45% safety buffer
```
**Higher = safer rule** (tolerates more variation)

### Perfect Rules Count
```
141 rules with 100% accuracy found
```
**More rules = more robust** (pattern is clear, many solutions work)

---

## 💡 Pro Tips

1. **Start simple**: Use `compare_optimizers.py --quick` for rapid iteration
2. **Label variety**: Include soft/loud, early/late open hi-hats
3. **Validate on test set**: Check rules work on different song
4. **Check correlations**: Features that correlate can substitute for each other
5. **Trust the margins**: 40%+ margin = production ready
6. **3-feature rules**: Usually best trade-off (robust but not over-complicated)

---

## 🔗 Quick Links

| Document | Purpose |
|----------|---------|
| [QUICK_START.md](QUICK_START.md) | User guide (5 min) |
| [README.md](README.md) | Comprehensive docs |
| [OPTIMIZATION_METHODS.md](OPTIMIZATION_METHODS.md) | Algorithm details |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design |
| [WHATS_NEW.md](WHATS_NEW.md) | Recent features |

---

## ⚡ Speed Comparison

```
Method              Time    Thoroughness   Sample Efficiency
──────────────────────────────────────────────────────────
classifier.py       <1s     N/A           N/A (analysis only)
threshold_opt       12s     ★★★★★         ★★☆☆☆
random_search       3s      ★★★★☆         ★★★☆☆
bayesian_opt        9s      ★★★☆☆         ★★★★★
compare_all         30s     ★★★★★         ★★★★☆
```

---

## 📋 File Size Reference

```
data.csv                    ~15 KB  (160 samples)
optimal_rules.csv           ~5 KB   (30 rules)
bayesian_optimal_rules.csv  ~3 KB   (8 rules)
random_search_results.csv   ~4 KB   (30 rules)
*.png visualizations        ~100 KB each
```

---

## 🎓 Learning Path

1. **Beginner**: Follow QUICK_START.md, use `compare_optimizers.py`
2. **Intermediate**: Read README.md, try individual optimizers
3. **Advanced**: Study OPTIMIZATION_METHODS.md, modify algorithms
4. **Expert**: Read ARCHITECTURE.md, extend system

---

## 🆘 Getting Help

1. Check [README.md](README.md) FAQ section
2. Run `python <script>.py --help`
3. Read [OPTIMIZATION_METHODS.md](OPTIMIZATION_METHODS.md) for theory
4. Review [ARCHITECTURE.md](ARCHITECTURE.md) for system design
5. Open GitHub issue with:
   - Your `data.csv` (first 20 lines)
   - Command you ran
   - Error message
   - Python version: `python --version`

---

**Last Updated**: 2025-10-31  
**Quick Start**: `python compare_optimizers.py`
