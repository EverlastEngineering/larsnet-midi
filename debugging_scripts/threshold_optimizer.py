"""
Threshold Optimizer for Hi-Hat Classification

This script performs exhaustive grid search to find optimal threshold combinations
that maximize classification accuracy while maintaining safety margins.

Approach:
1. Generate all reasonable threshold combinations for key features
2. Evaluate each combination's performance (precision, recall, margins)
3. Rank by a composite score that values both accuracy and safety
4. Output the best rules with detailed analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product

# Load data
script_dir = Path(__file__).parent
df = pd.read_csv(script_dir / 'data.csv')

# Label the open hi-hat events
open_hihat_times = [1.962, 7.755, 13.944, 19.574, 25.960, 31.753]
df['OpenHH'] = df['Time'].round(3).isin(open_hihat_times).astype(int)

# Get statistics for range exploration
open_hh = df[df['OpenHH'] == 1]
closed_hh = df[df['OpenHH'] == 0]

print('='*80)
print('EXHAUSTIVE THRESHOLD OPTIMIZATION')
print('='*80)

print('\nData Summary:')
print(f'  Open hi-hats: {len(open_hh)}')
print(f'  Closed hi-hats: {len(closed_hh)}')

print('\nOpen Hi-Hat Value Ranges:')
for feat in ['GeoMean', 'SustainMs', 'BodyE', 'SizzleE']:
    print(f'  {feat:12s}: min={open_hh[feat].min():7.1f}, max={open_hh[feat].max():7.1f}, mean={open_hh[feat].mean():7.1f}')

print('\nClosed Hi-Hat Value Ranges (for separation):')
for feat in ['GeoMean', 'SustainMs', 'BodyE', 'SizzleE']:
    print(f'  {feat:12s}: min={closed_hh[feat].min():7.1f}, max={closed_hh[feat].max():7.1f}, mean={closed_hh[feat].mean():7.1f}')

# ============================================================================
# GRID SEARCH: Test all combinations of thresholds
# ============================================================================

print('\n' + '='*80)
print('GRID SEARCH: Testing all threshold combinations...')
print('='*80)

# Define search ranges based on data
# Start from slightly below the minimum open value, go up to well above
geomean_thresholds = np.arange(300, 600, 25)  # Test every 25 units
sustain_thresholds = np.arange(80, 180, 10)   # Test every 10ms
bodye_thresholds = np.arange(50, 200, 10)     # Test every 10 units

# Store results
results = []

# Test 1-feature rules
print('\nTesting 1-feature rules...')
for gm_thresh in geomean_thresholds:
    df['pred'] = (df['GeoMean'] > gm_thresh).astype(int)
    
    tp = ((df['OpenHH'] == 1) & (df['pred'] == 1)).sum()
    fp = ((df['OpenHH'] == 0) & (df['pred'] == 1)).sum()
    fn = ((df['OpenHH'] == 1) & (df['pred'] == 0)).sum()
    
    if tp > 0:  # Only keep rules that catch at least one open hi-hat
        recall = tp / 6
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Calculate margin: distance from minimum open value
        margin = open_hh['GeoMean'].min() - gm_thresh
        margin_pct = (margin / open_hh['GeoMean'].min()) * 100
        
        results.append({
            'rule': f'GeoMean > {gm_thresh}',
            'n_features': 1,
            'recall': recall,
            'precision': precision,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'margin_score': margin_pct,
            'GeoMean': gm_thresh,
            'SustainMs': None,
            'BodyE': None
        })

# Test 2-feature rules: GeoMean + SustainMs
print('Testing 2-feature rules (GeoMean + SustainMs)...')
for gm_thresh, sus_thresh in product(geomean_thresholds, sustain_thresholds):
    df['pred'] = ((df['GeoMean'] > gm_thresh) & (df['SustainMs'] > sus_thresh)).astype(int)
    
    tp = ((df['OpenHH'] == 1) & (df['pred'] == 1)).sum()
    fp = ((df['OpenHH'] == 0) & (df['pred'] == 1)).sum()
    fn = ((df['OpenHH'] == 1) & (df['pred'] == 0)).sum()
    
    if tp > 0:
        recall = tp / 6
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Calculate combined margin score (average of both margins)
        gm_margin_pct = ((open_hh['GeoMean'].min() - gm_thresh) / open_hh['GeoMean'].min()) * 100
        sus_margin_pct = ((open_hh['SustainMs'].min() - sus_thresh) / open_hh['SustainMs'].min()) * 100
        margin_score = (gm_margin_pct + sus_margin_pct) / 2
        
        results.append({
            'rule': f'GeoMean > {gm_thresh} AND SustainMs > {sus_thresh}',
            'n_features': 2,
            'recall': recall,
            'precision': precision,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'margin_score': margin_score,
            'GeoMean': gm_thresh,
            'SustainMs': sus_thresh,
            'BodyE': None
        })

# Test 3-feature rules: GeoMean + SustainMs + BodyE
print('Testing 3-feature rules (GeoMean + SustainMs + BodyE)...')
# Use sparser grid for 3-feature to keep runtime reasonable
for gm_thresh, sus_thresh, body_thresh in product(
    geomean_thresholds[::2],  # Every other value
    sustain_thresholds[::2], 
    bodye_thresholds[::2]
):
    df['pred'] = ((df['GeoMean'] > gm_thresh) & 
                  (df['SustainMs'] > sus_thresh) & 
                  (df['BodyE'] > body_thresh)).astype(int)
    
    tp = ((df['OpenHH'] == 1) & (df['pred'] == 1)).sum()
    fp = ((df['OpenHH'] == 0) & (df['pred'] == 1)).sum()
    fn = ((df['OpenHH'] == 1) & (df['pred'] == 0)).sum()
    
    if tp > 0:
        recall = tp / 6
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Calculate combined margin score
        gm_margin_pct = ((open_hh['GeoMean'].min() - gm_thresh) / open_hh['GeoMean'].min()) * 100
        sus_margin_pct = ((open_hh['SustainMs'].min() - sus_thresh) / open_hh['SustainMs'].min()) * 100
        body_margin_pct = ((open_hh['BodyE'].min() - body_thresh) / open_hh['BodyE'].min()) * 100
        margin_score = (gm_margin_pct + sus_margin_pct + body_margin_pct) / 3
        
        results.append({
            'rule': f'GeoMean > {gm_thresh} AND SustainMs > {sus_thresh} AND BodyE > {body_thresh}',
            'n_features': 3,
            'recall': recall,
            'precision': precision,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'margin_score': margin_score,
            'GeoMean': gm_thresh,
            'SustainMs': sus_thresh,
            'BodyE': body_thresh
        })

print(f'Total rules tested: {len(results)}')

# ============================================================================
# RANK AND DISPLAY RESULTS
# ============================================================================

results_df = pd.DataFrame(results)

# Filter to only perfect recall (catches all 6 open hi-hats)
perfect_recall = results_df[results_df['recall'] == 1.0].copy()

print(f'\nRules with 100% recall (catch all 6 open hi-hats): {len(perfect_recall)}')

# Among perfect recall, filter to perfect precision (no false positives)
perfect_rules = perfect_recall[perfect_recall['precision'] == 1.0].copy()

print(f'Rules with 100% recall AND 100% precision: {len(perfect_rules)}')

if len(perfect_rules) > 0:
    # Sort by margin score (higher is better - means safer thresholds)
    perfect_rules = perfect_rules.sort_values('margin_score', ascending=False)
    
    print('\n' + '='*80)
    print('TOP 15 RULES (100% accuracy, ranked by safety margin)')
    print('='*80)
    print('\nMargin Score = Average % distance below minimum open hi-hat value')
    print('Higher margin = safer rule (further from edge cases)\n')
    
    for idx, (_, row) in enumerate(perfect_rules.head(15).iterrows(), 1):
        print(f'\n{idx}. {row["rule"]}')
        print(f'   Margin Safety Score: {row["margin_score"]:.1f}%')
        print(f'   Features used: {row["n_features"]}')
        print(f'   Performance: {row["tp"]}/6 detected, {row["fp"]} false positives')
        
        # Show distance to minimum
        if row['GeoMean']:
            dist = open_hh['GeoMean'].min() - row['GeoMean']
            print(f'   GeoMean margin: {dist:.1f} units ({(dist/open_hh["GeoMean"].min())*100:.1f}% below min)')
        if row['SustainMs']:
            dist = open_hh['SustainMs'].min() - row['SustainMs']
            print(f'   SustainMs margin: {dist:.1f} ms ({(dist/open_hh["SustainMs"].min())*100:.1f}% below min)')
        if row['BodyE']:
            dist = open_hh['BodyE'].min() - row['BodyE']
            print(f'   BodyE margin: {dist:.1f} units ({(dist/open_hh["BodyE"].min())*100:.1f}% below min)')

    print('\n' + '='*80)
    print('RECOMMENDATIONS BY USE CASE')
    print('='*80)
    
    # Best 1-feature rule (simplest)
    best_1feat = perfect_rules[perfect_rules['n_features'] == 1].iloc[0] if len(perfect_rules[perfect_rules['n_features'] == 1]) > 0 else None
    if best_1feat is not None:
        print(f'\nBest SIMPLE rule (1 feature):')
        print(f'  {best_1feat["rule"]}')
        print(f'  Margin: {best_1feat["margin_score"]:.1f}%')
    
    # Best 2-feature rule (good balance)
    best_2feat = perfect_rules[perfect_rules['n_features'] == 2].iloc[0] if len(perfect_rules[perfect_rules['n_features'] == 2]) > 0 else None
    if best_2feat is not None:
        print(f'\nBest BALANCED rule (2 features):')
        print(f'  {best_2feat["rule"]}')
        print(f'  Margin: {best_2feat["margin_score"]:.1f}%')
    
    # Best 3-feature rule (most robust)
    best_3feat = perfect_rules[perfect_rules['n_features'] == 3].iloc[0] if len(perfect_rules[perfect_rules['n_features'] == 3]) > 0 else None
    if best_3feat is not None:
        print(f'\nBest ROBUST rule (3 features):')
        print(f'  {best_3feat["rule"]}')
        print(f'  Margin: {best_3feat["margin_score"]:.1f}%')
    
    # Export top 30 to CSV for further analysis
    perfect_rules.head(30).to_csv(script_dir / 'optimal_rules.csv', index=False)
    print(f'\n✓ Top 30 rules exported to: optimal_rules.csv')

else:
    print('\nNo rules achieved 100% recall AND precision.')
    print('Showing best rules by recall and precision...')
    
    # Sort by composite score
    results_df['composite_score'] = results_df['recall'] * 0.6 + results_df['precision'] * 0.4
    results_df = results_df.sort_values('composite_score', ascending=False)
    
    print('\nTop 10 rules by composite score:')
    for idx, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
        print(f'\n{idx}. {row["rule"]}')
        print(f'   Recall: {row["recall"]*100:.0f}%, Precision: {row["precision"]*100:.0f}%')
        print(f'   {row["tp"]}/6 detected, {row["fp"]} false positives, {row["fn"]} missed')

print('\n' + '='*80)
print('ANALYSIS COMPLETE')
print('='*80)
