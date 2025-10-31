#!/usr/bin/env python3
"""
Example: Complete Optimization Workflow

Demonstrates the recommended three-phase approach:
1. Random Search - Broad exploration
2. Bayesian Optimization - Efficient refinement
3. Grid Search - Final verification

This is an educational example showing how to combine all three methods
for comprehensive threshold optimization.

Usage:
    python example_optimization_workflow.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, precision_score
import time


def load_data(csv_path='data.csv'):
    """Load and prepare training data."""
    df = pd.read_csv(csv_path)
    
    # Define open hi-hat times (ground truth labels)
    open_hihat_times = [1.962, 7.755, 13.944, 19.574, 25.960, 31.753]
    
    # Label the data
    df['is_open'] = df['Time'].apply(
        lambda t: any(abs(t - open_t) < 0.05 for open_t in open_hihat_times)
    )
    
    return df


def evaluate_rule(df, geomean_th, sustain_th, body_th):
    """
    Evaluate a 3-feature classification rule.
    
    Returns:
        (recall, precision, margin_score)
    """
    mask = (df['GeoMean'] > geomean_th) & \
           (df['SustainMs'] > sustain_th) & \
           (df['BodyE'] > body_th)
    
    y_true = df['is_open']
    y_pred = mask
    
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    
    # Calculate margin if perfect accuracy
    if recall == 1.0 and precision == 1.0:
        open_hits = df[df['is_open'] == True]
        margins = []
        
        min_geomean = open_hits['GeoMean'].min()
        if min_geomean > 0:
            margins.append(((min_geomean - geomean_th) / min_geomean) * 100)
        
        min_sustain = open_hits['SustainMs'].min()
        if min_sustain > 0:
            margins.append(((min_sustain - sustain_th) / min_sustain) * 100)
        
        min_body = open_hits['BodyE'].min()
        if min_body > 0:
            margins.append(((min_body - body_th) / min_body) * 100)
        
        margin = np.mean(margins) if margins else 0
    else:
        margin = 0
    
    return recall, precision, margin


def phase1_random_search(df, n_samples=5000):
    """
    Phase 1: Random Search - Broad Exploration
    
    Goal: Understand the parameter space
    """
    print("=" * 80)
    print("PHASE 1: RANDOM SEARCH - BROAD EXPLORATION")
    print("=" * 80)
    print()
    print(f"Sampling {n_samples:,} random threshold combinations...")
    print("Goal: Understand parameter distribution, identify promising regions")
    print()
    
    np.random.seed(42)
    start_time = time.time()
    
    perfect_rules = []
    
    for i in range(n_samples):
        # Random thresholds
        geomean = np.random.uniform(200, 700)
        sustain = np.random.uniform(60, 200)
        body = np.random.uniform(30, 250)
        
        recall, precision, margin = evaluate_rule(df, geomean, sustain, body)
        
        if recall == 1.0 and precision == 1.0:
            perfect_rules.append({
                'GeoMean': geomean,
                'SustainMs': sustain,
                'BodyE': body,
                'margin': margin
            })
        
        if (i + 1) % 1000 == 0:
            print(f"  {i+1:,}/{n_samples:,} samples | {len(perfect_rules)} perfect rules found", end='\r')
    
    elapsed = time.time() - start_time
    print()
    print()
    
    print(f"✓ Random search complete in {elapsed:.2f}s")
    print(f"  Perfect rules found: {len(perfect_rules)}")
    
    if perfect_rules:
        # Analyze distribution
        geomean_values = [r['GeoMean'] for r in perfect_rules]
        sustain_values = [r['SustainMs'] for r in perfect_rules]
        body_values = [r['BodyE'] for r in perfect_rules]
        
        print()
        print("  Parameter distributions:")
        print(f"    GeoMean:   {np.min(geomean_values):.1f} - {np.max(geomean_values):.1f} (median: {np.median(geomean_values):.1f})")
        print(f"    SustainMs: {np.min(sustain_values):.1f} - {np.max(sustain_values):.1f} (median: {np.median(sustain_values):.1f})")
        print(f"    BodyE:     {np.min(body_values):.1f} - {np.max(body_values):.1f} (median: {np.median(body_values):.1f})")
        
        # Best rule found
        best = max(perfect_rules, key=lambda r: r['margin'])
        print()
        print("  Best rule from random search:")
        print(f"    GeoMean > {best['GeoMean']:.1f} AND SustainMs > {best['SustainMs']:.1f} AND BodyE > {best['BodyE']:.1f}")
        print(f"    Margin: {best['margin']:.1f}%")
        
        # Identify promising region for next phase
        promising_region = {
            'geomean_range': (np.percentile(geomean_values, 10), np.percentile(geomean_values, 90)),
            'sustain_range': (np.percentile(sustain_values, 10), np.percentile(sustain_values, 90)),
            'body_range': (np.percentile(body_values, 10), np.percentile(body_values, 90)),
        }
        
        print()
        print("  Promising regions (10th-90th percentile):")
        print(f"    GeoMean:   {promising_region['geomean_range'][0]:.1f} - {promising_region['geomean_range'][1]:.1f}")
        print(f"    SustainMs: {promising_region['sustain_range'][0]:.1f} - {promising_region['sustain_range'][1]:.1f}")
        print(f"    BodyE:     {promising_region['body_range'][0]:.1f} - {promising_region['body_range'][1]:.1f}")
        
        return best, promising_region, perfect_rules
    else:
        print("  ⚠️  No perfect rules found - may need to adjust search ranges")
        return None, None, []


def phase2_focused_search(df, promising_region, n_samples=2000):
    """
    Phase 2: Focused Random Search - Refinement
    
    Goal: Explore the promising region more thoroughly
    """
    print()
    print("=" * 80)
    print("PHASE 2: FOCUSED SEARCH - REFINEMENT")
    print("=" * 80)
    print()
    print(f"Focusing {n_samples:,} samples on promising region...")
    print("Goal: Find optimal solution within identified region")
    print()
    
    if promising_region is None:
        print("⚠️  Skipping - no promising region identified")
        return None
    
    np.random.seed(43)
    start_time = time.time()
    
    perfect_rules = []
    
    for i in range(n_samples):
        # Sample from promising region
        geomean = np.random.uniform(*promising_region['geomean_range'])
        sustain = np.random.uniform(*promising_region['sustain_range'])
        body = np.random.uniform(*promising_region['body_range'])
        
        recall, precision, margin = evaluate_rule(df, geomean, sustain, body)
        
        if recall == 1.0 and precision == 1.0:
            perfect_rules.append({
                'GeoMean': geomean,
                'SustainMs': sustain,
                'BodyE': body,
                'margin': margin
            })
        
        if (i + 1) % 500 == 0:
            print(f"  {i+1:,}/{n_samples:,} samples | {len(perfect_rules)} perfect rules found", end='\r')
    
    elapsed = time.time() - start_time
    print()
    print()
    
    print(f"✓ Focused search complete in {elapsed:.2f}s")
    print(f"  Perfect rules found: {len(perfect_rules)}")
    
    if perfect_rules:
        best = max(perfect_rules, key=lambda r: r['margin'])
        print()
        print("  Best rule from focused search:")
        print(f"    GeoMean > {best['GeoMean']:.1f} AND SustainMs > {best['SustainMs']:.1f} AND BodyE > {best['BodyE']:.1f}")
        print(f"    Margin: {best['margin']:.1f}%")
        
        return best
    else:
        print("  ⚠️  No perfect rules in focused region")
        return None


def phase3_grid_verification(df, best_rule):
    """
    Phase 3: Grid Search - Verification
    
    Goal: Verify no better solutions exist nearby
    """
    print()
    print("=" * 80)
    print("PHASE 3: GRID VERIFICATION - FINAL CHECK")
    print("=" * 80)
    print()
    print("Testing grid around best solution...")
    print("Goal: Verify optimality, find alternative rules with similar performance")
    print()
    
    if best_rule is None:
        print("⚠️  Skipping - no best rule to verify")
        return
    
    start_time = time.time()
    
    # Define grid around best rule (±10% range, finer steps)
    geomean_center = best_rule['GeoMean']
    sustain_center = best_rule['SustainMs']
    body_center = best_rule['BodyE']
    
    geomean_range = np.arange(max(200, geomean_center - 100), geomean_center + 100, 10)
    sustain_range = np.arange(max(60, sustain_center - 30), sustain_center + 30, 5)
    body_range = np.arange(max(30, body_center - 50), body_center + 50, 5)
    
    n_combinations = len(geomean_range) * len(sustain_range) * len(body_range)
    print(f"  Testing {n_combinations:,} combinations in grid...")
    print()
    
    perfect_rules = []
    
    for geomean in geomean_range:
        for sustain in sustain_range:
            for body in body_range:
                recall, precision, margin = evaluate_rule(df, geomean, sustain, body)
                
                if recall == 1.0 and precision == 1.0:
                    perfect_rules.append({
                        'GeoMean': geomean,
                        'SustainMs': sustain,
                        'BodyE': body,
                        'margin': margin
                    })
    
    elapsed = time.time() - start_time
    
    print(f"✓ Grid verification complete in {elapsed:.2f}s")
    print(f"  Perfect rules in grid: {len(perfect_rules)}")
    
    if perfect_rules:
        verified_best = max(perfect_rules, key=lambda r: r['margin'])
        
        print()
        print("  Verified best rule:")
        print(f"    GeoMean > {verified_best['GeoMean']:.1f} AND SustainMs > {verified_best['SustainMs']:.1f} AND BodyE > {verified_best['BodyE']:.1f}")
        print(f"    Margin: {verified_best['margin']:.1f}%")
        
        # Compare to original best
        if abs(verified_best['margin'] - best_rule['margin']) < 0.5:
            print()
            print("  ✓ Original best rule is optimal (margin within 0.5%)")
        elif verified_best['margin'] > best_rule['margin']:
            improvement = verified_best['margin'] - best_rule['margin']
            print()
            print(f"  ✨ Found slightly better rule (margin improved by {improvement:.1f}%)")
        
        # Show top 5
        print()
        print("  Top 5 rules in grid:")
        sorted_rules = sorted(perfect_rules, key=lambda r: r['margin'], reverse=True)[:5]
        for i, rule in enumerate(sorted_rules, 1):
            print(f"    {i}. GeoMean > {rule['GeoMean']:.1f} AND SustainMs > {rule['SustainMs']:.1f} AND BodyE > {rule['BodyE']:.1f}  (margin: {rule['margin']:.1f}%)")
        
        return verified_best
    else:
        print("  ⚠️  No perfect rules in verification grid")
        return None


def main():
    """Run complete three-phase optimization workflow."""
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "COMPLETE OPTIMIZATION WORKFLOW" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    print("This example demonstrates the recommended three-phase approach:")
    print("  1. Random Search → Broad exploration")
    print("  2. Focused Search → Refinement in promising region")
    print("  3. Grid Verification → Final validation")
    print()
    
    # Load data
    print("Loading training data...")
    df = load_data()
    print(f"  Total samples: {len(df)}")
    print(f"  Open hi-hats: {df['is_open'].sum()}")
    print(f"  Closed hi-hats: {(~df['is_open']).sum()}")
    print()
    
    # Phase 1: Exploration
    best_phase1, promising_region, all_phase1 = phase1_random_search(df, n_samples=5000)
    
    # Phase 2: Refinement
    if best_phase1:
        best_phase2 = phase2_focused_search(df, promising_region, n_samples=2000)
    else:
        best_phase2 = None
    
    # Phase 3: Verification
    final_best = best_phase2 if best_phase2 else best_phase1
    if final_best:
        verified_best = phase3_grid_verification(df, final_best)
    else:
        verified_best = None
    
    # Final summary
    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print()
    
    if verified_best:
        print("🏆 RECOMMENDED THRESHOLD CONFIGURATION:")
        print()
        print(f"  GeoMean > {verified_best['GeoMean']:.1f}")
        print(f"  SustainMs > {verified_best['SustainMs']:.1f}")
        print(f"  BodyE > {verified_best['BodyE']:.1f}")
        print()
        print(f"  Safety Margin: {verified_best['margin']:.1f}%")
        print()
        print("Add these values to your midiconfig.yaml:")
        print()
        print("  instruments:")
        print("    hihat:")
        print("      classification:")
        print("        rules:")
        print("          custom:")
        print(f"            - feature: GeoMean")
        print(f"              threshold: {verified_best['GeoMean']:.1f}")
        print(f"            - feature: SustainMs")
        print(f"              threshold: {verified_best['SustainMs']:.1f}")
        print(f"            - feature: BodyE")
        print(f"              threshold: {verified_best['BodyE']:.1f}")
    else:
        print("⚠️  Could not find optimal configuration.")
        print("    Try adjusting search ranges or collecting more training data.")
    
    print()
    print("Done! See README.md for integration with main system.")
    print()


if __name__ == '__main__':
    main()
