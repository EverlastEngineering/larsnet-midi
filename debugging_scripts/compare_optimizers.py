#!/usr/bin/env python3
"""
Compare Different Optimization Approaches

Runs all three optimizers (grid search, random search, Bayesian optimization)
and compares their results to help you choose the best approach.

Usage:
    python compare_optimizers.py
    python compare_optimizers.py --quick  # Faster but less thorough
"""

import pandas as pd
import numpy as np
import argparse
import time
import subprocess
import sys
from pathlib import Path


def run_grid_search():
    """Run the grid search optimizer."""
    print("=" * 80)
    print("RUNNING GRID SEARCH")
    print("=" * 80)
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, 'threshold_optimizer.py'],
            capture_output=True,
            text=True,
            timeout=120
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(result.stdout)
            return {
                'method': 'Grid Search',
                'success': True,
                'time': elapsed,
                'output_file': 'optimal_rules.csv'
            }
        else:
            print(f"Error: {result.stderr}")
            return {'method': 'Grid Search', 'success': False, 'time': elapsed}
    
    except subprocess.TimeoutExpired:
        print("Grid search timed out")
        return {'method': 'Grid Search', 'success': False, 'time': 120}
    except Exception as e:
        print(f"Error running grid search: {e}")
        return {'method': 'Grid Search', 'success': False, 'time': 0}


def run_random_search(n_samples=10000):
    """Run the random search optimizer."""
    print("=" * 80)
    print("RUNNING RANDOM SEARCH")
    print("=" * 80)
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, 'random_search_optimizer.py', '--n-samples', str(n_samples)],
            capture_output=True,
            text=True,
            timeout=300
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(result.stdout)
            return {
                'method': 'Random Search',
                'success': True,
                'time': elapsed,
                'output_file': 'random_search_results.csv',
                'n_samples': n_samples
            }
        else:
            print(f"Error: {result.stderr}")
            return {'method': 'Random Search', 'success': False, 'time': elapsed}
    
    except subprocess.TimeoutExpired:
        print("Random search timed out")
        return {'method': 'Random Search', 'success': False, 'time': 300}
    except Exception as e:
        print(f"Error running random search: {e}")
        return {'method': 'Random Search', 'success': False, 'time': 0}


def run_bayesian_optimization(n_calls=100):
    """Run the Bayesian optimizer."""
    print("=" * 80)
    print("RUNNING BAYESIAN OPTIMIZATION")
    print("=" * 80)
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, 'bayesian_optimizer.py', '--n-calls', str(n_calls)],
            capture_output=True,
            text=True,
            timeout=300
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(result.stdout)
            return {
                'method': 'Bayesian Optimization',
                'success': True,
                'time': elapsed,
                'output_file': 'bayesian_optimal_rules.csv',
                'n_calls': n_calls
            }
        else:
            print(f"Error: {result.stderr}")
            # Check if it's just missing scikit-optimize
            if 'scikit-optimize' in result.stderr:
                print()
                print("💡 Install scikit-optimize to use Bayesian optimization:")
                print("   pip install scikit-optimize")
                print()
            return {'method': 'Bayesian Optimization', 'success': False, 'time': elapsed}
    
    except subprocess.TimeoutExpired:
        print("Bayesian optimization timed out")
        return {'method': 'Bayesian Optimization', 'success': False, 'time': 300}
    except Exception as e:
        print(f"Error running Bayesian optimization: {e}")
        return {'method': 'Bayesian Optimization', 'success': False, 'time': 0}


def load_results(output_file):
    """Load and analyze results from an optimizer."""
    try:
        df = pd.read_csv(output_file)
        
        # Find best rule
        if 'margin' in df.columns:
            best_idx = df['margin'].idxmax()
            best = df.iloc[best_idx]
        elif 'margin_score' in df.columns:
            best_idx = df['margin_score'].idxmax()
            best = df.iloc[best_idx]
        else:
            return None
        
        return {
            'best_rule': best.get('rule', 'N/A'),
            'best_margin': best.get('margin', best.get('margin_score', 0)),
            'n_perfect_rules': len(df[df.get('recall', df.get('tp', 0) / 6) == 1.0]),
            'top_10_margin_avg': df.head(10)['margin' if 'margin' in df.columns else 'margin_score'].mean()
        }
    
    except Exception as e:
        print(f"Error loading {output_file}: {e}")
        return None


def compare_results(results):
    """Generate comparison report."""
    print()
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print()
    
    # Performance comparison
    print("⏱️  PERFORMANCE")
    print("-" * 80)
    print(f"{'Method':<30} {'Time (s)':>12} {'Status':<20}")
    print("-" * 80)
    
    for r in results:
        if r['success']:
            status = "✓ Success"
        else:
            status = "✗ Failed"
        print(f"{r['method']:<30} {r['time']:>12.2f} {status:<20}")
    
    print()
    
    # Quality comparison
    print("🎯 RESULT QUALITY")
    print("-" * 80)
    print(f"{'Method':<30} {'Best Margin':>12} {'Perfect Rules':>15} {'Top 10 Avg':>15}")
    print("-" * 80)
    
    for r in results:
        if r['success'] and 'output_file' in r:
            analysis = load_results(r['output_file'])
            if analysis:
                print(f"{r['method']:<30} {analysis['best_margin']:>11.1f}% "
                      f"{analysis['n_perfect_rules']:>15} {analysis['top_10_margin_avg']:>14.1f}%")
                
                # Store for final recommendation
                r['analysis'] = analysis
    
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS")
    print("-" * 80)
    print()
    
    successful = [r for r in results if r['success']]
    
    if not successful:
        print("⚠️  No optimizers completed successfully")
        return
    
    # Find best by margin
    best_margin = max(
        (r for r in successful if 'analysis' in r),
        key=lambda r: r['analysis']['best_margin'],
        default=None
    )
    
    # Find fastest
    fastest = min(successful, key=lambda r: r['time'])
    
    # Find most thorough (most perfect rules found)
    most_thorough = max(
        (r for r in successful if 'analysis' in r),
        key=lambda r: r['analysis']['n_perfect_rules'],
        default=None
    )
    
    if best_margin:
        print(f"🏆 Best margin: {best_margin['method']}")
        print(f"   {best_margin['analysis']['best_margin']:.1f}% safety margin")
        print(f"   Rule: {best_margin['analysis']['best_rule']}")
        print()
    
    if fastest:
        print(f"⚡ Fastest: {fastest['method']}")
        print(f"   Completed in {fastest['time']:.1f} seconds")
        print()
    
    if most_thorough:
        print(f"🔍 Most thorough: {most_thorough['method']}")
        print(f"   Found {most_thorough['analysis']['n_perfect_rules']} perfect rules")
        print()
    
    print("WHEN TO USE EACH METHOD:")
    print()
    print("📊 Grid Search")
    print("   ✓ Exhaustive - guarantees finding all solutions in search space")
    print("   ✓ Deterministic - same results every time")
    print("   ✗ Slow for high-dimensional problems")
    print("   → Best for: Final verification, small search spaces")
    print()
    
    print("🎲 Random Search")
    print("   ✓ Fast - can sample millions of points quickly")
    print("   ✓ Good for high dimensions")
    print("   ✓ Can find unexpected regions")
    print("   ✗ May miss optimal solutions")
    print("   → Best for: Initial exploration, many features")
    print()
    
    print("🧠 Bayesian Optimization")
    print("   ✓ Intelligent - learns from previous evaluations")
    print("   ✓ Sample efficient - fewer evaluations needed")
    print("   ✓ Provides uncertainty estimates")
    print("   ✗ Requires scikit-optimize library")
    print("   → Best for: Expensive evaluation functions, continuous spaces")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare different optimization approaches"
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help="Run faster with fewer samples (less thorough)"
    )
    parser.add_argument(
        '--skip-grid',
        action='store_true',
        help="Skip grid search (if it's too slow)"
    )
    parser.add_argument(
        '--skip-random',
        action='store_true',
        help="Skip random search"
    )
    parser.add_argument(
        '--skip-bayesian',
        action='store_true',
        help="Skip Bayesian optimization"
    )
    
    args = parser.parse_args()
    
    # Determine parameters based on mode
    if args.quick:
        n_samples = 1000
        n_calls = 50
        print("🚀 Quick mode: Running with reduced samples")
    else:
        n_samples = 10000
        n_calls = 100
        print("🔬 Full mode: Running comprehensive optimization")
    
    print()
    
    results = []
    
    # Run optimizers
    if not args.skip_grid:
        results.append(run_grid_search())
    
    if not args.skip_random:
        results.append(run_random_search(n_samples=n_samples))
    
    if not args.skip_bayesian:
        results.append(run_bayesian_optimization(n_calls=n_calls))
    
    # Compare
    if results:
        compare_results(results)
    else:
        print("No optimizers were run")
    
    print()
    print("Done!")


if __name__ == '__main__':
    main()
