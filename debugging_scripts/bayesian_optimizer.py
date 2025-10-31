#!/usr/bin/env python3
"""
Bayesian Optimization for Hi-Hat Classification Thresholds

Uses Gaussian Process-based Bayesian optimization to intelligently explore
the threshold parameter space. More efficient than grid search - learns from
each evaluation to guide the next search location.

Install: pip install scikit-optimize

Features:
- Adaptive exploration vs exploitation trade-off
- Handles continuous parameter spaces efficiently
- Visualizes optimization landscape
- Provides uncertainty estimates
- Converges faster than random/grid search

Usage:
    python bayesian_optimizer.py
    python bayesian_optimizer.py --n-calls 200  # More thorough search
    python bayesian_optimizer.py --visualize     # Show optimization plots
"""

import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, precision_score
import argparse
import sys

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    from skopt.plots import plot_convergence, plot_objective, plot_evaluations
    import matplotlib.pyplot as plt
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("ERROR: scikit-optimize not installed")
    print("Install with: pip install scikit-optimize")
    sys.exit(1)


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


def calculate_margin_score(df, thresholds, feature_names):
    """
    Calculate safety margin for a threshold combination.
    Higher margin = more robust to variations.
    """
    open_hits = df[df['is_open'] == True]
    
    if len(open_hits) == 0:
        return 0.0
    
    margins = []
    for i, (threshold, feature) in enumerate(zip(thresholds, feature_names)):
        if threshold is None:
            continue
        min_open_value = open_hits[feature].min()
        if min_open_value > 0:
            margin = ((min_open_value - threshold) / min_open_value) * 100
            margins.append(margin)
    
    return np.mean(margins) if margins else 0.0


def evaluate_rule(df, thresholds, feature_names):
    """
    Evaluate a classification rule.
    
    Returns:
        score: Combined metric to maximize (recall + precision + margin)
        recall: Fraction of open hi-hats detected
        precision: Fraction of detections that are correct
        margin: Safety buffer percentage
    """
    # Build mask for classification
    mask = pd.Series([True] * len(df), index=df.index)
    
    for threshold, feature in zip(thresholds, feature_names):
        if threshold is not None and threshold > 0:
            mask &= (df[feature] > threshold)
    
    y_true = df['is_open']
    y_pred = mask
    
    # Calculate metrics
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    
    # Only calculate margin if we have perfect accuracy
    if recall == 1.0 and precision == 1.0:
        margin = calculate_margin_score(df, thresholds, feature_names)
    else:
        margin = 0.0
    
    # Combined score: prioritize accuracy, then margin
    # Perfect accuracy gets bonus, then margin matters
    if recall == 1.0 and precision == 1.0:
        score = 100.0 + margin  # 100 bonus + margin
    else:
        # Penalize imperfect accuracy heavily
        score = (recall * 50) + (precision * 50) - 50  # Max 50 if one is perfect
    
    return score, recall, precision, margin


class BayesianThresholdOptimizer:
    """Bayesian optimization for threshold discovery."""
    
    def __init__(self, df, feature_names, feature_ranges):
        """
        Initialize optimizer.
        
        Args:
            df: Training data
            feature_names: List of features to optimize (e.g., ['GeoMean', 'SustainMs'])
            feature_ranges: Dict mapping feature to (min, max) range
        """
        self.df = df
        self.feature_names = feature_names
        self.feature_ranges = feature_ranges
        self.results = []
        
        # Define search space
        self.space = []
        for feature in feature_names:
            min_val, max_val = feature_ranges[feature]
            self.space.append(Real(min_val, max_val, name=feature))
    
    def objective(self, params):
        """
        Objective function to MINIMIZE.
        
        Bayesian optimization minimizes, so we negate our score
        (which we want to maximize).
        """
        thresholds = params
        score, recall, precision, margin = evaluate_rule(
            self.df, thresholds, self.feature_names
        )
        
        # Store results for analysis
        result = {
            'thresholds': dict(zip(self.feature_names, thresholds)),
            'score': score,
            'recall': recall,
            'precision': precision,
            'margin': margin
        }
        self.results.append(result)
        
        # Return negative score (to minimize)
        return -score
    
    def optimize(self, n_calls=100, n_random_starts=20, verbose=True):
        """
        Run Bayesian optimization.
        
        Args:
            n_calls: Total number of evaluations
            n_random_starts: Number of random samples before using GP
            verbose: Print progress
        """
        if verbose:
            print(f"Optimizing {len(self.feature_names)}-feature rule...")
            print(f"Features: {', '.join(self.feature_names)}")
            print(f"Running {n_calls} evaluations ({n_random_starts} random starts)")
            print()
        
        # Run optimization
        result = gp_minimize(
            self.objective,
            self.space,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            random_state=42,
            verbose=False
        )
        
        self.gp_result = result
        
        # Find best result from our tracking
        best_idx = np.argmax([r['score'] for r in self.results])
        best = self.results[best_idx]
        
        if verbose:
            print(f"✓ Optimization complete!")
            print(f"  Best score: {best['score']:.2f}")
            print(f"  Recall: {best['recall']:.2f}")
            print(f"  Precision: {best['precision']:.2f}")
            print(f"  Margin: {best['margin']:.2f}%")
            print()
            print("  Best thresholds:")
            for feat, val in best['thresholds'].items():
                print(f"    {feat:12s} > {val:.1f}")
        
        return best, result
    
    def plot_convergence(self, save_path=None):
        """Plot optimization convergence."""
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_convergence(self.gp_result, ax=ax)
        ax.set_title(f"Convergence: {' + '.join(self.feature_names)}")
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved convergence plot to {save_path}")
        else:
            plt.show()
    
    def plot_objective(self, save_path=None):
        """Plot objective function landscape."""
        fig = plot_objective(self.gp_result)
        fig.suptitle(f"Objective Landscape: {' + '.join(self.feature_names)}")
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved objective plot to {save_path}")
        else:
            plt.show()
    
    def plot_evaluations(self, save_path=None):
        """Plot evaluation points."""
        fig = plot_evaluations(self.gp_result)
        fig.suptitle(f"Evaluation Points: {' + '.join(self.feature_names)}")
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved evaluations plot to {save_path}")
        else:
            plt.show()


def run_comprehensive_search(df, n_calls=100, visualize=False):
    """
    Run Bayesian optimization for 1, 2, and 3-feature rules.
    """
    # Determine sensible search ranges from data
    closed_hits = df[df['is_open'] == False]
    open_hits = df[df['is_open'] == True]
    
    feature_ranges = {
        'GeoMean': (200, 700),
        'SustainMs': (60, 200),
        'BodyE': (30, 250),
        'SizzleE': (500, 4000),
        'Total': (500, 4000),
        'Amp': (0.05, 0.35),
    }
    
    all_results = {}
    
    # 1-feature rules
    print("=" * 70)
    print("SINGLE-FEATURE RULES")
    print("=" * 70)
    print()
    
    for feature in ['GeoMean', 'SustainMs', 'BodyE']:
        opt = BayesianThresholdOptimizer(df, [feature], feature_ranges)
        best, gp_result = opt.optimize(n_calls=n_calls, n_random_starts=10)
        all_results[f"1feat_{feature}"] = (best, opt)
        
        if visualize:
            opt.plot_convergence(f"convergence_1feat_{feature}.png")
        print()
    
    # 2-feature rules
    print("=" * 70)
    print("TWO-FEATURE RULES")
    print("=" * 70)
    print()
    
    two_feat_combos = [
        ['GeoMean', 'SustainMs'],
        ['GeoMean', 'BodyE'],
        ['SustainMs', 'BodyE'],
    ]
    
    for features in two_feat_combos:
        opt = BayesianThresholdOptimizer(df, features, feature_ranges)
        best, gp_result = opt.optimize(n_calls=n_calls, n_random_starts=20)
        all_results[f"2feat_{'_'.join(features)}"] = (best, opt)
        
        if visualize:
            opt.plot_convergence(f"convergence_2feat_{'_'.join(features)}.png")
            opt.plot_objective(f"objective_2feat_{'_'.join(features)}.png")
        print()
    
    # 3-feature rule
    print("=" * 70)
    print("THREE-FEATURE RULES")
    print("=" * 70)
    print()
    
    three_feat_combos = [
        ['GeoMean', 'SustainMs', 'BodyE'],
        ['GeoMean', 'SustainMs', 'SizzleE'],
    ]
    
    for features in three_feat_combos:
        opt = BayesianThresholdOptimizer(df, features, feature_ranges)
        best, gp_result = opt.optimize(n_calls=n_calls, n_random_starts=30)
        all_results[f"3feat_{'_'.join(features)}"] = (best, opt)
        
        if visualize:
            opt.plot_convergence(f"convergence_3feat_{'_'.join(features)}.png")
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY: TOP RULES BY SCORE")
    print("=" * 70)
    print()
    
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1][0]['score'],
        reverse=True
    )
    
    print(f"{'Rank':<6} {'Features':<40} {'Score':>8} {'Margin':>8} {'Recall':>8} {'Prec':>8}")
    print("-" * 80)
    
    for rank, (name, (best, opt)) in enumerate(sorted_results[:15], 1):
        feature_str = ' + '.join(opt.feature_names)
        print(f"{rank:<6} {feature_str:<40} {best['score']:>8.2f} {best['margin']:>7.1f}% {best['recall']:>8.2f} {best['precision']:>8.2f}")
    
    print()
    
    # Best overall
    best_name, (best_overall, best_opt) = sorted_results[0]
    print("🏆 BEST RULE FOUND:")
    print()
    print(f"  Rule: {' AND '.join([f'{k} > {v:.1f}' for k, v in best_overall['thresholds'].items()])}")
    print(f"  Score: {best_overall['score']:.2f}")
    print(f"  Margin: {best_overall['margin']:.1f}%")
    print(f"  Recall: {best_overall['recall']:.2f}")
    print(f"  Precision: {best_overall['precision']:.2f}")
    print()
    
    # Export to CSV
    export_data = []
    for name, (best, opt) in sorted_results:
        row = {
            'rule': ' AND '.join([f'{k} > {v:.1f}' for k, v in best['thresholds'].items()]),
            'n_features': len(opt.feature_names),
            'score': best['score'],
            'margin': best['margin'],
            'recall': best['recall'],
            'precision': best['precision'],
        }
        for feat, val in best['thresholds'].items():
            row[feat] = val
        export_data.append(row)
    
    results_df = pd.DataFrame(export_data)
    results_df.to_csv('bayesian_optimal_rules.csv', index=False)
    print("✓ Results exported to bayesian_optimal_rules.csv")
    print()
    
    return all_results, best_overall


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian optimization for hi-hat classification thresholds"
    )
    parser.add_argument(
        '--n-calls',
        type=int,
        default=100,
        help="Number of optimization iterations (default: 100)"
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help="Generate convergence and objective plots"
    )
    parser.add_argument(
        '--csv',
        default='data.csv',
        help="Path to training data CSV (default: data.csv)"
    )
    
    args = parser.parse_args()
    
    # Load data
    print("Loading training data...")
    df = load_data(args.csv)
    print(f"  Total samples: {len(df)}")
    print(f"  Open hi-hats: {df['is_open'].sum()}")
    print(f"  Closed hi-hats: {(~df['is_open']).sum()}")
    print()
    
    # Run optimization
    all_results, best = run_comprehensive_search(
        df,
        n_calls=args.n_calls,
        visualize=args.visualize
    )
    
    print("Done!")


if __name__ == '__main__':
    main()
