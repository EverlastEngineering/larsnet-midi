#!/usr/bin/env python3
"""
Random Search with Multi-Dimensional Feature Space Exploration

Complements grid search and Bayesian optimization with:
- Pure random sampling across feature space
- Multi-dimensional visualization
- Feature interaction analysis
- Statistical significance testing

Usage:
    python random_search_optimizer.py
    python random_search_optimizer.py --n-samples 10000
    python random_search_optimizer.py --analyze-interactions
"""

import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, precision_score
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from itertools import combinations


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
    """Calculate safety margin for a threshold combination."""
    open_hits = df[df['is_open'] == True]
    
    if len(open_hits) == 0:
        return 0.0
    
    margins = []
    for threshold, feature in zip(thresholds, feature_names):
        if threshold is not None:
            min_open_value = open_hits[feature].min()
            if min_open_value > 0:
                margin = ((min_open_value - threshold) / min_open_value) * 100
                margins.append(margin)
    
    return np.mean(margins) if margins else 0.0


def evaluate_rule(df, thresholds, feature_names):
    """Evaluate a classification rule."""
    mask = pd.Series([True] * len(df), index=df.index)
    
    for threshold, feature in zip(thresholds, feature_names):
        if threshold is not None:
            mask &= (df[feature] > threshold)
    
    y_true = df['is_open']
    y_pred = mask
    
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    
    if recall == 1.0 and precision == 1.0:
        margin = calculate_margin_score(df, thresholds, feature_names)
        score = 100.0 + margin
    else:
        margin = 0.0
        score = (recall * 50) + (precision * 50) - 50
    
    return score, recall, precision, margin


class RandomSearchOptimizer:
    """Random search with comprehensive feature space exploration."""
    
    def __init__(self, df, feature_names, feature_ranges):
        """
        Initialize optimizer.
        
        Args:
            df: Training data
            feature_names: Features to optimize
            feature_ranges: Dict mapping feature to (min, max)
        """
        self.df = df
        self.feature_names = feature_names
        self.feature_ranges = feature_ranges
        self.samples = []
        self.perfect_rules = []
    
    def random_search(self, n_samples=10000, seed=42):
        """
        Pure random search across parameter space.
        
        More efficient than grid search for high-dimensional spaces.
        """
        np.random.seed(seed)
        
        print(f"Running random search with {n_samples:,} samples...")
        print(f"Features: {', '.join(self.feature_names)}")
        print()
        
        for i in range(n_samples):
            # Random thresholds
            thresholds = []
            for feature in self.feature_names:
                min_val, max_val = self.feature_ranges[feature]
                threshold = np.random.uniform(min_val, max_val)
                thresholds.append(threshold)
            
            # Evaluate
            score, recall, precision, margin = evaluate_rule(
                self.df, thresholds, self.feature_names
            )
            
            sample = {
                'sample_id': i,
                'thresholds': dict(zip(self.feature_names, thresholds)),
                'score': score,
                'recall': recall,
                'precision': precision,
                'margin': margin,
                'is_perfect': recall == 1.0 and precision == 1.0
            }
            
            self.samples.append(sample)
            
            if sample['is_perfect']:
                self.perfect_rules.append(sample)
            
            # Progress
            if (i + 1) % 1000 == 0:
                perfect_count = len(self.perfect_rules)
                print(f"  {i+1:,}/{n_samples:,} samples | {perfect_count} perfect rules found", end='\r')
        
        print()
        print(f"✓ Search complete!")
        print(f"  Perfect rules found: {len(self.perfect_rules)}")
        print()
        
        return self.perfect_rules
    
    def analyze_feature_distribution(self):
        """Analyze distribution of successful threshold values."""
        if not self.perfect_rules:
            print("No perfect rules found - cannot analyze distribution")
            return
        
        print("=" * 70)
        print("FEATURE DISTRIBUTION ANALYSIS")
        print("=" * 70)
        print()
        
        for feature in self.feature_names:
            values = [r['thresholds'][feature] for r in self.perfect_rules]
            
            print(f"{feature}:")
            print(f"  Min:    {np.min(values):.2f}")
            print(f"  Q1:     {np.percentile(values, 25):.2f}")
            print(f"  Median: {np.median(values):.2f}")
            print(f"  Q3:     {np.percentile(values, 75):.2f}")
            print(f"  Max:    {np.max(values):.2f}")
            print(f"  Mean:   {np.mean(values):.2f} ± {np.std(values):.2f}")
            print()
    
    def find_robust_region(self):
        """
        Find the most robust region in parameter space.
        
        Robust = many nearby points also achieve perfect accuracy.
        """
        if len(self.perfect_rules) < 10:
            print("Not enough perfect rules for robust region analysis")
            return None
        
        print("=" * 70)
        print("ROBUST REGION ANALYSIS")
        print("=" * 70)
        print()
        
        # Convert to array for distance calculations
        threshold_array = np.array([
            [r['thresholds'][f] for f in self.feature_names]
            for r in self.perfect_rules
        ])
        
        # Normalize to [0, 1] for fair distance calculation
        normalized = threshold_array.copy()
        for i, feature in enumerate(self.feature_names):
            min_val, max_val = self.feature_ranges[feature]
            normalized[:, i] = (threshold_array[:, i] - min_val) / (max_val - min_val)
        
        # Find point with most neighbors
        k = min(20, len(self.perfect_rules) // 2)
        best_density = 0
        best_idx = 0
        
        for i in range(len(normalized)):
            # Calculate distances to all other points
            distances = np.sqrt(np.sum((normalized - normalized[i]) ** 2, axis=1))
            # Count neighbors within radius
            density = np.sum(distances < 0.1)  # 0.1 in normalized space
            
            if density > best_density:
                best_density = density
                best_idx = i
        
        robust_rule = self.perfect_rules[best_idx]
        
        print(f"Most robust rule (has {best_density} nearby perfect rules):")
        print()
        for feat, val in robust_rule['thresholds'].items():
            print(f"  {feat:12s} > {val:.1f}")
        print(f"\n  Margin: {robust_rule['margin']:.1f}%")
        print()
        
        return robust_rule
    
    def plot_2d_space(self, feat1, feat2, save_path=None):
        """Visualize 2D slice of parameter space."""
        if feat1 not in self.feature_names or feat2 not in self.feature_names:
            print(f"Features {feat1} and {feat2} not in search space")
            return
        
        # Prepare data
        all_points = np.array([
            [s['thresholds'][feat1], s['thresholds'][feat2], s['score']]
            for s in self.samples
        ])
        
        perfect_points = np.array([
            [r['thresholds'][feat1], r['thresholds'][feat2], r['margin']]
            for r in self.perfect_rules
        ])
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # All samples colored by score
        scatter1 = ax1.scatter(
            all_points[:, 0],
            all_points[:, 1],
            c=all_points[:, 2],
            s=10,
            alpha=0.5,
            cmap='viridis'
        )
        ax1.set_xlabel(feat1)
        ax1.set_ylabel(feat2)
        ax1.set_title(f'All Samples (colored by score)\n{len(self.samples):,} points')
        plt.colorbar(scatter1, ax=ax1, label='Score')
        
        # Perfect rules colored by margin
        if len(perfect_points) > 0:
            scatter2 = ax2.scatter(
                perfect_points[:, 0],
                perfect_points[:, 1],
                c=perfect_points[:, 2],
                s=50,
                alpha=0.7,
                cmap='RdYlGn'
            )
            ax2.set_xlabel(feat1)
            ax2.set_ylabel(feat2)
            ax2.set_title(f'Perfect Rules (colored by margin)\n{len(perfect_points)} points')
            plt.colorbar(scatter2, ax=ax2, label='Margin %')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved 2D space plot to {save_path}")
        else:
            plt.show()
    
    def plot_3d_space(self, save_path=None):
        """Visualize 3D parameter space."""
        if len(self.feature_names) < 3:
            print("Need at least 3 features for 3D visualization")
            return
        
        feat1, feat2, feat3 = self.feature_names[:3]
        
        perfect_points = np.array([
            [r['thresholds'][feat1], r['thresholds'][feat2], 
             r['thresholds'][feat3], r['margin']]
            for r in self.perfect_rules
        ])
        
        if len(perfect_points) == 0:
            print("No perfect rules to visualize")
            return
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            perfect_points[:, 0],
            perfect_points[:, 1],
            perfect_points[:, 2],
            c=perfect_points[:, 3],
            s=50,
            alpha=0.6,
            cmap='RdYlGn'
        )
        
        ax.set_xlabel(feat1)
        ax.set_ylabel(feat2)
        ax.set_zlabel(feat3)
        ax.set_title(f'Perfect Rules in 3D Space\n{len(perfect_points)} points')
        
        plt.colorbar(scatter, ax=ax, label='Margin %', shrink=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved 3D space plot to {save_path}")
        else:
            plt.show()
    
    def analyze_feature_interactions(self):
        """Analyze how features interact in successful rules."""
        if len(self.perfect_rules) < 10:
            print("Not enough perfect rules for interaction analysis")
            return
        
        print("=" * 70)
        print("FEATURE INTERACTION ANALYSIS")
        print("=" * 70)
        print()
        
        # Calculate correlations between threshold values
        threshold_matrix = np.array([
            [r['thresholds'][f] for f in self.feature_names]
            for r in self.perfect_rules
        ])
        
        corr_matrix = np.corrcoef(threshold_matrix.T)
        
        print("Threshold correlations:")
        print("(How threshold values co-vary in successful rules)")
        print()
        
        for i, feat1 in enumerate(self.feature_names):
            for j, feat2 in enumerate(self.feature_names):
                if i < j:
                    corr = corr_matrix[i, j]
                    print(f"  {feat1:12s} vs {feat2:12s}: {corr:+.3f}", end='')
                    if abs(corr) > 0.5:
                        print(" ⚠️  Strong correlation!")
                    else:
                        print()
        
        print()
        
        # Plot correlation heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            xticklabels=self.feature_names,
            yticklabels=self.feature_names,
            ax=ax
        )
        ax.set_title('Feature Threshold Correlations')
        plt.tight_layout()
        plt.savefig('feature_correlations.png', dpi=150, bbox_inches='tight')
        print("✓ Saved correlation heatmap to feature_correlations.png")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Random search optimizer for threshold discovery"
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10000,
        help="Number of random samples (default: 10000)"
    )
    parser.add_argument(
        '--analyze-interactions',
        action='store_true',
        help="Perform feature interaction analysis"
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help="Generate visualization plots"
    )
    parser.add_argument(
        '--csv',
        default='data.csv',
        help="Path to training data CSV"
    )
    
    args = parser.parse_args()
    
    # Load data
    print("Loading training data...")
    df = load_data(args.csv)
    print(f"  Total samples: {len(df)}")
    print(f"  Open hi-hats: {df['is_open'].sum()}")
    print(f"  Closed hi-hats: {(~df['is_open']).sum()}")
    print()
    
    # Define search space
    feature_ranges = {
        'GeoMean': (200, 700),
        'SustainMs': (60, 200),
        'BodyE': (30, 250),
        'SizzleE': (500, 4000),
        'Total': (500, 4000),
    }
    
    # Run 3-feature search (most interesting)
    features = ['GeoMean', 'SustainMs', 'BodyE']
    optimizer = RandomSearchOptimizer(df, features, feature_ranges)
    
    # Random search
    perfect_rules = optimizer.random_search(n_samples=args.n_samples)
    
    if perfect_rules:
        # Analyze distribution
        optimizer.analyze_feature_distribution()
        
        # Find robust region
        robust_rule = optimizer.find_robust_region()
        
        # Feature interactions
        if args.analyze_interactions:
            optimizer.analyze_feature_interactions()
        
        # Visualizations
        if args.visualize:
            print("Generating visualizations...")
            optimizer.plot_2d_space('GeoMean', 'SustainMs', 'random_2d_geomean_sustain.png')
            optimizer.plot_2d_space('GeoMean', 'BodyE', 'random_2d_geomean_bodye.png')
            optimizer.plot_3d_space('random_3d_space.png')
            print()
        
        # Export best results
        top_rules = sorted(perfect_rules, key=lambda r: r['margin'], reverse=True)[:30]
        
        export_data = []
        for r in top_rules:
            row = {
                'rule': ' AND '.join([f'{k} > {v:.1f}' for k, v in r['thresholds'].items()]),
                'margin': r['margin'],
            }
            for feat, val in r['thresholds'].items():
                row[feat] = val
            export_data.append(row)
        
        results_df = pd.DataFrame(export_data)
        results_df.to_csv('random_search_results.csv', index=False)
        print("✓ Results exported to random_search_results.csv")
    else:
        print("⚠️  No perfect rules found. Try:")
        print("   - Increasing --n-samples")
        print("   - Adjusting feature_ranges")
        print("   - Checking if data is correctly labeled")
    
    print()
    print("Done!")


if __name__ == '__main__':
    main()
