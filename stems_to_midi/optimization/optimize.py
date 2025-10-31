#!/usr/bin/env python3
"""
Bayesian Optimization for Threshold Learning

Learns optimal thresholds from labeled ground truth data using Bayesian optimization.
Maximizes classification accuracy and safety margin.

Usage:
    python -m stems_to_midi.optimization.optimize 4 --stem hihat
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_labeled_data(csv_path: Path) -> pd.DataFrame:
    """Load labeled CSV with ground truth."""
    df = pd.read_csv(csv_path)
    
    # Get labeled samples
    labeled_open = df[df['actual_open'] == 'x'].copy()
    labeled_closed = df[df['actual_closed'] == 'x'].copy()
    
    # Get detected samples (these passed the filter - we assume they're correctly kept)
    detected = df[(df['detected_closed'] == 'x') | (df['detected_open'] == 'x')].copy()
    
    # Get rejected samples (these were filtered out - we want to keep most rejected)
    rejected = df[(df['detected_closed'] != 'x') & (df['detected_open'] != 'x')].copy()
    
    # Build training set:
    # 1. Labeled open (8) - should be detected as open
    # 2. Labeled closed (4) - should be detected as closed  
    # 3. Sample of rejected (to learn what to filter) - should stay rejected
    
    labeled = pd.concat([labeled_open, labeled_closed], ignore_index=True)
    labeled['is_open'] = (labeled['actual_open'] == 'x').astype(int)
    labeled['is_closed'] = (labeled['actual_closed'] == 'x').astype(int)
    labeled['should_detect'] = 1  # All labeled samples should be detected
    
    # Add a sample of rejected onsets as negative examples
    # These should NOT be detected (stay filtered out)
    rejected_sample = rejected.sample(min(50, len(rejected)), random_state=42).copy()
    rejected_sample['is_open'] = 0
    rejected_sample['is_closed'] = 0
    rejected_sample['should_detect'] = 0  # These should stay filtered out
    
    # Combine
    training_df = pd.concat([labeled, rejected_sample], ignore_index=True)
    
    return training_df


def evaluate_thresholds(
    df: pd.DataFrame,
    geomean_threshold: float,
    open_geomean_min: float,
    open_sustain_min: float
) -> Dict:
    """
    Evaluate classification performance with given thresholds.
    
    Includes both detection accuracy (keep vs reject) and classification (open vs closed).
    
    Returns dict with:
        - detection_accuracy: Correctly detected + correctly rejected
        - classification_accuracy: Of detected, correctly classified as open/closed
        - overall_accuracy: Combined score
    """
    # Step 1: Apply detection filter (GeoMean >= threshold)
    df['passes_filter'] = (df['GeoMean'] >= geomean_threshold).astype(int)
    
    # Step 2: Evaluate detection (should_detect vs passes_filter)
    detection_correct = (df['should_detect'] == df['passes_filter']).sum()
    detection_accuracy = detection_correct / len(df)
    
    # Get samples that passed filter
    detected = df[df['passes_filter'] == 1].copy()
    
    if len(detected) == 0:
        return {
            'detection_accuracy': detection_accuracy,
            'classification_accuracy': 0.0,
            'overall_accuracy': detection_accuracy * 0.5,  # 50% weight to detection
            'false_positives': 0,
            'false_negatives': (df['should_detect'] == 1).sum(),
            'correct': detection_correct,
            'total': len(df)
        }
    
    # Step 3: Classify detected samples as open or closed
    detected['predicted_open'] = (
        (detected['GeoMean'] >= open_geomean_min) & 
        (detected['SustainMs'] >= open_sustain_min)
    ).astype(int)
    detected['predicted_closed'] = 1 - detected['predicted_open']
    
    # Step 4: Calculate classification accuracy on detected samples
    # Only evaluate classification on samples that should be detected
    detected_positives = detected[detected['should_detect'] == 1]
    
    if len(detected_positives) > 0:
        correct_open = ((detected_positives['is_open'] == 1) & (detected_positives['predicted_open'] == 1)).sum()
        correct_closed = ((detected_positives['is_closed'] == 1) & (detected_positives['predicted_closed'] == 1)).sum()
        classification_correct = correct_open + correct_closed
        classification_accuracy = classification_correct / len(detected_positives)
    else:
        classification_accuracy = 0.0
    
    # Step 5: Count false positives (detected but should be rejected)
    false_positives = ((df['passes_filter'] == 1) & (df['should_detect'] == 0)).sum()
    false_negatives = ((df['passes_filter'] == 0) & (df['should_detect'] == 1)).sum()
    
    # Combined score: both detection and classification matter
    overall_accuracy = 0.7 * detection_accuracy + 0.3 * classification_accuracy
    
    return {
        'detection_accuracy': detection_accuracy,
        'classification_accuracy': classification_accuracy,
        'overall_accuracy': overall_accuracy,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'correct': detection_correct,
        'total': len(df)
    }


def objective_function(params: Tuple[float, float, float], df: pd.DataFrame) -> float:
    """
    Objective function for Bayesian optimization.
    
    Maximize overall accuracy while penalizing false positives.
    """
    geomean_threshold, open_geomean_min, open_sustain_min = params
    
    result = evaluate_thresholds(df, geomean_threshold, open_geomean_min, open_sustain_min)
    
    # Penalize false positives heavily (we want to avoid detecting noise)
    fp_penalty = result['false_positives'] / len(df) if len(df) > 0 else 0
    
    # Penalize false negatives (missing real hihats)
    fn_penalty = result['false_negatives'] / len(df) if len(df) > 0 else 0
    
    # Score: maximize accuracy, heavily penalize false positives
    score = result['overall_accuracy'] - (2.0 * fp_penalty) - (1.0 * fn_penalty)
    
    return max(0, score)  # Keep non-negative


def bayesian_optimize(df: pd.DataFrame, n_calls: int = 50) -> Dict:
    """
    Run Bayesian optimization to find optimal thresholds.
    
    Returns best parameters and score.
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError:
        print("Error: scikit-optimize not installed")
        print("Install with: pip install scikit-optimize")
        sys.exit(1)
    
    # Define search space
    # Note: geomean_threshold needs to catch the 4 labeled closed (GeoMean 4-20)
    # but reject most of the 289 rejected onsets (median GeoMean ~8)
    space = [
        Real(3.0, 25.0, name='geomean_threshold'),       # Detection threshold
        Real(200.0, 700.0, name='open_geomean_min'),     # Open classification - GeoMean
        Real(80.0, 180.0, name='open_sustain_min'),      # Open classification - Sustain
    ]
    
    # Objective wrapper for skopt (minimization)
    def objective_wrapper(params):
        score = objective_function(tuple(params), df)
        return -score  # Negative because skopt minimizes
    
    print("\nRunning Bayesian optimization...")
    print(f"  Search space:")
    print(f"    geomean_threshold: 1.0 - 30.0")
    print(f"    open_geomean_min: 200.0 - 700.0")
    print(f"    open_sustain_min: 100.0 - 180.0")
    print(f"  Number of evaluations: {n_calls}")
    print()
    
    # Run optimization
    result = gp_minimize(
        objective_wrapper,
        space,
        n_calls=n_calls,
        n_random_starts=10,
        random_state=42,
        verbose=True
    )
    
    # Extract best parameters
    best_geomean_threshold, best_open_geomean_min, best_open_sustain_min = result.x
    best_score = -result.fun
    
    # Evaluate with best parameters
    final_result = evaluate_thresholds(
        df, best_geomean_threshold, best_open_geomean_min, best_open_sustain_min
    )
    
    return {
        'geomean_threshold': best_geomean_threshold,
        'open_geomean_min': best_open_geomean_min,
        'open_sustain_min': best_open_sustain_min,
        'score': best_score,
        **final_result
    }


def main():
    parser = argparse.ArgumentParser(
        description="Optimize thresholds from labeled ground truth"
    )
    parser.add_argument(
        'project_number',
        type=int,
        help="Project number (e.g., 4)"
    )
    parser.add_argument(
        '--stem',
        required=True,
        help="Stem type (hihat)"
    )
    parser.add_argument(
        '--n-calls',
        type=int,
        default=50,
        help="Number of optimization iterations"
    )
    
    args = parser.parse_args()
    
    # Load labeled data
    csv_path = Path(f"user_files/{args.project_number} - */optimization/{args.stem}_features_actual.csv")
    csv_files = list(Path(".").glob(str(csv_path)))
    
    if not csv_files:
        print(f"Error: Could not find {csv_path}")
        sys.exit(1)
    
    csv_path = csv_files[0]
    print(f"Loading labeled data from: {csv_path}")
    
    df = load_labeled_data(csv_path)
    
    print(f"\nLabeled data:")
    print(f"  Total labeled hits: {len(df)}")
    print(f"  Open hits: {df['is_open'].sum()}")
    print(f"  Closed hits: {df['is_closed'].sum()}")
    print()
    
    # Run optimization
    best_params = bayesian_optimize(df, n_calls=args.n_calls)
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"\nBest parameters:")
    print(f"  geomean_threshold: {best_params['geomean_threshold']:.2f}")
    print(f"  open_geomean_min: {best_params['open_geomean_min']:.2f}")
    print(f"  open_sustain_min: {best_params['open_sustain_min']:.2f}")
    print(f"\nPerformance:")
    print(f"  Detection accuracy: {best_params['detection_accuracy']*100:.1f}%")
    print(f"  Classification accuracy: {best_params['classification_accuracy']*100:.1f}%")
    print(f"  Overall accuracy: {best_params['overall_accuracy']*100:.1f}%")
    print(f"  False positives: {best_params['false_positives']}")
    print(f"  False negatives: {best_params['false_negatives']}")
    print(f"  Correct: {best_params['correct']}/{best_params['total']}")
    
    # Save results
    output_dir = csv_path.parent
    output_file = output_dir / f"{args.stem}_optimal_thresholds.txt"
    
    with open(output_file, 'w') as f:
        f.write(f"# Optimal Thresholds for {args.stem}\n")
        f.write(f"# Learned from {len(df)} labeled examples\n\n")
        f.write(f"geomean_threshold: {best_params['geomean_threshold']:.2f}\n")
        f.write(f"open_geomean_min: {best_params['open_geomean_min']:.2f}\n")
        f.write(f"open_sustain_min: {best_params['open_sustain_min']:.2f}\n")
        f.write(f"\n# Performance\n")
        f.write(f"detection_accuracy: {best_params['detection_accuracy']*100:.1f}%\n")
        f.write(f"classification_accuracy: {best_params['classification_accuracy']*100:.1f}%\n")
        f.write(f"overall_accuracy: {best_params['overall_accuracy']*100:.1f}%\n")
        f.write(f"false_positives: {best_params['false_positives']}\n")
        f.write(f"false_negatives: {best_params['false_negatives']}\n")
    
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == '__main__':
    main()
