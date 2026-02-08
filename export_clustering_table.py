#!/usr/bin/env python3
"""
Export clustering data to CSV for analysis in spreadsheet.
"""

import librosa
import numpy as np
import csv
from stems_to_midi.optimization_core import optimize_threshold_by_clustering


def export_clustering_csv(
    audio_path: str,
    output_csv: str,
    threshold: float = 5.0,
    threshold_min: float = 3.0,
    threshold_max: float = 15.0,
    dbscan_eps: float = 1.2,  # Increased for 11-dimensional feature space (was 0.5)
    dbscan_min_samples: int = 5,
):
    """Export all onset features and cluster labels to CSV."""
    
    print(f"Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    
    # Ensure stereo
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.shape[0] != 2:
        audio = audio[:2]
    
    print(f"Running optimization...")
    result = optimize_threshold_by_clustering(
        audio,
        sr=sr,
        expected_clusters=2,
        initial_threshold=threshold,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        clustering_method='dbscan',
        clustering_kwargs={'eps': dbscan_eps, 'min_samples': dbscan_min_samples},
        merge_window_ms=100,
        max_iterations=20,
        tolerance=0,
        convergence_patience=5,
        feature_names=None,  # All features
        min_silhouette_score=0.3,
        max_noise_ratio=0.8,
        min_onset_interval_ms=50.0,
        use_quality_optimization=True,
    )
    
    features = result['final_features']
    labels = result['final_cluster_result']['labels']
    
    print(f"  Optimized threshold: {result['optimized_threshold']}")
    print(f"  Converged: {result['converged']} ({result['convergence_reason']})")
    print(f"  Iterations: {result['iterations']}")
    
    print(f"Writing CSV: {output_csv}")
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Index',
            'Time(s)',
            'Cluster',
            'Pan',
            'SpectralCentroid(Hz)',
            'SpectralRolloff(Hz)',
            'SpectralFlatness',
            'Pitch(Hz)',
            'TimeDelta(s)',
            'BodyEnergy',
            'BrillianceEnergy',
            'GeoMean',
            'TotalEnergy',
            'SustainMs',
        ])
        
        # Data rows
        for idx, (feature, label) in enumerate(zip(features, labels)):
            cluster_str = 'NOISE' if label == -1 else f'C{label}'
            
            def fmt(val, default=0):
                return default if val is None else val
            
            writer.writerow([
                idx + 1,
                fmt(feature.get('time'), 0),
                cluster_str,
                fmt(feature.get('pan_confidence'), 0),
                fmt(feature.get('spectral_centroid'), 0),
                fmt(feature.get('spectral_rolloff'), 0),
                fmt(feature.get('spectral_flatness'), 0),
                fmt(feature.get('pitch'), 0),
                fmt(feature.get('timing_delta'), 0),
                fmt(feature.get('body_energy'), 0),
                fmt(feature.get('brilliance_energy'), 0),
                fmt(feature.get('geomean'), 0),
                fmt(feature.get('total_energy'), 0),
                fmt(feature.get('sustain_ms'), 0),
            ])
    
    print(f"✓ Exported {len(features)} onsets to {output_csv}")
    print(f"  Valid clusters: {result['final_quality'].get('valid_clusters', 0)}")
    print(f"  Silhouette score: {result['final_quality'].get('silhouette_score', 0):.3f}")
    print(f"  Noise ratio: {result['final_quality'].get('noise_ratio', 0):.3f}")
    print(f"  Noise points: {np.sum(labels == -1)}")
    print(f"  Cluster 0: {np.sum(labels == 0)}")
    print(f"  Cluster 1: {np.sum(labels == 1)}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Export clustering results to CSV')
    parser.add_argument('--audio', default='user_files/1 - AC_DC_Thunderstruck_Drums/stems/AC_DC_Thunderstruck_Drums-cymbals.wav',
                        help='Path to audio file')
    parser.add_argument('--output', default='clustering_data.csv',
                        help='Output CSV file')
    parser.add_argument('--threshold', type=float, default=5.0,
                        help='Initial onset detection threshold')
    parser.add_argument('--threshold-min', type=float, default=3.0,
                        help='Minimum threshold')
    parser.add_argument('--threshold-max', type=float, default=15.0,
                        help='Maximum threshold')
    parser.add_argument('--eps', type=float, default=1.2,
                        help='DBSCAN eps parameter (tuned for 11-dimensional feature space)')
    parser.add_argument('--min-samples', type=int, default=5,
                        help='DBSCAN min_samples parameter')
    
    args = parser.parse_args()
    
    export_clustering_csv(
        audio_path=args.audio,
        output_csv=args.output,
        threshold=args.threshold,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        dbscan_eps=args.eps,
        dbscan_min_samples=args.min_samples,
    )
