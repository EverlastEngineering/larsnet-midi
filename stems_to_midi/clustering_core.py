"""
Clustering Core - Pure Functional Core

Pure functions for clustering onset features to identify distinct instruments.
All functions are deterministic with no side effects.

Architecture: Functional Core
- No I/O operations
- Deterministic (same input → same output)
- No external state or side effects
- Testable in isolation
"""

import numpy as np
from typing import List, Dict, Optional, Literal
from sklearn.cluster import DBSCAN, KMeans  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

# Import TypedDict from parent module
try:
    from midi_types import OnsetFeatures
except ImportError:
    # Running from stems_to_midi/ directly
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from midi_types import OnsetFeatures


__all__ = [
    'features_to_array',
    'cluster_dbscan',
    'cluster_kmeans',
    'cluster_onsets',
]


def features_to_array(
    features: List[OnsetFeatures],
    feature_names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Convert list of OnsetFeatures to numpy array for clustering.
    
    Pure function - no side effects.
    
    Args:
        features: List of OnsetFeatures dicts
        feature_names: Which features to include (default: all except 'time')
    
    Returns:
        2D array of shape (n_onsets, n_features)
    
    Examples:
        >>> features = [
        ...     {'time': 0.5, 'pan_confidence': -0.8, 'spectral_centroid': 2000.0, ...},
        ...     {'time': 1.0, 'pan_confidence': 0.7, 'spectral_centroid': 3000.0, ...}
        ... ]
        >>> array = features_to_array(features, ['pan_confidence', 'spectral_centroid'])
        >>> array.shape
        (2, 2)
    """
    if len(features) == 0:
        return np.array([]).reshape(0, 0)
    
    if feature_names is None:
        # Default: all features except 'time' (which is the onset position, not a clustering feature)
        feature_names = [
            'pan_confidence',
            'spectral_centroid',
            'spectral_rolloff',
            'spectral_flatness',
            'pitch',
            'timing_delta',
            # Domain-specific band energies (missing keys return 0.0)
            'body_energy',
            'wire_energy',
            'fundamental_energy',
            'attack_energy',
            'sizzle_energy',
            'brilliance_energy',
            'geomean',
            'total_energy',
            'sustain_ms'
        ]
    
    # Extract features, handling None values
    rows = []
    for f in features:
        row = []
        for name in feature_names:
            value = f.get(name)
            if value is None:
                # Replace None with 0 (or could use mean imputation later)
                row.append(0.0)
            else:
                row.append(float(value))
        rows.append(row)
    
    return np.array(rows)


def cluster_dbscan(
    feature_array: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 2,
    normalize: bool = True
) -> Dict[str, any]:
    """
    Cluster features using DBSCAN (Density-Based Spatial Clustering).
    
    DBSCAN finds core samples of high density and expands clusters from them.
    Good for:
    - Finding arbitrary-shaped clusters
    - Identifying outliers (label = -1)
    - No need to specify number of clusters
    
    Pure function - no side effects.
    
    Args:
        feature_array: 2D array of shape (n_samples, n_features)
        eps: Maximum distance between samples to be considered neighbors
        min_samples: Minimum samples in neighborhood to be core point
        normalize: Whether to standardize features (recommended)
    
    Returns:
        Dict with:
            - labels: Cluster label for each sample (-1 = noise)
            - n_clusters: Number of clusters found (excluding noise)
            - n_noise: Number of noise points
            - core_sample_indices: Indices of core samples
    
    Examples:
        >>> # Two distinct clusters
        >>> X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]])
        >>> result = cluster_dbscan(X, eps=2.0, min_samples=2)
        >>> result['n_clusters']
        2
    """
    if len(feature_array) == 0:
        return {
            'labels': np.array([]),
            'n_clusters': 0,
            'n_noise': 0,
            'core_sample_indices': np.array([])
        }
    
    # Normalize features for better distance calculations
    if normalize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_array)
    else:
        X_scaled = feature_array
    
    # Run DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(X_scaled)
    
    # Count clusters (excluding noise label -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    
    return {
        'labels': labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'core_sample_indices': clustering.core_sample_indices_ if hasattr(clustering, 'core_sample_indices_') else np.array([])
    }


def cluster_kmeans(
    feature_array: np.ndarray,
    n_clusters: int,
    normalize: bool = True,
    random_state: int = 42
) -> Dict[str, any]:
    """
    Cluster features using k-means clustering.
    
    K-means partitions data into k clusters where each sample belongs to
    the cluster with the nearest mean (centroid).
    Good for:
    - Known number of clusters
    - Spherical/convex cluster shapes
    - Fast and deterministic
    
    Pure function - no side effects (with fixed random_state).
    
    Args:
        feature_array: 2D array of shape (n_samples, n_features)
        n_clusters: Number of clusters to create
        normalize: Whether to standardize features (recommended)
        random_state: Random seed for reproducibility
    
    Returns:
        Dict with:
            - labels: Cluster label for each sample (0 to n_clusters-1)
            - n_clusters: Number of clusters (same as input)
            - centroids: Cluster centroids in original feature space
            - inertia: Sum of squared distances to nearest cluster center
    
    Examples:
        >>> # Two distinct clusters
        >>> X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]])
        >>> result = cluster_kmeans(X, n_clusters=2)
        >>> result['n_clusters']
        2
    """
    if len(feature_array) == 0:
        return {
            'labels': np.array([]),
            'n_clusters': 0,
            'centroids': np.array([]).reshape(0, feature_array.shape[1] if feature_array.ndim > 1 else 0),
            'inertia': 0.0
        }
    
    # Can't have more clusters than samples
    n_clusters = min(n_clusters, len(feature_array))
    
    # Normalize features
    if normalize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_array)
    else:
        X_scaled = feature_array
        scaler = None
    
    # Run k-means
    clustering = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = clustering.fit_predict(X_scaled)
    
    # Transform centroids back to original space if normalized
    if normalize and scaler is not None:
        centroids = scaler.inverse_transform(clustering.cluster_centers_)
    else:
        centroids = clustering.cluster_centers_
    
    return {
        'labels': labels,
        'n_clusters': n_clusters,
        'centroids': centroids,
        'inertia': float(clustering.inertia_)
    }


def cluster_onsets(
    features: List[OnsetFeatures],
    method: Literal['dbscan', 'kmeans'],
    n_clusters: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, any]:
    """
    Cluster onsets using specified method.
    
    Dispatcher function that converts features to array and calls
    appropriate clustering function.
    
    Pure function - no side effects.
    
    Args:
        features: List of OnsetFeatures dicts
        method: Clustering method ('dbscan' or 'kmeans')
        n_clusters: Number of clusters (required for kmeans, ignored for dbscan)
        feature_names: Which features to use (default: all except 'time')
        **kwargs: Additional arguments passed to clustering function
    
    Returns:
        Dict with clustering results (structure depends on method)
    
    Raises:
        ValueError: If method is invalid or n_clusters missing for kmeans
    
    Examples:
        >>> features = [...]  # List of OnsetFeatures
        >>> result = cluster_onsets(features, method='dbscan', eps=0.5)
        >>> result = cluster_onsets(features, method='kmeans', n_clusters=2)
    """
    if len(features) == 0:
        if method == 'dbscan':
            return {
                'labels': np.array([]),
                'n_clusters': 0,
                'n_noise': 0,
                'core_sample_indices': np.array([])
            }
        elif method == 'kmeans':
            return {
                'labels': np.array([]),
                'n_clusters': 0,
                'centroids': np.array([]),
                'inertia': 0.0
            }
        else:
            raise ValueError(f"Unknown clustering method: {method}")
    
    # Convert features to array
    feature_array = features_to_array(features, feature_names)
    
    # Call appropriate clustering function
    if method == 'dbscan':
        return cluster_dbscan(feature_array, **kwargs)
    
    elif method == 'kmeans':
        if n_clusters is None:
            raise ValueError("n_clusters is required for kmeans clustering")
        return cluster_kmeans(feature_array, n_clusters=n_clusters, **kwargs)
    
    else:
        raise ValueError(f"Unknown clustering method: {method}. Choose 'dbscan' or 'kmeans'.")
