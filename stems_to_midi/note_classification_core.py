"""
Note Classification — Functional Core

Classifies MIDI note assignments from stored spectral features on the
final KEPT event set. Runs identically in both the full pipeline and
the rebuild pipeline.

Pass 2 of the two-pass architecture:
  Pass 1: Detect onsets, compute spectral features, apply threshold filters.
  Pass 2: Classify notes from stored features on KEPT events only (this module).

Pure functions — no I/O, no audio, no side effects.
"""

from typing import Dict, List, Tuple

import numpy as np


# ============================================================================
# Hihat Classification (threshold-based)
# ============================================================================


def classify_hihat_notes(
    events: List[Dict],
    config: Dict,
) -> List[Dict]:
    """
    Classify hihat events as open or closed from stored spectral features.

    Uses geomean (sqrt(body_energy * sizzle_energy)) and sustain_ms
    to distinguish open from closed hits. Matches the logic in
    detection_shell.detect_hihat_state() but operates on stored data.

    Args:
        events: KEPT hihat event dicts with body_energy, sizzle_energy,
            sustain_ms, and optionally geomean fields.
        config: Full config dict. Reads hihat.open_geomean_min (default 262)
            and hihat.open_sustain_ms (default 150).

    Returns:
        Same events with 'hihat_state' field set to 'open' or 'closed'.
    """
    hihat_config = config.get('hihat', {})
    open_geomean_min = hihat_config.get('open_geomean_min', 262.0)
    open_sustain_ms = hihat_config.get('open_sustain_ms', 150.0)

    for event in events:
        # Use stored geomean if available, otherwise compute from energies
        geomean = event.get('geomean')
        if geomean is None:
            body = event.get('body_energy', 0)
            sizzle = event.get('sizzle_energy', 0)
            geomean = np.sqrt(body * sizzle) if (body > 0 and sizzle > 0) else 0

        sustain = event.get('sustain_ms', 0) or 0

        if geomean >= open_geomean_min and sustain >= open_sustain_ms:
            event['hihat_state'] = 'open'
        else:
            event['hihat_state'] = 'closed'

    return events


# ============================================================================
# Clustering Helpers
# ============================================================================


def _extract_feature_values(
    events: List[Dict],
    feature_key: str,
) -> Tuple[np.ndarray, List[int]]:
    """
    Extract a numeric feature from events, returning values and valid indices.

    Args:
        events: Event dicts.
        feature_key: Key to extract (e.g. 'spectral_centroid_hz').

    Returns:
        Tuple of (values array for valid events, list of valid indices).
    """
    values = []
    valid_indices = []
    for i, event in enumerate(events):
        val = event.get(feature_key)
        if val is not None and val > 0:
            values.append(float(val))
            valid_indices.append(i)
    return np.array(values), valid_indices


def _cluster_values(
    values: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Cluster 1-D values into k groups using k-means, sorted by magnitude.

    Returns classification indices 0..k-1 where 0 = lowest centroid.
    Falls back to percentile-based splitting when sklearn is unavailable
    or when fewer than k unique values exist.

    Args:
        values: 1-D array of positive numeric values.
        k: Number of clusters.

    Returns:
        Array of classification indices (same length as values).
    """
    if len(values) == 0:
        return np.array([], dtype=int)

    unique = np.unique(values)
    n_unique = len(unique)

    # Fewer unique values than clusters — assign by rank
    if n_unique < k:
        sorted_unique = np.sort(unique)
        # Spread unique values across classification range
        # e.g., 2 unique with k=3 → assign to indices 0 and 2
        if n_unique == 1:
            return np.zeros(len(values), dtype=int)
        spacing = (k - 1) / (n_unique - 1)
        rank_map = {v: int(round(i * spacing)) for i, v in enumerate(sorted_unique)}
        return np.array([rank_map[v] for v in values], dtype=int)

    # Try sklearn k-means
    try:
        from sklearn.cluster import KMeans

        X = values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)

        # Sort clusters by center value (0 = lowest)
        centers = kmeans.cluster_centers_.flatten()
        sorted_indices = np.argsort(centers)
        label_map = {old: new for new, old in enumerate(sorted_indices)}

        return np.array([label_map[lbl] for lbl in kmeans.labels_], dtype=int)

    except ImportError:
        # Fallback: percentile-based splitting
        percentiles = [100 * (i + 1) / k for i in range(k - 1)]
        thresholds = [np.percentile(values, p) for p in percentiles]

        classifications = np.zeros(len(values), dtype=int)
        for i, val in enumerate(values):
            for j, threshold in enumerate(thresholds):
                if val > threshold:
                    classifications[i] = j + 1
        return classifications


# ============================================================================
# Tom Classification (k-means on spectral centroid)
# ============================================================================


def classify_tom_notes(
    events: List[Dict],
    config: Dict,
) -> List[Dict]:
    """
    Classify tom events into low/mid/high using spectral centroid clustering.

    Uses k-means (k=3) on spectral_centroid_hz to separate tom types.
    Lower centroid → low tom, higher → high tom.

    Args:
        events: KEPT tom event dicts with spectral_centroid_hz field.
        config: Full config dict (reserved for future per-stem tuning).

    Returns:
        Same events with 'classification' field: 0=low, 1=mid, 2=high.
    """
    values, valid_indices = _extract_feature_values(events, 'spectral_centroid_hz')

    if len(values) == 0:
        # No valid centroid data — default all to mid (1)
        for event in events:
            event['classification'] = 1
        return events

    labels = _cluster_values(values, k=3)

    # Set default for events without valid centroid
    for event in events:
        event['classification'] = 1  # mid tom default

    # Apply cluster labels to valid events
    for idx, vi in enumerate(valid_indices):
        events[vi]['classification'] = int(labels[idx])

    return events


# ============================================================================
# Cymbal Classification (k-means on spectral centroid)
# ============================================================================


def classify_cymbal_notes(
    events: List[Dict],
    config: Dict,
) -> List[Dict]:
    """
    Classify cymbal events into crash/ride/chinese using spectral centroid.

    Uses k-means (k=3) on spectral_centroid_hz. Lower centroid → crash,
    mid → ride, higher → chinese.

    Args:
        events: KEPT cymbal event dicts with spectral_centroid_hz field.
        config: Full config dict.

    Returns:
        Same events with 'classification' field: 0=crash, 1=ride, 2=chinese.
    """
    values, valid_indices = _extract_feature_values(events, 'spectral_centroid_hz')

    if len(values) == 0:
        # No valid centroid data — default all to crash (0)
        for event in events:
            event['classification'] = 0
        return events

    labels = _cluster_values(values, k=3)

    # Set default for events without valid centroid
    for event in events:
        event['classification'] = 0  # crash default

    # Apply cluster labels to valid events
    for idx, vi in enumerate(valid_indices):
        events[vi]['classification'] = int(labels[idx])

    return events


# ============================================================================
# Snare Classification (k-means on spectral centroid)
# ============================================================================


def classify_snare_notes(
    events: List[Dict],
    config: Dict,
) -> List[Dict]:
    """
    Classify snare events into snare/rimshot/clap/clap+snare types.

    Uses k-means (k up to 4) on spectral_centroid_hz. Sorted by centroid:
    0=snare (lowest), 1=rimshot, 2=clap, 3=clap+snare (highest).

    Automatically reduces k when fewer unique centroid values exist.

    Args:
        events: KEPT snare event dicts with spectral_centroid_hz field.
        config: Full config dict.

    Returns:
        Same events with 'classification' field: 0-3.
    """
    values, valid_indices = _extract_feature_values(events, 'spectral_centroid_hz')

    if len(values) == 0:
        # No valid centroid data — default all to snare (0)
        for event in events:
            event['classification'] = 0
        return events

    # Use k = min(4, number of unique values) to avoid over-clustering
    n_unique = len(np.unique(values))
    k = min(4, n_unique)

    labels = _cluster_values(values, k=k)

    # Set default for events without valid centroid
    for event in events:
        event['classification'] = 0  # snare default

    # Apply cluster labels to valid events
    for idx, vi in enumerate(valid_indices):
        events[vi]['classification'] = int(labels[idx])

    return events


# ============================================================================
# Note Mapping
# ============================================================================


# Maps (stem_type, classification_index) → DrumMapping attribute name
_NOTE_MAP = {
    'hihat': {
        'open': 'hihat_open',
        'closed': 'hihat_closed',
        'handclap': 'handclap',
    },
    'toms': {
        0: 'tom_low',
        1: 'tom_mid',
        2: 'tom_high',
    },
    'cymbals': {
        0: 'crash',
        1: 'ride',
        2: 'chinese',
    },
    'snare': {
        0: 'snare',
        1: 'snare_rimshot',
        2: 'snare_clap',
        3: 'snare_clap_snare',
    },
}


def _map_note(
    event: Dict,
    stem_type: str,
    drum_mapping,
) -> int:
    """
    Map a classified event to its MIDI note number.

    Args:
        event: Event dict with 'hihat_state' (hihat) or 'classification' (others).
        stem_type: One of 'hihat', 'toms', 'cymbals', 'snare', 'kick'.
        drum_mapping: DrumMapping instance.

    Returns:
        MIDI note number.
    """
    if stem_type == 'hihat':
        state = event.get('hihat_state', 'closed')
        attr = _NOTE_MAP['hihat'].get(state, 'hihat_closed')
        return getattr(drum_mapping, attr)

    mapping = _NOTE_MAP.get(stem_type)
    if mapping is not None:
        cls = event.get('classification', 0)
        attr = mapping.get(cls)
        if attr is not None:
            return getattr(drum_mapping, attr)

    # Fallback: default note for stem type
    return getattr(drum_mapping, stem_type, 36)


# ============================================================================
# Main Entry Point
# ============================================================================


def classify_notes(
    events: List[Dict],
    stem_type: str,
    drum_mapping,
    config: Dict,
) -> List[Dict]:
    """
    Classify and assign MIDI notes to KEPT events based on stored features.

    This is the Pass 2 entry point. Runs the per-stem classifier, then
    maps classification results to MIDI note numbers via DrumMapping.

    Modifies events in place (adds 'note', 'hihat_state'/'classification' fields).

    Args:
        events: KEPT event dicts from a single stem, with stored spectral
            features (spectral_centroid_hz, sustain_ms, energy bands).
        stem_type: Stem type ('hihat', 'toms', 'cymbals', 'snare', 'kick').
        drum_mapping: DrumMapping instance with note number attributes.
        config: Full config dict.

    Returns:
        Same events with 'note' field set to the classified MIDI note.
    """
    if not events:
        return events

    # Kick has no sub-classification — always the same note
    if stem_type == 'kick':
        for event in events:
            event['note'] = drum_mapping.kick
        return events

    # Run per-stem classifier
    if stem_type == 'hihat':
        classify_hihat_notes(events, config)
    elif stem_type == 'toms':
        classify_tom_notes(events, config)
    elif stem_type == 'cymbals':
        classify_cymbal_notes(events, config)
    elif stem_type == 'snare':
        classify_snare_notes(events, config)

    # Map classification to MIDI note numbers
    for event in events:
        event['note'] = _map_note(event, stem_type, drum_mapping)

    return events
