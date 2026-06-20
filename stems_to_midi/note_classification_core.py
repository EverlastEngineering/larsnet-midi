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

from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Hihat Classification (threshold-based)
# ============================================================================


def classify_hihat_notes(
    events: List[Dict],
    config: Dict,
    force_reclassify: bool = False,
) -> List[Dict]:
    """
    Classify hihat events as open or closed from stored spectral features.

    Uses geomean (sqrt(body_energy * sizzle_energy)) and sustain_ms
    to distinguish open from closed hits. Matches the logic in
    detection_shell.detect_hihat_state() but operates on stored data.

    By default, events that already have a stored ``hihat_state`` keep it.
    This matches the behavior of other stems (snare/toms/cymbals keep their
    stored ``classification`` when re-running pass-2) and prevents
    threshold-slider changes from silently re-classifying the same event
    differently on rebuild.  Pass ``force_reclassify=True`` to override
    stored states (e.g. when the user has explicitly changed the sliders
    and wants the new thresholds applied to the existing data).

    Args:
        events: KEPT hihat event dicts with body_energy, sizzle_energy,
            sustain_ms, and optionally geomean fields.
        config: Full config dict. Reads hihat.open_decay_slope_max
            (default 2.0 dB/frame) for the slope rule. The legacy
            hihat.open_geomean_min (default 262) and
            hihat.open_sustain_ms (default 100) keys are also
            read by the defensive fallback path, which only
            fires when decay_slope_db is missing (older sidecars
            from before 2026-06-19). They are no longer in the
            settings schema or project yaml — current projects
            will never go down the fallback path.
        force_reclassify: If True, recompute hihat_state for every event
            even if one is already stored. Default False.

    Returns:
        Same events with 'hihat_state' field set to 'open' or 'closed'.
    """
    hihat_config = config.get('hihat', {})
    open_geomean_min = hihat_config.get('open_geomean_min', 262.0)
    open_sustain_ms = hihat_config.get('open_sustain_ms', 100.0)

    for event in events:
        # Preserve stored classification across rebuilds (parity with
        # snare/toms/cymbals which keep their stored 'classification').
        if not force_reclassify and event.get('hihat_state') in ('open', 'closed'):
            continue

        # 2026-06-19: broadband-envelope decay-slope rule wins when
        # the field is present. The PGA detector walks the broadband
        # contrast envelope forward from the event's peak frame and
        # stamps ``decay_slope_db`` (mean per-frame dB drop). Closed
        # hihats decay fast (3.4-3.6 dB/frame); open hihats ring out
        # so the next strike cuts in before the envelope drops, giving
        # a shallow slope (0.7-1.4). If decay_slope_db is below
        # ``hihat.open_decay_slope_max`` (default 2.0 dB/frame), the
        # event is open — no need to consult geomean/sustain.
        #
        # Falls back to the geomean+sustain rule when decay_slope_db
        # is missing (older sidecars from before 2026-06-19, or stems
        # where the detector didn't produce a per-event walk).
        slope = event.get('decay_slope_db')
        slope_max = hihat_config.get('open_decay_slope_max', 2.0)
        if slope is not None:
            event['hihat_state'] = 'open' if slope < slope_max else 'closed'
            continue

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
    allow_zero: bool = False,
) -> Tuple[np.ndarray, List[int]]:
    """
    Extract a numeric feature from events, returning values and valid indices.

    Args:
        events: Event dicts.
        feature_key: Key to extract (e.g. 'spectral_centroid_hz').
        allow_zero: If True, include zero values as valid. Use for features
            like stereo_width where 0.0 is meaningful (perfectly mono).

    Returns:
        Tuple of (values array for valid events, list of valid indices).
    """
    values = []
    valid_indices = []
    for i, event in enumerate(events):
        val = event.get(feature_key)
        if val is not None:
            if allow_zero or val > 0:
                values.append(float(val))
                valid_indices.append(i)
    return np.array(values), valid_indices


# Per-stem feature fallback priorities (first match with data wins)
# Note: pitch_hz is only available for toms (calculated in analysis_core.py)
_FEATURE_PRIORITIES = {
    'snare': ['stereo_width', 'spectral_centroid_hz'],
    'toms': ['pitch_hz', 'spectral_centroid_hz', 'stereo_width'],
    'cymbals': ['spectral_centroid_hz', 'stereo_width'],
}

# Features where 0.0 is a valid measurement (not missing data)
_ALLOW_ZERO_FEATURES = {'stereo_width', 'pan_confidence'}


def _resolve_cluster_feature(
    events: List[Dict],
    stem_type: str,
    config: Dict,
) -> Tuple[np.ndarray, List[int], Optional[str]]:
    """
    Resolve which feature to cluster on and extract its values.

    Reads config[stem_type]['cluster_feature']. When 'auto' (default),
    walks the priority list for the stem and uses the first feature with
    sufficient data. When a specific feature is named, tries it first
    then falls back to the priority chain.

    The 3rd return element (``actual_feature``) tells the caller which
    feature was *actually* used — which may differ from the user's
    explicit choice if their choice has no data. This makes the
    silent fallback observable: callers can log a warning when
    ``actual_feature != chosen``, which the user sees in the WebUI
    console log.

    User report (2026-06-08): picking "Pitch" in the snare Cluster
    By dropdown did nothing visible. Root cause: pitch detection was
    disabled, so no ``pitch_hz`` data existed; the resolver fell
    back to ``stereo_width`` silently. Same result as the default —
    looked like "doesn't work."

    Args:
        events: Event dicts with feature data.
        stem_type: Stem type for priority lookup.
        config: Full config dict.

    Returns:
        Tuple of (values array, valid indices list, actual_feature).
        ``actual_feature`` is the feature whose values are in
        ``values``, or None if no feature had any data.
    """
    stem_config = config.get(stem_type, {})
    chosen = stem_config.get('cluster_feature', 'auto')
    priorities = _FEATURE_PRIORITIES.get(stem_type, ['spectral_centroid_hz'])

    # Build ordered feature list: explicit choice first, then fallbacks
    if chosen and chosen != 'auto':
        feature_order = [chosen] + [f for f in priorities if f != chosen]
    else:
        feature_order = list(priorities)

    for feat in feature_order:
        allow_zero = feat in _ALLOW_ZERO_FEATURES
        values, valid_indices = _extract_feature_values(
            events, feat, allow_zero=allow_zero,
        )
        if len(values) > 0:
            return values, valid_indices, feat

    return np.array([]), [], None


def _warn_on_cluster_feature_fallback(
    stem_type: str,
    config: Dict,
    actual_feature: Optional[str],
    events: List[Dict],
) -> None:
    """
    Log a warning when the resolver had to fall back from the user's
    explicit cluster_feature choice because their choice had no data.

    User report (2026-06-08): picking "Pitch" in the snare Cluster
    By dropdown did nothing visible. Root cause: pitch detection was
    disabled, no ``pitch_hz`` data existed, the resolver fell back
    to ``stereo_width`` silently. This warning makes the fallback
    visible in the WebUI console log so the user can see why their
    selection didn't take effect.

    Args:
        stem_type: Stem type (snare, toms, cymbals, etc.).
        config: Full config dict — used to read the user's chosen
            cluster feature.
        actual_feature: The feature the resolver actually used (may
            be None if no feature had data at all).
        events: Event list — used to count how many events had the
            chosen feature (informational, so the user knows the
            scope of the problem).
    """
    chosen = config.get(stem_type, {}).get('cluster_feature', 'auto')
    if chosen == 'auto' or chosen == actual_feature:
        # No explicit choice, or no fallback — nothing to warn about.
        return

    if actual_feature is None:
        # Resolver found no feature with any data at all. Different
        # problem (no detection ran); the classification functions
        # already default every event to index 0. Skip the warning
        # to keep this message focused on the fallback case.
        return

    # Count events that had the chosen feature, so the user can
    # see "0 of 140 events had pitch_hz" rather than just "fell back".
    n_chosen = sum(
        1 for e in events if e.get(chosen) is not None
    )

    stem_label = {
        'snare': 'snare',
        'toms': 'tom',
        'cymbals': 'cymbal',
    }.get(stem_type, stem_type)

    print(
        f"WARNING: {stem_label} cluster_feature='{chosen}' was chosen "
        f"but only {n_chosen}/{len(events)} events have that data. "
        f"Falling back to '{actual_feature}'. "
        f"For pitch: enable {stem_type}.enable_pitch_detection AND "
        f"run a full Convert (rebuild alone does not re-detect features)."
    )


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
    force_reclassify: bool = False,
) -> List[Dict]:
    """
    Classify tom events into low/mid/high using configurable feature clustering.

    Default: k-means (k=3) on spectral_centroid_hz. The clustering feature
    can be overridden via config['toms']['cluster_feature'] (e.g. 'stereo_width'
    for panning-based separation). Falls through a priority chain when the
    chosen feature has insufficient data.

    Args:
        events: KEPT tom event dicts with spectral_centroid_hz field.
        config: Full config dict. Reads toms.expected_clusters (default 3)
            and toms.cluster_feature (default 'auto').

    Returns:
        Same events with 'classification' field: 0=low, 1=mid, 2=high.
    """
    toms_config = config.get('toms', {})
    expected_clusters = toms_config.get('expected_clusters') or 3
    expected_clusters = max(1, min(4, int(expected_clusters)))

    # T2 follow-up (2026-06-08): when the rebuild is invoked with the
    # same config, preserve stored classifications instead of
    # re-running k-means. T3 e2e found snare/tom/cymbal events being
    # silently reclassified on rebuild even when nothing changed —
    # likely cause is the same as hihat_state: classify_*_notes used
    # to always overwrite. The fix mirrors hihat_state's pattern:
    # events that already have a stored classification keep it unless
    # force_reclassify=True (which the rebuild path sets only when a
    # classification threshold actually changed).
    if not force_reclassify and any(
        e.get('classification') is not None for e in events
    ):
        # All events already have stored classifications — keep them.
        for event in events:
            event['classification'] = event.get('classification', 0)
        return events

    if expected_clusters == 1:
        for event in events:
            event['classification'] = 0
        return events

    values, valid_indices, actual_feature = _resolve_cluster_feature(
        events, 'toms', config,
    )
    _warn_on_cluster_feature_fallback(
        'toms', config, actual_feature, events,
    )

    if len(values) == 0:
        for event in events:
            event['classification'] = 1  # mid tom default
        return events

    n_unique = len(np.unique(values))
    k = min(expected_clusters, n_unique)
    labels = _cluster_values(values, k=k)

    for event in events:
        event['classification'] = 1  # mid tom default

    for idx, vi in enumerate(valid_indices):
        events[vi]['classification'] = int(labels[idx])

    return events


# ============================================================================
# Cymbal Classification (k-means on spectral centroid)
# ============================================================================


def classify_cymbal_notes(
    events: List[Dict],
    config: Dict,
    force_reclassify: bool = False,
) -> List[Dict]:
    """
    Classify cymbal events into crash/ride/chinese using configurable feature.

    Default: k-means on spectral_centroid_hz. The clustering feature can be
    overridden via config['cymbals']['cluster_feature'] (e.g. 'stereo_width'
    for L/R panning-based separation).

    Args:
        events: KEPT cymbal event dicts with spectral_centroid_hz field.
        config: Full config dict. Reads cymbals.expected_clusters (default 2)
            and cymbals.cluster_feature (default 'auto').

    Returns:
        Same events with 'classification' field: 0=crash, 1=ride, 2=chinese.
    """
    cymbal_config = config.get('cymbals', {})
    expected_clusters = cymbal_config.get('expected_clusters') or 2
    expected_clusters = max(1, min(4, int(expected_clusters)))

    # T2 follow-up (2026-06-08): preserve stored classifications on
    # rebuild (see classify_tom_notes for full rationale). T3 e2e
    # found cymbal events being silently reclassified even when
    # config didn't change.
    if not force_reclassify and any(
        e.get('classification') is not None for e in events
    ):
        for event in events:
            event['classification'] = event.get('classification', 0)
        return events

    if expected_clusters == 1:
        for event in events:
            event['classification'] = 0
        return events

    values, valid_indices, actual_feature = _resolve_cluster_feature(
        events, 'cymbals', config,
    )
    _warn_on_cluster_feature_fallback(
        'cymbals', config, actual_feature, events,
    )

    if len(values) == 0:
        for event in events:
            event['classification'] = 0  # crash default
        return events

    n_unique = len(np.unique(values))
    k = min(expected_clusters, n_unique)
    labels = _cluster_values(values, k=k)

    for event in events:
        event['classification'] = 0  # crash default

    for idx, vi in enumerate(valid_indices):
        events[vi]['classification'] = int(labels[idx])

    return events


# ============================================================================
# Snare Classification (k-means on stereo width)
# ============================================================================


def classify_snare_notes(
    events: List[Dict],
    config: Dict,
    force_reclassify: bool = False,
) -> List[Dict]:
    """
    Classify snare events into sub-types using stereo width.

    Uses k-means on stereo_width to distinguish mono snare hits from
    wide stereo claps/layered sounds. The number of clusters is
    controlled by config['snare']['expected_clusters'] (1-3, default 2).
    Sorted by width: 0=snare (narrowest/mono), 1=clap (wider stereo).

    When expected_clusters=1, all events are assigned classification=0
    (pure snare, no sub-type splitting).

    Falls back to spectral_centroid_hz when stereo_width data is absent
    (mono source audio).

    Args:
        events: KEPT snare event dicts with stereo_width and/or
            spectral_centroid_hz fields.
        config: Full config dict. Reads snare.expected_clusters (default 2).

    Returns:
        Same events with 'classification' field: 0-2.
    """
    snare_config = config.get('snare', {})
    expected_clusters = snare_config.get('expected_clusters') or 2
    expected_clusters = max(1, min(3, int(expected_clusters)))  # Clamp to 1-3

    # T2 follow-up (2026-06-08): preserve stored classifications on
    # rebuild. T3 e2e found 5/10 snare events being silently
    # reclassified as rimshot even when config didn't change — the
    # root cause was that k-means re-ran on every rebuild, producing
    # a different cluster assignment than the stored baseline. The
    # fix: if all events already have a stored classification and
    # force_reclassify=False, keep them. force_reclassify=True is set
    # by the rebuild path only when a classification threshold
    # actually changed (kick.geomean_threshold is a detection
    # threshold, not a classification one — only expected_clusters,
    # midi_note_*, cluster_feature affect classification).
    if not force_reclassify and any(
        e.get('classification') is not None for e in events
    ):
        for event in events:
            event['classification'] = event.get('classification', 0)
        return events

    # With 1 cluster, all events are plain snare — skip clustering
    if expected_clusters == 1:
        for event in events:
            event['classification'] = 0
        return events

    values, valid_indices, actual_feature = _resolve_cluster_feature(
        events, 'snare', config,
    )
    _warn_on_cluster_feature_fallback(
        'snare', config, actual_feature, events,
    )

    if len(values) == 0:
        for event in events:
            event['classification'] = 0
        return events

    # Use k = min(expected_clusters, number of unique values)
    n_unique = len(np.unique(values))
    k = min(expected_clusters, n_unique)

    labels = _cluster_values(values, k=k)

    # Set default for events without valid feature data
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
    },
}


def _map_note(
    event: Dict,
    stem_type: str,
    drum_mapping,
    config: Optional[Dict] = None,
) -> int:
    """
    Map a classified event to its MIDI note number.

    When config contains a cluster_note_map for the stem, use it directly
    (maps classification index → MIDI note number). Otherwise fall back
    to the default _NOTE_MAP → DrumMapping attribute lookup.

    Args:
        event: Event dict with 'hihat_state' (hihat) or 'classification' (others).
        stem_type: One of 'hihat', 'toms', 'cymbals', 'snare', 'kick'.
        drum_mapping: DrumMapping instance.
        config: Optional full config dict. When present, checks for
            config[stem_type]['cluster_note_map'] = {0: 38, 1: 39, ...}.

    Returns:
        MIDI note number.
    """
    if stem_type == 'hihat':
        state = event.get('hihat_state', 'closed')
        attr = _NOTE_MAP['hihat'].get(state, 'hihat_closed')
        return getattr(drum_mapping, attr)

    # Check for custom cluster→note mapping in config
    if config:
        stem_config = config.get(stem_type, {})
        cluster_note_map = stem_config.get('cluster_note_map')
        if cluster_note_map:
            cls = event.get('classification', 0)
            note = cluster_note_map.get(cls) or cluster_note_map.get(str(cls))
            if note is not None:
                return int(note)

    mapping = _NOTE_MAP.get(stem_type)
    if mapping is not None:
        cls = event.get('classification', 0)
        attr = mapping.get(cls)
        if attr is not None:
            return getattr(drum_mapping, attr)

    # Fallback: default note for stem type
    return getattr(drum_mapping, stem_type, 36)


# ============================================================================
# Cluster Analysis
# ============================================================================

# Features to analyze per cluster, in priority order for each stem
# Note: pitch_hz is only available for toms (calculated in analysis_core.py)
_CLUSTER_FEATURES = {
    'snare': ['stereo_width', 'pan_confidence', 'spectral_centroid_hz'],
    'toms': ['pitch_hz', 'spectral_centroid_hz', 'stereo_width', 'pan_confidence'],
    'cymbals': ['spectral_centroid_hz', 'stereo_width', 'pan_confidence'],
    'hihat': ['geomean', 'duration_sec'],
}

# Human-readable labels for features
_FEATURE_LABELS = {
    'stereo_width': 'Stereo Width',
    'pan_confidence': 'Pan Position',
    'spectral_centroid_hz': 'Brightness',
    'pitch_hz': 'Pitch',
    'geomean': 'Energy',
    'duration_sec': 'Duration',
    'total_energy': 'Total Energy',
}


def analyze_clusters(
    events: List[Dict],
    stem_type: str,
    drum_mapping,
) -> List[Dict]:
    """
    Analyze classified events and return per-cluster metadata.

    Examines which features best distinguish each cluster and provides
    statistics for the UI to display meaningful cluster descriptions.

    For hihat specifically, events are grouped by ``hihat_state`` (open /
    closed) instead of the integer ``classification`` field — the hihat
    pipeline does threshold-based open/closed detection, not k-means
    clustering, so its cluster info must reflect the binary state
    (bug A3, part 3 — hihat cluster visualization).

    Args:
        events: KEPT events that have already been classified (have
            'classification' and 'note' fields set, or for hihat a
            'hihat_state' field set to 'open' or 'closed').
        stem_type: Stem type for feature priority lookup.
        drum_mapping: DrumMapping instance for note labels.

    Returns:
        List of cluster info dicts sorted by classification index:
        [
            {
                'classification': 0,
                'note': 38,
                'note_label': 'Snare',
                'count': 200,
                'features': {
                    'stereo_width': {'mean': 0.03, 'min': 0.01, 'max': 0.08},
                    'pan_confidence': {'mean': -0.02, 'min': -0.1, 'max': 0.05},
                    ...
                },
                'distinguishing_feature': 'stereo_width',
                'distinguishing_label': 'Stereo Width',
                'description': 'Narrow stereo (mono)',
            },
            ...
        ]

        For hihat, ``classification`` is replaced with a string state
        ('open' or 'closed') and ``note_label`` is 'Open' or 'Closed'.
    """
    if not events:
        return []

    # Hihat groups by hihat_state, not by integer classification. The
    # other stems continue to use the integer field (set by k-means
    # clustering).
    if stem_type == 'hihat':
        return _analyze_hihat_clusters(events, drum_mapping)

    # Group events by classification
    clusters: Dict[int, List[Dict]] = {}
    for event in events:
        cls = event.get('classification', 0)
        clusters.setdefault(cls, []).append(event)

    features_to_check = _CLUSTER_FEATURES.get(stem_type, ['spectral_centroid_hz'])

    # Compute per-cluster feature stats
    cluster_infos = []
    for cls_idx in sorted(clusters.keys()):
        cluster_events = clusters[cls_idx]
        note_num = cluster_events[0].get('note')
        note_info = _NOTE_MAP.get(stem_type, {}).get(cls_idx)
        note_label = note_info.replace('_', ' ').title() if note_info else f'Type {cls_idx}'

        feature_stats = {}
        for feat in features_to_check:
            vals = [
                e[feat] for e in cluster_events
                if feat in e and e[feat] is not None
            ]
            if vals:
                feature_stats[feat] = {
                    'mean': round(float(np.mean(vals)), 4),
                    'min': round(float(np.min(vals)), 4),
                    'max': round(float(np.max(vals)), 4),
                }

        cluster_infos.append({
            'classification': cls_idx,
            'note': note_num,
            'note_label': note_label,
            'count': len(cluster_events),
            'features': feature_stats,
        })

    # Find the feature that best distinguishes the clusters
    _annotate_distinguishing_features(cluster_infos, features_to_check)

    return cluster_infos


def _analyze_hihat_clusters(
    events: List[Dict],
    drum_mapping,
) -> List[Dict]:
    """
    Build cluster info dicts for hihat events grouped by hihat_state.

    Hihat classification is binary (open vs closed) and uses stored
    spectral features (geomean, sustain_ms) rather than k-means. This
    helper produces the same cluster_info shape as the k-means path so
    the WebUI can render the cluster cards uniformly.

    Args:
        events: KEPT hihat events with hihat_state in {'open', 'closed'}.
        drum_mapping: DrumMapping for note lookup.

    Returns:
        List of cluster info dicts (one per state that has at least one
        event). Each dict has integer ``classification`` 0 or 1
        (preserving the rest of the schema's typing contract) plus a
        ``state`` field with the string ('open' or 'closed') for any
        downstream consumer that prefers strings.
    """
    if not events:
        return []

    # Map hihat_state -> integer classification (mirrors _NOTE_MAP for hihat)
    state_to_cls = {'closed': 0, 'open': 1}
    state_to_attr = {'closed': 'hihat_closed', 'open': 'hihat_open'}

    clusters: Dict[str, List[Dict]] = {}
    for event in events:
        state = event.get('hihat_state', 'closed')
        clusters.setdefault(state, []).append(event)

    features_to_check = _CLUSTER_FEATURES.get('hihat', ['geomean', 'duration_sec'])

    cluster_infos = []
    for state in ('closed', 'open'):  # Fixed order: closed first, then open
        if state not in clusters:
            continue
        cluster_events = clusters[state]
        note_attr = state_to_attr[state]
        note_num = getattr(drum_mapping, note_attr, None)

        feature_stats = {}
        for feat in features_to_check:
            vals = [
                e[feat] for e in cluster_events
                if feat in e and e[feat] is not None
            ]
            if vals:
                feature_stats[feat] = {
                    'mean': round(float(np.mean(vals)), 4),
                    'min': round(float(np.min(vals)), 4),
                    'max': round(float(np.max(vals)), 4),
                }

        cluster_infos.append({
            'classification': state_to_cls[state],
            'state': state,  # String for WebUI convenience
            'note': note_num,
            'note_label': state.title(),  # 'Open' / 'Closed'
            'count': len(cluster_events),
            'features': feature_stats,
        })

    # Annotate distinguishing features (works on integer classification
    # like the k-means path; pass features list directly).
    _annotate_distinguishing_features(cluster_infos, features_to_check)
    return cluster_infos


def _annotate_distinguishing_features(
    cluster_infos: List[Dict],
    features: List[str],
) -> None:
    """
    Annotate each cluster with the feature that most separates it from others.

    Uses the ratio of between-cluster to within-cluster variance for each
    feature. The feature with the highest ratio is the best discriminator.
    Also generates a human-readable description per cluster.
    """
    if len(cluster_infos) <= 1:
        for info in cluster_infos:
            info['distinguishing_feature'] = features[0] if features else None
            info['distinguishing_label'] = _FEATURE_LABELS.get(
                features[0], features[0]) if features else None
            info['description'] = f'{info["count"]} events'
        return

    # Find best discriminating feature across all clusters
    best_feature = features[0]
    best_separation = 0.0

    for feat in features:
        means = [
            info['features'][feat]['mean']
            for info in cluster_infos
            if feat in info['features']
        ]
        if len(means) < 2:
            continue

        # Separation = range of means / average within-cluster range
        mean_range = max(means) - min(means)
        within_ranges = []
        for info in cluster_infos:
            if feat in info['features']:
                s = info['features'][feat]
                within_ranges.append(s['max'] - s['min'])
        avg_within = np.mean(within_ranges) if within_ranges else 1e-9
        separation = mean_range / (avg_within + 1e-9)

        if separation > best_separation:
            best_separation = separation
            best_feature = feat

    # Annotate each cluster
    feat_label = _FEATURE_LABELS.get(best_feature, best_feature)
    for info in cluster_infos:
        info['distinguishing_feature'] = best_feature
        info['distinguishing_label'] = feat_label

        # Generate description based on relative position
        if best_feature in info['features']:
            mean_val = info['features'][best_feature]['mean']
            all_means = [
                ci['features'][best_feature]['mean']
                for ci in cluster_infos
                if best_feature in ci['features']
            ]
            all_means_sorted = sorted(all_means)
            rank = all_means_sorted.index(mean_val)
            total = len(all_means_sorted)

            info['description'] = _describe_cluster(
                best_feature, mean_val, rank, total,
            )
        else:
            info['description'] = f'{info["count"]} events'


def _describe_cluster(
    feature: str,
    mean_val: float,
    rank: int,
    total: int,
) -> str:
    """Generate a human-readable cluster description."""
    if total <= 1:
        return f'{_FEATURE_LABELS.get(feature, feature)}: {mean_val:.3f}'

    position = rank / (total - 1)  # 0.0 = lowest, 1.0 = highest

    descriptors = {
        'stereo_width': {
            0.0: 'Narrow (mono)',
            0.5: 'Medium width',
            1.0: 'Wide (stereo)',
        },
        'pan_confidence': {
            0.0: 'Panned left',
            0.5: 'Center',
            1.0: 'Panned right',
        },
        'spectral_centroid_hz': {
            0.0: 'Low pitch',
            0.5: 'Mid pitch',
            1.0: 'High pitch',
        },
        'geomean': {
            0.0: 'Low energy',
            0.5: 'Medium energy',
            1.0: 'High energy',
        },
        'duration_sec': {
            0.0: 'Short',
            0.5: 'Medium length',
            1.0: 'Long',
        },
    }

    feat_descriptors = descriptors.get(feature, {
        0.0: 'Low', 0.5: 'Medium', 1.0: 'High',
    })

    # Pick closest descriptor
    if position <= 0.25:
        desc = feat_descriptors[0.0]
    elif position >= 0.75:
        desc = feat_descriptors[1.0]
    else:
        desc = feat_descriptors[0.5]

    return desc


# ============================================================================
# Main Entry Point
# ============================================================================


def classify_notes(
    events: List[Dict],
    stem_type: str,
    drum_mapping,
    config: Dict,
    force_reclassify: bool = False,
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
        force_reclassify: If True, ignore any stored hihat_state/classification
            and recompute from current thresholds.  Default False preserves
            stored classification across rebuilds (see hihat bug A4 and the
            'hihat_classification_unchanged' event_overrides semantics).

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
        classify_hihat_notes(events, config, force_reclassify=force_reclassify)
    elif stem_type == 'toms':
        classify_tom_notes(events, config, force_reclassify=force_reclassify)
    elif stem_type == 'cymbals':
        classify_cymbal_notes(events, config, force_reclassify=force_reclassify)
    elif stem_type == 'snare':
        classify_snare_notes(events, config, force_reclassify=force_reclassify)

    # Map classification to MIDI note numbers
    for event in events:
        event['note'] = _map_note(event, stem_type, drum_mapping, config)

    return events
