"""
MIDI File Operations Module

Handles creation and reading of MIDI files for drum transcription.
Includes JSON sidecar export for spectral analysis data (Detection Output Contract).
"""

from midiutil import MIDIFile
import mido
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional

# Import helper function for event preparation
from .analysis_core import prepare_midi_events_for_writing

# 2026-06-19: hihat open/closed classifier (used by save_analysis_sidecar
# to stamp hihat_state on every hihat event so the sidecar and MIDI rule
# both consume the same signal. Driven by PGA broadband-envelope decay
# slope (decay_slope_db); falls back to geomean+sustain for older sidecars).
from .note_classification_core import classify_hihat_notes

# Import contract for validation
try:
    pass
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

__all__ = [
    'create_midi_file',
    'read_midi_notes',
    'save_analysis_sidecar',
    'load_analysis_sidecar',
    'save_envelope_data',
    'load_envelope_data',
    'save_contrast_envelope',
    'load_contrast_envelope',
]


def create_midi_file(
    events_by_stem: Dict[str, List[Dict]],
    output_path: Union[str, Path],
    tempo: float = 120.0,
    track_name: str = "Drums",
    config: Optional[Dict] = None
):
    """
    Create a MIDI file from detected drum events.
    
    Uses midiutil to write MIDI - stores note times as BEATS.
    With tempo=120 BPM, this preserves original timing when DAW reads it.
    
    Args:
        events_by_stem: Dictionary mapping stem names to lists of MIDI events
        output_path: Path to save MIDI file
        tempo: Tempo in BPM
        track_name: Name of the MIDI track
        config: Configuration dictionary (optional, loads default if not provided)
    """
    # Import here to avoid circular dependency
    from .config import load_config
    
    # Load config if not provided
    if config is None:
        config = load_config()
    
    # Create MIDI file with 1 track (midiutil)
    midi = MIDIFile(1)
    track = 0
    channel = 9  # Channel 10 (0-indexed as 9) is typically drums in MIDI
    time = 0
    
    midi.addTrackName(track, time, track_name)
    midi.addTempo(track, time, tempo)
    
    # Add a marker/text event at time 0 to anchor the MIDI file
    midi.addText(track, 0.0, "START")
    
    # Also add a very quiet anchor note at time 0 (velocity 1, not 0)
    very_short_duration = config.get('audio', {}).get('very_short_duration', 0.01)
    midi.addNote(
        track=track,
        channel=9,
        pitch=27,  # Very low note (outside typical drum range)
        time=0.0,  # At the very start
        duration=very_short_duration,  # Very short (beats)
        volume=1  # Very quiet but not silent (velocity 1)
    )
    
    # Prepare all events (convert times to beats using pure function)
    print(f"  DEBUG create_midi_file: events_by_stem keys={list(events_by_stem.keys())}")
    for _stem, _evs in events_by_stem.items():
        print(f"    {_stem}: {len(_evs)} events")
        for _i, _e in enumerate(_evs[:3]):
            print(f"      [{_i}] {_e}")
    prepared_events = prepare_midi_events_for_writing(events_by_stem, tempo)
    print(f"  DEBUG create_midi_file: prepared_events={len(prepared_events)}")
    for _p in prepared_events[:3]:
        print(f"      prepared: {_p}")
    
    # Add all prepared events to MIDI file
    for event in prepared_events:
        midi.addNote(
            track=track,
            channel=channel,
            pitch=event['note'],
            time=event['time_beats'],
            duration=event['duration_beats'],
            volume=event['velocity']
        )
    
    total_events = len(prepared_events)
    
    # Write to file
    with open(output_path, 'wb') as f:
        midi.writeFile(f)
    
    print(f"  Created MIDI file with {total_events} notes")


def read_midi_notes(midi_path: Union[str, Path], target_note: int) -> List[float]:
    """
    Read note times from a MIDI file for a specific MIDI note number.
    
    Args:
        midi_path: Path to MIDI file
        target_note: MIDI note number to extract (e.g., 38 for snare)
    
    Returns:
        List of note times in seconds
    """
    midi_file = mido.MidiFile(str(midi_path))
    note_times = []
    current_time = 0.0
    
    # Get ticks per beat for time conversion
    ticks_per_beat = midi_file.ticks_per_beat
    tempo = 500000  # Default tempo (120 BPM in microseconds per beat)
    
    for track in midi_file.tracks:
        current_time = 0.0
        for msg in track:
            current_time += mido.tick2second(msg.time, ticks_per_beat, tempo)
            
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'note_on' and msg.note == target_note and msg.velocity > 0:
                note_times.append(current_time)
    
    return sorted(note_times)


def _round_value(value, decimals: int):
    """Round numeric value to specified decimals, handle None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return round(value, decimals)
    return value


def _round_significant(value, sig_figs: int):
    """Round a float to N significant figures, then return as float.

    Unlike _round_value (which uses fixed decimal places), this
    preserves small-magnitude values like 4.35e-7 — important for
    the spectral snap_to_ring_ratio / snap_to_top_ratio fields
    whose real-data range is 1e-7 to 1e-2. Fixed 6-decimal
    rounding on those would zero out the small end. Used
    selectively for the spectral ratio fields (2026-06-10)."""
    if value is None or not isinstance(value, (int, float)):
        return value
    if value == 0:
        return 0.0
    import math
    magnitude = math.floor(math.log10(abs(value)))
    decimals = sig_figs - 1 - int(magnitude)
    return round(value, decimals)


def _serialize_onset_events(
    onset_data_list: list,
    midi_events: Optional[List[Dict]] = None,
) -> list:
    """
    Serialize onset data dicts into rounded JSON-ready event dicts.

    Args:
        onset_data_list: List of onset dicts from filter_onsets_by_spectral
        midi_events: Optional list of MIDI events to attach note/velocity to KEPT onsets.
                     Should exclude foot-close events (note 44).

    Returns:
        List of serialized event dicts with rounded numeric values.

    Notes:
        - ``duration_sec``, ``amplitude_at_start``, ``amplitude_at_end``,
          ``attack_sharpness``, ``envelope_continuity``, ``peak_prominence``,
          ``spectral_centroid_hz``, ``spectral_flux``, and
          ``gap_from_previous_sec`` are omitted when missing (older analyses
          may not have Phase 2 metadata at all).
        - ``pitch_hz``, ``pan_confidence``, and ``stereo_width`` are always
          written, with ``null`` when the underlying feature is not
          applicable (e.g. mono audio, pitch detection disabled, kick stem
          without pan). This guarantees a stable schema for downstream
          consumers (WebUI tuning panel, cluster features, JSON-driven
          analysis scripts).
    """
    events = []
    midi_idx = 0

    # Fields that are stem-relevant and should always be present in the
    # JSON (with null when missing). Bug B: pan_confidence/pitch_hz/
    # stereo_width are computed in the pipeline and must surface in the
    # sidecar even when the value happens to be 0.0 or None.
    # ``method`` is also always-present: it carries 'rms' / 'peak_hold' /
    # 'spectral_flux' (energy-detected) or 'spectral' (spectral-detected
    # survivor in method='both' or full method='spectral' projects). The
    # WebUI waveform viewer reads this key to color spectral survivors
    # magenta for the A/B-comparison overlay; the key MUST be present on
    # every event (with null when the pipeline didn't stamp it) so the
    # viewer doesn't have to feature-detect the field per record.
    ALWAYS_PRESENT_FIELDS = ('pan_confidence', 'stereo_width', 'pitch_hz', 'method')

    # Fields that are present-or-absent (older analyses may lack them).
    OPTIONAL_PHASE2_FIELDS = (
        'duration_sec', 'amplitude_at_start', 'amplitude_at_end',
        'attack_sharpness', 'envelope_continuity', 'peak_prominence',
        'spectral_centroid_hz', 'spectral_flux',
        'gap_from_previous_sec',
        # Toms PGA cleanup (2026-06-11 / 2026-06-12). These live on
        # PGA events (method='percentile_gated'). The
        # events_configured list for toms is the PGA list,
        # so these fields are always present-or-absent
        # there. midi_velocity is the integer that landed
        # in the MIDI file (per-file linear scale of the
        # PGA envelope_value into [min_velocity,
        # max_velocity]). filter_reason is a human-readable
        # explanation of why the event was dropped (e.g.
        # "below pga_min_prominence (800 < 1000)").
        # pga_filter_config (the per-event active-filter
        # dict) is NOT serialized to the sidecar — the
        # threshold is a config concern (yaml), not a
        # sidecar concern (output), so the sidecar
        # carries ONLY the per-event consequences of
        # the filter: status + filter_reason. Consumers
        # that need the active threshold re-read
        # midiconfig.yaml. The in-memory event still
        # carries pga_filter_config for any in-process
        # consumer that wants the per-event value.
        #
        # PGA per-event features (2026-06-12 bug fix). These
        # were attached in-memory by compute_event_features()
        # at the end of process_stem_to_midi() but were being
        # DROPPED at this serializer because they weren't in
        # the OPTIONAL list. The WebUI tooltip was showing
        # only the legacy event fields (Centroid, MIDI
        # velocity, Status, etc.) and missing the per-event
        # feature battery. Adding all the PGA per-event
        # fields here so they survive to the sidecar.
        #
        # The 4 PGA detector fields (frame, envelope_value,
        # prominence, iqr_threshold) attached directly in
        # processing_shell.py at the per-event dict build
        # (around line 1745-1751) were ALSO missing until
        # 2026-06-12. They live on the same set of PGA
        # events and the WebUI tooltip reads them at the
        # top of the PGA block.
        'midi_velocity', 'filter_reason',
        'duration_ms', 'attack_rise_ms', 'inter_onset_ms',
        'pitch_hz', 'pitch_confidence',
        'decay_t60_ms', 'spectral_flatness',
        'hr_peak_offset_ms', 'decay_envelope_energy',
        'decay_col_min_median_db',
        'frame', 'envelope_value', 'prominence', 'iqr_threshold',
        'pga_filter_config',
    )

    for onset_data in onset_data_list:
        event = {
            'time': _round_value(onset_data.get('time'), 4),
            'status': onset_data.get('status', 'UNKNOWN')
        }

        # Add spectral features with rounding
        # Band energy fields are dynamic per stem (e.g., body_energy, wire_energy)
        band_fields = [f'{b}_energy' for b in onset_data.get('geomean_bands', [])]
        # 2026-06-10: snap_delta and band_delta need HIGHER precision
        # than the default 2dp because the real signal is often in the
        # 0.0001-0.001 range (e.g. the user's calibration: snap_delta
        # values 0.000014 to 0.0004 round to 0.00 at 2dp, which
        # destroys the discriminator the user was tracking by). Use
        # 6-decimal fixed-point rounding for these.
        #
        # The derived ratios (snap_to_ring_ratio, snap_to_top_ratio)
        # need SIGNIFICANT-FIGURE rounding (6 sig figs) instead of
        # fixed-point — the real-data range is 1e-7 to 1e-2, so
        # fixed-point 6dp still zeros out the small end. 6 sig figs
        # preserves 4.35e-7 as 4.35e-7.
        #
        # band_max_ratio is unbounded above (the user's case had 459)
        # so 4dp fixed-point is the right balance: distinguishes
        # 18.99 from 459.12 in the JSON without bloating the file.
        HIGH_PRECISION_FIELDS = {'snap_delta', 'band_delta'}
        SIG_FIG_FIELDS = {'snap_to_ring_ratio', 'snap_to_top_ratio'}
        for field in (['strength', 'amplitude']
                      + band_fields
                      + ['geomean', 'total_energy', 'sustain_ms',
                         'bins_above_floor', 'max_db',
                         'band_max_idx', 'band_max_ratio',
                         'band_delta', 'snap_delta',
                         'snap_to_ring_ratio', 'snap_to_top_ratio']):
            value = onset_data.get(field)
            if value is not None:
                # bins_above_floor is an int count; round to int.
                # band_max_idx is also an int (0-4).
                if field in ('bins_above_floor', 'band_max_idx'):
                    event[field] = int(round(value))
                elif field in HIGH_PRECISION_FIELDS:
                    event[field] = _round_value(value, 6)
                elif field in SIG_FIG_FIELDS:
                    event[field] = _round_significant(value, 6)
                elif field == 'band_max_ratio':
                    event[field] = _round_value(value, 4)
                else:
                    event[field] = _round_value(value, 2)
        # band_powers is a list of 5 floats — serialize with 6-decimal
        # precision (sub-band power sums are small on quiet hits).
        bp_raw = onset_data.get('band_powers')
        if bp_raw is not None:
            event['band_powers'] = [
                _round_value(float(x), 6) for x in bp_raw
            ]

        # Optional Phase 2 metadata — present when the upstream pipeline
        # computed them, omitted otherwise.
        for field in OPTIONAL_PHASE2_FIELDS:
            value = onset_data.get(field)
            if value is not None:
                event[field] = _round_value(value, 4)

        # 2026-06-29: dynamic passthrough for per-event diagnostic
        # fields NOT in the OPTIONAL_PHASE2_FIELDS allowlist above.
        # Previously, every new per-event field required editing
        # this serializer (e.g. ``hihat_openness_score`` and the
        # KMeans classifier's ``hihat_kmeans_*`` fields silently
        # disappeared into the sidecar). Any key the explicit
        # passes already wrote is skipped; everything else survives
        # at 4-decimal rounding, matching the OPTIONAL convention.
        # Numpy 0-dim scalars get coerced via .item(); multi-dim
        # arrays are still skipped (the KMeans feature vector is
        # 1-D and would need explicit .tolist() — left for whoever
        # actually wants to inspect it post-hoc).
        for k, v in onset_data.items():
            if k in event:
                continue
            if v is None:
                continue
            if isinstance(k, str) and k.startswith('_'):
                continue  # private — should not land in the sidecar
            # Coerce numpy 0-dim scalars to native Python.
            if hasattr(v, 'item') and callable(v.item):
                try:
                    v = v.item()
                except (ValueError, TypeError):
                    continue
            if isinstance(v, bool):
                event[k] = v
            elif isinstance(v, (int, float)):
                event[k] = _round_value(v, 4)
            elif isinstance(v, str):
                event[k] = v
            elif isinstance(v, (list, dict)):
                event[k] = v
            # else: skip (n-dim arrays, custom objects, etc.)

        # Stem-relevant features — always present, with null when missing
        # so downstream consumers (WebUI, scripts) can rely on the key
        # existing in every event.
        for field in ALWAYS_PRESENT_FIELDS:
            value = onset_data.get(field)
            event[field] = _round_value(value, 4) if value is not None else None

        # Add MIDI fields for KEPT events (from midi_events by index)
        if midi_events is not None and event['status'] == 'KEPT':
            if midi_idx < len(midi_events):
                event['note'] = midi_events[midi_idx].get('note')
                event['velocity'] = midi_events[midi_idx].get('velocity')
                classification = midi_events[midi_idx].get('classification')
                if classification is not None:
                    event['classification'] = classification
                # T2 follow-up (2026-06-08): the hihat_state field is
                # only set on the in-memory event dict by
                # classify_hihat_notes. _serialize_onset_events used to
                # drop it on the way to the JSON sidecar, which meant
                # T2 A4's "preserve hihat_state on rebuild" was a no-op
                # for the initial-conversion case (the field was never
                # written to begin with). T3 e2e found this:
                # "hihat_state field missing from all 13 hihat KEPT
                # events in fresh conversion (baseline had 13/13)".
                hihat_state = midi_events[midi_idx].get('hihat_state')
                if hihat_state is not None:
                    event['hihat_state'] = hihat_state
                midi_idx += 1

        events.append(event)

    return events


def _serialize_spectral_events(spectral_events: list) -> list:
    """
    2026-06-20: function stub retained for one release to keep the
    public surface stable. Returns an empty list because the
    spectral-transient detector is no longer run by the main
    pipeline (PGA is universal). All callers that previously read
    events_spectral from the sidecar have been updated to no
    longer expect the key. Phase 7 will hard-delete this stub.
    """
    return []


def _serialize_pga_events(pga_events: list) -> list:
    """
    Serialize percentile-gated broad-attack events for the
    analysis.json sidecar (2026-06-10; updated 2026-06-15).

    After the 2026-06-15 refactor, ``events_pga`` in the sidecar
    carries ALL detected events with ``status='KEPT'`` — no filter
    is applied at detect time. The prominence filter is re-applied
    by the WebUI (client-side live preview) and by rebuild_core
    (server-side Save & Reconvert) using
    ``apply_pga_prominence_filter()``.

    Each event has these fields (all may be None)::

        {
            'time': float,            # 4 decimal places
            'method': 'percentile_gated',
            'status': 'KEPT',
            # Detector diagnostic (why the peak fired)
            'frame': int,             # STFT frame index
            'envelope_value': float,  # contrast envelope at peak
            'prominence': float,      # scipy find_peaks prominence
            'iqr_threshold': float,   # peak-pick threshold (q3 + 2.5*IQR)
            # Per-event features (for classification)
            'duration_ms': float,     # ring time via slope-of-decline
            'attack_rise_ms': float,  # 10-90% rise time
            'pitch_hz': float,        # YIN/pYIN fundamental
            'pitch_confidence': float,  # 0-1
            'decay_t60_ms': float,    # T60 in the broad band
            'spectral_centroid_hz': float,  # brightness
            'spectral_flatness': float,  # 0-1 attack-region flatness
            'hr_peak_offset_ms': float,  # PGA-vs-high-res timing offset
            'decay_envelope_energy': float,  # 15ms post-peak ring energy
            'decay_col_min_median_db': float,  # 15ms post-peak col_min
            'inter_onset_ms': float,  # time to next event
        }
    """
    out = []
    for ev in pga_events:
        out.append({
            'time': _round_value(ev.get('time'), 4),
            'method': ev.get('method', 'percentile_gated'),
            'status': ev.get('status', 'KEPT'),
            # Detector diagnostic
            'frame': ev.get('frame'),
            'envelope_value': _round_value(ev.get('envelope_value'), 4),
            'prominence': _round_value(ev.get('prominence'), 4),
            'iqr_threshold': _round_value(ev.get('iqr_threshold'), 4),
            # Per-event features — preserve None if absent,
            # 4-decimal rounding for floats in the user-facing
            # range (durations, pitches, frequencies in Hz).
            'duration_ms': _round_value(ev.get('duration_ms'), 2),
            'attack_rise_ms': _round_value(ev.get('attack_rise_ms'), 2),
            'pitch_hz': _round_value(ev.get('pitch_hz'), 2),
            'pitch_confidence': _round_value(ev.get('pitch_confidence'), 4),
            'decay_t60_ms': _round_value(ev.get('decay_t60_ms'), 2),
            'spectral_centroid_hz': _round_value(ev.get('spectral_centroid_hz'), 2),
            'spectral_flatness': _round_value(ev.get('spectral_flatness'), 4),
            'hr_peak_offset_ms': _round_value(ev.get('hr_peak_offset_ms'), 2),
            'decay_envelope_energy': _round_value(ev.get('decay_envelope_energy'), 2),
            'decay_col_min_median_db': _round_value(ev.get('decay_col_min_median_db'), 2),
            'inter_onset_ms': _round_value(ev.get('inter_onset_ms'), 2),
            # Peak bases (2026-06-19): STFT-frame indices of the
            # left/right valley around each peak, plus the
            # right-base-minus-peak gap in frames and ms. The
            # gap is the candidate open/closed hihat
            # discriminator. Diagnostic only — not used by any
            # classifier yet.
            'left_base_frame': ev.get('left_base_frame'),
            'right_base_frame': ev.get('right_base_frame'),
            'right_base_minus_peak_frames': ev.get('right_base_minus_peak_frames'),
            'right_base_minus_peak_ms': _round_value(
                ev.get('right_base_minus_peak_ms'), 2
            ),
            # Peak widths (2026-06-19): scipy.peak_widths at
            # rel_height=0.9. Bounded to a 10% slice around the
            # peak. left_ips/right_ips are floating-point frame
            # indices; attack_frames/decay_frames are the
            # per-event split. For hihat open vs closed,
            # decay_ms is the candidate discriminator.
            'peak_width_left_ip_frame': _round_value(
                ev.get('peak_width_left_ip_frame'), 4
            ),
            'peak_width_right_ip_frame': _round_value(
                ev.get('peak_width_right_ip_frame'), 4
            ),
            'attack_frames': _round_value(ev.get('attack_frames'), 4),
            'decay_frames': _round_value(ev.get('decay_frames'), 4),
            'attack_ms': _round_value(ev.get('attack_ms'), 2),
            'decay_ms': _round_value(ev.get('decay_ms'), 2),
            # Toms cleanup (2026-06-11): filter metadata.
            # FILTERED events are kept in the sidecar so the
            # WebUI can render them faded; the MIDI output
            # skips them. The active prominence threshold is
            # NOT written per-event here — it lives on the
            # events_pga dict as the top-level
            # 'pga_min_prominence' field (set in
            # save_analysis_sidecar from analysis.pga_min_prominence
            # and read in load_analysis_sidecar). 2026-06-13
            # split: source of truth moved out of the
            # per-event payload.
            'midi_velocity': ev.get('midi_velocity'),
            'pga_filter_config': ev.get('pga_filter_config'),
            'filter_reason': ev.get('filter_reason'),
            # 2026-06-19: per-event broadband-envelope walk
            # (open/closed hihat discriminator). The PGA detector
            # computed a contrast envelope for the whole stem;
            # we walk it forward and backward from each event's
            # peak frame and stamp these fields. The hihat
            # classifier (note_classification_core) reads
            # ``decay_slope_db`` against
            # ``hihat.open_decay_slope_max`` to decide
            # open vs closed. The other fields are diagnostic
            # surface for the WebUI tooltip.
            'decay_slope_db': _round_value(ev.get('decay_slope_db'), 4),
            'decay_slope_linear': _round_value(ev.get('decay_slope_linear'), 4),
            'decay_frames_walked': ev.get('decay_frames_walked'),
            'decay_pct_at_stop': _round_value(ev.get('decay_pct_at_stop'), 4),
            'decay_stop_reason': ev.get('decay_stop_reason'),
            # 2026-06-19: hihat open/closed classification (set by
            # classify_hihat_notes above for the hihat stem only).
            # WebUI tooltips + MIDI rule both read this. 'open' /
            # 'closed' / None (no rule fired for non-hihat stems).
            'hihat_state': ev.get('hihat_state'),
            'onset_crossed': ev.get('onset_crossed'),
            'onset_cross_ms': _round_value(ev.get('onset_cross_ms'), 2),
            # 2026-06-26: per-event Δ1/Δ2/Δ5 stability ratios
            # + combined warble score. Computed in
            # percentile_gated_detector.py and surfaced into
            # pga_onset_data by detect_pga_events; this
            # serializer is the canonical place that defines
            # the sidecar's per-event field set, so they
            # belong here too. -1.0 sentinel preserved as-is
            # (means "undefined" — no forward window or zero
            # peak). Used by the WebUI's warble-robustness
            # filter (real hits have positive combined_score,
            # warble FPs have negative). Rounded to 4 dp
            # to match prominence / iqr_threshold.
            'delta1_stability': _round_value(ev.get('delta1_stability'), 4),
            'delta2_stability': _round_value(ev.get('delta2_stability'), 4),
            'delta5_stability': _round_value(ev.get('delta5_stability'), 4),
            'combined_score': _round_value(ev.get('combined_score'), 4),
        })
        # 2026-06-29: dynamic passthrough for per-event fields NOT
        # in the explicit dict above. Avoids allowlist maintenance
        # for every new diagnostic field (hihat_openness_score,
        # hihat_kmeans_*, future additions). Anything the explicit
        # pass already wrote is skipped; everything else lands at
        # 4-decimal rounding (matches the OPTIONAL convention used
        # by _serialize_onset_events). Keys starting with '_'
        # (private) and None values are dropped. n-dim numpy
        # arrays (e.g. the 3-element KMeans feature vector) still
        # need explicit .tolist() in their callers if they want
        # to round-trip — left for whoever actually needs that.
        for k, v in ev.items():
            if k in out[-1]:
                continue
            if v is None:
                continue
            if isinstance(k, str) and k.startswith('_'):
                continue
            if hasattr(v, 'item') and callable(v.item):
                try:
                    v = v.item()
                except (ValueError, TypeError):
                    continue
            if isinstance(v, bool):
                out[-1][k] = v
            elif isinstance(v, (int, float)):
                out[-1][k] = _round_value(v, 4)
            elif isinstance(v, str):
                out[-1][k] = v
            elif isinstance(v, (list, dict)):
                out[-1][k] = v
            # else: skip (n-dim arrays, custom objects, etc.)
    return out


def save_analysis_sidecar(
    events_by_stem: Dict[str, List[Dict]],
    midi_path: Union[str, Path],
    tempo: float = 120.0,
    analysis_by_stem: Optional[Dict[str, Dict]] = None,
    config: Optional[Dict] = None,
) -> Path:
    """
    Save spectral analysis data as JSON sidecar file (v3 format).

    V3 Format:
        - Logic block per stem (thresholds, passes)
        - events_configured: All onsets from configured detection (KEPT + FILTERED)
        - events_sensitive: All onsets from max-sensitivity detection (for interactive tuning)
        - events_spectral: REMOVED 2026-06-20 — see Phase 5 of
          agent-plans/pga-cleanup-2026-06.plan.md
          (complementary signal, always computed alongside the energy
          detector so the WebUI can compare both candidate lists)
        - Numeric precision: times=4 decimals, features=2 decimals

    Args:
        events_by_stem: Dictionary mapping stem names to lists of MIDI events
        midi_path: Path to corresponding MIDI file (sidecar uses same name + .analysis.json)
        tempo: Tempo in BPM (for reference)
        analysis_by_stem: Dict with all_onset_data, sensitive_onset_data,
                          spectral_onset_data, and spectral_config per stem

    Returns:
        Path to created sidecar file
    """
    midi_path = Path(midi_path)
    sidecar_path = midi_path.with_suffix('.analysis.json')

    sidecar_data = {
        'version': '3.0',
        'tempo_bpm': round(tempo, 1),
        'stems': {}
    }

    total_configured = 0
    total_filtered = 0
    total_sensitive = 0

    for stem_type, events in events_by_stem.items():
        # Get analysis data for this stem
        analysis = analysis_by_stem.get(stem_type, {}) if analysis_by_stem else {}
        all_onset_data = analysis.get('all_onset_data', [])
        sensitive_onset_data = analysis.get('sensitive_onset_data', [])
        spectral_config = analysis.get('spectral_config')

        # 2026-06-20: the sidecar's `logic` block was a stale snapshot of
        # the old energy/spectral tuning knobs (geomean_threshold,
        # min_sustain_ms, decay_filter_enabled, statistical_enabled,
        # reverb_continuation_attack_threshold, open_geomean_min,
        # open_sustain_ms, expected_clusters, cluster_feature).
        # Those knobs no longer exist in midiconfig.yaml (Phase 2) or
        # the WebUI schema (Phase 3), and the pipeline that wrote them
        # was hard-deleted (Phase 1+7). The ONLY live knob on the
        # PGA path is `pga_min_prominence` (per-stem wins over global
        # onset_detection.pga_min_prominence).
        logic = {}
        if config:
            onset_config = config.get('onset_detection', {})
            global_pga = onset_config.get('pga_min_prominence')
            stem_pga = config.get(stem_type, {}).get('pga_min_prominence')
            if stem_pga is not None:
                logic['pga_min_prominence'] = stem_pga
            elif global_pga is not None:
                logic['pga_min_prominence'] = global_pga

        # Serialize configured events (KEPT + FILTERED from configured detection)
        # The processing_shell pipeline may have already prebuilt
        # events_configured based on the configured detection_method
        # (energy / spectral / both). When present, that list is the
        # source of truth — the sidecar must reflect the user-chosen
        # promotion (see webui/settings_schema.detection_method). When
        # absent (older code paths that bypass process_stem_to_midi),
        # fall back to building from all_onset_data + midi_events.
        prebuilt_configured = analysis.get('events_configured')
        if stem_type == 'toms':
            # Toms (2026-06-15): events_pga is the sole source of truth.
            # events_configured is absent for toms.
            configured_events = []
        elif prebuilt_configured is not None:
            # The prebuilt list is a flat list of onset-shaped dicts.
            # KEPT events still need note/velocity from midi_events;
            # the serializer handles that by KEPT-index.
            midi_events = [e for e in events if e.get('note') != 44]  # Exclude foot-close
            configured_events = _serialize_onset_events(
                prebuilt_configured, midi_events=midi_events,
            )
        elif all_onset_data:
            midi_events = [e for e in events if e.get('note') != 44]  # Exclude foot-close
            configured_events = _serialize_onset_events(all_onset_data, midi_events=midi_events)
        else:
            # Fallback: use events_by_stem directly if no all_onset_data.
            # Mirrors the always-present fields contract from the primary
            # serializer (pan_confidence / stereo_width / pitch_hz always
            # present, with null when missing).
            configured_events = []
            for midi_event in events:
                event = {
                    'time': _round_value(midi_event.get('time'), 4),
                    'note': midi_event.get('note'),
                    'velocity': midi_event.get('velocity'),
                    'status': 'KEPT',
                }
                band_fields = [f'{b}_energy' for b in midi_event.get('geomean_bands', [])]
                for field in ['onset_strength', 'peak_amplitude'] + band_fields + ['geomean',
                             'total_energy', 'sustain_ms']:
                    value = midi_event.get(field)
                    if value is not None:
                        event[field] = _round_value(value, 2)
                # Bug B: always present (null when missing)
                for field in ('pan_confidence', 'stereo_width', 'pitch_hz'):
                    value = midi_event.get(field)
                    event[field] = _round_value(value, 4) if value is not None else None
                configured_events.append(event)

        # Serialize sensitive events (all from max-sensitivity detection)
        sensitive_events = _serialize_onset_events(sensitive_onset_data) if sensitive_onset_data else []

        # 2026-06-20: spectral-transient events no longer serialized
        # to the sidecar. PGA is the only detector that runs in the
        # main pipeline now, and the WebUI removed its spectral
        # overlay in Phase 0.5. The events_spectral key is dropped
        # from the sidecar shape.

        # Serialize percentile-gated broad-attack events (2026-06-10).
        # PGA events have a simple shape — time, method, status. The
        # WebUI waveform viewer uses them as another color in the
        # marker stack, showing where the broadband percussive attack
        # fired independent of the energy/RING signal. PGA is the
        # THIRD complementary detector; it runs alongside energy +
        # spectral but isn't part of the 'promoted to configured'
        # pipeline.
        #
        # Sidecar content contract (2026-06-13): the sidecar
        # carries ONLY the per-event consequences of the PGA
        # prominence filter — ``status`` (KEPT/FILTERED) and
        # ``filter_reason`` (human-readable explanation). The
        # ACTIVE THRESHOLD is a config concern (yaml), not a
        # sidecar concern (output), so we do NOT persist it
        # here. Any consumer that needs the threshold re-reads
        # midiconfig.yaml. This is the
        # "yaml = config, sidecar = output" architecture rule.
        pga_onset_data = analysis.get('pga_onset_data', [])

        # 2026-06-19: Stamp hihat_state on every hihat event so the
        # sidecar and the MIDI rule consume the same signal. The
        # classifier runs on both lists (events_configured and
        # pga_onset_data) because the sidecar persists them as
        # separate arrays but they are two views of the same physical
        # hihat hits. Mutates in place — both arrays point to the
        # same dicts in pga_event_builder's output, so classifying
        # one and then the other is idempotent.
        if stem_type == 'hihat' and config is not None and pga_onset_data:
            classify_hihat_notes(pga_onset_data, config, force_reclassify=True)
        if stem_type == 'hihat' and config is not None and configured_events:
            classify_hihat_notes(configured_events, config, force_reclassify=True)

        pga_events = _serialize_pga_events(pga_onset_data) if pga_onset_data else []

        # Count totals
        total_configured += len(configured_events)
        total_filtered += sum(1 for e in configured_events if e.get('status') == 'FILTERED')
        total_sensitive += len(sensitive_events)
        total_pga_events = len(pga_events)
        # Assemble stem data. The stem block shape is
        # stable: ``logic``, ``events_configured``,
        # ``events_sensitive``,
        # ``events_pga``. No top-level threshold fields.
        # (events_spectral was removed 2026-06-20.)
        # Build stem dict — only include event arrays when non-empty.
        # Toms (2026-06-15): events_configured and events_sensitive
        # are absent for toms; events_pga is the sole source of truth.
        stem_dict = {}
        if logic:
            stem_dict['logic'] = logic
        if pga_events:
            stem_dict['events_pga'] = pga_events
        if configured_events:
            stem_dict['events_configured'] = configured_events
        if sensitive_events:
            stem_dict['events_sensitive'] = sensitive_events
        # Toms (2026-06-15): events_pga is the sole source of truth.
        # Drop empty arrays so the keys are absent from the sidecar JSON.
        if not configured_events:
            stem_dict.pop('events_configured', None)
        if not sensitive_events:
            stem_dict.pop('events_sensitive', None)
        sidecar_data['stems'][stem_type] = stem_dict

    # Write JSON
    with open(sidecar_path, 'w') as f:
        json.dump(sidecar_data, f, indent=2)

    print(f"  Saved analysis sidecar v3: {sidecar_path.name} "
          f"({total_pga_events} PGA events, {total_configured} configured events, {total_filtered} filtered, "
          f"{total_sensitive} sensitive events)")

    return sidecar_path


def load_analysis_sidecar(midi_path: Union[str, Path]) -> Optional[Dict]:
    """
    Load spectral analysis data from JSON sidecar file.

    Validates that every event in events_configured has a time within
    1ms of an event in events_sensitive (bug C — events_configured
    must be a subset of events_sensitive by time, since the configured
    detection is a stricter subset of the max-sensitivity detection).
    If any event violates this invariant, a warning is attached to
    the returned data so the WebUI can surface a toast notification.
    We do NOT silently fix the data — the user may have hand-edited
    the MIDI or YAML and we should not lose information.

    Args:
        midi_path: Path to MIDI file (will look for .analysis.json sidecar)

    Returns:
        Sidecar data dict, or None if not found. The dict may have a
        'data_integrity_warnings' top-level list with string messages
        when invariant violations are detected.
    """
    midi_path = Path(midi_path)
    sidecar_path = midi_path.with_suffix('.analysis.json')

    if not sidecar_path.exists():
        return None

    with open(sidecar_path, 'r') as f:
        data = json.load(f)

    # Validate events_configured ⊆ events_sensitive by time (within 1ms).
    warnings = _validate_events_subset(data)
    if warnings:
        data.setdefault('data_integrity_warnings', []).extend(warnings)

    return data


def _validate_events_subset(data: Dict, time_tolerance_sec: float = 0.012) -> List[str]:
    """
    Check that every event in events_configured has a matching time
    (within ``time_tolerance_sec``) in events_sensitive for the same stem.

    This is a structural invariant: the configured detection runs at
    the user's chosen thresholds, while the sensitive detection runs
    at maximum sensitivity. If an event appears in events_configured
    but not in events_sensitive, it suggests the data was edited by
    hand or written by a buggy code path. Bug C — surface the
    inconsistency as a warning the WebUI can toast, don't silently
    fix it.

    Tolerance rationale (round 2 of bug C, 2026-06-08):
        The configured and sensitive passes are two separate calls to
        ``detect_onsets_energy_based()`` with different thresholds. For
        stereo stems, the L/R peak merge
        (``stems_to_midi/energy_detection_core.py:507``) picks
        ``min(left_peak_time, right_peak_time)``. The two passes can
        find different sets of L/R peaks (sensitive catches quieter
        hits the configured pass missed), so the merged onset time
        can land on a different hop for the same physical hit.

        At hop_length=512 / sr=44100, the hop duration is 11.61ms.
        So the maximum legitimate gap between the two arrays is
        ~12ms. The old 1ms tolerance was tighter than the actual
        quantization step and produced false-positive toasters for
        every stereo Convert run.

        The proper architectural fix is to make the merge
        deterministic (TODO in agent-plans/bug-tracking.md) so
        events_configured is a true subset of events_sensitive by
        time. Until that lands, this wider tolerance matches the
        pipeline's actual behavior. Real data-integrity issues
        (hand-edited analysis.json, events written to the wrong
        array) still trigger warnings — those gaps are at least
        hundreds of ms.

    Args:
        data: Parsed analysis.json dict (v3 format).
        time_tolerance_sec: Maximum allowed time difference (default 1ms).

    Returns:
        List of human-readable warning strings. Empty when the data
        passes the check.
    """
    warnings: List[str] = []
    stems = data.get('stems', {})
    if not stems:
        return warnings

    for stem_type, stem_data in stems.items():
        configured = stem_data.get('events_configured', [])
        sensitive = stem_data.get('events_sensitive', [])

        if not configured or not sensitive:
            # If one is empty we can't validate the subset relationship
            # but the case itself is suspicious enough to note when the
            # other side is non-empty.
            if configured and not sensitive:
                warnings.append(
                    f"data integrity: stem '{stem_type}' has "
                    f"{len(configured)} events_configured but no "
                    f"events_sensitive — re-run full detection to "
                    f"regenerate the sensitive pool."
                )
            continue

        # Build a quick lookup of sensitive times
        sensitive_times = [e.get('time', 0.0) for e in sensitive]

        missing = []
        for ev in configured:
            t = ev.get('time', 0.0)
            if not any(abs(t - st) <= time_tolerance_sec for st in sensitive_times):
                missing.append(t)

        if missing:
            n = len(missing)
            sample = ', '.join(f"{t:.4f}" for t in missing[:3])
            warnings.append(
                f"data integrity: stem '{stem_type}' has {n} "
                f"event(s) in events_configured with no matching time "
                f"in events_sensitive (samples: {sample}). "
                f"This usually means the analysis was edited by hand "
                f"or a bug wrote events to the wrong array. The WebUI "
                f"will display this as a toast. Use the tuning panel "
                f"to filter or rerun detection to regenerate."
            )

    return warnings


def save_envelope_data(
    envelope_by_stem: Dict[str, Dict],
    midi_path: Union[str, Path]
) -> List[Path]:
    """
    Save per-stem energy envelope arrays as .npz files for waveform visualization.
    
    Each stem gets its own file: {base}.{stem_type}.envelope.npz containing
    the L/R energy envelope arrays, time axis, and detection parameters.
    
    Args:
        envelope_by_stem: Dict mapping stem_type to envelope data dict with keys:
            - times: np.ndarray of frame times in seconds
            - left: np.ndarray of left channel energy values
            - right: np.ndarray of right channel energy values
            - sr: int sample rate
            - hop_length: int hop length used
            - method: str energy calculation method ('rms', 'peak_hold', etc.)
        midi_path: Path to corresponding MIDI file (used to derive output paths)
    
    Returns:
        List of paths to created .npz files
    """
    midi_path = Path(midi_path)
    base = midi_path.with_suffix('')  # Remove .mid extension
    saved_paths = []
    
    for stem_type, envelope in envelope_by_stem.items():
        if envelope is None:
            continue
        
        times = envelope.get('times')
        left = envelope.get('left')
        right = envelope.get('right')
        
        # Skip if no envelope data (e.g. librosa detection path)
        if times is None or left is None or right is None:
            continue
        
        npz_path = Path(f"{base}.{stem_type}.envelope.npz")
        np.savez_compressed(
            npz_path,
            times=np.asarray(times, dtype=np.float32),
            left=np.asarray(left, dtype=np.float32),
            right=np.asarray(right, dtype=np.float32),
            sr=np.array(envelope.get('sr', 44100)),
            hop_length=np.array(envelope.get('hop_length', 512)),
            method=np.array(envelope.get('method', 'rms'))
        )
        saved_paths.append(npz_path)
    
    if saved_paths:
        stem_names = [p.suffixes[-2].lstrip('.') for p in saved_paths]
        print(f"  Saved envelope data: {', '.join(stem_names)} ({len(saved_paths)} files)")
    
    return saved_paths


def load_envelope_data(
    midi_path: Union[str, Path],
    stem_type: str
) -> Optional[Dict]:
    """
    Load energy envelope data for a specific stem.
    
    Args:
        midi_path: Path to MIDI file (used to derive .npz path)
        stem_type: Stem type to load ('kick', 'snare', etc.)
    
    Returns:
        Dict with keys: times, left, right, sr, hop_length, method.
        Returns None if file not found.
    """
    midi_path = Path(midi_path)
    base = midi_path.with_suffix('')
    npz_path = Path(f"{base}.{stem_type}.envelope.npz")
    
    if not npz_path.exists():
        return None
    
    data = np.load(npz_path, allow_pickle=False)
    return {
        'times': data['times'],
        'left': data['left'],
        'right': data['right'],
        'sr': int(data['sr']),
        'hop_length': int(data['hop_length']),
        'method': str(data['method'])
    }


# Contrast envelope (2026-06-19): the broadband contrast envelope
# built by detect_percentile_gated_broad_attacks — sum of
# bin-level (s_db - floor) over the [broad_min_hz, broad_max_hz]
# band, where floor is the per-bin p5 noise. Different from the
# L/R RMS envelope saved above (which is the WebUI's waveform
# visualization). Saved so post-hoc tools can walk the
# envelope around each KEPT event's frame without re-running
# detection. Compressed with savez_compressed (~30-50KB per
# stem for typical 4-minute songs).
#
# The contrast envelope is the basis for:
#   - pga_event_builder's per-event decay features
#   - the open/closed hihat walk diagnostic
#   - any future per-event "shape of the ring" analysis
#
# Filename: {base}.{stem_type}.contrast_envelope.npz (distinct
# from {base}.{stem_type}.envelope.npz so both can coexist
# alongside the L/R RMS WebUI viz data).


def save_contrast_envelope(
    contrast_envelope_by_stem: Dict[str, Dict],
    midi_path: Union[str, Path],
) -> List[Path]:
    """
    Save per-stem broadband contrast envelope arrays as .npz
    files (2026-06-19). Used by the open/closed hihat walk
    diagnostic — see :func:`save_envelope_data` for the L/R RMS
    WebUI viz counterpart.

    Each stem gets its own file:
    ``{base}.{stem_type}.contrast_envelope.npz`` containing:
        - ``envelope``: 1D float32 array, broadband contrast
          envelope at hop=256 sample stride (PGA STFT).
        - ``sr``: int, sample rate.
        - ``hop_length``: int, hop in samples (256 for PGA).
        - ``n_fft``: int, FFT size used (1024 for PGA).

    Args:
        contrast_envelope_by_stem: Dict mapping stem_type to
            a dict with keys: envelope, sr, hop_length, n_fft.
        midi_path: Path to corresponding MIDI file (used to
            derive output paths).

    Returns:
        List of paths to created .npz files.
    """
    midi_path = Path(midi_path)
    base = midi_path.with_suffix('')
    saved_paths: List[Path] = []

    for stem_type, env_data in contrast_envelope_by_stem.items():
        if env_data is None:
            continue
        envelope = env_data.get('envelope')
        if envelope is None:
            continue
        npz_path = Path(f"{base}.{stem_type}.contrast_envelope.npz")
        np.savez_compressed(
            npz_path,
            envelope=np.asarray(envelope, dtype=np.float32),
            sr=np.array(env_data.get('sr', 44100)),
            hop_length=np.array(env_data.get('hop_length', 256)),
            n_fft=np.array(env_data.get('n_fft', 1024)),
        )
        saved_paths.append(npz_path)

    if saved_paths:
        # Filename is {base}.{stem_type}.contrast_envelope.npz;
        # the stem name is the segment between the last '.' and
        # '.contrast_envelope'. Split on the suffix to recover it.
        stem_names = [
            p.name.replace('.contrast_envelope.npz', '').split('.')[-1]
            for p in saved_paths
        ]
        print(
            f"  Saved contrast envelopes: "
            f"{', '.join(stem_names)} ({len(saved_paths)} files)"
        )

    return saved_paths


def load_contrast_envelope(
    midi_path: Union[str, Path],
    stem_type: str,
) -> Optional[Dict]:
    """
    Load the broadband contrast envelope for a single stem
    (2026-06-19). See :func:`save_contrast_envelope`.

    Returns:
        Dict with keys: envelope, sr, hop_length, n_fft.
        Returns None if file not found.
    """
    midi_path = Path(midi_path)
    base = midi_path.with_suffix('')
    npz_path = Path(f"{base}.{stem_type}.contrast_envelope.npz")

    if not npz_path.exists():
        return None

    data = np.load(npz_path, allow_pickle=False)
    return {
        'envelope': data['envelope'],
        'sr': int(data['sr']),
        'hop_length': int(data['hop_length']),
        'n_fft': int(data['n_fft']),
    }

