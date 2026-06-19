"""
Percentile-gated broad-attack pipeline (PGA-only).

Minimal path for stems that use the PGA detector exclusively:
load audio -> PGA detection -> build MIDI from pga_kept.

No energy/spectral/pan detection. No geomean/sustain filtering.
All raw PGA events stored in events_pga (all-KEPT at detect time).

The YAML pga_min_prominence threshold controls which events survive
in the rebuild path.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .pga_event_builder import  _build_pga_events_with_filter
from .energy_detection_core import calculate_energy_envelope
# 2026-06-19: hihat open/closed classifier. Stamps hihat_state
# on every PGA event using the broadband-envelope decay-slope
# rule (falls back to geomean+sustain when decay_slope_db is
# absent). The MIDI note loop below reads hihat_state to flip
# drum_mapping.hihat (42) -> drum_mapping.hihat_open (46).
from .note_classification_core import classify_hihat_notes


def process_percentile_gated(
    audio_path: Union[str, Path],
    drum_mapping,  # DrumMapping instance
    config: Dict,
    min_velocity: int = 80,
    max_velocity: int = 110,
    stem_type: str = 'toms',
) -> Dict:
    """Run the PGA-only pipeline on a stem.

    Args:
        audio_path: Path to the stem audio file.
        drum_mapping: DrumMapping instance — the ``stem_type``
            field on this mapping (``drum_mapping.snare``,
            ``drum_mapping.toms``, etc.) is used as the MIDI
            note for every detected event. Previously this
            function was hard-coded to ``toms``; the
            ``stem_type`` parameter (2026-06-18) generalizes
            it to any stem that opts into
            ``<stem_type>.use_pga_detection: true`` in the
            project midiconfig.
        config: Project config dict. Reads
            ``<stem_type>.pga_min_prominence`` and
            ``<stem_type>.pga_abs_envelope_threshold`` (and
            their global ``onset_detection`` fallbacks) via
            the pga_event_builder helpers. Also reads
            ``<stem_type>.timing_offset`` and
            ``<stem_type>.max_note_duration`` for the MIDI
            event construction.
        stem_type: Which stem this is. Defaults to ``'toms'``
            for back-compat with the original (toms-only)
            call. ``process_stem_to_midi`` passes the actual
            stem_type in.

    Returns:
        Dict with:
            'events': MIDI events from pga_kept
            'events_configured': [] (absent for this pipeline)
            'all_onset_data': []
            'sensitive_onset_data': []
            'spectral_onset_data': []
            'spectral_config': None
            'envelope_data': None
            'pga_onset_data': all raw PGA events (all-KEPT)
    """
    # Load audio
    from .processing_shell import _load_and_validate_audio
    audio, sr = _load_and_validate_audio(audio_path, config, stem_type, max_duration=None)
    if audio is None:
        return _empty_result()

    # Mono mix for PGA detector
    if audio.ndim == 2:
        audio_mono = audio.mean(axis=1).astype(audio.dtype)
    else:
        audio_mono = audio

    # Run PGA detection
    # _build_pga_events_with_filter: filtered split (for MIDI output)
    # 2026-06-19: capture the 4th return value (debug dict) so
    # we can extract the broadband contrast envelope for the
    # CLI to cache to {stem}.contrast_envelope.npz. Post-hoc
    # walk diagnostics (open/closed hihat) read from that npz
    # — no need to re-run detection.
    pga_raw, pga_kept, pga_filtered, pga_debug = _build_pga_events_with_filter(
        audio_mono, sr, config, stem_type=stem_type,
    )
    pga_envelope_data: Optional[Dict[str, Any]] = None
    if pga_debug is not None and pga_debug.get('envelope') is not None:
        pga_envelope_data = {
            'envelope': np.asarray(pga_debug['envelope'], dtype=np.float32),
            'sr': int(sr),
            'hop_length': 256,
            'n_fft': 1024,
        }

    # 2026-06-19: Build envelope_data for the WebUI's
    # detection analysis waveform viewer. The legacy energy
    # pipeline in processing_shell.py computes this from
    # the same calculate_energy_envelope function. The PGA
    # pipeline has its own contrast envelope internally
    # but the WebUI's renderer is calibrated to the energy
    # envelope's shape/scale, so we use the same function
    # here. Minimal change: ~10 lines, no test impact
    # (envelope saving is a side effect). User explicitly
    # requested this over Option A (PGA contrast envelope)
    # because the contrast envelope's shape is too different
    # from the energy envelope for the WebUI to render
    # correctly. (energy envelope: peak-hold, sharp
    # transients, fast time resolution. contrast envelope:
    # broadband sum, very different vertical scale and
    # shape, would render as a flat-ish blob.)
    stem_cfg = config.get(stem_type, {}) or {}
    envelope_method = stem_cfg.get('energy_method', 'peak_hold')
    envelope_peak_hold_ms = float(stem_cfg.get('peak_hold_ms', 3.0))
    envelope_hop = config.get('onset_detection', {}).get('hop_length', 512)
    env_times, env_energy = calculate_energy_envelope(
        audio_mono, sr,
        frame_length=2048,
        hop_length=envelope_hop,
        method=envelope_method,
        peak_hold_ms=envelope_peak_hold_ms,
    )
    # Match the legacy shape: 'times'/'left'/'right' keys,
    # where left/right are the same mono envelope (PGA
    # detector runs on mono, no stereo channel separation).
    # The WebUI renders whichever channel it has; duplicating
    # the mono envelope into both fields is the safe choice
    # for downstream consumers that may key on either.
    envelope_data = {
        'times': env_times,
        'left': env_energy,
        'right': env_energy,
    }

    # Build MIDI events from pga_kept
    note = int(getattr(drum_mapping, stem_type))
    note_open = int(getattr(drum_mapping, 'hihat_open', note))
    timing_offset = config.get(stem_type, {}).get('timing_offset', 0.0)
    max_duration = config.get(stem_type, {}).get(
        'max_note_duration', config.get('midi', {}).get('max_note_duration', 0.5))
    default_duration = config.get('audio', {}).get('default_note_duration', 0.1)

    # 2026-06-19: hihat open/closed via broadband-envelope decay
    # slope. Stamp hihat_state on every PGA event in place so
    # the MIDI note loop below can pick the right note. Also
    # covers the sidecar's events_pga list (built from pga_raw
    # in the return dict — pga_raw is the same list object as
    # pga_kept + pga_filtered before any classification runs,
    # so classifying it stamps the sidecar's hihat_state too).
    if stem_type == 'hihat' and pga_kept:
        classify_hihat_notes(pga_kept, config, force_reclassify=True)

    midi_events = []
    for i, ev in enumerate(pga_kept):
        midi_time = float(ev['time']) + timing_offset
        velocity = int(ev.get('midi_velocity', min_velocity))
        if ev.get('duration_ms') is not None:
            duration = min(ev['duration_ms'] / 1000.0, max_duration)
        elif i < len(pga_kept) - 1:
            duration = min(pga_kept[i + 1]['time'] - ev['time'], max_duration)
        else:
            duration = default_duration
        # 2026-06-19: open hihats use the open note (46), the
        # rest use the default hihat note (42 = closed). The
        # classifier above stamped hihat_state on this event.
        ev_note = note_open if ev.get('hihat_state') == 'open' else note
        midi_events.append({
            'time': float(midi_time),
            'note': ev_note,
            'velocity': int(velocity),
            'duration': float(duration),
            'hihat_state': ev.get('hihat_state'),
        })

    print(f"    [percentile_gated] Built {len(midi_events)} MIDI events from PGA")

    return {
        'events': midi_events,
        'events_configured': [],
        'all_onset_data': [],
        'sensitive_onset_data': [],
        'spectral_onset_data': [],
        'spectral_config': None,
        'envelope_data': envelope_data,
        'pga_onset_data': list(pga_raw),
        # 2026-06-19: cached broadband contrast envelope from
        # the PGA detector. CLI writes this to
        # {stem}.contrast_envelope.npz so post-hoc walk
        # diagnostics (open/closed hihat) can run without
        # re-detecting. May be None if pga_debug wasn't
        # populated (defensive; the detector always populates
        # it in practice).
        'pga_envelope_data': pga_envelope_data,
    }


def _empty_result() -> Dict:
    return {
        'events': [],
        'events_configured': [],
        'all_onset_data': [],
        'sensitive_onset_data': [],
        'spectral_onset_data': [],
        'spectral_config': None,
        'envelope_data': None,
        'pga_onset_data': [],
        'pga_envelope_data': None,
    }
