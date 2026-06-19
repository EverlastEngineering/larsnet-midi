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
from typing import Dict, List, Union

from .pga_event_builder import  _build_pga_events_with_filter
from .energy_detection_core import calculate_energy_envelope


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
    pga_raw, pga_kept, pga_filtered, _ = _build_pga_events_with_filter(
        audio_mono, sr, config, stem_type=stem_type,
    )

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
    timing_offset = config.get(stem_type, {}).get('timing_offset', 0.0)
    max_duration = config.get(stem_type, {}).get(
        'max_note_duration', config.get('midi', {}).get('max_note_duration', 0.5))
    default_duration = config.get('audio', {}).get('default_note_duration', 0.1)

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
        midi_events.append({
            'time': float(midi_time),
            'note': note,
            'velocity': int(velocity),
            'duration': float(duration),
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
    }
