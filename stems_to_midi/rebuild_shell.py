"""
Rebuild MIDI from Analysis — Imperative Shell

Handles I/O for the rebuild-from-analysis pipeline: loading analysis.json,
reading overrides, applying config updates, writing MIDI, and updating
sidecar files.

This module is the thin I/O wrapper around rebuild_core.py.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import load_config
from .event_overrides import (
    clean_overrides,
    load_event_overrides,
    save_event_overrides,
)
from .midi import create_midi_file, load_analysis_sidecar
from .rebuild_core import rebuild_events_from_analysis


__all__ = ['rebuild_midi_for_project']


def _find_midi_path(midi_dir: Path) -> Optional[Path]:
    """Find the primary MIDI file in a project's midi directory."""
    midi_files = list(midi_dir.glob("*.mid"))
    # Exclude learning mode files
    midi_files = [f for f in midi_files if '_learning' not in f.stem]
    if not midi_files:
        return None
    # If multiple, pick the first (there should typically be one)
    return midi_files[0]


def _load_overrides(midi_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load event_overrides.json if it exists."""
    project_dir = midi_dir.parent
    return load_event_overrides(project_dir) or {}


def _clean_overrides(
    overrides: Dict[str, Dict[str, Any]],
    analysis_data: Dict,
) -> Dict[str, Dict[str, Any]]:
    """Drop redundant override entries. Wrapper that handles
    the None case (no overrides → return empty dict)."""
    if not overrides:
        return {}
    return clean_overrides(overrides, analysis_data)


def _persist_overrides_if_changed(
    midi_dir: Path,
    before: Dict[str, Dict[str, Any]],
    after: Dict[str, Dict[str, Any]],
) -> int:
    """Write the cleaned override dict back to disk if it differs
    from the input. Returns the number of entries removed
    (0 if no change, positive if entries were dropped)."""
    if before == after:
        return 0
    project_dir = midi_dir.parent
    if not after:
        # No overrides left — delete the file rather than
        # writing an empty object.
        path = midi_dir / 'event_overrides.json'
        if path.exists():
            path.unlink()
    else:
        save_event_overrides(project_dir, after)
    # Count removed entries
    before_count = sum(len(t) for t in before.values())
    after_count = sum(len(t) for t in after.values())
    return before_count - after_count


def _apply_config_overrides(config: Dict, overrides: Dict) -> None:
    """
    Write WebUI slider values into a loaded config dict.

    Keys are dotted paths matching the YAML structure:
        'filtering.reverb_continuation_attack_threshold'
        'kick.geomean_threshold'
        'hihat.open_geomean_min'
        'hihat.open_sustain_ms'
        'snare.expected_clusters'
        'snare.cluster_feature'
        etc.

    Existing nested dicts are preserved; missing sections are created.
    Mutates ``config`` in place.

    Args:
        config: The loaded midiconfig.yaml dict (mutated).
        overrides: Mapping of dotted path → value.
    """
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        parts = dotted_key.split('.')
        node = config
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value


def _save_analysis(analysis_data: Dict, midi_path: Path) -> Path:
    """Write updated analysis.json sidecar."""
    sidecar_path = midi_path.with_suffix('.analysis.json')
    with open(sidecar_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    return sidecar_path


def rebuild_midi_for_project(
    project_dir: Path,
    config_path: Optional[Path] = None,
    stem_types: Optional[List[str]] = None,
    honor_overrides: bool = True,
    config_overrides: Optional[Dict] = None,
) -> Dict:
    """
    Rebuild MIDI from cached analysis data without re-running detection.

    This is the main entry point called by the API endpoint. It:
    1. Loads the project's analysis.json and config
    2. Optionally loads event_overrides.json
    3. Calls rebuild_events_from_analysis() (pure function)
    4. Writes updated MIDI file and analysis.json

    Args:
        project_dir: Path to project directory (contains midi/ subfolder).
        config_path: Path to midiconfig.yaml. If None, looks in project_dir
            then falls back to root config.
        stem_types: Optional list of stems to rebuild (None = all).
        honor_overrides: Whether to apply manual event overrides.
        config_overrides: Optional mapping of config keys to override
            values. Keys use the YAML structure (e.g. 'filtering.
            reverb_continuation_attack_threshold', 'kick.geomean_threshold',
            'hihat.open_geomean_min'). Used by the WebUI tuning panel
            to pass the slider values the user just moved — without this
            the UI filter would diverge from the server result (bug D).

    Returns:
        Dict with:
        - success: bool
        - stems_rebuilt: list of stem types that were rebuilt
        - elapsed_ms: int, rebuild time in milliseconds
        - analysis_data: updated analysis dict (for API response)
        - events_by_stem: MIDI events by stem (for API response)
        - error: str (only if success is False)
    """
    start = time.monotonic()

    midi_dir = project_dir / 'midi'
    if not midi_dir.exists():
        # T2 follow-up (2026-06-08): a project with no midi/ directory
        # yet has never been through stems-to-midi. Return a clean
        # 409 via the route, not a 500. The requires_full_pipeline
        # flag signals the WebUI to show "run full conversion first".
        return {
            'success': False,
            'error': 'No midi directory found in project',
            'requires_full_pipeline': True,
            'stems_rebuilt': [],
            'elapsed_ms': 0,
        }

    # Find MIDI file (needed for sidecar paths)
    midi_path = _find_midi_path(midi_dir)
    if midi_path is None:
        return {
            'success': False,
            'error': 'No MIDI file found in project',
            'requires_full_pipeline': True,
            'stems_rebuilt': [],
            'elapsed_ms': 0,
        }

    # Load analysis sidecar
    analysis_data = load_analysis_sidecar(midi_path)
    if analysis_data is None:
        return {
            'success': False,
            'error': 'No analysis.json found. Run full detection first.',
            'requires_full_pipeline': True,
            'stems_rebuilt': [],
            'elapsed_ms': 0,
        }

    # Capture any data integrity warnings from the loader (bug C) so the
    # API response can surface them to the WebUI. These are warnings, not
    # errors — rebuild proceeds regardless.
    integrity_warnings = analysis_data.pop('data_integrity_warnings', [])

    # Load config
    if config_path is None:
        config_path = project_dir / 'midiconfig.yaml'
    if not config_path.exists():
        # Fall back to root config
        config_path = Path(__file__).parent.parent / 'midiconfig.yaml'
    config = load_config(config_path)

    # Apply WebUI slider overrides (bug D: ensures UI and server agree on
    # the threshold values used for filtering). Keys are YAML paths
    # (e.g. 'filtering.reverb_continuation_attack_threshold') and are
    # written into the matching nested config section.
    if config_overrides:
        _apply_config_overrides(config, config_overrides)

    # Load overrides (post-2026-06-30 shape: {stem: {time: {status,
    # [classification]?}}}). The legacy string-valued file format
    # was deleted per the user's "nuke the old files and start
    # fresh" direction.
    overrides = _load_overrides(midi_dir) if honor_overrides else {}

    # Run the rebuild (pure function)
    try:
        updated_analysis, midi_events_by_stem = rebuild_events_from_analysis(
            analysis_data=analysis_data,
            overrides=overrides,
            config=config,
            stem_types=stem_types,
        )
    except ValueError as e:
        return {
            'success': False,
            'error': str(e),
            'requires_full_pipeline': 'version' in str(e).lower(),
            'stems_rebuilt': [],
            'elapsed_ms': int((time.monotonic() - start) * 1000),
        }

    # 2026-06-30: clean up overrides that now match the sidecar's
    # natural state (e.g. user toggled a FILTERED event to KEPT,
    # then lowered the prominence filter so the event's natural
    # state became KEPT — the override is redundant). The file
    # stays intentionally minimal.
    cleaned_overrides = _clean_overrides(overrides, updated_analysis)
    _cleanup_result = _persist_overrides_if_changed(
        midi_dir, overrides, cleaned_overrides,
    )

    # Write updated MIDI file
    tempo = config.get('midi', {}).get('default_tempo', 120.0)
    create_midi_file(
        midi_events_by_stem,
        midi_path,
        tempo=tempo,
        track_name=f"Drums - {midi_path.stem}",
        config=config,
    )

    # Write updated analysis sidecar
    _save_analysis(updated_analysis, midi_path)

    elapsed_ms = int((time.monotonic() - start) * 1000)
    stems_rebuilt = list(midi_events_by_stem.keys())
    total_events = sum(len(events) for events in midi_events_by_stem.values())

    print(f"  Rebuild complete: {total_events} MIDI events across "
          f"{len(stems_rebuilt)} stems in {elapsed_ms}ms")

    return {
        'success': True,
        'stems_rebuilt': stems_rebuilt,
        'elapsed_ms': elapsed_ms,
        'analysis_data': updated_analysis,
        'events_by_stem': midi_events_by_stem,
        'data_integrity_warnings': integrity_warnings,
        'event_overrides': cleaned_overrides,
        'event_overrides_removed': _cleanup_result,
    }
