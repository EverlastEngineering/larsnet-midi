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
from typing import Dict, List, Optional

from .config import load_config
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


def _load_overrides(midi_dir: Path) -> Dict[str, Dict[str, str]]:
    """Load event_overrides.json if it exists."""
    overrides_path = midi_dir / 'event_overrides.json'
    if not overrides_path.exists():
        return {}
    with open(overrides_path, 'r') as f:
        return json.load(f)


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
        return {
            'success': False,
            'error': 'No midi directory found in project',
            'stems_rebuilt': [],
            'elapsed_ms': 0,
        }

    # Find MIDI file (needed for sidecar paths)
    midi_path = _find_midi_path(midi_dir)
    if midi_path is None:
        return {
            'success': False,
            'error': 'No MIDI file found in project',
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

    # Load config
    if config_path is None:
        config_path = project_dir / 'midiconfig.yaml'
    if not config_path.exists():
        # Fall back to root config
        config_path = Path(__file__).parent.parent / 'midiconfig.yaml'
    config = load_config(config_path)

    # Load overrides
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
    }
