"""
Rebuild MIDI from an existing analysis.json sidecar.

Reads the sidecar written by the full pipeline, re-applies all filter
thresholds from the project's midiconfig.yaml, and writes the updated
sidecar and MIDI file.

Usage:
    python -m stems_to_midi.rebuild_cli 4 --stems toms
    python -m stems_to_midi.rebuild_cli 4 --stems toms hihat kick
    python -m stems_to_midi.rebuild_cli 4                    # rebuild all stems
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path for project_manager resolution
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stems_to_midi.rebuild_shell import rebuild_midi_for_project
from project_manager import get_project_by_number, USER_FILES_DIR


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild MIDI from an existing analysis.json sidecar. "
                    "Reads filter thresholds from the project's midiconfig.yaml.",
        prog='stems_to_midi.rebuild_cli',
    )
    parser.add_argument(
        'project_number',
        type=int,
        nargs='?',
        default=None,
        help="Project number to rebuild (auto-detected if omitted)",
    )
    parser.add_argument(
        '--stems',
        nargs='+',
        choices=['kick', 'snare', 'toms', 'hihat', 'cymbals'],
        help="Which stems to rebuild (default: all found in sidecar)",
    )
    return parser


def main():
    parser = _build_argparser()
    args = parser.parse_args()

    # Resolve project
    if args.project_number is not None:
        project = get_project_by_number(args.project_number, USER_FILES_DIR)
        if project is None:
            print(f"Error: project {args.project_number} not found in {USER_FILES_DIR}")
            sys.exit(1)
    else:
        project = None  # auto-detect

    if project is None:
        from project_manager import select_project
        project = select_project(USER_FILES_DIR)
        if project is None:
            print("No project selected")
            sys.exit(1)

    project_dir = Path(project['path'])
    print(f"Rebuilding project: {project_dir.name}")

    # Determine stem types
    stem_types = args.stems  # None = rebuild_shell default (all)

    # Run rebuild
    result = rebuild_midi_for_project(
        project_dir,
        stem_types=stem_types,
        honor_overrides=True,
        config_overrides={},  # Use YAML values only
    )

    if not result['success']:
        print(f"Error: {result.get('error', 'unknown error')}")
        if result.get('requires_full_pipeline'):
            print("  Run the full pipeline first (stems_to_midi_cli) to generate analysis.json")
        sys.exit(1)

    print(f"  Rebuilt stems: {result['stems_rebuilt']}")
    print(f"  Total MIDI events: {sum(len(evs) for evs in result['events_by_stem'].values())}")
    print(f"  Elapsed: {result['elapsed_ms']}ms")

    if result.get('data_integrity_warnings'):
        for w in result['data_integrity_warnings']:
            print(f"  Warning: {w}")


if __name__ == '__main__':
    main()
