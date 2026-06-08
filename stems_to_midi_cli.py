"""
Convert separated drum stems to MIDI tracks.

Uses project-based workflow: automatically detects projects with stems
and generates MIDI files in the project/midi/ directory.

Architecture: Modular Design (Functional Core, Imperative Shell)
- stems_to_midi/ submodules: Core conversion logic
- project_manager: Project discovery and management
- stems_to_midi_cli.py (this file): CLI orchestration

CLI flags are now generated from the centralized settings schema
(see webui.settings_schema + webui.cli_builder). Adding a new
``SettingDefinition`` with a non-empty ``cli_flag`` is the ONLY place
a new CLI flag needs to be declared.

Usage:
    python stems_to_midi_cli.py              # Auto-detect project
    python stems_to_midi_cli.py 1            # Process specific project
    python stems_to_midi_cli.py --learn      # Learning mode
    python stems_to_midi_cli.py --help       # List all schema-driven flags
"""

from pathlib import Path
import argparse
from typing import List
import sys

# Import modules (thin orchestration layer)
from stems_to_midi.config import DrumMapping
from stems_to_midi.midi import create_midi_file, save_analysis_sidecar, save_envelope_data, load_analysis_sidecar
from stems_to_midi.processing_shell import process_stem_to_midi
from stems_to_midi.rebuild_core import rebuild_events_from_analysis

# Import project manager
from project_manager import (
    select_project,
    get_project_by_number,
    get_project_config,
    update_project_metadata,
    USER_FILES_DIR
)

# Schema-driven CLI builder
from webui.cli_builder import (
    build_cli_parser,
    count_cli_flags,
    apply_cli_overrides,
    validate_args,
)


def stems_to_midi_for_project(
    project: dict,
    config: dict = None,
    stems_to_process: List[str] = None,
    max_duration: float = None,
    learning_mode: bool = False,
):
    """
    Convert separated drum stems to MIDI files for a specific project.

    Args:
        project: Project info dictionary from project_manager.
        config: Fully-resolved config dict (CLI overrides already applied).
            If None, the project's midiconfig.yaml is loaded fresh.
        stems_to_process: List of stem types to process (default: all).
        max_duration: Maximum duration in seconds (for faster learning).
        learning_mode: Enable learning mode (export all detections).
    """
    project_dir = project["path"]

    print(f"\n{'='*60}")
    print(f"Converting Stems to MIDI - Project {project['number']}: {project['name']}")
    print(f"{'='*60}\n")

    # Load project-specific config if not provided
    if config is None:
        config_path = get_project_config(project_dir, "midiconfig.yaml")
        if config_path is None:
            print("ERROR: midiconfig.yaml not found in project or root directory")
            sys.exit(1)
        print(f"Using config: {config_path}")
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"ERROR: Failed to load config: {e}")
            sys.exit(1)
    else:
        print("Using CLI-overridden config")

    # Use cleaned stems if available, otherwise use regular stems
    stems_source = project_dir / "cleaned"
    if not stems_source.exists() or not any(stems_source.iterdir()):
        stems_source = project_dir / "stems"

    if not stems_source.exists():
        print("ERROR: No stems found in project. Run separate.py first.")
        sys.exit(1)

    print(f"Using stems from: {stems_source}")

    # Output to project/midi/ directory
    midi_dir = project_dir / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)

    # Process using existing logic
    _process_stems_to_midi(
        stems_source=stems_source,
        midi_dir=midi_dir,
        project_name=project["name"],
        config=config,
        stems_to_process=stems_to_process,
        max_duration=max_duration,
        learning_mode=learning_mode
    )

    # Update project metadata
    update_project_metadata(project_dir, {
        "status": {
            "separated": project["metadata"]["status"].get("separated", False) if project["metadata"] else False,
            "cleaned": project["metadata"]["status"].get("cleaned", False) if project["metadata"] else False,
            "midi_generated": True,
            "video_rendered": project["metadata"]["status"].get("video_rendered", False) if project["metadata"] else False
        }
    })

    print("Status Update: MIDI conversion complete!")
    print(f"  MIDI files saved to: {midi_dir}")
    print("  Project status updated\n")


def _process_stems_to_midi(
    stems_source: Path,
    midi_dir: Path,
    project_name: str,
    config: dict,
    stems_to_process: List[str],
    max_duration: float,
    learning_mode: bool,
):
    """
    Internal function to process stems to MIDI.

    All tuning parameters come from ``config`` (which has been merged
    with CLI overrides). The legacy ``onset_threshold`` etc. CLI args
    are now applied to the config dict via the schema's yaml_path, so
    we read them from config like the pipeline does.
    """
    # Apply learning mode if enabled
    if learning_mode:
        config['learning_mode'] = config.get('learning_mode', {})
        config['learning_mode']['enabled'] = True

    # Default stems to process
    if stems_to_process is None:
        stems_to_process = ['kick', 'snare', 'toms', 'hihat', 'cymbals']

    # Initialize drum mapping from config
    drum_mapping = DrumMapping.from_config(config)

    # Find stem files in the stems_source directory
    # Expected pattern: project_name-kick.wav, project_name-snare.wav, etc.
    stem_files = list(stems_source.glob("*.wav"))

    if not stem_files:
        raise RuntimeError(f"No WAV files found in {stems_source}")

    # Pull all onset / midi parameters from config (single source of truth).
    onset_threshold = config['onset_detection']['threshold']
    onset_delta = config['onset_detection']['delta']
    onset_wait = config['onset_detection']['wait']
    hop_length = config['onset_detection']['hop_length']
    min_velocity = config['midi'].get('min_velocity', 80)
    max_velocity = config['midi'].get('max_velocity', 110)
    tempo = config['midi'].get('default_tempo') or config['midi'].get('tempo')

    print("Settings:")
    print(f"  Onset threshold: {onset_threshold}")
    print(f"  Onset delta: {onset_delta}")
    print(f"  Onset wait: {onset_wait}")
    print(f"  Hop length: {hop_length}")
    print(f"  Velocity range: {min_velocity}-{max_velocity}")
    print(f"  Tempo: {tempo} BPM")
    if max_duration is not None:
        print(f"  Max duration: {max_duration} seconds (fast learning mode)")
    print()

    # Group stem files by base name (everything before the last hyphen and stem type)
    from collections import defaultdict
    files_by_song = defaultdict(dict)

    for stem_file in stem_files:
        # Parse filename: "song_name-stem_type.wav"
        name_without_ext = stem_file.stem
        for stem_type in stems_to_process:
            if name_without_ext.endswith(f"-{stem_type}"):
                base_name = name_without_ext[:-len(f"-{stem_type}")]
                files_by_song[base_name][stem_type] = stem_file
                break

    if not files_by_song:
        print("No stem files found matching expected pattern (name-stemtype.wav)")
        return

    total_songs = len(files_by_song)
    for song_idx, (base_name, stem_files_dict) in enumerate(files_by_song.items(), 1):
        print(f"Processing: {base_name}")

        # Progress: start of song processing
        song_start_progress = int((song_idx - 1) / total_songs * 90)
        print(f"Progress: {song_start_progress}%")

        events_by_stem = {}
        analysis_by_stem = {}
        envelope_by_stem = {}

        # Process each stem type
        total_stems = len(stems_to_process)
        processed_stems = 0
        for stem_type in stems_to_process:
            if stem_type not in stem_files_dict:
                print(f"  Warning: {stem_type} file not found, skipping...")
                processed_stems += 1
                continue

            stem_file = stem_files_dict[stem_type]

            result = process_stem_to_midi(
                stem_file,
                stem_type,
                drum_mapping,
                config,
                onset_threshold=onset_threshold,
                onset_delta=onset_delta,
                onset_wait=onset_wait,
                hop_length=hop_length,
                min_velocity=min_velocity,
                max_velocity=max_velocity,
                max_duration=max_duration
            )

            if result and result.get('events'):
                events_by_stem[stem_type] = result['events']
                # Store analysis data for sidecar v3
                # (configured + sensitive + spectral-transient).
                # The spectral detector runs on every stem regardless
                # of detection_method; its candidates are written to
                # stems.<stem>.events_spectral in the sidecar.
                analysis_by_stem[stem_type] = {
                    'all_onset_data': result.get('all_onset_data', []),
                    'sensitive_onset_data': result.get('sensitive_onset_data', []),
                    'spectral_onset_data': result.get('spectral_onset_data', []),
                    'spectral_config': result.get('spectral_config')
                }
                # Store envelope data for waveform visualization
                if result.get('envelope_data'):
                    envelope_by_stem[stem_type] = result['envelope_data']

            # Progress: after each stem (0-90% of total)
            processed_stems += 1
            stem_progress = int((song_idx - 1) / total_songs * 90 + (processed_stems / total_stems) * (90 / total_songs))
            print(f"Progress: {stem_progress}%")

        # Create MIDI file using rebuild logic (same as "Save & Reconvert")
        if events_by_stem:
            # Add suffix for learning mode
            if learning_mode:
                suffix = config.get('learning_mode', {}).get('learning_midi_suffix', '_learning')
                midi_path = midi_dir / f"{base_name}{suffix}.mid"
            else:
                midi_path = midi_dir / f"{base_name}.mid"

            # Step 1: Save analysis sidecar FIRST (Detection Output Contract v3)
            save_analysis_sidecar(
                events_by_stem, midi_path, tempo=tempo,
                analysis_by_stem=analysis_by_stem if analysis_by_stem else None,
                config=config,
            )

            # Step 2: Load the analysis sidecar and rebuild MIDI from it
            analysis_data = load_analysis_sidecar(midi_path)
            if not analysis_data:
                raise RuntimeError(f"Failed to save analysis sidecar for {midi_path}")

            # Rebuild MIDI from analysis data (honors thresholds, applies all filters)
            updated_analysis, rebuild_events = rebuild_events_from_analysis(
                analysis_data=analysis_data,
                overrides={},  # No manual overrides for initial conversion
                config=config,
            )

            # Create MIDI from rebuilt events
            create_midi_file(
                rebuild_events,
                midi_path,
                tempo=tempo,
                track_name=f"Drums - {base_name}",
                config=config
            )

            # Save energy envelope data for waveform visualization
            if envelope_by_stem:
                save_envelope_data(envelope_by_stem, midi_path)

            # Progress: after MIDI creation (90-100% of total)
            midi_progress = int(90 + (song_idx / total_songs) * 10)
            print(f"Progress: {midi_progress}%")

            if learning_mode:
                print(f"  Saved LEARNING MIDI: {midi_path}")
                print(f"  ** Load in DAW, delete false positives (velocity=1 hits), save as: {base_name}_edited.mid **\n")
            else:
                print(f"  Saved: {midi_path}\n")
        else:
            print("  No events detected, skipping MIDI creation\n")


def _build_argparser() -> argparse.ArgumentParser:
    """
    Build the CLI parser: schema-driven flags + orchestration flags.
    Schema-driven flags are generated by webui.cli_builder; orchestration
    flags (--learn, --maxtime, --stems, project) live here because they
    don't correspond to a single yaml_path.
    """
    parser = build_cli_parser(
        prog='stems_to_midi_cli',
        description=(
            "Convert separated drum stems to MIDI tracks. "
            "Every setting flag is generated from the centralized "
            "settings schema (webui.settings_schema)."
        ),
    )

    # Orchestration flags (not in the schema because they don't map to
    # a single yaml_path).
    parser.add_argument('project_number', type=int, nargs='?', default=None,
                        help="Project number to process (optional, auto-detects if omitted)")
    parser.add_argument('--stems', type=str, nargs='+',
                        choices=['kick', 'snare', 'toms', 'hihat', 'cymbals'],
                        help="Specific stems to process (default: all)")

    learning_group = parser.add_argument_group('Threshold Learning Mode')
    learning_group.add_argument('--learn', action='store_true',
                                help="Enable learning mode (exports all detections, rejected=velocity 1).")
    learning_group.add_argument('--maxtime', type=float, default=None,
                                help="Maximum duration in seconds to analyze (for faster learning on long tracks).")

    return parser


if __name__ == '__main__':
    n_flags = count_cli_flags()
    print(f"{n_flags} CLI flags available, run --help to see them")

    parser = _build_argparser()
    args = parser.parse_args()

    # Validate args via SettingDefinition.validate
    errors = validate_args(args)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        sys.exit(1)

    # Cross-field validation
    if hasattr(args, 'min_velocity') and hasattr(args, 'max_velocity'):
        if args.min_velocity is not None and args.max_velocity is not None:
            if args.min_velocity > args.max_velocity:
                print("ERROR: --min-velocity cannot be greater than --max-velocity")
                sys.exit(1)

    # Select project
    if args.project_number is not None:
        project = get_project_by_number(args.project_number, USER_FILES_DIR)
        if project is None:
            print(f"ERROR: Project {args.project_number} not found")
            sys.exit(1)
    else:
        # Auto-select project
        project = select_project(None, USER_FILES_DIR, allow_interactive=True)
        if project is None:
            print("\nNo projects found in user_files/")
            print("Run separate.py first to create stems!")
            sys.exit(0)

    # Check that project has stems
    has_stems = (project["path"] / "stems").exists()
    has_cleaned = (project["path"] / "cleaned").exists()

    if not has_stems and not has_cleaned:
        print(f"\nERROR: Project {project['number']} has no stems.")
        print("Run separate.py first!")
        sys.exit(1)

    # Process the project
    if args.learn:
        print("=== LEARNING MODE ENABLED ===")
        print("All detections will be exported. Rejected hits have velocity=1.")
        print("Load MIDI in DAW, delete false positives, then use calibrated settings.\n")

    # Load project config (so we have a dict to merge CLI overrides into)
    config_path = get_project_config(project["path"], "midiconfig.yaml")
    if config_path is None:
        print("ERROR: midiconfig.yaml not found in project or root directory")
        sys.exit(1)
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}

    # Apply CLI overrides to the config dict (flag → yaml_path)
    n_applied, applied = apply_cli_overrides(args, config)
    if n_applied:
        print(f"Applied {n_applied} CLI override(s): {', '.join(applied)}")

    stems_to_midi_for_project(
        project=project,
        config=config,
        stems_to_process=args.stems,
        max_duration=args.maxtime,
        learning_mode=args.learn,
    )


def _load_project_config_for_project(project):
    """Load the project's midiconfig.yaml (per-project first, then root).

    Lives in stems_to_midi_cli.py (not webui.api.operations) because
    the /api/stems-to-midi work function loads this file via importlib
    util.spec_from_file_location and accesses helpers through the
    loaded module's namespace — see webui/api/operations.py:run_stems_to_midi.
    A helper defined in webui.api.operations is invisible to that
    importlib-loaded module, which crashed the WebUI Convert button
    with 'module stems_to_midi_cli has no attribute _load_project_config_for_project'
    on 2026-06-08.
    """
    config_path = get_project_config(project["path"], "midiconfig.yaml")
    if config_path is None:
        # Fall back to empty config — let the pipeline's per-stem defaults apply
        return {}
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def _apply_cli_overrides_to_config(config, overrides):
    """Apply dotted-YAML-path overrides to a config dict.

    Mirrors webui.cli_builder.apply_cli_overrides but without requiring
    the full schema SettingDefinition machinery. The route path
    doesn't need schema validation — the values come from the JS,
    which already validated against the schema when the form was
    built.

    Lives in stems_to_midi_cli.py for the same reason as
    _load_project_config_for_project above — the importlib-loaded
    module is the namespace the WebUI work function uses.
    """
    for path, value in overrides.items():
        if value is None:
            continue
        parts = path.split('.')
        d = config
        for part in parts[:-1]:
            if part not in d or not isinstance(d[part], dict):
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
