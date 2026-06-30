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
from typing import Dict, List, Optional, Tuple
import sys

# Import modules (thin orchestration layer)
from stems_to_midi.config import DrumMapping
from stems_to_midi.midi import create_midi_file, save_analysis_sidecar, save_envelope_data, save_contrast_envelope, load_analysis_sidecar
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

    # Pull live MIDI output parameters from config.
    # 2026-06-20: the legacy onset_threshold / onset_delta / onset_wait /
    # hop_length prints were removed along with those YAML keys (they
    # were the energy/spectral detector's tuning knobs — PGA has
    # pga_min_prominence instead, see onset_detection.pga_min_prominence).
    # The CLI's "Settings:" block now shows only the LIVE knobs.
    min_velocity = config['midi'].get('min_velocity', 80)
    max_velocity = config['midi'].get('max_velocity', 110)
    tempo = config['midi'].get('default_tempo') or config['midi'].get('tempo')

    print("Settings:")
    pga_min_prom = (
        config.get('onset_detection', {}).get('pga_min_prominence')
        or config.get('toms', {}).get('pga_min_prominence')
        or 'unset'
    )
    print(f"  PGA min prominence: {pga_min_prom}")
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
        # 2026-06-19: per-stem broadband contrast envelope cache
        # (PGA STFT, hop=256). Saved to
        # {stem}.contrast_envelope.npz for post-hoc walks.
        contrast_envelope_by_stem = {}

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
                # events_configured MUST be forwarded too — without it,
                # save_analysis_sidecar falls back to all_onset_data
                # (energy only) and the spectral events never make it
                # into the sidecar. Bug caught by e2e verifier on
                # plan_e0953a25, 2026-06-08.
                # Toms (2026-06-15): events_pga is the single source of truth.
                # events_configured and events_sensitive are absent — do not include them.
                if stem_type == 'toms':
                    analysis_by_stem[stem_type] = {
                        'pga_onset_data': result.get('pga_onset_data', []),
                        'all_onset_data': result.get('all_onset_data', []),
                        'spectral_onset_data': result.get('spectral_onset_data', []),
                        'spectral_config': result.get('spectral_config'),
                    }
                else:
                    analysis_by_stem[stem_type] = {
                        'events_configured': result.get('events_configured', []),
                        'all_onset_data': result.get('all_onset_data', []),
                        'sensitive_onset_data': result.get('sensitive_onset_data', []),
                        'spectral_onset_data': result.get('spectral_onset_data', []),
                        'spectral_config': result.get('spectral_config'),
                        'pga_onset_data': result.get('pga_onset_data', []),
                    }
                # Store envelope data for waveform visualization
                if result.get('envelope_data'):
                    envelope_by_stem[stem_type] = result['envelope_data']
                # 2026-06-19: store broadband contrast envelope
                # (PGA STFT) for post-hoc walk diagnostics.
                if result.get('pga_envelope_data'):
                    contrast_envelope_by_stem[stem_type] = (
                        result['pga_envelope_data']
                    )

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

            # 2026-06-30: when the user invoked --stems <subset>, the
            # per-stem loop above only processed those stems. Without
            # this block, save_analysis_sidecar would overwrite the
            # existing sidecar with just the re-processed stems,
            # erasing kick/hihat/toms/cymbals from the sidecar (and
            # therefore the MIDI — the rebuild step reads the sidecar).
            #
            # Load the existing sidecar and merge its non-reprocessed
            # stems into events_by_stem + analysis_by_stem. Learning
            # mode is excluded: --learn changes per-event semantics
            # (velocity=1 for FPs, etc.) so merging the pre-learn
            # sidecar would corrupt the learning-mode output.
            if not learning_mode:
                # 2026-06-30: pass midi_path (the .mid path), NOT
                # midi_path.with_suffix('.analysis.json'). The loader
                # applies with_suffix internally — if we pre-apply it,
                # the loader looks for .analysis.analysis.json which
                # doesn't exist and returns None. (Caught after the
                # first attempt with double-debug output.)
                existing_sidecar = (
                    load_analysis_sidecar(midi_path) or {}
                )
                stems_to_preserve = [
                    s for s in (existing_sidecar.get('stems') or {}).keys()
                    if s not in stems_to_process
                ]
                if stems_to_preserve:
                    preserved_midi, preserved_analysis = (
                        _deserialize_sidecar_stems_for_merge(
                            existing_sidecar,
                            stems_to_preserve=stems_to_preserve,
                            config=config,
                        )
                    )
                    if preserved_midi:
                        events_by_stem.update(preserved_midi)
                        print(
                            f"  Preserving {len(preserved_midi)} non-reprocessed stem(s) "
                            f"from existing sidecar: {sorted(preserved_midi.keys())}"
                        )
                    if preserved_analysis:
                        analysis_by_stem.update(preserved_analysis)

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

            # 2026-06-19: save broadband contrast envelopes
            # (PGA STFT) for post-hoc per-event walk
            # diagnostics. ~30-50KB per stem compressed.
            if contrast_envelope_by_stem:
                save_contrast_envelope(contrast_envelope_by_stem, midi_path)

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


def _deserialize_sidecar_stems_for_merge(
    existing_sidecar: Dict,
    stems_to_preserve: List[str],
    config: Optional[Dict] = None,
) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict]]:
    """Build ``(midi_events_by_stem, analysis_by_stem)`` for stems
    that were NOT re-processed in the current run, so the new
    sidecar preserves them instead of erasing them.

    2026-06-30: the CLI's ``--stems <subset>`` flag re-runs detection
    on the requested stems and calls ``save_analysis_sidecar`` with
    only those stems — erasing the others from the sidecar and the
    MIDI. The fix is to load the existing sidecar before saving,
    take its ``events_pga`` for the non-reprocessed stems, and pass
    it through as both the MIDI events (KEPT subset) and the
    analysis dict (full events_pga) so the existing save pipeline
    serializes everything unchanged.

    Pure function. No I/O, no audio, no mutation of inputs.

    Defined BEFORE the ``if __name__ == '__main__':`` block (which
    sits below this helper) because the ``__main__`` block runs at
    module-load time when the file is invoked directly. The first
    attempt put the helper after ``__main__`` and the call from
    ``_process_stems_to_midi`` failed with NameError because Python
    had not yet executed the def statement below ``__main__``.

    Args:
        existing_sidecar: The output of ``load_analysis_sidecar`` —
            a sidecar-shaped dict with top-level ``version`` /
            ``tempo_bpm`` / ``stems``. ``stems`` maps stem type to
            ``{events_pga, events_configured?, logic?, ...}``.
        stems_to_preserve: Stems whose existing sidecar data should
            be carried through to the new sidecar. Typically
            ``[s for s in existing_sidecar['stems'] if s not in stems_to_process]``.
        config: The current run's config dict. Used to resolve
            ``<stem>.max_note_duration`` (per-stem wins over global
            ``midi.max_note_duration``, default 0.5) and
            ``audio.default_note_duration`` (default 0.1) when
            reconstructing MIDI events from KEPT events that lack
            a ``duration_ms`` field (older sidecars pre-2026-06-19).

    Returns:
        Tuple of:
          - ``midi_events_by_stem``: ``{stem: [midi_event, ...]}``
            where each ``midi_event`` has ``time``, ``note``,
            ``velocity`` (mapped from ``midi_velocity``),
            ``duration`` (mapped from ``duration_ms`` / 1000,
            clamped to max_note_duration), and ``hihat_state`` (for
            hihat only). Only KEPT events are included — the MIDI
            only carries notes for KEPT events.
          - ``analysis_by_stem``: ``{stem: {'pga_onset_data': events_pga}}``
            where ``events_pga`` is the FULL list (KEPT + FILTERED
            + all per-event diagnostic fields). The re-serialize
            step in ``save_analysis_sidecar`` handles the rest.
    """
    midi_events_by_stem: Dict[str, List[Dict]] = {}
    analysis_by_stem: Dict[str, Dict] = {}

    if not existing_sidecar:
        return midi_events_by_stem, analysis_by_stem

    existing_stems = existing_sidecar.get('stems', {}) or {}
    preserve_set = set(stems_to_preserve or [])

    # Default-duration resolution matches the live pipeline
    # (process_percentile_gated.py:218-220).
    global_max_note_duration = (
        (config or {}).get('midi', {}).get('max_note_duration', 0.5)
    )
    default_note_duration = (
        (config or {}).get('audio', {}).get('default_note_duration', 0.1)
    )

    for stem_type, stem_data in existing_stems.items():
        if stem_type not in preserve_set:
            continue

        events_pga = stem_data.get('events_pga', []) or []
        if not events_pga:
            continue

        # Per-stem max_note_duration wins over global default.
        per_stem_max = (
            (config or {}).get(stem_type, {}).get('max_note_duration')
        )
        max_note_duration = (
            per_stem_max if per_stem_max is not None
            else global_max_note_duration
        )

        midi_events: List[Dict] = []
        for ev in events_pga:
            if ev.get('status') != 'KEPT':
                continue  # MIDI only carries notes for KEPT events

            # Duration: prefer duration_ms; fall back to default_note_duration
            # for older sidecars that pre-date the 2026-06-19 duration field.
            raw_duration_ms = ev.get('duration_ms')
            if raw_duration_ms is None:
                duration_sec = float(default_note_duration)
            else:
                duration_sec = float(raw_duration_ms) / 1000.0
            duration_sec = min(duration_sec, float(max_note_duration))

            midi_event: Dict = {
                'time': float(ev['time']),
                'note': ev.get('note'),
                'velocity': int(ev.get('midi_velocity', 80)),
                'duration': float(duration_sec),
            }
            # hihat_state only applies to hihat; carry through if present
            # so the MIDI loop can pick note_open vs default hihat note.
            if stem_type == 'hihat' and ev.get('hihat_state') is not None:
                midi_event['hihat_state'] = ev['hihat_state']
            midi_events.append(midi_event)

        if midi_events:
            midi_events_by_stem[stem_type] = midi_events
        # Pass the full events_pga (KEPT + FILTERED) through as
        # pga_onset_data — save_analysis_sidecar's existing serialization
        # path consumes this and preserves all per-event fields
        # (stereo_width, pitch_hz, classification, filter_reason, etc.).
        analysis_by_stem[stem_type] = {'pga_onset_data': events_pga}

    return midi_events_by_stem, analysis_by_stem


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
