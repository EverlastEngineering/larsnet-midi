"""
Convert separated drum stems to MIDI tracks.

Uses project-based workflow: automatically detects projects with stems
and generates MIDI files in the project/midi/ directory.

Architecture: Modular Design (Functional Core, Imperative Shell)
- stems_to_midi/ submodules: Core conversion logic
- project_manager: Project discovery and management
- stems_to_midi.py (this file): CLI orchestration

Usage:
    python stems_to_midi.py              # Auto-detect project
    python stems_to_midi.py 1            # Process specific project
    python stems_to_midi.py --learn      # Learning mode
"""

from pathlib import Path
import argparse
from typing import Union, List
import sys

# Import modules (thin orchestration layer)
from stems_to_midi.config import load_config, DrumMapping
from stems_to_midi.midi import create_midi_file
from stems_to_midi.learning import learn_threshold_from_midi, save_calibrated_config
from stems_to_midi.processor import process_stem_to_midi

# Import project manager
from project_manager import (
    discover_projects,
    select_project,
    get_project_by_number,
    get_project_config,
    update_project_metadata,
    USER_FILES_DIR
)


def stems_to_midi_for_project(
    project: dict,
    onset_threshold: float = None,
    onset_delta: float = None,
    onset_wait: int = None,
    hop_length: int = None,
    min_velocity: int = 80,
    max_velocity: int = 110,
    tempo: float = None,
    detect_hihat_open: bool = False,
    stems_to_process: List[str] = None,
    max_duration: float = None,
    learning_mode: bool = False
):
    """
    Convert separated drum stems to MIDI files for a specific project.
    
    Args:
        project: Project info dictionary from project_manager
        onset_threshold: Threshold for onset detection (None = use config)
        onset_delta: Peak picking sensitivity (None = use config)
        onset_wait: Minimum frames between peaks (None = use config)
        hop_length: Samples between frames (None = use config)
        min_velocity: Minimum MIDI velocity
        max_velocity: Maximum MIDI velocity
        tempo: Tempo in BPM (None = use config)
        detect_hihat_open: Try to detect open hi-hat hits
        stems_to_process: List of stem types to process (default: all)
        max_duration: Maximum duration in seconds (for faster learning)
        learning_mode: Enable learning mode (export all detections)
    """
    project_dir = project["path"]
    
    print(f"\n{'='*60}")
    print(f"Converting Stems to MIDI - Project {project['number']}: {project['name']}")
    print(f"{'='*60}\n")
    
    # Load project-specific config
    config_path = get_project_config(project_dir, "midiconfig.yaml")
    if config_path is None:
        print("ERROR: midiconfig.yaml not found in project or root directory")
        sys.exit(1)
    
    print(f"Using config: {config_path}")
    
    # Load configuration
    try:
        import yaml # type: ignore
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        sys.exit(1)
    
    # Use cleaned stems if available, otherwise use regular stems
    stems_source = project_dir / "cleaned"
    if not stems_source.exists() or not any(stems_source.iterdir()):
        stems_source = project_dir / "stems"
    
    if not stems_source.exists():
        print(f"ERROR: No stems found in project. Run separate.py first.")
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
        onset_threshold=onset_threshold,
        onset_delta=onset_delta,
        onset_wait=onset_wait,
        hop_length=hop_length,
        min_velocity=min_velocity,
        max_velocity=max_velocity,
        tempo=tempo,
        detect_hihat_open=detect_hihat_open,
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
    
    print(f"Status Update: MIDI conversion complete!")
    print(f"  MIDI files saved to: {midi_dir}")
    print(f"  Project status updated\n")


def _process_stems_to_midi(
    stems_source: Path,
    midi_dir: Path,
    project_name: str,
    config: dict,
    onset_threshold: float,
    onset_delta: float,
    onset_wait: int,
    hop_length: int,
    min_velocity: int,
    max_velocity: int,
    tempo: float,
    detect_hihat_open: bool,
    stems_to_process: List[str],
    max_duration: float,
    learning_mode: bool
):
    """
    Internal function to process stems to MIDI (extracted from original stems_to_midi).
    
    This handles the core conversion logic, called by stems_to_midi_for_project().
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
    
    # Set onset detection params from config if not provided
    if onset_threshold is None:
        onset_threshold = config['onset_detection']['threshold']
    if onset_delta is None:
        onset_delta = config['onset_detection']['delta']
    if onset_wait is None:
        onset_wait = config['onset_detection']['wait']
    if hop_length is None:
        hop_length = config['onset_detection']['hop_length']
    if tempo is None:
        tempo = config['midi']['default_tempo']
    
    print(f"Settings:")
    print(f"  Onset threshold: {onset_threshold}")
    print(f"  Onset delta: {onset_delta}")
    print(f"  Onset wait: {onset_wait}")
    print(f"  Hop length: {hop_length}")
    print(f"  Velocity range: {min_velocity}-{max_velocity}")
    print(f"  Tempo: {tempo} BPM")
    print(f"  Detect open hi-hat: {detect_hihat_open}")
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
        print(f"No stem files found matching expected pattern (name-stemtype.wav)")
        return
    
    total_songs = len(files_by_song)
    for song_idx, (base_name, stem_files_dict) in enumerate(files_by_song.items(), 1):
        print(f"Processing: {base_name}")
        
        # Progress: start of song processing
        song_start_progress = int((song_idx - 1) / total_songs * 90)
        print(f"Progress: {song_start_progress}%")
        
        events_by_stem = {}
        
        # Process each stem type
        total_stems = len(stems_to_process)
        processed_stems = 0
        for stem_type in stems_to_process:
            if stem_type not in stem_files_dict:
                print(f"  Warning: {stem_type} file not found, skipping...")
                processed_stems += 1
                continue
            
            stem_file = stem_files_dict[stem_type]
            
            # For hihat, check config for detect_open setting (can be overridden by command-line flag)
            hihat_detect = detect_hihat_open
            if stem_type == 'hihat' and not detect_hihat_open:
                # If not set via command-line, check config
                hihat_detect = config.get('hihat', {}).get('detect_open', False)
            
            events = process_stem_to_midi(
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
                detect_hihat_open=hihat_detect,
                max_duration=max_duration
            )
            
            if events:
                events_by_stem[stem_type] = events
            
            # Progress: after each stem (0-90% of total)
            processed_stems += 1
            stem_progress = int((song_idx - 1) / total_songs * 90 + (processed_stems / total_stems) * (90 / total_songs))
            print(f"Progress: {stem_progress}%")
        
        # Create MIDI file
        if events_by_stem:
            # Add suffix for learning mode
            if learning_mode:
                suffix = config.get('learning_mode', {}).get('learning_midi_suffix', '_learning')
                midi_path = midi_dir / f"{base_name}{suffix}.mid"
            else:
                midi_path = midi_dir / f"{base_name}.mid"
            
            create_midi_file(
                events_by_stem,
                midi_path,
                tempo=tempo,
                track_name=f"Drums - {base_name}",
                config=config
            )
            
            # Progress: after MIDI creation (90-100% of total)
            midi_progress = int(90 + (song_idx / total_songs) * 10)
            print(f"Progress: {midi_progress}%")
            
            if learning_mode:
                print(f"  Saved LEARNING MIDI: {midi_path}")
                print(f"  ** Load in DAW, delete false positives (velocity=1 hits), save as: {base_name}_edited.mid **\n")
            else:
                print(f"  Saved: {midi_path}\n")
        else:
            print(f"  No events detected, skipping MIDI creation\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert separated drum stems to MIDI tracks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - auto-detect project
  python stems_to_midi.py
  
  # Process specific project
  python stems_to_midi.py 1
  
  # More sensitive onset detection
  python stems_to_midi.py -t 0.2
  
  # Less sensitive (fewer false positives)
  python stems_to_midi.py -t 0.5
  
  # Full velocity range
  python stems_to_midi.py --min-vel 1 --max-vel 127
  
  # Specific tempo
  python stems_to_midi.py --tempo 140
  
  # Learning mode with 50 seconds limit (faster for long tracks)
  python stems_to_midi.py --learn --maxtime 50

MIDI Note Mapping (General MIDI):
  Kick:    36 (C1)  - Bass Drum 1
  Snare:   38 (D1)  - Acoustic Snare
  Toms:    45 (A1)  - Low Tom
  Hi-Hat:  42 (F#1) - Closed Hi-Hat
           46 (A#1) - Open Hi-Hat
  Cymbals: 49 (C#2) - Crash Cymbal 1
        """
    )
    
    parser.add_argument('project_number', type=int, nargs='?', default=None,
                        help="Project number to process (optional)")
    parser.add_argument('-t', '--threshold', type=float, default=None,
                        help="Onset detection threshold (0-1, lower = more sensitive). If not specified, uses value from midiconfig.yaml.")
    parser.add_argument('--delta', type=float, default=None,
                        help="Peak picking sensitivity for onset detection (lower = more sensitive). If not specified, uses value from midiconfig.yaml.")
    parser.add_argument('--wait', type=int, default=None,
                        help="Minimum frames between detected peaks (controls minimum spacing, 1 ≈ 11ms). If not specified, uses value from midiconfig.yaml.")
    parser.add_argument('--hop-length', type=int, default=None,
                        help="Number of samples between frames for onset detection (affects time resolution). If not specified, uses value from midiconfig.yaml.")
    parser.add_argument('--min-vel', type=int, default=40,
                        help="Minimum MIDI velocity (1-127, default: 40).")
    parser.add_argument('--max-vel', type=int, default=127,
                        help="Maximum MIDI velocity (1-127, default: 127).")
    parser.add_argument('--tempo', type=float, default=None,
                        help="Tempo in BPM for MIDI timing (default: read from midiconfig.yaml).")
    parser.add_argument('--detect-hihat-open', action='store_true',
                        help="Enable open/closed hi-hat detection (disabled by default - most hits will be closed).")
    parser.add_argument('--stems', type=str, nargs='+',
                        choices=['kick', 'snare', 'toms', 'hihat', 'cymbals'],
                        help="Specific stems to process (default: all).")
    
    # Learning mode arguments
    learning_group = parser.add_argument_group('Threshold Learning Mode')
    learning_group.add_argument('--learn', action='store_true',
                               help="Enable learning mode (exports all detections, rejected=velocity 1).")
    learning_group.add_argument('--maxtime', type=float, default=None,
                               help="Maximum duration in seconds to analyze (for faster learning on long tracks).")
    
    args = parser.parse_args()
    
    # Validate
    if args.threshold is not None and not (0.0 <= args.threshold <= 1.0):
        print("ERROR: --threshold must be between 0.0 and 1.0")
        sys.exit(1)
    if not (1 <= args.min_vel <= 127):
        print("ERROR: --min-vel must be between 1 and 127")
        sys.exit(1)
    if not (1 <= args.max_vel <= 127):
        print("ERROR: --max-vel must be between 1 and 127")
        sys.exit(1)
    if args.min_vel > args.max_vel:
        print("ERROR: --min-vel cannot be greater than --max-vel")
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
    
    stems_to_midi_for_project(
        project=project,
        onset_threshold=args.threshold,
        onset_delta=args.delta,
        onset_wait=args.wait,
        hop_length=args.hop_length,
        min_velocity=args.min_vel,
        max_velocity=args.max_vel,
        tempo=args.tempo,
        detect_hihat_open=args.detect_hihat_open,
        stems_to_process=args.stems,
        max_duration=args.maxtime,
        learning_mode=args.learn
    )
