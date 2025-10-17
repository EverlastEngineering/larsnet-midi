"""
Convert separated drum stems to MIDI tracks.

Main entry point and command-line interface for the stems-to-MIDI conversion system.

Architecture: Modular Design (Functional Core, Imperative Shell)
- stems_to_midi_config.py: Configuration loading and drum mapping
- stems_to_midi_detection.py: Onset detection and drum classification
- stems_to_midi_helpers.py: Pure helper functions (testable, no side effects)
- stems_to_midi_midi.py: MIDI file creation
- stems_to_midi_learning.py: Threshold learning and calibration
- stems_to_midi_processor.py: Main audio processing pipeline
- stems_to_midi.py (this file): CLI orchestration (thin imperative shell)
"""

from pathlib import Path
import argparse
from typing import Union, List

# Import modules (thin orchestration layer)
from stems_to_midi.config import load_config, DrumMapping
from stems_to_midi.midi import create_midi_file
from stems_to_midi.learning import learn_threshold_from_midi, save_calibrated_config
from stems_to_midi.processor import process_stem_to_midi


def stems_to_midi(
    stems_dir: Union[str, Path],
    output_dir: Union[str, Path],
    onset_threshold: float,
    onset_delta: float,
    onset_wait: int,
    hop_length: int,
    min_velocity: int = 80,
    max_velocity: int = 110,
    tempo: float = 120.0,
    detect_hihat_open: bool = False,
    stems_to_process: List[str] = None
):
    """
    Convert separated drum stems to MIDI files.
    
    Args:
        stems_dir: Directory containing separated stems (subdirs: kick/, snare/, etc.)
        output_dir: Directory to save MIDI files
        onset_threshold: Threshold for onset detection (0-1, lower = more sensitive)
        min_velocity: Minimum MIDI velocity
        max_velocity: Maximum MIDI velocity
        tempo: Tempo in BPM (for MIDI timing)
        detect_hihat_open: Try to detect open hi-hat hits
        stems_to_process: List of stem types to process (default: all)
    """
    stems_dir = Path(stems_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    try:
        config = load_config()
        print(f"Loaded configuration from: midiconfig.yaml")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Creating default config file...")
        # Config file will be created by default in the same directory
        raise
    
    # Default stems to process
    if stems_to_process is None:
        stems_to_process = ['kick', 'snare', 'toms', 'hihat', 'cymbals']
    
    # Initialize drum mapping
    drum_mapping = DrumMapping()
    
    # Find all audio file directories (new structure: stems_dir/input_name/)
    # Each subdirectory should contain files like input_name-kick.wav, input_name-snare.wav, etc.
    audio_dirs = [d for d in stems_dir.iterdir() if d.is_dir()]
    
    if not audio_dirs:
        raise RuntimeError(f"No subdirectories found in {stems_dir}")
    
    audio_files_to_process = []
    for audio_dir in audio_dirs:
        # Check if this directory has the stem files we need
        has_stems = False
        for stem_type in stems_to_process:
            expected_file = audio_dir / f"{audio_dir.name}-{stem_type}.wav"
            if expected_file.exists():
                has_stems = True
                break
        
        if has_stems:
            audio_files_to_process.append(audio_dir)
    
    if not audio_files_to_process:
        raise RuntimeError(f"No valid stem directories found in {stems_dir} with expected naming pattern (e.g., name/name-kick.wav)")
    
    # Set onset detection params from config if not provided
    if onset_threshold is None:
        onset_threshold = config['onset_detection']['threshold']
    if onset_delta is None:
        onset_delta = config['onset_detection']['delta']
    if onset_wait is None:
        onset_wait = config['onset_detection']['wait']
    if hop_length is None:
        hop_length = config['onset_detection']['hop_length']
    print(f"Processing {len(audio_files_to_process)} file(s)...")
    print(f"Settings:")
    print(f"  Onset threshold: {onset_threshold}")
    print(f"  Onset delta: {onset_delta}")
    print(f"  Onset wait: {onset_wait}")
    print(f"  Hop length: {hop_length}")
    print(f"  Velocity range: {min_velocity}-{max_velocity}")
    print(f"  Tempo: {tempo} BPM")
    print(f"  Detect open hi-hat: {detect_hihat_open}")
    print()
    
    for audio_dir in audio_files_to_process:
        base_name = audio_dir.name
        print(f"Processing: {base_name}")
        
        events_by_stem = {}
        
        # Process each stem type
        for stem_type in stems_to_process:
            stem_file = audio_dir / f"{base_name}-{stem_type}.wav"
            
            if not stem_file.exists():
                print(f"  Warning: {stem_type} file not found, skipping...")
                continue
            
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
                detect_hihat_open=hihat_detect
            )
            
            if events:
                events_by_stem[stem_type] = events
        
        # Create MIDI file
        if events_by_stem:
            # Add suffix for learning mode
            learning_mode = config.get('learning_mode', {}).get('enabled', False)
            if learning_mode:
                suffix = config['learning_mode']['learning_midi_suffix']
                midi_path = output_dir / f"{base_name}{suffix}.mid"
            else:
                midi_path = output_dir / f"{base_name}.mid"
            
            create_midi_file(
                events_by_stem,
                midi_path,
                tempo=tempo,
                track_name=f"Drums - {base_name}",
                config=config
            )
            
            if learning_mode:
                print(f"  Saved LEARNING MIDI: {midi_path}")
                print(f"  ** Load in DAW, delete false positives (velocity=1 hits), save as: {audio_file.stem}_edited.mid **\n")
            else:
                print(f"  Saved: {midi_path}\n")
        else:
            print(f"  No events detected, skipping MIDI creation\n")
    
    print(f"Done! MIDI files saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert separated drum stems to MIDI tracks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python stems_to_midi.py -i cleaned_stems/ -o midi_output/
  
  # More sensitive onset detection
  python stems_to_midi.py -i cleaned_stems/ -o midi_output/ -t 0.2
  
  # Less sensitive (fewer false positives)
  python stems_to_midi.py -i cleaned_stems/ -o midi_output/ -t 0.5
  
  # Full velocity range
  python stems_to_midi.py -i cleaned_stems/ -o midi_output/ --min-vel 1 --max-vel 127
  
  # Specific tempo
  python stems_to_midi.py -i cleaned_stems/ -o midi_output/ --tempo 140

MIDI Note Mapping (General MIDI):
  Kick:    36 (C1)  - Bass Drum 1
  Snare:   38 (D1)  - Acoustic Snare
  Toms:    45 (A1)  - Low Tom
  Hi-Hat:  42 (F#1) - Closed Hi-Hat
           46 (A#1) - Open Hi-Hat
  Cymbals: 49 (C#2) - Crash Cymbal 1
        """
    )
    
    parser.add_argument('-i', '--input_dir', type=str, required=False,
                        help="Directory containing separated stems (must have kick/, snare/, etc. subdirectories).")
    parser.add_argument('-o', '--output_dir', type=str, default='midi_output',
                        help="Directory to save MIDI files (default: midi_output).")
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
    learning_group.add_argument('--learn-from-midi', type=str, nargs=3, metavar=('AUDIO', 'ORIG_MIDI', 'EDITED_MIDI'),
                               help="Learn thresholds from edited MIDI. Args: audio_file original_midi edited_midi")
    learning_group.add_argument('--learn-stem', type=str, default='snare',
                               choices=['kick', 'snare', 'toms', 'hihat', 'cymbals'],
                               help="Stem type for learning (default: snare).")
    
    args = parser.parse_args()
    
    # Validate that -i is provided unless using --learn-from-midi
    if not args.learn_from_midi and not args.input_dir:
        parser.error("-i/--input_dir is required unless using --learn-from-midi")
    
    # Load config to get default tempo if not specified
    if args.tempo is None:
        config = load_config()
        args.tempo = config['midi']['default_tempo']
        print(f"Using tempo from config: {args.tempo} BPM\n")
    
    # Validate
    if args.threshold is not None and not (0.0 <= args.threshold <= 1.0):
        parser.error("--threshold must be between 0.0 and 1.0")
    if not (1 <= args.min_vel <= 127):
        parser.error("--min-vel must be between 1 and 127")
    if not (1 <= args.max_vel <= 127):
        parser.error("--max-vel must be between 1 and 127")
    if args.min_vel > args.max_vel:
        parser.error("--min-vel cannot be greater than --max-vel")
        parser.add_argument('--delta', type=float, default=None,
                            help="Onset peak picking sensitivity (default: from config). Lower = more sensitive.")
        parser.add_argument('--wait', type=int, default=None,
                            help="Minimum frames between peaks (default: from config). 1 = ~11ms.")
        parser.add_argument('--hop-length', type=int, default=None,
                            help="Samples between frames (default: from config). Affects time resolution.")
        parser.add_argument('-t', '--threshold', type=float, default=None,
                            help="Samples between frames (default: from config). Affects time resolution.")
    # Set detection params from args or config for all entry points
    config = load_config()
    onset_threshold = args.threshold if args.threshold is not None else config['onset_detection']['threshold']
    onset_delta = args.delta if args.delta is not None else config['onset_detection']['delta']
    onset_wait = args.wait if args.wait is not None else config['onset_detection']['wait']
    hop_length = args.hop_length if args.hop_length is not None else config['onset_detection']['hop_length']

    if args.learn_from_midi:
        # Learn thresholds from edited MIDI
        from pathlib import Path
        audio_file = Path(args.learn_from_midi[0])
        orig_midi = Path(args.learn_from_midi[1])
        edited_midi = Path(args.learn_from_midi[2])
        if not audio_file.exists():
            parser.error(f"Audio file not found: {audio_file}")
        if not orig_midi.exists():
            parser.error(f"Original MIDI not found: {orig_midi}")
        if not edited_midi.exists():
            parser.error(f"Edited MIDI not found: {edited_midi}")
        drum_mapping = DrumMapping()
        learned = learn_threshold_from_midi(
            audio_file,
            orig_midi,
            edited_midi,
            args.learn_stem,
            config,
            drum_mapping
        )
        if learned:
            output_config = Path(config['learning_mode']['calibrated_config_output'])
            save_calibrated_config(config, {args.learn_stem: learned}, output_config)
    elif args.learn:
        import tempfile
        import shutil
        config['learning_mode']['enabled'] = True
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            temp_config = f.name
        try:
            print("=== LEARNING MODE ENABLED ===")
            print("All detections will be exported. Rejected hits have velocity=1.")
            print("Load MIDI in DAW, delete false positives, save as *_edited.mid")
            print("Then run: python stems_to_midi.py --learn-from-midi <audio> <original_midi> <edited_midi>\n")
            stems_to_midi(
                stems_dir=args.input_dir,
                output_dir=args.output_dir,
                onset_threshold=onset_threshold,
                onset_delta=onset_delta,
                onset_wait=onset_wait,
                hop_length=hop_length,
                min_velocity=args.min_vel,
                max_velocity=args.max_vel,
                tempo=args.tempo,
                detect_hihat_open=args.detect_hihat_open,
                stems_to_process=args.stems
            )
        finally:
            Path(temp_config).unlink()
    else:
        stems_to_midi(
            stems_dir=args.input_dir,
            output_dir=args.output_dir,
            onset_threshold=onset_threshold,
            onset_delta=onset_delta,
            onset_wait=onset_wait,
            hop_length=hop_length,
            min_velocity=args.min_vel,
            max_velocity=args.max_vel,
            tempo=args.tempo,
            detect_hihat_open=args.detect_hihat_open,
            stems_to_process=args.stems
        )
