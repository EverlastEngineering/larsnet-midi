"""
Sidechain compression to reduce bleed between stems.

Uses the separated snare track as a sidechain trigger to duck the kick track
when the snare is playing, effectively removing snare bleed from the kick.

Uses project-based workflow: automatically detects projects with stems
and creates cleaned versions in the project/cleaned/ directory.

Usage:
    python sidechain_cleanup.py              # Auto-detect project
    python sidechain_cleanup.py 1            # Process specific project
"""

from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal
import argparse
import sys
from typing import Union, Tuple

# Import project manager
from project_manager import (
    discover_projects,
    select_project,
    get_project_by_number,
    get_project_config,
    update_project_metadata,
    USER_FILES_DIR
)


def envelope_follower(audio: np.ndarray, sr: int, attack_ms: float = 5.0, release_ms: float = 50.0) -> np.ndarray:
    """
    Create an envelope follower for the audio signal.
    
    Args:
        audio: Input audio (mono or stereo)
        sr: Sample rate
        attack_ms: Attack time in milliseconds
        release_ms: Release time in milliseconds
    
    Returns:
        Envelope of the audio signal
    """
    # Convert to mono if stereo
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    
    # Get absolute values (rectify)
    rectified = np.abs(audio)
    
    # Calculate coefficients
    attack_coef = np.exp(-1.0 / (sr * attack_ms / 1000.0))
    release_coef = np.exp(-1.0 / (sr * release_ms / 1000.0))
    
    # Apply envelope follower
    envelope = np.zeros_like(rectified)
    envelope[0] = rectified[0]
    
    for i in range(1, len(rectified)):
        if rectified[i] > envelope[i-1]:
            # Attack
            envelope[i] = attack_coef * envelope[i-1] + (1 - attack_coef) * rectified[i]
        else:
            # Release
            envelope[i] = release_coef * envelope[i-1] + (1 - release_coef) * rectified[i]
    
    return envelope


def sidechain_compress(
    main_audio: np.ndarray,
    sidechain_audio: np.ndarray,
    sr: int,
    threshold_db: float = -30.0,
    ratio: float = 10.0,
    attack_ms: float = 1.0,
    release_ms: float = 100.0,
    makeup_gain_db: float = 0.0,
    knee_db: float = 3.0
) -> np.ndarray:
    """
    Apply sidechain compression to main audio based on sidechain audio.
    
    Args:
        main_audio: Audio to be compressed (kick track)
        sidechain_audio: Audio that triggers compression (snare track)
        sr: Sample rate
        threshold_db: Threshold in dB below which no compression occurs
        ratio: Compression ratio (higher = more aggressive ducking)
        attack_ms: How quickly compression kicks in when snare hits
        release_ms: How quickly compression releases after snare stops
        makeup_gain_db: Gain to apply after compression
        knee_db: Soft knee width in dB
    
    Returns:
        Compressed audio
    """
    # Get envelope of sidechain (snare)
    sidechain_envelope = envelope_follower(sidechain_audio, sr, attack_ms, release_ms)
    
    # Convert to dB
    epsilon = 1e-10
    sidechain_db = 20 * np.log10(sidechain_envelope + epsilon)
    
    # Calculate gain reduction
    threshold = threshold_db
    gain_reduction_db = np.zeros_like(sidechain_db)
    
    for i in range(len(sidechain_db)):
        if sidechain_db[i] > threshold + knee_db:
            # Above knee - full compression
            over_threshold = sidechain_db[i] - threshold
            gain_reduction_db[i] = -over_threshold * (1 - 1/ratio)
        elif sidechain_db[i] > threshold - knee_db:
            # In knee - soft compression
            over_threshold = sidechain_db[i] - threshold + knee_db
            gain_reduction_db[i] = -over_threshold**2 * (1 - 1/ratio) / (4 * knee_db)
        # else: below threshold, no gain reduction
    
    # Convert gain reduction to linear
    gain_linear = 10 ** (gain_reduction_db / 20.0)
    
    # Apply makeup gain
    makeup_gain_linear = 10 ** (makeup_gain_db / 20.0)
    gain_linear *= makeup_gain_linear
    
    # Apply gain reduction to main audio
    if main_audio.ndim == 2:
        # Stereo - apply same gain to both channels
        compressed = main_audio * gain_linear[:, np.newaxis]
    else:
        # Mono
        compressed = main_audio * gain_linear
    
    return compressed


def process_stems(
    stems_dir: Union[str, Path],
    output_dir: Union[str, Path],
    threshold_db: float = -30.0,
    ratio: float = 10.0,
    attack_ms: float = 1.0,
    release_ms: float = 100.0,
    dry_wet: float = 1.0
):
    """
    Process separated stems to remove bleed using sidechain compression.
    
    Args:
        stems_dir: Directory containing separated stems (should have subdirs like: trackname/trackname-kick.wav)
        output_dir: Directory to save cleaned stems
        threshold_db: Sidechain threshold in dB
        ratio: Compression ratio (higher = more aggressive)
        attack_ms: Attack time in milliseconds
        release_ms: Release time in milliseconds
        dry_wet: Mix between original (0.0) and processed (1.0)
    """
    stems_dir = Path(stems_dir)
    output_dir = Path(output_dir)
    
    # Find all subdirectories containing stems (new structure: stems_dir/trackname/)
    track_dirs = [d for d in stems_dir.iterdir() if d.is_dir()]
    
    if not track_dirs:
        raise RuntimeError(f"No subdirectories found in {stems_dir}")
    
    # Find tracks that have both kick and snare files
    tracks_to_process = []
    for track_dir in track_dirs:
        base_name = track_dir.name
        kick_file = track_dir / f"{base_name}-kick.wav"
        snare_file = track_dir / f"{base_name}-snare.wav"
        
        if kick_file.exists() and snare_file.exists():
            tracks_to_process.append(track_dir)
        else:
            print(f"Warning: Skipping {base_name} - missing kick or snare file")
    
    if not tracks_to_process:
        raise RuntimeError(f"No tracks found with both kick and snare files in {stems_dir}")
    
    print(f"Processing {len(tracks_to_process)} track(s)...")
    print(f"Settings:")
    print(f"  Threshold: {threshold_db} dB")
    print(f"  Ratio: {ratio}:1")
    print(f"  Attack: {attack_ms} ms")
    print(f"  Release: {release_ms} ms")
    print(f"  Dry/Wet: {dry_wet * 100:.0f}% processed")
    print()
    
    for track_dir in tracks_to_process:
        base_name = track_dir.name
        kick_file = track_dir / f"{base_name}-kick.wav"
        snare_file = track_dir / f"{base_name}-snare.wav"
        
        print(f"Processing: {base_name}")
        
        # Load audio files
        kick_audio, sr = sf.read(str(kick_file))
        snare_audio, sr_snare = sf.read(str(snare_file))
        
        if sr != sr_snare:
            print(f"  Warning: Sample rate mismatch! Kick: {sr}Hz, Snare: {sr_snare}Hz")
            continue
        
        # Ensure same length
        min_length = min(len(kick_audio), len(snare_audio))
        kick_audio = kick_audio[:min_length]
        snare_audio = snare_audio[:min_length]
        
        # Apply sidechain compression
        print(f"  Applying sidechain compression...")
        kick_compressed = sidechain_compress(
            kick_audio,
            snare_audio,
            sr,
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms
        )
        
        # Dry/wet mix
        kick_final = dry_wet * kick_compressed + (1 - dry_wet) * kick_audio
        
        # Save - create output directory for this track
        output_track_dir = output_dir / base_name
        output_track_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_track_dir / f'{base_name}-kick.wav'
        
        sf.write(str(output_file), kick_final, sr)
        print(f"  Saved: {output_file}")
        
        # Copy other stems unchanged
        for stem_name in ['snare', 'toms', 'hihat', 'cymbals']:
            stem_file = track_dir / f"{base_name}-{stem_name}.wav"
            if stem_file.exists():
                # Copy unchanged
                stem_audio, stem_sr = sf.read(str(stem_file))
                output_file = output_track_dir / f'{base_name}-{stem_name}.wav'
                sf.write(str(output_file), stem_audio, stem_sr)
    
    print(f"\nDone! Processed stems saved to: {output_dir}")


def cleanup_project_stems(
    project_number: int = None,
    threshold_db: float = -30.0,
    ratio: float = 10.0,
    attack_ms: float = 1.0,
    release_ms: float = 100.0,
    dry_wet: float = 1.0
):
    """
    Clean up stems for a project using sidechain compression.
    
    Args:
        project_number: Specific project to process, or None for auto-select
        threshold_db: Sidechain threshold in dB
        ratio: Compression ratio (higher = more aggressive)
        attack_ms: Attack time in milliseconds
        release_ms: Release time in milliseconds
        dry_wet: Mix between original (0.0) and processed (1.0)
    """
    # Select project
    project_info = select_project(project_number)
    if not project_info:
        print("No project available for cleanup.")
        sys.exit(1)
    
    project_folder = project_info['path']
    song_name = project_info['name']
    
    print(f"\n{'='*60}")
    print(f"Sidechain Cleanup: {song_name}")
    print(f"Project: {project_folder.name}")
    print(f"{'='*60}\n")
    
    # Check for stems directory
    stems_dir = project_folder / 'stems'
    if not stems_dir.exists():
        print(f"❌ No stems directory found in project.")
        print(f"   Expected: {stems_dir}")
        print(f"   Run separate.py first to generate stems.")
        sys.exit(1)
    
    # Create cleaned directory
    cleaned_dir = project_folder / 'cleaned'
    cleaned_dir.mkdir(exist_ok=True)
    
    # Process the stems
    process_stems(
        stems_dir=stems_dir,
        output_dir=cleaned_dir,
        threshold_db=threshold_db,
        ratio=ratio,
        attack_ms=attack_ms,
        release_ms=release_ms,
        dry_wet=dry_wet
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Remove snare bleed from kick track using sidechain compression.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process most recent project with default settings
  python sidechain_cleanup.py
  
  # Process specific project
  python sidechain_cleanup.py 1
  
  # Aggressive ducking
  python sidechain_cleanup.py -t -40 -r 20 --attack 0.5
  
  # Gentle/subtle ducking
  python sidechain_cleanup.py -t -25 -r 4 --dry-wet 0.5
        """
    )
    
    parser.add_argument('project_number', type=int, nargs='?', default=None,
                        help="Project number to process (auto-selects most recent if not provided).")
    parser.add_argument('-t', '--threshold', type=float, default=-30.0,
                        help="Sidechain threshold in dB. Lower = more sensitive (default: -30).")
    parser.add_argument('-r', '--ratio', type=float, default=10.0,
                        help="Compression ratio. Higher = more aggressive ducking (default: 10).")
    parser.add_argument('--attack', type=float, default=1.0,
                        help="Attack time in milliseconds. Lower = faster response (default: 1).")
    parser.add_argument('--release', type=float, default=100.0,
                        help="Release time in milliseconds. Lower = faster recovery (default: 100).")
    parser.add_argument('--dry-wet', type=float, default=1.0,
                        help="Mix between original (0.0) and processed (1.0). Default: 1.0 (fully processed).")
    
    args = parser.parse_args()
    
    # Validate
    if not (0.0 <= args.dry_wet <= 1.0):
        parser.error("--dry-wet must be between 0.0 and 1.0")
    if args.ratio < 1.0:
        parser.error("--ratio must be >= 1.0")
    
    cleanup_project_stems(
        project_number=args.project_number,
        threshold_db=args.threshold,
        ratio=args.ratio,
        attack_ms=args.attack,
        release_ms=args.release,
        dry_wet=args.dry_wet
    )
