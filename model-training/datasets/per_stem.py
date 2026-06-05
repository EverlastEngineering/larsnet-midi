"""
Per-stem MIDI filter and target encoder for approach 05.

A "stem" is one of: kick, snare, hihat, toms, cymbals.
For each stem, we have a list of MIDI pitches that belong to it.
Per-stem training filters the ground-truth MIDI to only events whose
pitch is in the stem's pitch set, and encodes those as a 2-channel
target tensor (onset probability + velocity).

Per-stem model outputs:
  channel 0: onset probability (binary)
  channel 1: velocity value (0.0 - 1.0)
"""

from typing import Any, List, Optional
import torch

# MIDI pitches that belong to each stem (per model-training/label_encoder.py:27)
STEM_PITCHES = {
    "kick":    [35, 36],
    "snare":   [37, 38, 39, 40],   # includes clap & rim
    "hihat":   [22, 26, 42, 44, 46],  # closed, pedal, open
    "toms":    [41, 43, 45, 47, 48, 50],  # all 3 toms
    "cymbals": [49, 51, 52, 53, 55, 57, 59],  # crash, ride, china, splash
}

# Canonical MIDI pitch emitted for each stem hit (used for MIDI output).
# For multi-pitch stems we just emit the lowest pitch; subclassification
# (snare vs rim, etc.) is a future enhancement.
STEM_CANONICAL_PITCH = {
    "kick":    36,
    "snare":   38,
    "hihat":   42,
    "toms":    47,
    "cymbals": 51,
}

# Onset buffer in frames (causal smear for per-stem is shorter: [1.0, 0.5]
# because per-stem models train on isolated audio without class confusion,
# so the smear is less necessary).
PER_STEM_SMEAR = [1.0, 0.5]


def filter_notes_to_stem(midi_notes: List[Any], stem: str) -> List[Any]:
    """Return only the notes whose pitch is in STEM_PITCHES[stem]."""
    pitches = set(STEM_PITCHES[stem])
    return [n for n in midi_notes if getattr(n, "pitch", None) in pitches]


def build_per_stem_targets(
    stem_notes: List[Any],
    total_frames: int,
    hop_length: int = 512,
    sr: int = 44100,
    smear: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Build a [2, total_frames] target tensor for a single stem.

    channel 0: binary onset with optional causal smear
    channel 1: normalized velocity at exact hit frames, 0 elsewhere
    """
    if smear is None:
        smear = PER_STEM_SMEAR

    targets = torch.zeros((2, total_frames))
    seconds_per_frame = hop_length / sr

    for note in stem_notes:
        hit_frame = int(getattr(note, "start_time", 0.0) / seconds_per_frame)
        if not (0 <= hit_frame < total_frames):
            continue
        targets[0, hit_frame] = 1.0
        for offset, val in enumerate(smear):
            f = hit_frame + offset
            if 0 <= f < total_frames:
                targets[0, f] = max(targets[0, f], val)
        velocity = getattr(note, "velocity", 100)
        targets[1, hit_frame] = (velocity / 127.0) ** 0.7

    return targets


def load_stem_pairs(project_dir: str) -> dict:
    """
    Given a project directory (with /stems/<name>-{stem}.wav and /midi/<name>.mid),
    return a dict of {stem: (stem_wav_path, midi_path)} pairs.

    Skips stems that don't have a corresponding WAV.
    """
    from pathlib import Path
    project = Path(project_dir)
    stems_dir = project / "stems"
    midi_dir = project / "midi"

    # Find the MIDI file (project.mid or <name>.mid)
    midi_files = list(midi_dir.glob("*.mid"))
    if not midi_files:
        raise FileNotFoundError(f"No .mid in {midi_dir}")
    midi_path = midi_files[0]
    base_name = midi_path.stem  # e.g. "Metallica_Cyanide_Drums"

    pairs = {}
    for stem in STEM_PITCHES:
        wav_path = stems_dir / f"{base_name}-{stem}.wav"
        if wav_path.exists():
            pairs[stem] = (str(wav_path), str(midi_path))
    return pairs


if __name__ == "__main__":
    # Quick sanity check
    from pathlib import Path
    project = "/Users/jasoncopp/Source/GitHub/larsnet/user_files/3 - Metallica_Cyanide_Drums"
    pairs = load_stem_pairs(project)
    print(f"Discovered {len(pairs)} stem pairs in {project}:")
    for stem, (wav, midi) in pairs.items():
        print(f"  {stem:8s} -> {Path(wav).name}")
    print(f"\nCanonical pitch per stem: {STEM_CANONICAL_PITCH}")
