"""
Per-stem inference orchestrator for approach 05.

For a given audio file (or pre-separated stems), run each per-stem
model on its corresponding stem and merge the events into one MIDI
file.

Two modes:
  1. --stems-dir: per-stem WAVs already exist (e.g. from MDX23C output
     in user_files/<project>/stems/). Orchestrator just runs each model.
  2. --mix: original mix WAV. Orchestrator calls separate() first,
     then runs each model.

The current implementation is mode (1) only. Mode (2) is a follow-up
that requires running the stem separator (which is compute-intensive
and the user's responsibility to invoke separately).
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import numpy as np

MT_DIR = Path(__file__).parent
sys.path.insert(0, str(MT_DIR))

from config import DEVICE  # noqa: E402
from feature_extractor import get_input_tensor  # noqa: E402
from model_stem import StemTranscriber  # noqa: E402
from inference_core import find_peaks_with_onset_snap  # noqa: E402
from midi_shell import load_midi_file  # noqa: E402
from datasets.per_stem import (  # noqa: E402
    STEM_PITCHES, STEM_CANONICAL_PITCH, load_stem_pairs,
)


STEMS = list(STEM_PITCHES.keys())  # canonical order


def find_onset_frames(pred: np.ndarray, threshold: float = 0.5) -> list:
    """
    Convert model output [T, 2] (onset, velocity) into a list of
    (frame, velocity) tuples by sigmoid + peak detection.
    """
    onset_probs = 1.0 / (1.0 + np.exp(-pred[:, 0]))
    vel_probs = 1.0 / (1.0 + np.exp(-pred[:, 1]))

    peaks = find_peaks_with_onset_snap(onset_probs, threshold, min_distance=5)
    return [(int(frame), int(vel_probs[frame] * 127)) for frame, _ in peaks]


def write_per_stem_midi(events: list, output_path: str, bpm: float = 120.0):
    """
    Write a list of (time_seconds, midi_pitch, velocity) tuples to a MIDI file.
    """
    from midiutil import MIDIFile

    midi = MIDIFile(1)
    midi.addTrackName(0, 0, "Drums (per-stem)")
    midi.addTempo(0, 0, bpm)
    midi.addText(0, 0.0, "START")
    # Anchor note
    midi.addNote(0, 9, 27, 0.0, 0.01, 100)

    events_sorted = sorted(events, key=lambda e: e[0])
    for time_s, pitch, vel in events_sorted:
        beats = time_s * (bpm / 60.0)
        midi.addNote(0, 9, int(pitch), beats, 0.1, max(1, min(127, int(vel))))

    with open(output_path, "wb") as f:
        midi.writeFile(f)


def infer_one_stem(
    stem: str,
    stem_wav: str,
    ckpt_path: str,
    threshold: float = 0.3,
    device: str = None,
) -> list:
    """
    Run per-stem inference on one stem WAV. Returns list of
    (time_seconds, midi_pitch, velocity) tuples.
    """
    if device is None:
        device = DEVICE

    spec = get_input_tensor(stem_wav).unsqueeze(0).to(device)  # [1, 3, 128, T]
    model = StemTranscriber(num_classes=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        pred = model(spec).cpu().numpy()[0]  # [T, 2]

    from config import SECONDS_PER_FRAME
    frames = find_onset_frames(pred, threshold=threshold)
    pitch = STEM_CANONICAL_PITCH[stem]
    events = [(frame * SECONDS_PER_FRAME, pitch, vel) for frame, vel in frames]
    return events


def transcribe_project(
    project_dir: str,
    ckpt_dir: str = "models_stem",
    threshold: float = 0.3,
    output_midi: str = None,
) -> list:
    """
    Run all 5 per-stem models on a user_files project, return merged events.
    """
    pairs = load_stem_pairs(project_dir)
    if not pairs:
        raise FileNotFoundError(f"No stem pairs found in {project_dir}")

    all_events = []
    for stem, (wav, midi) in pairs.items():
        ckpt_path = Path(ckpt_dir) / f"stem_{stem}.ckpt"
        if not ckpt_path.exists():
            print(f"  [SKIP] {stem}: no checkpoint at {ckpt_path}")
            continue
        events = infer_one_stem(stem, wav, str(ckpt_path), threshold=threshold)
        print(f"  [OK]   {stem}: {len(events)} detected events")
        all_events.extend(events)

    all_events.sort(key=lambda e: e[0])
    if output_midi:
        write_per_stem_midi(all_events, output_midi)
        print(f"\nWrote {len(all_events)} events to {output_midi}")
    return all_events


# -------- CLI --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True,
                        help="Path to a user_files project with /stems and /midi")
    parser.add_argument("--ckpt-dir", default="models_stem")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--output", help="Output MIDI path; default: <project>/per_stem_pred.mid")
    args = parser.parse_args()

    if not args.output:
        args.output = str(Path(args.project) / "per_stem_pred.mid")

    print(f"Transcribing project: {args.project}")
    print(f"Checkpoints: {args.ckpt_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {args.output}\n")

    events = transcribe_project(
        project_dir=args.project,
        ckpt_dir=args.ckpt_dir,
        threshold=args.threshold,
        output_midi=args.output,
    )
    print(f"\nTotal events: {len(events)}")
