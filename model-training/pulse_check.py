"""
pulse_check.py — Quick sanity check on a checkpoint.

Usage:
    python pulse_check.py models/dl-1.ckpt
    python pulse_check.py models/dl-1.ckpt --audio dl-1.wav --midi dl-1.mid
    python pulse_check.py models/dl-1.ckpt --threshold 0.3
"""

import argparse
import torch
from pathlib import Path

from config import DEVICE
from io_utils import pulse_check
from model import DrumTranscriber
from train_utils import load_audio


def main():
    parser = argparse.ArgumentParser(description='Pulse check on a checkpoint')
    parser.add_argument('checkpoint', help='Path to checkpoint file')
    parser.add_argument('--audio', '-a', help='Audio file to test against')
    parser.add_argument('--midi', '-m', help='Ground truth MIDI file')
    parser.add_argument('--threshold', '-t', type=float, default=0.5)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = DrumTranscriber()
    model.load_state_dict(ckpt['model_state_dict'])

    if args.audio:
        pulse_check(model, args.audio, midi_path=args.midi, threshold=args.threshold, device=DEVICE)
    else:
        print(f"=== Checkpoint: {ckpt_path.name} ===")
        print(f"  epoch: {ckpt.get('epoch', '?')}")
        print(f"  loss:  {ckpt.get('loss', '?')}")
        print(f"  layers: {len(ckpt['model_state_dict'])}")


if __name__ == '__main__':
    main()
