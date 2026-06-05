"""
Per-stem training script for approach 05 (stems-as-input).

Trains one StemTranscriber per stem (kick, snare, hihat, toms, cymbals)
on per-stem audio (WAV) + per-stem-filtered MIDI (the project MIDI with
notes filtered to that stem's pitches).

Usage:
    # Smoke test on user_files project (small, fast)
    conda run -n drumtomidi python train_stem.py --stem kick \
        --project "/path/to/project" --epochs 500

    # Full training on a manifest of e-GMD projects (after running
    # separate.py on each mix to produce per-stem WAVs)
    conda run -n drumtomidi python train_stem.py --stem kick \
        --manifest path/to/manifest.txt --epochs 30
"""

import argparse
import time
import tempfile
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

# Add model-training to path
MT_DIR = Path(__file__).parent
sys.path.insert(0, str(MT_DIR))

from config import DEVICE  # noqa: E402
from feature_extractor import get_input_tensor  # noqa: E402
from model_stem import StemTranscriber, PerStemLoss  # noqa: E402
from train_utils import load_midi_notes  # noqa: E402
from datasets.per_stem import (  # noqa: E402
    STEM_PITCHES, STEM_CANONICAL_PITCH,
    filter_notes_to_stem, build_per_stem_targets,
    load_stem_pairs,
)


# -------- Dataset --------
class PerStemDataset(Dataset):
    """
    Yields (spec, target) pairs for one stem, given the stem WAV
    and the project MIDI. MIDI is filtered to the stem's pitches.
    """
    def __init__(self, stem: str, stem_wav: str, midi_path: str, max_seconds: float = 60.0):
        self.stem = stem
        # Load + filter MIDI
        all_notes, _ = load_midi_notes(midi_path)
        self.notes = filter_notes_to_stem(all_notes, stem)

        # Load audio as mel-spec
        spec = get_input_tensor(stem_wav)  # [3, 128, T]
        # Truncate to max_seconds (training window; keeps memory bounded)
        max_frames = int(max_seconds * 44100 / 512)
        if spec.shape[2] > max_frames:
            spec = spec[:, :, :max_frames]
        self.spec = spec
        # Pre-compute the target for the truncated spec
        self.target = build_per_stem_targets(self.notes, spec.shape[2])

    def __len__(self):
        return 1  # one example per project for now

    def __getitem__(self, idx):
        return self.spec.unsqueeze(0), self.target  # [1, 3, 128, T], [2, T]


# -------- Training --------
def train_stem(
    stem: str,
    stem_wav: str,
    midi_path: str,
    epochs: int = 200,
    lr: float = 1e-3,
    chunk_frames: int = 8000,
    max_seconds: float = 60.0,
    device: str = None,
) -> tuple:
    """Train a per-stem model. Returns (final_loss, model, ckpt_path)."""
    if device is None:
        device = DEVICE

    print(f"\n[TRAIN-STEM {stem}] stem={stem} wav={stem_wav} midi={midi_path}")
    print(f"[TRAIN-STEM {stem}] epochs={epochs} lr={lr} chunk={chunk_frames} max_sec={max_seconds}")

    dataset = PerStemDataset(stem, stem_wav, midi_path, max_seconds=max_seconds)
    spec, target = dataset[0]
    spec = spec.to(device)  # [1, 3, 128, T]
    target = target.to(device)  # [2, T]

    print(f"[TRAIN-STEM {stem}] spec shape: {spec.shape}, target shape: {target.shape}")
    print(f"[TRAIN-STEM {stem}] positive targets in onset channel: "
          f"{(target[0] >= 0.99).sum().item()}")

    model = StemTranscriber(num_classes=1).to(device)
    criterion = PerStemLoss(velocity_weight=1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20,
    )

    T = spec.shape[3]
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_chunks = 0
        for chunk_start in range(0, T, chunk_frames):
            chunk_end = min(chunk_start + chunk_frames, T)
            spec_chunk = spec[:, :, :, chunk_start:chunk_end]
            target_chunk = target[:, chunk_start:chunk_end].unsqueeze(0).permute(0, 2, 1)
            # target_chunk: [1, T_chunk, 2]

            optimizer.zero_grad()
            pred = model(spec_chunk)  # [1, T_chunk, 2]
            loss, ld = criterion(pred, target_chunk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_chunks += 1
        avg_loss = epoch_loss / max(n_chunks, 1)
        losses.append(avg_loss)
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"[TRAIN-STEM {stem}] epoch {epoch+1:3d}/{epochs} | "
                  f"loss={avg_loss:.4f} | onset={ld['onset_loss']:.4f} "
                  f"vel={ld['velocity_loss']:.4f}")
        scheduler.step(avg_loss)

    final_loss = sum(losses[-10:]) / 10  # avg of last 10 epochs
    ckpt_path = Path(tempfile.gettempdir()) / f"stem_{stem}.ckpt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "loss": final_loss,
        "stem": stem,
        "num_classes": 1,
    }, ckpt_path)
    print(f"[TRAIN-STEM {stem}] DONE. Final loss: {final_loss:.4f}, ckpt: {ckpt_path}")
    return final_loss, model, ckpt_path


# -------- CLI --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", required=True, choices=list(STEM_PITCHES.keys()),
                        help="Which stem to train")
    parser.add_argument("--project", help="Path to a user_files project dir (with /stems and /midi)")
    parser.add_argument("--stem-wav", help="Direct path to a stem WAV (alternative to --project)")
    parser.add_argument("--midi", help="Direct path to a MIDI file (alternative to --project)")
    parser.add_argument("--manifest", help="Manifest of project dirs (one per line)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-seconds", type=float, default=60.0,
                        help="Max audio length per training sample (memory bound)")
    parser.add_argument("--out", default="models_stem", help="Where to save the final ckpt")
    args = parser.parse_args()

    # Resolve stem WAV + MIDI
    if args.stem_wav and args.midi:
        stem_wav, midi = args.stem_wav, args.midi
    elif args.project:
        pairs = load_stem_pairs(args.project)
        if args.stem not in pairs:
            print(f"ERROR: stem '{args.stem}' not in project. "
                  f"Available: {list(pairs.keys())}")
            sys.exit(1)
        stem_wav, midi = pairs[args.stem]
    else:
        parser.error("Either --project or both --stem-wav and --midi required")

    print(f"Using stem WAV: {stem_wav}")
    print(f"Using MIDI:     {midi}")

    final_loss, model, ckpt_path = train_stem(
        stem=args.stem,
        stem_wav=stem_wav,
        midi_path=midi,
        epochs=args.epochs,
        lr=args.lr,
        max_seconds=args.max_seconds,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    final_path = out_dir / f"stem_{args.stem}.ckpt"
    import shutil
    shutil.copy(ckpt_path, final_path)
    print(f"\nSaved final checkpoint to {final_path}")
