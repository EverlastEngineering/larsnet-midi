# Test: Bug Isolation Grid

> Systematic 2×2×2 ablation matrix to identify which architectural
> choice is responsible when `03-test-prove-overfit-first.md` fails and
> bisection doesn't immediately produce a clear answer.
>
> Schema follows `00-overview.md`.

---

## Premise

The current pipeline has at least three independent design choices that
each could be contributing to the inference failure:

1. **Input channels** — mono (current) vs 3-channel (Left, Right, Width per roadmap §1)
2. **Label smear** — `[1.0, 0.8, 0.5, 0.2]` causal smear (current) vs hard `[1.0]` only
3. **Output head** — joint onset+velocity (current 20 channels) vs onset-only (10 channels)

These factors **may interact**. Mono + smear might be fine; mono + smear +
velocity head might collapse. The only reliable way to find which
combination is bad is a full 2×2×2 grid: 8 training runs, identical except
for these axes.

This is cheap on a 10-second overfit target: 8 runs × 200 epochs × ~3
min/run = 30 minutes total on CPU.

---

## Architecture

```
                 ┌───────────────────────────────┐
                 │  8 training runs in parallel  │
                 │  on the same 10s drum loop   │
                 └───────────────┬───────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
  ┌──────────┐            ┌──────────┐            ┌──────────┐
  │ mono     │            │ mono     │            │ mono     │
  │ smear    │            │ no smear │            │ smear    │
  │ +vel     │            │ +vel     │            │ no vel   │
  └─────┬────┘            └─────┬────┘            └─────┬────┘
        │                       │                       │
        ...etc (8 configs)
        │
        ▼
   ┌─────────────────────────────────────┐
   │  Score each config:                 │
   │  - Final train loss                 │
   │  - Inference F1 vs ground truth     │
   │  - Velocity correlation             │
   │  - Per-class recall                 │
   └──────────────────┬──────────────────┘
                      │
                      ▼
        Tabulate; identify which axis (or
        interaction) flips PASS → FAIL.
```

---

## Why this should work

Factorial-design experimentation is the bread-and-butter of empirical
debugging. If a particular combination is broken, the table will show:

```
                 mono   3ch
                  ┌─────┬─────┐
   smear,  +vel   │ FAIL│ PASS│   <-- mono is the cause
                  ├─────┼─────┤
   smear,  -vel   │ PASS│ PASS│   <-- velocity head + mono interacts
                  ├─────┼─────┤
   no smear, +vel │ PASS│ PASS│
                  ├─────┼─────┤
   no smear, -vel │ PASS│ PASS│
                  └─────┴─────┘
```

A single failed cell pinpoints the interaction. Multiple failed cells
identify the dominant factor.

---

## What could go wrong

1. **8 configs is not enough surface area** — if the bug interacts with
   pos_weight, learning rate, or chunk_frames, the grid won't see it.
   Mitigation: only add axes after the first 8-cell grid eliminates the
   obvious ones.

2. **10-second overfit target is too easy** — all 8 configs may all
   reach loss < 0.01 even with bugs, hiding the problem until inference.
   Mitigation: the score is *inference F1*, not training loss; this
   already encodes the failure mode of interest.

3. **Stochasticity** — different random seeds may produce different
   results. Mitigation: use a fixed seed for all 8 runs; if results
   look noisy, run each cell 3 times and average.

4. **Implementation overhead** — adding switches to mono/stereo, smear
   on/off, velocity-head on/off requires changes to `feature_extractor.py`,
   `label_encoder.py`, `model.py`, `train_utils.py`. May take a day to
   wire up. Mitigation: keep the changes minimal and behind feature flags
   so they can be reverted after the experiment.

---

## Prerequisites

- `03-test-prove-overfit-first.md` test harness in place.
- Test fixture audio + MIDI (10s loop).
- Conda env `drumtomidi`.
- ~2 hours of CPU time + 1–2 hours of implementation time.

---

## Implementation steps

### Step 1: Add feature flags

#### `feature_extractor.py` — add channel mode

```python
# New optional parameter at function signature
def get_input_tensor(audio_path, sample_rate=44100, channels='mono'):
    """channels: 'mono' (default, current behavior) or 'stereo3' (L, R, width)"""
    # ... existing load + resample code ...

    if channels == 'mono':
        mono = waveform.mean(dim=0, keepdim=True)
        return amplitude_to_db(mel_transform(mono))  # [1, 128, T]
    elif channels == 'stereo3':
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        spec_l = amplitude_to_db(mel_transform(waveform[0:1]))
        spec_r = amplitude_to_db(mel_transform(waveform[1:2]))
        side = waveform[0:1] - waveform[1:2]
        spec_w = amplitude_to_db(mel_transform(side))
        return torch.cat([spec_l, spec_r, spec_w], dim=0)  # [3, 128, T]
    else:
        raise ValueError(f"Unknown channels={channels}")
```

#### `label_encoder.py` — add smear mode

```python
def midi_to_frame_array(midi_notes, total_frames, hop_length=512, sr=44100,
                         smear='causal'):
    """smear: 'causal' (default [1.0, 0.8, 0.5, 0.2]) or 'hard' ([1.0] only)"""
    labels = torch.zeros((20, total_frames))
    seconds_per_frame = hop_length / sr
    for note in midi_notes:
        if note.pitch in MAPPING:
            hit_frame = int(note.start_time / seconds_per_frame)
            idx = MAPPING[note.pitch]
            if 0 <= hit_frame < total_frames:
                labels[idx, hit_frame] = 1.0
                if smear == 'causal':
                    if hit_frame + 1 < total_frames: labels[idx, hit_frame+1] = 0.8
                    if hit_frame + 2 < total_frames: labels[idx, hit_frame+2] = 0.5
                    if hit_frame + 3 < total_frames: labels[idx, hit_frame+3] = 0.2
                # else 'hard': only the impact frame
                vel_channel = idx + 10
                labels[vel_channel, hit_frame] = (note.velocity / 127.0) ** 0.7
    return labels
```

#### `model.py` — accept variable in_channels and output dim

```python
class DrumTranscriber(nn.Module):
    def __init__(self, in_channels=1, output_dim=20):
        """
        in_channels: 1 (mono) or 3 (L/R/Width)
        output_dim: 10 (onset only) or 20 (onset + velocity)
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.rnn = nn.GRU(2048, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 3, 1, 2).flatten(2)
        x, _ = self.rnn(x)
        return self.fc(x)
```

#### `train_utils.py` — handle output_dim variant in loss

```python
class MultiTaskDrumLoss(nn.Module):
    def __init__(self, velocity_weight=2.0, device='cpu', use_velocity=True):
        # ... existing init ...
        self.use_velocity = use_velocity

    def forward(self, pred, target):
        # pred: [B, T, 10] or [B, T, 20]
        onset_pred = pred[:, :, :10]
        onset_target = target[:, :, :10]
        onset_loss = self.onset_criterion(onset_pred, onset_target)

        if not self.use_velocity or pred.shape[-1] == 10:
            return onset_loss, {'onset_loss': onset_loss.item(), 'velocity_loss': 0.0}

        # ... existing velocity branch ...
```

### Step 2: Grid runner script

Create `model-training/tools/bug_isolation_grid.py`:

```python
"""
Run the 2x2x2 ablation grid and produce a results table.

Usage:
    conda run -n drumtomidi python tools/bug_isolation_grid.py \
        --audio dl-1.wav --midi dl-1.mid --epochs 200
"""

import argparse
import itertools
import json
from pathlib import Path
import subprocess

import torch
import numpy as np


def run_one_config(audio, midi, channels, smear, use_velocity, epochs, seed=42):
    """
    Train one config, then run inference, then evaluate.
    Returns dict with metrics.
    """
    torch.manual_seed(seed)
    # 1) build feature extractor + model + label encoder per the config
    # 2) run training loop for `epochs`
    # 3) capture final train loss
    # 4) run inference on the same file
    # 5) evaluate with mir_eval
    # 6) return {'config': {...}, 'train_loss': ..., 'f1': ..., 'velocity_corr': ...,
    #            'per_class_recall': {...}}
    ...


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', required=True)
    parser.add_argument('--midi', required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--output', default='grid_results.json')
    args = parser.parse_args()

    grid = list(itertools.product(
        ['mono', 'stereo3'],       # channels
        ['hard', 'causal'],         # smear
        [False, True],              # use_velocity
    ))

    results = []
    for i, (ch, sm, uv) in enumerate(grid, 1):
        print(f"\n=== Config {i}/8: channels={ch}, smear={sm}, vel={uv} ===")
        r = run_one_config(args.audio, args.midi, ch, sm, uv, args.epochs)
        results.append(r)
        # Save incrementally so a crash doesn't lose progress
        Path(args.output).write_text(json.dumps(results, indent=2))

    # Print table
    print("\n=== GRID RESULTS ===")
    print(f"{'channels':<10} {'smear':<8} {'velocity':<10} {'train_loss':<12} {'F1':<8} {'vel_corr':<10}")
    print('-' * 70)
    for r in results:
        c = r['config']
        print(f"{c['channels']:<10} {c['smear']:<8} {str(c['use_velocity']):<10} "
              f"{r['train_loss']:<12.4f} {r['f1']:<8.3f} {r['velocity_corr']:<10.3f}")


if __name__ == '__main__':
    main()
```

### Step 3: Run the grid

```bash
cd model-training
conda run -n drumtomidi python tools/bug_isolation_grid.py \
    --audio dl-1.wav --midi dl-1.mid --epochs 200 \
    --output agent-plans/next-attempt/grid_results.json
```

Expected runtime: ~30 minutes on CPU.

### Step 4: Interpret

Look at the F1 column. The configuration that FIRST passes F1 > 0.5
identifies the necessary changes:

- **If `stereo3` configs all pass and `mono` configs all fail** →
  channel collapse (Theory T1) is the dominant cause. Fix feature_extractor.py
  permanently.
- **If `hard` smear passes and `causal` smear fails** → the causal smear
  shape is wrong. Investigate.
- **If `use_velocity=False` passes and `=True` fails** → velocity head
  poisons gradients. Reduce velocity_weight or train onset-first then
  add velocity.
- **Multiple axes flip** → interaction. The minimum-config that passes
  is your new baseline.

### Step 5: Persist findings

Append a results section to `agent-plans/next-attempt/grid_results.json`
*and* to this document, so future agents see the empirical answer.

---

## Evaluation

The grid IS the evaluation. The outcome is a table; the interpretation
gives you a fix or a starting baseline for approaches 05/06/etc.

If even the "best" config in the grid fails to reach F1 > 0.5, then
the bug is in something the grid doesn't cover (loss function, optimizer,
fundamental data corruption). At that point, switch to approach 09
(ADTOF port) — a known-working architecture that bypasses all the
custom code.

---

## Estimated effort

| Subtask | Time |
|---------|------|
| Wire feature flags into 4 files | 2 hours |
| Write `bug_isolation_grid.py` | 1 hour |
| Run the grid | 30 minutes |
| Interpret + decide next action | 30 minutes |
| **Total** | **~4 hours** |

CPU sufficient. No external dependencies beyond what's already in `drumtomidi`.

---

## Optional extensions (only after the 2×2×2 grid)

- **Add `pos_weight` axis**: `dampened [2..10]` vs `inverse-freq [5..156]` vs `uniform [1.0]`.
- **Add `learning_rate` axis**: 1e-2 vs 1e-3 vs 1e-4.
- **Add `model_size` axis**: GRU 64 vs 128 vs 256 hidden units.

Each axis doubles the runtime. Resist the urge to test everything at
once. Confirm the 2×2×2 first.

---

## Escalation paths

- **If grid produces a clear winner**: apply the winning config as the
  new baseline; remove feature flags; proceed to approach 05 or 06.
- **If grid shows no config works**: the bug is elsewhere. Most likely
  the data pipeline (sample-rate, MIDI alignment) or the optimizer.
  Switch to approach 09 (ADTOF port — known-good architecture).
- **If grid shows everything works on 10s loop but not on real data**:
  the 10s loop is unrepresentative. Run the grid on a 60s loop with
  more drum variety, then retry approach 05.
