# Approach 8: Port Onsets-and-Frames (Magenta)

> Adopt Google Magenta's *Onsets and Frames* architecture (Hawthorne et al.,
> 2018). It's the de-facto baseline for music transcription, has a clean
> reference implementation, and its **gated dual-head design directly
> addresses the failure mode** where the velocity head poisons the onset
> head.
>
> Schema follows `00-overview.md`.

---

## Premise

The single most cited paper in modern music transcription is *Onsets and
Frames: Dual-Objective Piano Transcription* (arXiv:1710.11153). Its key
architectural insight: **don't let the frame-level prediction head fire
unless the onset head fires first.** The onset head is a binary "is there
a note attack right now?" detector. The frame head is "is this note
sustaining right now?" The two are linked by GATING the frame logits with
the onset logits.

For drums this maps cleanly:
- Onset head = "is there a drum hit at this frame?"
- Frame head = "what is the velocity of the hit at this frame?"

This is *exactly* the architecture the previous attempt tried to build,
but without the gating. The gate is what makes joint training stable.

Reference implementation:
- Original (TensorFlow): https://github.com/magenta/magenta/tree/main/magenta/models/onsets_frames_transcription
- PyTorch port (well-maintained): https://github.com/jongwook/onsets-and-frames
- Apache 2.0 license — clean to port.

---

## Architecture (Hawthorne 2018, simplified for drums)

```
[Input: mel-spec, [1, 229, T]]
                │
                ▼
[Acoustic Model: ConvStack → BiLSTM, ~1M params]
                │
        ┌───────┴───────┐
        ▼               ▼
[Onset Stack]      [Frame Stack]
  - Conv + LSTM     - Conv + LSTM
  - Sigmoid         - Sigmoid
  - Output: P(onset)  - Output: P(active)
        │               │
        │   ┌───────────┘
        ▼   ▼
   [Element-wise multiply: onset gate * frame logit]
        │
        ▼
[Combined output, used for inference]

[Velocity Stack] (parallel branch)
  - Conv + LSTM
  - Linear
  - Output: velocity ∈ [0, 1]
  - Only loss-active when onset target = 1 (masked)
```

For drums with 10 classes:
- Each head outputs 10-channel logits
- Total params: ~10M (10× the previous attempt's ~600k)
- Input mel-bins: 229 (paper standard), n_fft=2048, hop=512

---

## Why this should work

1. **Battle-tested.** Hundreds of papers cite this; the architecture is
   known to converge.
2. **Gating prevents head poisoning.** This is the specific issue
   diagnosed in `01-critique-and-theories.md` Theory T7. Gating makes
   joint multi-task learning stable.
3. **Drum adaptation is straightforward.** The piano version has 88
   pitch classes; drums have 10 drum-class equivalents. Change one
   number.
4. **Published reference numbers exist.** F1 ≥ 0.95 onset F1 on MAESTRO
   piano data. Drums are not piano, but the same architecture *family*
   has been used for drums (see also "Onsets and Velocities" by Brunner
   2020 for the exact velocity variant).
5. **PyTorch port already exists** — no TensorFlow detour.
6. **The community has debugged this code for 5+ years.** Edge cases
   like NaN gradients, learning-rate warmup, and label smoothing are
   already handled.

---

## What could go wrong

1. **Drum-specific class imbalance is more extreme than piano.** Piano
   pitches are all played ~equally; drums have 30× imbalance between
   kicks and Crash2. Mitigation: keep the dampened pos_weight from the
   rescue commits.
2. **Onset gating may suppress quiet hits.** Ghost-note snares may not
   trigger the onset head strongly enough to gate the frame head on.
   Mitigation: lower the onset threshold for inference.
3. **10M parameters needs a GPU for reasonable training time.**
   Mitigation: rent an A100 for the training run. ~$10-30 total.
4. **The PyTorch port may have subtle bugs or be outdated.** Mitigation:
   compare loss curves against the paper's reported numbers; if they
   diverge significantly, switch to the official TF version via TF-PyTorch
   ONNX bridge.

---

## Prerequisites

- Working Python env with PyTorch (existing `drumtomidi` env).
- The `tests/test_overfit_reproduction.py` harness from approach 03.
- mir_eval evaluation wrapper from `02-tooling-wishlist.md`.
- e-GMD dataset.
- GPU recommended: Lambda Labs A100 ($1.10/hr), free Kaggle P100, or
  similar. CPU works but training takes ~10× longer.

---

## Implementation steps

### Phase 1: Clone the reference and inspect

```bash
git clone https://github.com/jongwook/onsets-and-frames external/onsets-and-frames
# Read external/onsets-and-frames/onsets_and_frames/transcriber.py
# Read external/onsets-and-frames/onsets_and_frames/lstm.py
# Note: it uses MAESTRO loader; we'll replace with e-GMD loader.
```

### Phase 2: Adapt for drums

Create `model-training/oaf_drum_transcriber.py`:

```python
"""
Drum-adapted Onsets and Frames transcriber.

Adapted from Hawthorne et al. 2018 (arXiv:1710.11153) and the PyTorch
port at https://github.com/jongwook/onsets-and-frames (MIT License).
"""

import torch
import torch.nn as nn

OUTPUT_CLASSES = 10  # drum classes (Kick, Snare, HHC, HHO, ...)

class ConvStack(nn.Module):
    # 3 conv layers + dropout + fc to model_size
    # (identical to the reference; copy verbatim)
    ...

class OnsetsAndFramesDrum(nn.Module):
    def __init__(self, input_features=229, output_classes=10, model_size=768):
        super().__init__()
        self.onset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.LSTM(model_size, model_size, batch_first=True, bidirectional=True),
            nn.Linear(model_size * 2, output_classes),
            nn.Sigmoid(),
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_classes),
            nn.Sigmoid(),
        )
        self.combined_stack = nn.Sequential(
            nn.LSTM(output_classes * 2, model_size, batch_first=True, bidirectional=True),
            nn.Linear(model_size * 2, output_classes),
            nn.Sigmoid(),
        )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_classes),
        )

    def forward(self, mel):
        # mel: [B, T, F]
        onset_pred = self.onset_stack(mel)            # [B, T, 10]
        frame_pred_raw = self.frame_stack(mel)         # [B, T, 10]
        # Gate: combined input = [onset_pred.detach(), frame_pred_raw]
        combined_input = torch.cat([onset_pred.detach(), frame_pred_raw], dim=-1)
        frame_pred = self.combined_stack(combined_input)  # [B, T, 10]
        velocity_pred = self.velocity_stack(mel)       # [B, T, 10]
        return {
            'onset': onset_pred,
            'frame': frame_pred,
            'velocity': velocity_pred,
        }
```

### Phase 3: Adapt loss function

```python
def oaf_loss(pred, target, velocity_weight=1.0):
    """
    target dict: {'onset': [B,T,10] binary, 'frame': [B,T,10] binary,
                  'velocity': [B,T,10] [0,1] regression}
    """
    onset_loss = nn.functional.binary_cross_entropy(pred['onset'], target['onset'])
    frame_loss = nn.functional.binary_cross_entropy(pred['frame'], target['frame'])
    # Velocity loss is masked by onset_target
    velocity_mask = target['onset']
    velocity_loss = ((pred['velocity'] - target['velocity']) ** 2 * velocity_mask).sum() / (velocity_mask.sum() + 1e-8)
    return onset_loss + frame_loss + velocity_weight * velocity_loss
```

### Phase 4: Adapt data loader

Build a new dataset class `OAFDrumDataset` that produces:
- `mel`: [T, 229] mel-spectrogram
- `target['onset']`: [T, 10] binary, 1 at exactly the hit frame
- `target['frame']`: [T, 10] binary, 1 for hit_frame to hit_frame+sustain_frames
  (use a short fixed sustain like 4 frames since drums don't really sustain)
- `target['velocity']`: [T, 10] float in [0, 1]

### Phase 5: Train

```bash
# CPU
conda run -n drumtomidi python train_oaf_drum.py \
    --manifest batch1_shuffled.txt --val val1_shuffed.txt --epochs 20

# GPU (rented A100 setup, after rsync of code+data)
python train_oaf_drum.py --manifest ... --epochs 50 --batch-size 32 --device cuda
```

Expected training time: 5-15 hours on a single A100 for 50 epochs on full
e-GMD. ~$15-30 total cost.

### Phase 6: Inference

```python
def inference(audio_path, model, threshold_onset=0.5, threshold_frame=0.5):
    mel = extract_mel(audio_path)  # [1, T, 229]
    with torch.no_grad():
        out = model(mel)
    onset_thresh = out['onset'] > threshold_onset
    # Peak-pick onset (gated): for each class, find frames where onset > threshold
    # and is a local max.
    events = []
    for class_idx in range(10):
        peaks = find_peaks(onset_thresh[0, :, class_idx])
        for frame in peaks:
            velocity = int(out['velocity'][0, frame, class_idx].sigmoid().item() * 127)
            events.append((frame * seconds_per_frame, INDEX_TO_MIDI[class_idx], velocity))
    return events
```

### Phase 7: Evaluate

Run `tools/eval_with_mir_eval.py` on e-GMD test split.

Target: F1 ≥ 0.85.

---

## Evaluation

| Metric | Target | Notes |
|--------|--------|-------|
| Overfit smoke test (per `03-test-prove-overfit-first.md`) | PASS | Required gate |
| F1 on e-GMD test (±50ms) | ≥0.85 | Reference numbers from drum-OAF variants |
| Per-class recall (rare classes) | ≥0.60 (HHO, TomMid, Crash2) | Class imbalance check |
| Velocity R² | ≥0.75 | Independent velocity head |

---

## Estimated effort

| Phase | Time | Compute |
|-------|------|---------|
| 1: Clone reference | 1h | CPU |
| 2: Port architecture for drums | 4-8h | CPU |
| 3: Loss adaptation | 2h | CPU |
| 4: Data loader | 4-6h | CPU |
| 5: Train | 6-15h | GPU |
| 6: Inference wrapper | 2-4h | CPU |
| 7: Eval | 2h | CPU |
| **Total** | **5-7 days** | mostly GPU |

---

## Escalation paths

- **If overfit smoke test fails**: bug in your port. Compare layer-by-layer
  with the reference; common issues are LSTM initialization,
  feature normalization, and BCE vs BCEWithLogits choice.
- **If training diverges**: lower learning rate to 1e-4, add gradient
  clipping (the paper uses 3.0).
- **If F1 stuck below 0.7**: switch to approach 09 (ADTOF — drum-specific
  reference, may be a better fit out of the box).
- **If F1 above 0.85**: combine with approach 12 (curriculum learning)
  to push further.
