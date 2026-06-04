# Approach 5: Stems-as-Input (Per-Stem Transcribers)

> ★ **TOP RECOMMENDATION** ★
>
> The user independently arrived at this idea, and the math agrees: by
> feeding pre-separated drum stems to specialized per-stem transcribers,
> we turn one hard 10-class problem into five trivial 2–3 class problems
> that should converge easily even on modest data and compute.
>
> Schema follows `00-overview.md`.

---

## Premise

**The fundamental insight**: the user's repo already has a working drum
stem separator (`separate.py` + `mdx23c_*.py`). It splits a drum mix into
five clean stems: kick, snare, hihat, toms, cymbals. The classification
ambiguity that broke the unified 10-class model — "is this transient a
snare or a tom or a clap?" — **does not exist when the input is already
labeled as 'this is the snare stem'.**

The per-stem transcription problem is:
- **kick.wav** → binary onset detection. 1 class (kick). Easy.
- **snare.wav** → 3 classes (snare, rimshot, clap). Already partly solved
  by `stems_to_midi/note_classification_core.py`. Trivial-to-easy.
- **hihat.wav** → 2 classes (closed, open). The trickiest case because
  closed/open are a continuous spectrum. Medium.
- **toms.wav** → 3 classes (low, mid, high). Spectral centroid is hugely
  informative. Easy.
- **cymbals.wav** → 3 classes (ride, crash, splash/china). Trickiest after
  hihat. Medium.

A small CNN/CRNN per stem (≤500k params each) trained on stem-isolated
data is dramatically easier than a 10-class CRNN on a stereo drum mix.
Each per-stem model converges with thousands of training examples per
class — not millions.

---

## Architecture

```
                  ┌─────────────────────────┐
                  │  Drum mix WAV (stereo)  │
                  └───────────┬─────────────┘
                              │
                              ▼
                ┌──────────────────────────────┐
                │  Existing stem separator     │
                │  (separate.py / MDX23C)      │
                │  ALREADY WORKS, ALREADY      │
                │  IN PRODUCTION               │
                └─────────────┬────────────────┘
                              │ produces 5 stems
                              ▼
        ┌──────────┬──────────┬──────────┬──────────┬──────────┐
        ▼          ▼          ▼          ▼          ▼          
   kick.wav   snare.wav  hihat.wav  toms.wav  cymbals.wav
        │          │          │          │          │
        ▼          ▼          ▼          ▼          ▼
   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
   │ Kick   │ │ Snare  │ │ Hihat  │ │ Toms   │ │ Cymbals│
   │ Model  │ │ Model  │ │ Model  │ │ Model  │ │ Model  │
   │ 1-cls  │ │ 3-cls  │ │ 2-cls  │ │ 3-cls  │ │ 3-cls  │
   │ +vel   │ │ +vel   │ │ +vel   │ │ +vel   │ │ +vel   │
   └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘
        │          │          │          │          │
        └──────────┴──────────┼──────────┴──────────┘
                              ▼
                  ┌─────────────────────────┐
                  │  Merge events → MIDI    │
                  │  (simple union by time) │
                  └─────────────────────────┘
```

### Per-stem model spec

Each per-stem model is structurally identical, only differs in number of
output classes:

```python
class StemTranscriber(nn.Module):
    """
    Per-stem onset + velocity model.
    Input:  [B, 1, 128, T] mono stem mel-spectrogram
    Output: [B, T, 2N] where N = num_classes for this stem
              channels 0..N-1   = onset logits
              channels N..2N-1  = velocity logits (sigmoid → 0..1)
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(32),                  # NEW: stabilizes training
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 1))
        )
        # Bigger GRU now that we have less classification work to do
        self.rnn = nn.GRU(2048, 256, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, num_classes * 2)
        self.num_classes = num_classes
```

| Stem | Classes | Output dim | Params |
|------|---------|-----------|--------|
| kick | 1 | 2 | ~2.2M |
| snare | 3 (snare, rimshot, clap) | 6 | ~2.2M |
| hihat | 2 (closed, open) | 4 | ~2.2M |
| toms | 3 (low, mid, high) | 6 | ~2.2M |
| cymbals | 3 (ride, crash, splash) | 6 | ~2.2M |
| **Total** | 12 | 24 | **~11M** |

Note: total params (11M) is comparable to Onsets-and-Frames (10M), but
**distributed across 5 specialized models**. Each trains independently
on isolated data.

---

## Why this should work

### 1. The classification problem becomes trivial

The hardest part of "is this a kick or a tom?" was always disambiguating
spectrally similar drums. With stems, that's gone. The kick model never
sees a snare. The cymbal model never sees a tom. The snare model only
has to distinguish snare-center from snare-rim from clap — and the
clap-vs-snare distinction (the original 3-channel "Width" feature was
designed for this) is already encoded in the stem separation: claps and
snares get separated by MDX23C upstream.

### 2. The onset detection problem becomes well-conditioned

Drum stems have **very high signal-to-noise** ratios for their own
transients (because the separator filtered out other instruments) and
**very low SNR** for other drums (which is good — fewer false positives).
A 1-class onset detector on a clean kick stem is essentially solved by
energy-thresholding; a tiny neural model will trivially beat that.

### 3. Class imbalance is manageable

In a unified 10-class model, kick has 88k examples and Crash2 has 2,878
(30× imbalance). With per-stem models:
- Cymbals model: ride (51k) vs crash1+2 (9k) vs splash (~1k). 51× imbalance
  is still annoying but only across 3 classes, not 10.
- Toms model: ~5k/5k/5k roughly balanced.
- Hihat model: closed (118k) vs open (14k). 8× imbalance, well within
  `pos_weight` territory.

### 4. Failure isolation

If one stem model underperforms, you isolate the problem to that stem
and that stem only. The kick model breaking doesn't poison snare
transcription. This is unlike the unified model where one class's
gradient could swamp another's.

### 5. Training cost stays modest

5 models × ~200k params per model = comparable total to the previous
unified attempt. But: each model trains independently, can be parallelized
across cores/GPUs, and converges faster because the problem is simpler.

### 6. The user has already implemented half of this

`stems_to_midi/note_classification_core.py` already classifies:
- snare hits into {snare, rimshot, clap} based on spectral features
- toms into {low, mid, high} based on `spectral_centroid_hz` and `fundamental_energy`
- cymbals into {ride, crash, splash} based on `body_energy` and `brilliance_energy`
- hihat into {closed, open} based on `sustain_ms` and `sizzle_energy`

These calibrated heuristics are the **ground truth labels** the per-stem
models will learn to imitate, but with full generalization power.

---

## What could go wrong

### 1. Stem separator quality is the ceiling

If MDX23C bleeds 10% of snare energy into the kick stem, the kick model
will learn to fire on snares. Mitigation:
- Run an inventory: take a clean e-GMD multitrack, sum the original kick
  to a mix, separate with MDX23C, measure leakage.
- If leakage is severe, fine-tune MDX23C on e-GMD data, or use a better
  separator (Demucs v4 drums model is the current SOTA).

### 2. Per-stem MIDI labels need to be reconstructed

e-GMD provides one MIDI file per multitrack. To get per-stem labels, we
need to filter the MIDI by which drum class each note maps to.

The mapping is already established in `model-training/label_encoder.py:27`:
```python
MAPPING = {
    36:0, 35:0,           # Kick → kick stem
    38:1, 40:1, 37:1, 39:1, # Snare → snare stem
    42:2, 44:2, 22:2,     # HH Closed → hihat stem
    46:3, 26:3,           # HH Open → hihat stem
    48:4, 50:4,           # Tom High → toms stem
    45:5, 47:5,           # Tom Mid → toms stem
    43:6, 58:6,           # Tom Low → toms stem
    49:7, 55:7,           # Crash 1 → cymbals stem
    52:8, 57:8,           # Crash 2 → cymbals stem
    51:9, 53:9, 59:9      # Ride → cymbals stem
}
```

A new helper `split_midi_by_stem(midi_path)` produces 5 stem-specific
MIDI files in ~20 lines of code.

### 3. The "easy" stems might still fail

If the kick model fails on a clean kick stem, the bug is in the
infrastructure (feature extraction, loss, label encoding). Use the kick
model as a continuous smoke test — it should always pass first.

### 4. Stem separation is slow

MDX23C is ~10× slower than realtime on CPU. For training on 444 hours of
e-GMD this is a meaningful preprocessing cost. Mitigation:
- Run separation once, cache stems to disk (compress as opus or flac for
  manageable disk).
- e-GMD already has source-separated stems by design (drums-only audio,
  no other instruments). For e-GMD training we may not need MDX23C at
  all — we just need to split the MIDI per drum class.

### 5. Inference latency compounds

5 model forward passes per inference. Still trivial — total ~100 MB of
weights, ~50ms per file on CPU. Not a concern.

---

## Prerequisites

- `03-test-prove-overfit-first.md` test harness working (so each per-stem
  model can be smoke-tested).
- `02-tooling-wishlist.md` Tier 1 tools (mir_eval eval wrapper).
- e-GMD dataset accessible at `/Volumes/1TB SSD 1/e-gmd-v1.0.0/` (or wherever).
- Working stem separator (existing).
- For training stem models: ~5-50 GB scratch disk per stem (cached
  spectrograms).
- Compute: CPU for kick/toms, recommended GPU for snare/hihat/cymbals.

---

## Implementation steps

### Phase 0: Confirm e-GMD has stem-isolated audio

```bash
ls /Volumes/1TB\ SSD\ 1/e-gmd-v1.0.0/drummer1/session1/ | head -20
# Look for *_kick.wav, *_snare.wav, etc. patterns OR per-multitrack subdirectories.
```

If e-GMD provides multitracks, use them directly (skip MDX23C entirely
during training). If only mixes are provided, run MDX23C once to cache
stems.

**Spec:** e-GMD does provide separate audio per drum kit element in the
multitrack subdirectories — this is documented in the dataset paper.
Confirm before scaling up.

### Phase 1: Build per-stem dataset

Create `model-training/datasets/stem_dataset.py`:

```python
"""
StemDataset: yields (stem_audio_tensor, stem_midi_label_tensor) pairs
for one stem type ('kick', 'snare', 'hihat', 'toms', 'cymbals').
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path

STEM_TO_CLASSES = {
    'kick':    [(36, 0), (35, 0)],  # both kick variants → class 0
    'snare':   [(38, 0), (40, 0), (37, 0), (39, 1)],  # snare→0, rimshot variants→0, clap→1
    'hihat':   [(42, 0), (44, 0), (22, 0), (46, 1), (26, 1)],  # closed→0, open→1
    'toms':    [(43, 0), (58, 0), (41, 0), (45, 1), (47, 1), (48, 2), (50, 2)],  # low/mid/high
    'cymbals': [(51, 0), (53, 0), (59, 0), (49, 1), (55, 1), (57, 1), (52, 2)],  # ride/crash/splash
}

class StemDataset(Dataset):
    def __init__(self, stem_type: str, manifest_file: str):
        self.stem_type = stem_type
        self.classes = STEM_TO_CLASSES[stem_type]
        self.entries = self._load_manifest(manifest_file)

    def _load_manifest(self, path):
        # Each line: <stem_audio_wav>\t<full_midi_for_session>
        # Returns list of (audio_path, midi_path) pairs
        ...

    def __getitem__(self, idx):
        audio_path, midi_path = self.entries[idx]
        # Load audio (mono, since stems are mono)
        spec = get_input_tensor(audio_path)   # [1, 128, T]
        # Load MIDI, filter to relevant pitches, map to local class indices
        notes = load_midi_notes(midi_path)
        filtered = [n for n in notes if any(n.pitch == p for p, _ in self.classes)]
        targets = build_targets_per_stem(filtered, spec.shape[2], self.classes)
        return spec, targets  # spec: [1, 128, T], targets: [T, 2*num_classes]
```

### Phase 2: Build the per-stem trainer

`model-training/train_stem.py`:

```python
"""
Train one per-stem transcriber.

Usage:
    python train_stem.py --stem kick --manifest manifests/kick_train.txt --val manifests/kick_val.txt
"""

import argparse
import torch
from model_stem import StemTranscriber  # new
from train_utils_stem import MultiTaskStemLoss, setup_training  # adapted from train_utils.py
from datasets.stem_dataset import StemDataset, STEM_TO_CLASSES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stem', required=True, choices=['kick', 'snare', 'hihat', 'toms', 'cymbals'])
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--val', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--out', default='models')
    args = parser.parse_args()

    num_classes = len(set(c for _, c in STEM_TO_CLASSES[args.stem]))
    model = StemTranscriber(num_classes=num_classes)

    # ... standard training loop with validation ...

    torch.save(model.state_dict(), f"{args.out}/stem_{args.stem}_best.pt")


if __name__ == '__main__':
    main()
```

### Phase 3: Per-stem ablation runs (smoke-test each)

```bash
# Use the 03-test-prove-overfit-first.md harness adapted per-stem
for stem in kick snare hihat toms cymbals; do
    conda run -n drumtomidi python train_stem.py \
        --stem $stem --manifest manifests/${stem}_overfit.txt \
        --val manifests/${stem}_overfit.txt --epochs 500
    conda run -n drumtomidi pytest tests/test_stem_overfit.py::test_${stem} -v
done
```

If any stem fails its overfit test, fix it before proceeding to full
training. **Kick should pass first** — if it doesn't, the bug is in the
common infrastructure.

### Phase 4: Full training (one model per stem)

```bash
for stem in kick snare hihat toms cymbals; do
    conda run -n drumtomidi python train_stem.py \
        --stem $stem \
        --manifest manifests/${stem}_train.txt \
        --val manifests/${stem}_val.txt \
        --epochs 30
done
```

Estimated time: 2–4 hours per stem on CPU, 30 min per stem on GPU.

### Phase 5: Build inference orchestrator

`model-training/inference_stem_orchestrator.py`:

```python
"""
Inference orchestrator:
  1. Load input drum-mix audio
  2. Run separator to produce 5 stems
  3. Run each per-stem model on its stem
  4. Merge events into one MIDI file
"""

from separate import separate_audio  # existing
from model_stem import StemTranscriber

def transcribe(audio_path, out_mid_path):
    stems = separate_audio(audio_path)  # returns dict {kick: tensor, snare: tensor, ...}
    all_events = []
    for stem_name, audio in stems.items():
        model = load_stem_model(stem_name)
        events = run_inference(model, audio)
        all_events.extend(events)
    write_midi(sorted(all_events), out_mid_path)
```

### Phase 6: End-to-end evaluation

```bash
# Use the mir_eval harness from 02-tooling-wishlist.md
conda run -n drumtomidi python tools/eval_with_mir_eval.py \
    --pred predicted.mid --gt ground_truth.mid
```

Expected F1: ≥0.85 on e-GMD test split.

---

## Evaluation

| Metric | Target | Measurement |
|--------|--------|-------------|
| Per-stem overfit (smoke test) | F1 ≥ 0.95 on a 10s loop | `tests/test_stem_overfit.py` per stem |
| Per-stem validation F1 | ≥ 0.85 (kick), ≥ 0.80 (toms), ≥ 0.70 (snare/hihat/cymbals) | held-out e-GMD val split |
| End-to-end MIDI F1 | ≥ 0.80 overall, ±50ms tolerance | mir_eval on full test set |
| Velocity correlation | ≥ 0.70 | per-class velocity scatter |
| Inference latency | <5s for a 3min file on CPU | wall-clock |

---

## Estimated effort

| Phase | Time | Compute |
|-------|------|---------|
| Phase 0 (audit e-GMD structure) | 1 hour | CPU |
| Phase 1 (StemDataset class) | 4 hours | CPU |
| Phase 2 (per-stem trainer) | 4 hours | CPU |
| Phase 3 (per-stem smoke tests) | 4 hours | CPU |
| Phase 4 (full per-stem training, 5 models) | 10-20 hours | CPU (or 2-5h GPU) |
| Phase 5 (inference orchestrator) | 3 hours | CPU |
| Phase 6 (end-to-end eval) | 2 hours | CPU |
| **Total** | **3-5 days** | mostly CPU |

Renting an A100 for phase 4 would compress that to 1 day total and cost
~$10–20.

---

## Escalation paths

- **If kick model fails the smoke test**: bug is in the per-stem
  infrastructure. Use `04-test-bug-isolation-grid.md` to find the cause.
- **If one stem (e.g., cymbals) consistently underperforms**: that stem's
  3-class problem is the actual hardest part of drum transcription.
  Consider escalating just that stem to approach 10 (pretrained encoder).
- **If end-to-end F1 < 0.6 despite good per-stem F1**: the merging step
  has bugs (overlapping events, lost velocities) — debug the orchestrator.
- **If overall F1 stuck at 0.6-0.7**: the stem separator is the
  bottleneck. Consider fine-tuning MDX23C on e-GMD (separate effort)
  or switching to Demucs v4 drums.
- **If everything works but you want better**: combine with approach 12
  (curriculum learning) to add synthetic data, or approach 07 (distill
  from classical pipeline) for additional training signal.

---

## Why this is the top recommendation (vs the other 9 approaches)

| Criterion | Approach 5 (stems) | Approach 6 (DSP+ML) | Approach 8/9 (port papers) | Approach 10 (pretrained) |
|-----------|--------|-----------|----------|----------|
| Reuses existing working code | ✓ separator | ✓ DSP detector | ✗ | ✗ |
| Problem decomposition makes it easier | ✓✓✓ | ✓✓ | ✓ | ✓ |
| No GPU required | ✓ | ✓ | depends | needs GPU |
| Per-component failure isolation | ✓✓✓ | ✓✓ | ✗ | ✗ |
| Risk of fundamental blocker | Low | Low | Medium (porting bugs) | Medium (env complexity) |
| Time-to-first-working-result | 3-5 days | 2-4 days | 5-7 days | 4-6 days |
| Reuses your data prep | ✓ | ✓ | ✗ | ✗ |
| Match to user's intuition | ★ user proposed this | also reasonable | unfamiliar | unfamiliar |

The user's instinct to "run the stem splitter AND then run my own
inference on the stems" is the right call. This document gives that
instinct a concrete implementation plan.
