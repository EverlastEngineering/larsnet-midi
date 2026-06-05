---
name: drum-transcription-debug
description: Use when debugging why a trained drum transcription model in model-training/ fails to reproduce its training MIDI, or when the overfit smoke test fails. Walks the 4-test bisection procedure that isolates the bug to features (Step 1), labels (Step 3), or inference post-processing.
---

# Drum Transcription Debugging Skill

The model under `model-training/` has a known failure pattern: training
loss converges low (~0.17) but inference on a training file produces
unusable MIDI. The original roadmap stated this means "a fundamental bug
in Step 1 (features) or Step 3 (label encoding)" — but the inference
post-processing surface area in `inference_core.py` is a third candidate.

This skill walks the bisection deterministically. **Stop at the first
failed test** and fix the root cause before continuing.

## Required reading before running any test

Open these in order:

1. `model-training/agent-plans/next-attempt/01-critique-and-theories.md` —
   the ranked theory table at the bottom. Note the leading theory
   (T1 — channel collapse) and the test recipe attached to each.
2. `model-training/agent-plans/next-attempt/03-test-prove-overfit-first.md` —
   the test harness this skill operates against.

## Environment

All commands run inside `conda run -n drumtomidi`. The env is already
installed; verify with `conda env list | grep drumtomidi`.

## The bisection sequence

### Test 0: Sanity — does the feature extractor return the expected shape?

This catches the channel-collapse bug immediately. **Run this first;
takes 5 seconds.**

```bash
conda run -n drumtomidi python -c "
import sys; sys.path.insert(0, 'model-training')
from feature_extractor import get_input_tensor
t = get_input_tensor('model-training/dl-1.wav')
print(f'Feature shape: {t.shape}')
print(f'Expected per roadmap: torch.Size([3, 128, T])')
print(f'Actual right now (pre-fix): torch.Size([1, 128, T])')
"
```

If shape is `[1, 128, T]` → the channel-collapse bug is unfixed. See
`01-critique-and-theories.md` Critique 1 for the exact lines to change
in `feature_extractor.py` and `model.py`. Fix and continue.

### Test 1: Does training loss actually descend?

```bash
conda run -n drumtomidi python model-training/smoke_test.py \
    --audio model-training/dl-1.wav \
    --midi model-training/dl-1.mid \
    --epochs 50 2>&1 | tee /tmp/smoke.log
grep "Loss:" /tmp/smoke.log | tail -5
```

Interpretation:
- Loss stuck near 1.0 across all 50 epochs → optimizer / loss / device bug.
  Check learning rate (`config.yaml` says 0.001), gradient clipping,
  model device placement (currently forced to CPU per `config.py:_DEVICE_CACHE`).
- Loss starts <0.1 → labels are mostly zeros and the loss is trivially
  low. Check `build_targets` output shape (`[1, T, 20]`) and value range
  (channels 0-9 in `[0, 1]`, channels 10-19 in `[0, 1]`).
- Loss diverges or NaN → numerical instability. The rescue notes flag
  cuda/mps as producing this; CPU is the safe path right now.

### Test 2: Does the alignment visualizer show hits aligned with transients?

```bash
conda run -n drumtomidi python -c "
import sys; sys.path.insert(0, 'model-training')
from feature_extractor import get_input_tensor
from train_utils import load_midi_notes, build_targets
from visualizer import plot_alignment_check

spec = get_input_tensor('model-training/dl-1.wav')
notes, _ = load_midi_notes('model-training/dl-1.mid')
targets = build_targets(notes, spec.shape[2])  # [1, T, 20]
plot_alignment_check(spec, targets[0, :, :11].T)
"
ls -la /tmp/alignment_check.png
```

Open the PNG. **The agent cannot see images; ask the user to look.** The
expected output:
- Top panel (spectrogram) shows visible transients (vertical bright stripes at drum hits).
- Bottom panel (label heatmap) has hot spots **directly under** those transients.
- If the hot spots are shifted left or right of the transients → alignment bug.
  Most likely culprits:
  - Hop length mismatch (`feature_extractor.py:46` says 512; `label_encoder.py:55` defaults to 512 — verify they match)
  - Sample-rate mismatch (`feature_extractor.py:38` resampling is conditional — verify source is actually 44100 Hz)

### Test 3: Are inference outputs above threshold at known hits?

After training (e.g. from `models/dl-1.ckpt` or a fresh `models/overfit.ckpt`):

```bash
conda run -n drumtomidi python -c "
import sys; sys.path.insert(0, 'model-training')
import torch, numpy as np
from feature_extractor import get_input_tensor
from model import DrumTranscriber

CKPT = 'model-training/models/dl-1.ckpt'
model = DrumTranscriber()
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

spec = get_input_tensor('model-training/dl-1.wav').unsqueeze(0)
with torch.no_grad():
    logits = model(spec).numpy()[0]
probs = 1 / (1 + np.exp(-logits[:, :10]))

names = ['Kick','Snare','HHC','HHO','TomH','TomM','TomL','Cr1','Cr2','Rd']
print('Per-class sigmoid output:')
for i, name in enumerate(names):
    print(f'  {name:5s}: max={probs[:,i].max():.3f}, above_0.5={int((probs[:,i]>0.5).sum())}, above_0.8={int((probs[:,i]>0.8).sum())}')
"
```

Interpretation:
- Max sigmoid < 0.5 for any class with ground-truth hits → model didn't learn that class. Likely pos_weight too low or class missing from training set.
- Max sigmoid > 0.5 but < 0.8 → threshold of 0.8 (config.yaml default) is throwing away correct predictions. Try threshold=0.4 in inference.
- Max sigmoid > 0.8 but inference still produces wrong MIDI → the bug is in `inference_core.find_peaks_with_onset_snap` or `write_midi`, not the model.

### Test 4: Is the smear shape preventing distinct peaks?

Generate per-class probability traces and inspect for plateaus instead of peaks:

```bash
conda run -n drumtomidi python -c "
import sys; sys.path.insert(0, 'model-training')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, numpy as np
from feature_extractor import get_input_tensor
from model import DrumTranscriber

CKPT = 'model-training/models/dl-1.ckpt'
model = DrumTranscriber()
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

spec = get_input_tensor('model-training/dl-1.wav').unsqueeze(0)
with torch.no_grad():
    logits = model(spec).numpy()[0]
probs = 1 / (1 + np.exp(-logits[:, :10]))

fig, axes = plt.subplots(10, 1, figsize=(15, 20), sharex=True)
for i, ax in enumerate(axes):
    ax.plot(probs[:, i])
    ax.axhline(0.8, color='r', alpha=0.3)
    ax.axhline(0.5, color='orange', alpha=0.3)
    ax.set_title(f'Class {i} sigmoid output')
plt.tight_layout()
plt.savefig('/tmp/per_class_probs.png', dpi=100)
print('saved /tmp/per_class_probs.png')
"
```

Interpretation:
- Distinct peaks rising from ~0 to near 1 and back → model is working;
  bug is downstream (threshold or post-processing).
- Smooth plateaus instead of distinct peaks → smear `[1.0, 0.8, 0.5, 0.2]`
  smeared the model's output into continuous bumps. `find_peaks` can't
  find peaks in a plateau. Mitigation: shorten smear to `[1.0, 0.5]`
  or remove entirely.
- Flat lines near 0 → that class wasn't learned. See Theory T4.
- Flat lines near 1 → that class is hallucinated everywhere. Loss
  function or pos_weight bug.

## Decision tree after bisection

| Failed test | Most likely cause | Where to fix |
|-------------|-------------------|--------------|
| Test 0 (shape) | Channel collapse | `feature_extractor.py:52` (re-add stereo+width), `model.py:29` (Conv2d(1→3)) |
| Test 1 (loss) | Optimizer / loss / device | `config.py` (DEVICE), `train_utils.py:setup_training` |
| Test 2 (alignment) | hop/SR mismatch | `feature_extractor.py:38,46` vs `label_encoder.py:51-56` |
| Test 3 (threshold) | Inference threshold/peak | `config.yaml:16` (threshold=0.8 → 0.4) or `inference_core.py:78,40-47` |
| Test 4 (smear) | Causal-smear shape | `label_encoder.py:84-90` (smear array) |

## Anti-patterns

- **Do not skip Test 0.** The channel-collapse bug is the highest-likelihood
  cause and is detected in 5 seconds. Don't waste an hour on training-loop
  debugging before checking it.
- **Do not run all 4 tests then look at the results.** Stop at the first
  failure; the later tests' interpretations assume the earlier ones passed.
- **Do not "fix" things you didn't measure.** Hyperparameter thrash
  caused half the previous attempt's confusion (see rescue commit
  history). Each fix should be motivated by a specific failed test.

## Reference points

- The roadmap: `model-training/Deep Learning Roadmap.md`
- The forensic critique: `model-training/agent-plans/next-attempt/01-critique-and-theories.md`
- The mandatory test: `model-training/agent-plans/next-attempt/03-test-prove-overfit-first.md`
- The ablation grid (if Test 0–4 don't isolate the cause): `model-training/agent-plans/next-attempt/04-test-bug-isolation-grid.md`
