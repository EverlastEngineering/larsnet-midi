# Test: Prove Overfit First (the Missing Step 8)

> **THE MANDATORY FIRST STEP.** Do this before any modeling work in
> approaches 05–14. The user spent ~1 day training and the result was a
> model that can't reproduce a single training file. This document
> defines the test that should always have existed, and the bisection
> procedure to find the bug it exposes.
>
> Schema follows `00-overview.md`: Premise → Architecture → Why →
> Risks → Prereqs → Implementation → Evaluation → Effort → Escalation.

---

## Premise

**The fundamental sanity check for any neural transcription system**:
take a single audio file, train the model until loss approaches zero,
then run inference on that same file. The output should match the
ground truth nearly perfectly. If it doesn't, the bug is in your code
(features, labels, or inference post-processing), not your model
architecture or your dataset.

The original roadmap states this explicitly:

> *"Proves that the data pipe is leak-proof. If the model can't memorize
> one single 30-second file, there is a fundamental bug in Step 1
> (features) or Step 3 (label encoding)."*  
> — `Deep Learning Roadmap.md` §8

The previous attempt's `smoke_test.py` measures only that loss descends
during training. It never runs inference on the trained model and never
compares output MIDI to ground truth. **This is the missing test.**

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 1: Train                                                  │
│  ──────────────                                                  │
│  Input: 10-second drum loop (audio + MIDI ground truth)         │
│  Process: existing smoke_test pipeline, 500-1000 epochs         │
│  Success: training loss < 0.01                                  │
│  Output: overfit_checkpoint.ckpt                                │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 2: Inference                                              │
│  ──────────────────                                              │
│  Input: same 10-second drum loop audio                          │
│         overfit_checkpoint.ckpt                                 │
│  Process: existing inference.py pipeline                        │
│  Output: predicted.mid                                          │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 3: Assert (mir_eval)                                      │
│  ─────────────────────────                                       │
│  Compare predicted.mid vs ground_truth.mid                      │
│  Assert: F1 ≥ 0.95 within ±20ms onset tolerance                 │
│  Assert: velocity correlation ≥ 0.80                            │
│  Assert: each ground-truth class produces ≥1 prediction         │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                  ┌───────────┴───────────┐
                  ▼                       ▼
              ┌───────┐               ┌────────┐
              │ PASS  │               │  FAIL  │
              └───┬───┘               └────┬───┘
                  │                        │
                  ▼                        ▼
            existing pipeline           run the bisection in
            CAN memorize.               §"Failure Modes & Bisection"
            Move to 04 to find          below to find which stage
            inference-time bugs         is broken.
            on real data.
```

---

## Why this should work

Every machine learning system that produces sensible output for novel
data must, by construction, also produce sensible output for data it
was trained on. The contrapositive: if it fails on training data, it
will fail worse on novel data, and modeling approaches won't help.

This test takes ~10 minutes on CPU and gives you a single
unambiguous yes/no signal. The bisection procedure below converts a
"fail" outcome into a specific bug location in 30 minutes.

---

## What could go wrong

1. **The 10-second loop is too easy** — model memorizes via overfit, hides
   bugs. Mitigation: also test on a 60-second loop with multiple time
   signatures.

2. **The test environment differs from the training environment** —
   e.g., different sample rate, different normalization. Mitigation:
   use the existing `train_utils.load_audio` and `inference.py`
   end-to-end; don't reimplement.

3. **The synthetic test data masks problems** — a perfectly aligned
   synthetic file won't expose the human-played MIDI timing issues in
   e-GMD. Mitigation: run the test on BOTH a synthetic loop (rules out
   alignment bugs) AND on `dl-1.wav` (real-world data).

4. **Pytest fixtures get expensive** — training 1000 epochs in a fixture
   is slow. Mitigation: cache the trained checkpoint and reuse across
   test functions in the same session.

---

## Prerequisites

- **Conda env**: `drumtomidi` (already exists; see `environment.yml`).
- **Dependencies to add**: `pip install mir_eval pretty_midi pytest`.
  Likely already present; verify with `conda run -n drumtomidi pip list | grep -iE 'mir.eval|pretty.midi|pytest'`.
- **Test data**:
  - `model-training/dl-1.wav` + `dl-1.mid` (already in repo). Real-world
    drum performance. Use for "true overfit" test.
  - A synthetic 10-second loop generated per `02-tooling-wishlist.md`
    Tool 2. Use for "clean baseline" test. If you don't have this yet,
    use `dl-1` and accept the noise floor.
- **Disk space**: ~25 MB for one checkpoint.
- **Runtime**: 10–20 minutes on CPU per train+test cycle.

---

## Implementation steps

### Step 1: Establish ground-truth WAV/MIDI pair

```bash
# Option A: Use dl-1 (real performance, well-known)
cd model-training
ls dl-1.wav dl-1.mid

# Option B: Generate a 10s synthetic loop (clean labels)
# (requires Tool 2 from 02-tooling-wishlist.md)
python tools/synth_drum_loop.py \
    --bpm 120 --bars 4 --pattern simple_kick_snare \
    --out_wav tests/fixtures/10s_loop.wav \
    --out_mid tests/fixtures/10s_loop.mid
```

### Step 2: Create the test fixture and assertion

Create `model-training/tests/test_overfit_reproduction.py`:

```python
"""
Mandatory smoke test: a trained model must reproduce its training MIDI.

This is the missing Step 8 from Deep Learning Roadmap.md. Per the roadmap:
"If the model can't memorize one single 30-second file, there is a
fundamental bug in Step 1 or Step 3."

Run: conda run -n drumtomidi pytest tests/test_overfit_reproduction.py -v
"""

import os
import sys
import pytest
from pathlib import Path

# Add parent dir to path so we can import model-training modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import mir_eval
import pretty_midi

from smoke_test import run_smoke_test
from inference import run_inference
from config import DEVICE


# -------- Acceptance thresholds --------
ONSET_TOLERANCE_S = 0.020      # 20ms; tighter than mir_eval default
MIN_F1_OVERFIT = 0.95          # near-perfect on memorized data
MIN_VELOCITY_CORR = 0.80       # correlation, not absolute match
TRAINING_EPOCHS = 500          # enough to memorize a 10s loop
TARGET_TRAIN_LOSS = 0.01       # if loss > this, training didn't even succeed


# -------- Fixtures --------
@pytest.fixture(scope="module")
def fixture_audio():
    """Path to ground-truth audio file."""
    path = Path(__file__).parent / "fixtures" / "10s_loop.wav"
    if not path.exists():
        # Fall back to dl-1
        path = Path(__file__).parent.parent / "dl-1.wav"
    if not path.exists():
        pytest.skip(f"No test audio available at {path}")
    return path


@pytest.fixture(scope="module")
def fixture_midi(fixture_audio):
    """Path to ground-truth MIDI file (matched to audio)."""
    return fixture_audio.with_suffix(".mid")


@pytest.fixture(scope="module")
def overfit_checkpoint(tmp_path_factory, fixture_audio, fixture_midi):
    """
    Train the existing pipeline to memorize the fixture audio.
    Returns: path to checkpoint, plus the final training loss.
    """
    ckpt_dir = tmp_path_factory.mktemp("overfit")
    print(f"\n[OVERFIT-TRAIN] Training {TRAINING_EPOCHS} epochs on {fixture_audio.name}...")

    final_loss, model, optimizer = run_smoke_test(
        audio_path=str(fixture_audio),
        midi_path=str(fixture_midi),
        epochs=TRAINING_EPOCHS,
        device=DEVICE,
    )

    ckpt_path = ckpt_dir / "overfit.ckpt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'loss': final_loss,
    }, ckpt_path)

    print(f"[OVERFIT-TRAIN] Final loss: {final_loss:.6f}")
    return {'path': ckpt_path, 'loss': final_loss}


# -------- Assertions --------
def test_training_actually_converges(overfit_checkpoint):
    """
    First gate: did training even succeed?
    If loss > 0.01 after 500 epochs on a 10s file, the optimizer is broken.
    """
    assert overfit_checkpoint['loss'] < TARGET_TRAIN_LOSS, (
        f"Training loss {overfit_checkpoint['loss']:.4f} > {TARGET_TRAIN_LOSS}. "
        f"The model is failing to memorize a 10s file. "
        f"Bug is in: optimizer, loss function, or model capacity."
    )


def test_raw_logits_distinguish_hits_from_silence(overfit_checkpoint, fixture_audio, fixture_midi):
    """
    Second gate: do the trained logits actually peak at known hit frames?
    This isolates whether the model learned the right shape (regardless of
    inference post-processing).
    """
    from feature_extractor import get_input_tensor
    from model import DrumTranscriber
    from train_utils import load_midi_notes, build_targets

    model = DrumTranscriber().to(DEVICE)
    ckpt = torch.load(overfit_checkpoint['path'], map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    spec = get_input_tensor(str(fixture_audio)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(spec).cpu().numpy()[0]  # [T, 20]
    onset_probs = 1.0 / (1.0 + np.exp(-logits[:, :10]))  # sigmoid

    notes, _ = load_midi_notes(str(fixture_midi))
    target = build_targets(notes, spec.shape[3]).numpy()[0]  # [T, 20]
    onset_targets = target[:, :10]

    # Per-class: at frames where ground truth has an onset, what is the
    # average predicted probability?
    for class_idx in range(10):
        gt_frames = np.where(onset_targets[:, class_idx] >= 0.99)[0]  # exact-hit frames
        if len(gt_frames) == 0:
            continue  # no hits of this class in this file
        avg_prob_at_hits = onset_probs[gt_frames, class_idx].mean()
        avg_prob_at_silence = onset_probs[
            np.setdiff1d(np.arange(len(onset_probs)), gt_frames), class_idx
        ].mean()
        ratio = avg_prob_at_hits / max(avg_prob_at_silence, 1e-6)
        assert avg_prob_at_hits > 0.5, (
            f"Class {class_idx} (~{int(onset_targets[:, class_idx].sum())} hits): "
            f"prob at hits = {avg_prob_at_hits:.3f}, expected > 0.5. "
            f"Bug is in: label encoding, smear shape, or loss function."
        )
        assert ratio > 5.0, (
            f"Class {class_idx} signal-to-noise ratio = {ratio:.1f}, expected > 5.0. "
            f"Hits-vs-silence contrast too low. Bug likely in pos_weight or capacity."
        )


def test_inference_recovers_training_midi(overfit_checkpoint, fixture_audio, fixture_midi, tmp_path):
    """
    The headline test: run inference, write MIDI, compare to ground truth.
    """
    out_mid = tmp_path / "predicted.mid"

    # Run inference with a relaxed threshold first to maximize recall;
    # if THIS fails, the model is dead.
    notes = run_inference(
        audio_path=str(fixture_audio),
        output_path=str(out_mid),
        checkpoint_path=str(overfit_checkpoint['path']),
        threshold=0.3,   # deliberately low; we want EVERYTHING
        device=DEVICE,
    )

    # Glob to find the actual output (run_inference adds _v{N}_t{thresh})
    candidates = list(tmp_path.glob("predicted_v*_t*.mid"))
    assert candidates, "Inference did not produce a MIDI file"
    out_mid = candidates[0]

    # Evaluate with mir_eval
    pred_pm = pretty_midi.PrettyMIDI(str(out_mid))
    gt_pm = pretty_midi.PrettyMIDI(str(fixture_midi))

    pred_intervals, pred_pitches = _midi_to_arrays(pred_pm)
    gt_intervals, gt_pitches = _midi_to_arrays(gt_pm)

    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        gt_intervals, gt_pitches,
        pred_intervals, pred_pitches,
        onset_tolerance=ONSET_TOLERANCE_S,
        pitch_tolerance=0.5,
        offset_ratio=None,
    )
    print(f"\n[INFERENCE-TEST] Precision={p:.3f} Recall={r:.3f} F1={f:.3f}")
    print(f"  Ground-truth notes: {len(gt_intervals)}, Predicted notes: {len(pred_intervals)}")

    assert f >= MIN_F1_OVERFIT, (
        f"F1 = {f:.3f} (required >= {MIN_F1_OVERFIT}). "
        f"Model trained to loss {overfit_checkpoint['loss']:.4f} but inference "
        f"can't reproduce training MIDI. The bug is in: inference post-processing "
        f"(see inference_core.heatmap_to_notes), threshold, or peak detection."
    )


def test_velocity_correlation(overfit_checkpoint, fixture_audio, fixture_midi, tmp_path):
    """
    Velocity head should reproduce ground-truth velocities to within
    correlation 0.8 on a memorized file.
    """
    out_mid = tmp_path / "predicted_vel.mid"

    notes = run_inference(
        audio_path=str(fixture_audio),
        output_path=str(out_mid),
        checkpoint_path=str(overfit_checkpoint['path']),
        threshold=0.3,
        device=DEVICE,
    )

    candidates = list(tmp_path.glob("predicted_vel_v*_t*.mid"))
    assert candidates
    pred_pm = pretty_midi.PrettyMIDI(str(candidates[0]))
    gt_pm = pretty_midi.PrettyMIDI(str(fixture_midi))

    # Match each gt note to its nearest pred note of same pitch within tolerance
    matches = []
    for gt_inst in gt_pm.instruments:
        for gt_note in gt_inst.notes:
            best = None
            for pred_inst in pred_pm.instruments:
                for pred_note in pred_inst.notes:
                    if pred_note.pitch != gt_note.pitch:
                        continue
                    dt = abs(pred_note.start - gt_note.start)
                    if dt <= 0.05 and (best is None or dt < best[0]):
                        best = (dt, gt_note.velocity, pred_note.velocity)
            if best is not None:
                matches.append((best[1], best[2]))

    assert len(matches) >= 10, f"Only {len(matches)} velocity matches; need >= 10"
    gt_vels = np.array([m[0] for m in matches])
    pred_vels = np.array([m[1] for m in matches])

    if gt_vels.std() < 1 or pred_vels.std() < 1:
        pytest.skip("Insufficient velocity variance in either source")

    corr = np.corrcoef(gt_vels, pred_vels)[0, 1]
    print(f"\n[VELOCITY-CORR] r = {corr:.3f} over {len(matches)} matched notes")
    assert corr >= MIN_VELOCITY_CORR, (
        f"Velocity correlation {corr:.3f} < {MIN_VELOCITY_CORR}. "
        f"Bug is in: velocity scaling (label_encoder.py ^0.7), "
        f"velocity un-scaling (inference_core.py), or velocity clamp at 35."
    )


# -------- Helpers --------
def _midi_to_arrays(pm):
    """Convert PrettyMIDI to (intervals[N,2], pitches[N]) for mir_eval."""
    intervals, pitches = [], []
    for inst in pm.instruments:
        for note in inst.notes:
            intervals.append([note.start, note.end])
            pitches.append(float(note.pitch))
    if not intervals:
        return np.zeros((0, 2)), np.zeros((0,))
    return np.array(intervals), np.array(pitches)
```

### Step 3: Run it

```bash
cd model-training
conda run -n drumtomidi pytest tests/test_overfit_reproduction.py -v -s
```

Expected runtime: 10–20 minutes on CPU.

### Step 4: Interpret the result

| Outcome | Meaning | Next action |
|---------|---------|-------------|
| All 4 tests pass | Pipeline is healthy on memorized data | Proceed to approach 05 or 06 |
| `test_training_actually_converges` fails | Optimizer/loss/capacity broken | Bisect with §"Failure Modes" below |
| `test_raw_logits_distinguish_hits_from_silence` fails | Model isn't learning the target shape | Check `label_encoder.py`, `MultiTaskDrumLoss` |
| `test_inference_recovers_training_midi` fails (but logits test passes) | Inference post-processing throws away correct predictions | Check `heatmap_to_notes`, threshold, snap-back |
| `test_velocity_correlation` fails (but onset test passes) | Velocity head decoupled from velocity ground truth | Check the 5-step velocity transform chain |

---

## Failure modes & bisection procedure

If any test fails, **do not** skip to a modeling approach. Bisect first.

### Bisection 1: is the loss going down at all?

```bash
conda run -n drumtomidi python smoke_test.py --audio dl-1.wav --midi dl-1.mid --epochs 50 2>&1 | tee /tmp/smoke.log
grep "Loss:" /tmp/smoke.log | tail -5
```

- If loss starts at ~1.0 and stays there → optimizer can't make progress.
  Check: learning rate, gradient clipping, model device placement.
- If loss starts low (<0.1) → labels are mostly zeros and the loss is
  trivially low. Check `build_targets` shape, target value range.
- If loss diverges or NaNs → numerical instability. Check device (CPU is
  safe, mps/cuda may NaN per the rescue notes).

### Bisection 2: is the model learning the right shape at peaks?

Use `model-training/visualizer.py` on a 30-second slice of dl-1.

```bash
conda run -n drumtomidi python -c "
from feature_extractor import get_input_tensor
from train_utils import load_midi_notes, build_targets
from visualizer import plot_alignment_check
import torch

spec = get_input_tensor('dl-1.wav')
notes, _ = load_midi_notes('dl-1.mid')
targets = build_targets(notes, spec.shape[2])  # [1, T, 20]
plot_alignment_check(spec, targets[0, :, :11].T)
"
ls /tmp/alignment_check.png
```

Open the PNG. The top panel (spectrogram) should have visible
transients. The bottom panel (label heatmap) should have hot spots
*directly under* those transients. If the hot spots are shifted left or
right of the transients, you have an alignment bug. Likely culprits:

- Hop length mismatch (`feature_extractor.py:46` says 512; `label_encoder.py:55` defaults to 512 — check they match)
- Sample-rate mismatch (resampling in `feature_extractor.py:38` is conditional; verify the source file's sample rate)

### Bisection 3: are inference outputs above threshold at known hits?

```python
# After training, before testing the full inference path:
import torch
import numpy as np
from feature_extractor import get_input_tensor
from model import DrumTranscriber

model = DrumTranscriber()
ckpt = torch.load('models/overfit.ckpt', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

spec = get_input_tensor('dl-1.wav').unsqueeze(0)
with torch.no_grad():
    logits = model(spec).numpy()[0]
probs = 1 / (1 + np.exp(-logits[:, :10]))

print("Per-class max sigmoid output:")
for i, name in enumerate(['Kick','Snare','HHC','HHO','TH','TM','TL','Cr1','Cr2','Rd']):
    print(f"  {name:6s}: max={probs[:,i].max():.3f}, above_0.5={int((probs[:,i]>0.5).sum())}, above_0.8={int((probs[:,i]>0.8).sum())}")
```

- If max sigmoid < 0.5 for any class with ground-truth hits → model didn't
  learn that class. Likely pos_weight too low or class missing from
  training set.
- If max sigmoid > 0.5 but < 0.8 → threshold of 0.8 (config.yaml default)
  is throwing away correct predictions. Try threshold=0.4.
- If max sigmoid > 0.8 but inference still produces wrong MIDI → the bug
  is in `find_peaks_with_onset_snap` or `write_midi`, not the model.

### Bisection 4: is the smear shape preventing distinct peaks?

```python
# Same setup as Bisection 3:
import matplotlib.pyplot as plt

fig, axes = plt.subplots(10, 1, figsize=(15, 20), sharex=True)
for i, ax in enumerate(axes):
    ax.plot(probs[:, i])
    ax.axhline(0.8, color='r', alpha=0.3, label='current threshold')
    ax.axhline(0.5, color='orange', alpha=0.3, label='loose threshold')
    ax.set_title(f"Class {i} sigmoid output over time")
plt.tight_layout()
plt.savefig('/tmp/per_class_probs.png', dpi=100)
```

- If the probability curves are smooth plateaus instead of distinct peaks
  → the smear shape `[1.0, 0.8, 0.5, 0.2]` has smeared the model's output
  into a continuous bump. `find_peaks` can't find peaks in a plateau.
  Mitigation: shorten smear to `[1.0, 0.5]` or remove entirely.

---

## Evaluation

The test harness IS the evaluation. Either all 4 assertions pass or they
don't. There is no partial credit.

If they pass: proceed to approach 05 (stems-as-input) or 06 (classical-
onset-then-classify). Both should now ALSO pass these same assertions
after their own training run.

If they fail: do not proceed to any approach until you know which
assertion fails and you've fixed the root cause. The bisection procedure
above tells you exactly what to look at.

---

## Estimated effort

| Subtask | Time |
|---------|------|
| Write test_overfit_reproduction.py | 1 hour |
| Generate synthetic 10s loop (if tooling exists) | 0.5 hour |
| Initial run + interpret result | 30 min |
| Bisection (if it fails — and it probably will) | 1–3 hours |
| Fix root cause + verify | 1–4 hours |
| **Total**: | **0.5–1 day** |

CPU is sufficient. No external dependencies beyond `mir_eval` + `pretty_midi`.

---

## Escalation paths

- **If the test passes the first time**: surprising. Validate by
  manually opening the predicted MIDI in a DAW and confirming it sounds
  right. If it does, the failure was in evaluation methodology — likely
  `inference.compare_midi` not being mir_eval-equivalent.

- **If multiple bisections all "succeed" but the test still fails**:
  there's a subtle multi-component bug. Run approach 04 (bug-isolation-grid)
  to factorial-design the ablation.

- **If the bug is unfixable with current architecture**: switch to
  approach 05 or 06 (which sidestep the failure modes of the current
  monolithic model) or to approach 08/09 (port a reference architecture
  that's known to work).
