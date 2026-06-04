# Critique & Failure Theories

> Diagnostic forensics for why `model-training/` did not produce a working
> drum transcription model despite ~150 epochs of training over 145+ files.
> Read this **after** `00-overview.md`. The ranked theory table at the
> bottom is the most actionable part.

---

## Part 1 — Intent (what was supposed to be built)

Per `model-training/Deep Learning Roadmap.md`, the system was designed as:

```
[Stereo drum stem WAV]
        │
        ▼  (Section 1: Feature Engineering)
[3-channel Mel-spec: Left, Right, Stereo-Width]  shape: [3, 128, T]
        │
        ▼  (Section 5: CRNN)
[Bi-directional GRU over temporal sequence]
        │
        ▼  (Section 5: Linear)
[Per-frame probability per drum class]  shape: [T, 11]
        │
        ▼  (Section 6 & 7: Calibration + Post-Process)
[MIDI file]
```

The plan was sound. The intent was clear. The dataset choice (Roland e-GMD,
444 hours) was canonical. The model size (~600k parameters) was small but
not absurd. The training compute (CPU-on-a-laptop) was aspirational but
not necessarily wrong for an overfit-then-iterate workflow.

**Nothing about the intent is wrong.** The failures are all in execution.

---

## Part 2 — Execution critique (the unflinching version)

### Critique 1 (high severity): the input channel collapse

**Roadmap says**:

> *"We use a 3-channel approach to preserve spatial information (Stereo
> Width) which is critical for differentiating centered Snares from wide
> Claps/Reverbs."*  
> — `Deep Learning Roadmap.md` §1

**Code says** (`model-training/feature_extractor.py:52-53`):

```python
mono = waveform.mean(dim=0, keepdim=True)  # [1, samples]
spec = amplitude_to_db(mel_transform(mono))  # [1, 128, Time]
```

The stereo information is **averaged away on line 52** before the
Mel-spectrogram is even computed. The function returns `[1, 128, T]`, not
`[3, 128, T]`.

**Model says** (`model-training/model.py:29`):

```python
nn.Conv2d(1, 32, 3, padding=1),
```

First Conv2d accepts exactly 1 channel.

**But model's own docstring says** (`model-training/model.py:53, 61, 65`):

```python
"""x: Input tensor of shape [Batch, Channels, Freq, Time]
   Channels should be 3 (L, R, Width)
   ...
# Conv block: [B, 3, 128, T] -> [B, 64, 32, T]
"""
```

The docstring lies. The actual conv layer accepts 1 channel.

**Why this matters**:
- Snare (centered) vs hand-clap (wide stereo) are spectrally similar but
  spatially different. Without the side-channel feature, the model has
  no way to distinguish them — they look identical to the Mel-spec.
- Same logic for distinguishing similar cymbals positioned differently
  in the stereo field.
- The 2-conv 32→64 filter stack is sized for richer input than mono mel.

**Confidence this is at least part of the problem**: high. It's a
silent regression from the design spec, and the design spec was explicit
about *why* the third channel mattered. There's also no test that would
have caught this — the docstring matches the spec, only the implementation
diverges.

### Critique 2 (highest severity): the smoke test never tested the failure mode

The original roadmap §8 specifies the smoke test like this:

> *"Proves that the data pipe is leak-proof. If the model can't memorize
> one single 30-second file, there is a fundamental bug in Step 1 or
> Step 3."*

`README.md:5-6` claims:

> *"Status: Implementation Complete — Smoke Test Passing. The core
> pipeline has been implemented and verified. The model can overfit a
> single sample, confirming data pipes are leak-free."*

`README.md:48`:

> *"10-epoch result: 0.849 → 0.086 (loss converging, pipeline verified)"*

**Read carefully**: the test that was actually run measures *training loss
descent*. It does not measure whether **running inference on the trained
model reproduces the training MIDI**. Those are different things.

The user's report from this session ("couldn't get it to generate anything
useful from one of the files I actually trained on") **is the failed
smoke test of the original roadmap**. The smoke test never actually ran
the assertion that matters.

This is the single most consequential process failure. Every downstream
decision (add velocity head, scale to 145 files, run for 150 epochs) was
built on the unverified assumption that the basics worked.

**Where to fix this**: `03-test-prove-overfit-first.md` defines the test
that should always have existed.

### Critique 3 (high severity): velocity head added without re-verifying memorization

Per `multi-task-velocity.SHIPPED.plan.md` (the SHIPPED plan, this dual-head
design *was* implemented), the output layer changed from `Linear(256, 10)`
(onset only) to `Linear(256, 20)` (onset + velocity). The loss changed from
single-task BCEWithLogitsLoss to `MultiTaskDrumLoss` (BCEWithLogitsLoss +
masked MSE).

What didn't happen: a re-run of the smoke test after this change. So when
the existing `smoke_test.py` was modified to support the new 20-channel
target, nobody confirmed the model could still memorize a single file.

The masked MSE has its own subtlety: the mask is `(onset_target > 0.5)`.
For the causal smear `[1.0, 0.8, 0.5, 0.2]`, only the first 2 frames of
each smear contribute to velocity loss. If the model puts velocity into
frame `X+1` (where the smear is 0.8) but inference reads velocity from
frame `X` (the peak), there's an off-by-one.

### Critique 4 (medium severity): inference-time post-processing has surface area never validated

`model-training/inference_core.py` has three post-processing steps that
are not tested against the training-time target encoding:

**a. Sigmoid + threshold** (line 78, 56):

```python
onset_probs = 1.0 / (1.0 + np.exp(-pred_np[:, class_idx]))
# ...
peaks = find_peaks_with_onset_snap(onset_probs, threshold, min_distance=1)
```

Default `threshold = 0.8` (`config.yaml:16`). With `pos_weight=[2.0..10.0]`
(dampened from the inverse-frequency `[5.10..156.13]`), it's unclear
whether the trained logits actually saturate above sigmoid 0.8. Possibly
the model is correctly predicting hits at sigmoid 0.5–0.7 and the
threshold throws them away.

**b. Onset snap-back** (lines 40-47):

```python
onset_start = max(0, peak_idx - 5)
gradient = np.gradient(probabilities[onset_start:peak_idx + 1])
steepest_local = onset_start + np.argmax(gradient)
results.append((steepest_local, probabilities[steepest_local]))
```

Training teaches "peak at frame X = onset at frame X" (the smear is
0.8 at X+1, 0.5 at X+2 — DECREASING). The peak IS the onset. Snapping
backward by up to 5 frames searches for the steepest rise, which for
a clean onset should also be at frame X (the rise from 0 to 1.0).
But this code can silently shift onsets earlier by 1–5 frames if the
prediction is noisy, which destroys the ±20 ms tolerance match in MIDI
diff.

**c. Velocity un-scaling** (lines 88-90):

```python
raw_vel = pred_np[frame, class_idx + 10]
velocity_value = 1.0 / (1.0 + np.exp(-raw_vel))  # sigmoid
velocity = int(min(127, max(35, (velocity_value ** (1.0 / 0.7)) * 127)))
```

The pipeline: `midi_vel/127 → ^0.7` (label encoding, `label_encoder.py:96`)
`→ sigmoid loss → trained → sigmoid → ^(1/0.7) → *127 → clamp[35,127]`.

Five non-linear transforms, and a hard minimum of 35 — which **silently
discards every ghost note and soft hit**. The whole point of the velocity
head was to capture dynamics; clamping to ≥35 throws away the bottom 27%
of the dynamic range.

### Critique 5 (medium severity): trained on 145 files, no validation set

The final training run (`train_checkpoint_v497.ckpt`) shows:
- 145 files per epoch
- 150 completed epochs
- 21,750 per-file results recorded
- **No `val_loss` field** — meaning training was run without `--val-list`

The user mentioned "thousands of files" in the broader experiment, but
the checkpoint records 145. There may have been multiple separate runs;
the latest one used a small subset.

Even with a larger run, the no-validation issue is fatal: training loss
kept descending (`mean=0.173`, `min=0.066`) which looks good in isolation
but says nothing about generalization. The model could be memorizing
spectrogram patterns rather than learning drum-onset features.

### Critique 6 (medium severity): hyperparameter thrash mid-experiment

The rescue cleanup commits captured this snapshot of the state when
training stopped:
- `VELOCITY_WEIGHT`: bounced 5.0 → 2.0
- `MAX_TRAIN_SECONDS`: bounced 30 → 300
- `scheduler_patience`: bounced 10 → 3
- `chunk_frames`: bounced 2000 → 8000
- `learning_rate`: setup default 1e-4 → 1e-3, but checkpoint-resume path forced 1e-4 (inconsistent within one file)

These adjustments were happening *during* the training run. Without a
validation set to confirm whether each change helped or hurt, this is
fishing-in-the-dark.

### Critique 7 (medium severity): model capacity may be too small

Reference points:
- **DrumTranscriber** (current): Conv(1→32→64) + BiGRU(2048→128) + Linear(256→20). ~600k parameters.
- **Onsets and Frames** (Hawthorne 2018, piano transcription baseline): ~10M params.
- **ADTOF CRNN** (Cuisinier 2021, drum-specific): ~5M params.
- **MT3** (Gardner et al 2022, multi-instrument transcription, SOTA): ~30M params.

The DrumTranscriber is 8–50× smaller than published baselines for similar
tasks. With mono input (effectively half-resolution feature space) and a
shallow conv stack, it may simply lack the capacity to learn 10 fine-grained
drum classes.

### Critique 8 (low severity): reinventing what's been published as open-source

Both **Magenta's Onsets-and-Frames** (Apache 2.0) and **ADTOF** (MIT)
provide reference PyTorch implementations of CRNN-based music
transcription, with paper-quality numbers. The choice to build from scratch
is defensible *if* you're learning the domain, but doubly punishing when
the from-scratch implementation has subtle bugs (per critiques 1-4) that
the published implementations don't have.

### Critique 9 (low severity): the dataset reference is fragile

The training manifests (`batch1.txt`, `val1_shuffed.txt`, etc.) contain
absolute paths like `/Volumes/1TB SSD 1/e-gmd-v1.0.0/drummer1/session1/...`.
These won't survive a different machine, a renamed drive, or a fresh
e-GMD download. The data lineage is documented (good) but not portable
(bad).

---

## Part 3 — Ranked theories for the observed failure

Each theory has a falsifiable test. Run the tests *in order*; stop when
you find a confirmed root cause and fix it before moving on.

### Theory T1 (very high confidence): channel collapse + capacity starvation

**Hypothesis**: the model is operating on mono input it was never
designed for. Stereo-width information for distinguishing similar drums
is gone. The 600k-parameter network may be too small to compensate.

**Test recipe**:

```bash
# Step 1: confirm the channel collapse is real
conda run -n drumtomidi python -c "
from feature_extractor import get_input_tensor
t = get_input_tensor('dl-1.wav')
print(f'Feature shape: {t.shape}')   # expect [1, 128, T]; should be [3, 128, T] per roadmap
"

# Step 2: train a 3-channel version on the same data
# - Update feature_extractor.py to return [3, 128, T] per roadmap §1
# - Update model.py:29 to nn.Conv2d(3, 32, ...)
# - Run smoke_test.py on dl-1 for 200 epochs
# - Compare final loss + inference quality vs mono baseline
```

**Confidence**: very high that this contributes. Whether it's the
*sole* cause is less certain.

### Theory T2 (high confidence): inference-side threshold/peak/snap-back bug

**Hypothesis**: the model is correctly producing the trained pattern,
but `heatmap_to_notes` is throwing it away due to:
- threshold too high (0.8 default in config.yaml)
- snap-back shifting onsets out of tolerance
- velocity clamp at 35 erasing soft hits

**Test recipe**:

```python
# After loading a trained checkpoint and running model(spec):
import numpy as np

# 1. Dump raw logits and sigmoid probabilities
logits = model(spec).detach().cpu().numpy()[0]  # [T, 20]
onset_probs = 1 / (1 + np.exp(-logits[:, :10]))   # [T, 10]

# 2. Print per-class max sigmoid value
for i, name in enumerate(['Kick','Snare','HHC','HHO','TH','TM','TL','Cr1','Cr2','Rd']):
    print(f'{name}: max_prob={onset_probs[:, i].max():.4f}, '
          f'frames_above_0.5={int((onset_probs[:, i] > 0.5).sum())}, '
          f'frames_above_0.8={int((onset_probs[:, i] > 0.8).sum())}')

# 3. Compare to expected: e.g., dl-1.mid has ~600 kick hits.
#    If max_prob < 0.8 for any class with ground-truth hits, threshold is wrong.
```

**Confidence**: high. The threshold/clamp/snap chain has surface area
that nobody verified end-to-end.

### Theory T3 (medium confidence): causal smear softens targets to mush

**Hypothesis**: the smear `[1.0, 0.8, 0.5, 0.2]` creates a 4-frame "soft"
target. With pos_weight pushing for high confidence, the model may learn
to predict ~0.5 *everywhere* near hits as a least-bad solution under the
BCE loss, never producing the distinct peaks that `find_peaks` needs.

**Test recipe**: train two models on dl-1, one with the current smear,
one with hard targets (1.0 only at hit frame). Compare:
- Per-class max sigmoid value at hit frames
- Per-class min sigmoid value at silence frames (1+ second from any hit)
- Visual: overlay predicted probability on ground-truth onsets

**Confidence**: medium. Smearing is a standard trick in onset detection
and usually helps; it could equally be that the smear shape is wrong
(too long, or wrong polarity) but the principle is fine.

### Theory T4 (medium confidence): pos_weight rebalance starved rare classes

**Hypothesis**: the dampened weights `[2.0..10.0]` are too soft for the
8.6×–55× class-imbalance ratio in e-GMD. Rare classes (HHO, TomMid,
Crash2) may have learned to always predict 0.

**Test recipe**: with any trained checkpoint, compute per-class F1
on a held-out file. If HHO/TomMid/Crash2 all have recall = 0, T4 is
confirmed.

**Confidence**: medium. The rescue commit explicitly flags this LR
mismatch as a known issue.

### Theory T5 (medium confidence): catastrophic overfitting in the no-val run

**Hypothesis**: 150 epochs × 145 files without validation = standard
overfitting failure mode. Train loss converges, test/val/inference
quality collapses.

**Test recipe**: take any current train_checkpoint and run inference on
a file *not* in any batch list. If F1 << 0.5 on a truly held-out file
but inference on a training file gives F1 > 0.7, this is overfitting.
**The user's report — failure even on a training file — argues AGAINST
this theory.** Overfitting would memorize the training file.

**Confidence**: medium. Real but probably not the primary cause given
the symptom.

### Theory T6 (low confidence): MIDI/audio alignment drift

**Hypothesis**: a sample-rate, tempo, or hop-length mismatch silently
shifts MIDI labels relative to audio frames. The model trains on
misaligned data and produces predictions that don't match.

**Test recipe**: run `visualizer.py` (`alignment_check.png` generator)
on dl-1. The peaks in the spectrogram should align vertically with the
hot spots in the label grid. **The user already has `alignment_check.png`
in the repo from a prior run; visually inspect it.**

**Confidence**: low. The `midi_shell` infrastructure has been
production-tested in `stems_to_midi/`. The most likely failure mode
would have shown up as obvious mis-alignment in the visualizer.

### Theory T7 (low confidence): velocity head poisons onset gradient

**Hypothesis**: joint multi-task learning of onset + velocity sometimes
fails when the loss balance is wrong, causing the velocity head to
dominate the gradient and degrade onset quality.

**Test recipe**: train an onset-only model (remove channels 10-19 from
both target and loss). Compare onset F1 to the joint model. If
onset-only significantly outperforms joint, T7 is confirmed.

**Confidence**: low. The velocity_weight=2.0 is conservative; masked
MSE only computes loss on hit frames, so its gradient magnitude is
naturally bounded.

---

## Part 4 — What the user's evidence tells us

The user said:
> *"I ran a training for a full day on thousands of files about 8 or 9
> loops through and I couldn't get it to generate anything useful from
> one of the files I actually trained on."*

This single sentence is enormously diagnostic:

1. **"A full day on thousands of files, 8-9 loops"** → real training time
   on a real dataset. The scale is correct.
2. **"Couldn't get it to generate anything useful"** → either output is
   empty (threshold issue → T2) or output is wrong notes (label encoding
   or capacity issue → T1, T3, T4) or output is wildly off-time
   (alignment issue → T6 or snap-back → T2).
3. **"From one of the files I actually trained on"** → this **rules out
   overfitting** (T5). A model that overfits would memorize training
   files perfectly. The fact that it can't reproduce a training file
   means the failure is in either feature extraction (T1), label encoding
   (T3), or inference post-processing (T2). Per the original roadmap's
   own statement, this is "a fundamental bug in Step 1 or Step 3."

**Most likely root cause based on evidence**: T1 (channel collapse) + T2
(inference post-processing). These two together explain "trained model
doesn't reproduce training data" much better than any single cause.

**Action**: run the test in `03-test-prove-overfit-first.md`. It is
designed to bisect between T1 and T2 specifically. After that, pick
between approaches 05/06 as the actual modeling path.

---

## Part 5 — Hard truths

1. **The roadmap was good. The execution was rushed.** Skipping the real
   smoke test (Step 8) was the single most expensive shortcut.

2. **"Loss converged" is not "model works."** Training loss is a proxy.
   The only ground truth is: does inference on the trained model produce
   sensible output?

3. **Reinventing onsets-and-frames or ADTOF was the wrong call.** Both
   are open-source, both have published F1 numbers in the 0.85+ range
   on similar tasks, both could have been ported in 5 days. The
   from-scratch CRNN was a learning exercise that got branded as
   production work.

4. **The hardest problem (where is the hit?) was already solved** by
   `stems_to_midi/` and by every onset detector since Bello 2005. Asking
   a neural network to learn that from scratch when a calibrated DSP
   detector exists in the same repo is needlessly hard.

5. **Compute being CPU-only was a constraint and is no longer.** The user
   has confirmed access to bigger CPUs and rentable GPUs. This changes
   the modeling-decision tree substantially.

The good news: every one of these problems is fixable. The infrastructure
(stem separator, classical pipeline, MIDI loader, dataset access) is solid.
The plans in the sibling files 03–14 lay out concrete paths forward,
each ranked by risk and expected payoff.
