# Approach 9: Port ADTOF

> Adopt the *Automatic Drum Transcription of Polyphonic Music* (ADTOF)
> reference CRNN by Cuisinier et al. (2021). It is **purpose-built for
> drums** (unlike Onsets-and-Frames which is piano-first), has a
> pretrained checkpoint we can fine-tune, and reports F1=0.85 on
> standard splits.
>
> Schema follows `00-overview.md`.

---

## Premise

The single best published reference for our exact problem (polyphonic
drum transcription from audio) is the ADTOF paper. The author released:

- A drum-specific MIDI dataset annotated against polyphonic music
  (i.e., drums-with-other-instruments, harder than e-GMD's isolated stems)
- A reference PyTorch CRNN implementation
- A pretrained checkpoint with reported F1 = 0.85 on Vogl-train/test splits

Repo: `https://github.com/MZehren/ADTOF` (MIT License, as of 2024).
Paper: Cuisinier et al. 2021, ICME.

Two ways to use this:

**Mode A (fastest path to known-good baseline)**: download the pretrained
checkpoint, run it on user's test files, measure F1. If it works well
out-of-the-box → ship it; you're done. If it underperforms on user's
data → fine-tune.

**Mode B (replicate + adapt)**: port the architecture into our `model-training/`
namespace, retrain from scratch on e-GMD with the ADTOF training recipe.
Compare against our own attempts.

---

## Architecture (ADTOF CRNN, simplified)

```
[Input: log mel-spec, [1, 84, T], n_fft=2048, hop=512, n_mels=84]
                │
                ▼
[Conv2d(1→32, 3x3) + ReLU + BN + Dropout]
                │
                ▼
[Conv2d(32→64, 3x3) + ReLU + BN + Dropout + MaxPool2d((3,1))]
                │
                ▼
[Conv2d(64→64, 3x3) + ReLU + BN + Dropout]
                │
                ▼
[Conv2d(64→64, 3x3) + ReLU + BN + Dropout + MaxPool2d((3,1))]
                │
                ▼
[Flatten freq dim → [B, T, 64*9=576]]
                │
                ▼
[BiGRU(576 → 60, 3 layers)]
                │
                ▼
[Dense(120 → N_classes), Sigmoid]
                │
                ▼
[Per-frame onset probabilities, [B, T, N]]
```

Default N=5 in the paper (kick, snare, hihat, toms, cymbals — exactly
our stem-level taxonomy, conveniently). Approx 1.2M parameters. **Smaller
than Onsets-and-Frames** (10M) but specifically tuned for drums.

For our 10-class taxonomy: change last Dense to output 10.

---

## Why this should work

1. **Drum-specific design choices.** The MaxPool kernel `(3, 1)` is
   chosen to preserve time resolution (which matters for drums) while
   collapsing frequency aggressively (which is fine since drum
   spectral signatures are wideband).
2. **A pretrained checkpoint exists.** Run it on a test file in 5
   minutes. Even if it's only F1=0.5 out-of-the-box, that's a
   *meaningful* baseline to beat.
3. **The dataset includes polyphonic context.** Unlike e-GMD's isolated
   drum stems, ADTOF training data includes drums-with-music. Closer to
   real-world inference conditions (after stem separation).
4. **Hand-tuned hyperparameters.** Paper reports specific learning
   rate, dropout schedule, augmentation choices that worked. We can
   start there instead of re-discovering them.
5. **Smaller than Onsets-and-Frames.** Fits on a small GPU or a beefy CPU.

---

## What could go wrong

1. **The pretrained checkpoint expects 5 classes, not 10.** Adaptation
   needed for full 10-class output.
2. **Different mel-spec parameters** (n_mels=84, not 128). All training
   data must be regenerated with matching feature config.
3. **The repo may be unmaintained / break with current PyTorch.**
   Mitigation: pin to the PyTorch version mentioned in the repo's
   `requirements.txt`; isolate in a separate conda env if needed.
4. **License compatibility**: MIT-licensed code merging into our
   Apache-2.0 (or whatever) repo is fine, but track the attribution.

---

## Prerequisites

- Internet access to `git clone` the ADTOF repo.
- Storage: ~5 GB for the pretrained checkpoint + cached features.
- Compute: pretrained inference runs on CPU; fine-tuning recommended on
  GPU.
- e-GMD dataset for fine-tuning.

---

## Implementation steps

### Phase 1: Get the pretrained model running (1 day)

```bash
mkdir -p external && cd external
git clone https://github.com/MZehren/ADTOF.git
cd ADTOF
# Read README, install per their instructions (likely Conda env)
# Download pretrained checkpoint (path documented in README)
```

```python
# Test on a user file
from adtof.inference import transcribe
midi_out = transcribe("dl-1.wav", model="pretrained.pt", output="/tmp/adtof_baseline.mid")
```

Run mir_eval on the output:

```bash
python tools/eval_with_mir_eval.py --pred /tmp/adtof_baseline.mid --gt dl-1.mid
```

**Decision gate**:
- If F1 > 0.7 out-of-the-box → ship the pretrained model. You're done.
- If F1 < 0.7 → proceed to fine-tuning (Phase 2).

### Phase 2: Fine-tune on e-GMD (2-3 days)

Build an e-GMD data loader compatible with ADTOF's feature pipeline
(n_mels=84 instead of our 128, otherwise similar).

```python
# Fine-tuning loop
model.load_state_dict(torch.load("pretrained.pt"))
# Freeze conv backbone for first 5 epochs (preserve learned features)
for p in model.conv_layers.parameters(): p.requires_grad = False
train(model, e_gmd_loader, epochs=5, lr=1e-3)
# Unfreeze everything for fine-tuning
for p in model.parameters(): p.requires_grad = True
train(model, e_gmd_loader, epochs=15, lr=1e-4)
```

### Phase 3: Adapt output classes (optional)

If you need 10 drum classes instead of ADTOF's 5:
- Replace the final dense layer: `nn.Linear(120, 10)`
- Train with random init on the new head while keeping the rest fine-tuned

### Phase 4: Evaluate

Standard mir_eval evaluation on e-GMD held-out test split.

Target: F1 ≥ 0.85 (matching the paper).

---

## Evaluation

| Metric | Target |
|--------|--------|
| F1 of pretrained checkpoint on user's files | reportable baseline |
| F1 after e-GMD fine-tune | ≥0.85 |
| Per-class F1 (especially rare classes) | ≥0.70 for Crash2/TomMid/HHO |
| Inference latency | <2s for 3min file on CPU |

---

## Estimated effort

| Phase | Time | Compute |
|-------|------|---------|
| 1: Pretrained baseline | 1 day | CPU |
| 2: Fine-tune | 2-3 days | GPU recommended |
| 3: 10-class adaptation | 1 day | GPU |
| 4: Eval | 0.5 day | CPU |
| **Total** | **3-5 days** | mixed |

---

## Escalation paths

- **If pretrained F1 is already great**: stop and ship. Combine with
  approaches 05 or 06 for per-stem refinement.
- **If pretrained is bad and fine-tuning doesn't help**: the architecture
  is the ceiling. Escalate to approach 10 (pretrained audio encoder)
  for a stronger backbone.
- **If 10-class adaptation degrades the 5-class performance**: keep both
  models. Run the 5-class for coarse classification, the 10-class for
  refinement.
- **If the ADTOF repo is broken / unmaintained**: switch to approach 08
  (Onsets-and-Frames) which has a more actively maintained PyTorch port.
