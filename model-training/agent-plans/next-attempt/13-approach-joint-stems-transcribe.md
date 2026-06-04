# Approach 13: Joint Stems + Transcription Model

> Train a single end-to-end model that does BOTH stem separation AND
> transcription jointly. Architecturally elegant; potentially
> data-efficient because the two tasks share low-level features. Highest
> risk and most novel of the approaches.
>
> Schema follows `00-overview.md`.

---

## Premise

Approaches 05 and 06 use separation as a fixed preprocessing step. This
has costs:

- Errors in separation propagate to transcription with no recovery
- The separator was trained without transcription in mind; its features
  may not be ideal for the downstream task
- Two-stage pipelines have failure modes neither stage can detect

A *joint* model could:
- Share the feature extractor between separation and transcription
- Optimize separation to help transcription (the joint loss includes
  transcription F1)
- Detect inconsistencies (if the separator outputs a clean "kick" stem
  but the transcriber predicts no kicks, something is wrong)

This is the architecture behind end-to-end music understanding systems
(see ByteDance's "End-to-end automatic drum transcription" Wei 2021,
which jointly learns separation and transcription).

---

## Architecture

```
[Input: stereo drum mix, [B, 2, T]]
                │
                ▼
[Shared Encoder: convolutional U-Net trunk]
                │
        ┌───────┴───────┐
        ▼               ▼
[Separation head]   [Transcription head]
  - Predict 5      - Take encoded features
    spectrogram      directly
    masks (one per   - Per-stem decoders
    stem)            - Output: per-stem
                       onset + velocity
        │               │
        ▼               ▼
[5 stem WAVs]      [10-channel MIDI events]
        │               │
        └──────┬────────┘
               │
               ▼
[Joint loss: separation L1 + transcription BCE + transcription velocity MSE]
```

A simpler variant (recommended starting point):

```
[Input: stereo drum mix]
                │
                ▼
[Shared encoder: same U-Net trunk]
                │
        ┌───────┴───────┐
        ▼               ▼
[Sep head]          [Transcription head: per-stem decoders]
                            │
                            ▼
                    [Combined output: 10-channel onset + velocity]
```

Use Demucs v4 (Meta's open-source separator, MIT license) as the U-Net
trunk. Add transcription heads to its bottleneck features.

---

## Why this should work

1. **Shared features.** A snare-detection feature is useful for both
   "isolate the snare track" and "label this snare hit". Learning them
   jointly gives more gradient signal per parameter.
2. **End-to-end optimization.** The separator can subtly improve its
   output to help the transcriber, even if that means slightly worse
   pure-separation quality.
3. **Single inference call.** No two-stage latency.
4. **Joint pretraining is well-established.** UVR-MDX, Demucs, and
   newer hybrid models all use shared encoders for separation tasks;
   adding a transcription head is a small extension.

---

## What could go wrong

1. **Joint loss balancing is fragile.** Separation loss may dominate
   transcription loss (or vice versa), and the wrong balance hurts both.
   Mitigation: scale losses, use uncertainty-weighted multi-task loss
   (Kendall et al. 2018).
2. **Larger model = more compute and longer training.** Demucs v4 is
   ~80M params; adding 5 transcription heads adds another 5-20M.
   Mitigation: train on GPU (mandatory for this approach).
3. **Two failure modes per output.** If the joint model is bad, is the
   problem in separation or transcription? Mitigation: track both
   losses separately; if one drops while the other stalls, the imbalance
   is identified.
4. **Pretraining complications.** Starting from a pretrained Demucs is
   the right approach but requires careful loading and freezing
   schedule.
5. **Inference is more complex.** Need to ensure both separation and
   transcription outputs are usable.

---

## Prerequisites

- Working Demucs v4 install: `pip install demucs`.
- GPU strongly required (Demucs v4 alone needs ~12 GB VRAM).
- e-GMD dataset.
- For evaluation: multi-track audio (so you can compare predicted stems
  to real stems). MUSDB18-HQ drums is the standard, but e-GMD provides
  multitracks too.

---

## Implementation steps

### Phase 1: Get Demucs running as a baseline

```bash
pip install demucs
demucs --two-stems=drums dl-1.wav
# Output: separated drums isolated from any other instruments
```

### Phase 2: Inspect Demucs internals

```python
from demucs.pretrained import get_model
model = get_model('htdemucs')  # hybrid transformer demucs
print(model)
# Identify the bottleneck layer (last conv before the up-conv path)
```

### Phase 3: Add transcription heads

```python
class JointSepAndTranscribe(nn.Module):
    def __init__(self, demucs_model):
        super().__init__()
        self.demucs = demucs_model  # full separator
        # Bottleneck features are ~512-dim
        self.transcription_heads = nn.ModuleDict({
            'kick': nn.Linear(512, 2),      # 1 class + 1 velocity
            'snare': nn.Linear(512, 6),     # 3 classes + 3 velocities
            'hihat': nn.Linear(512, 4),
            'toms': nn.Linear(512, 6),
            'cymbals': nn.Linear(512, 6),
        })

    def forward(self, mix):
        # Forward through demucs to get separation + bottleneck
        sep_output, bottleneck = self.demucs(mix, return_bottleneck=True)
        # Transcription per stem
        transcription = {}
        for stem_name, head in self.transcription_heads.items():
            transcription[stem_name] = head(bottleneck)
        return {'separated': sep_output, 'transcription': transcription}
```

### Phase 4: Joint loss

```python
def joint_loss(pred, target, lambda_sep=1.0, lambda_trans=1.0):
    sep_loss = nn.functional.l1_loss(pred['separated'], target['separated_stems'])
    trans_loss = 0
    for stem in TRANSCRIPTION_STEMS:
        bce = nn.functional.binary_cross_entropy_with_logits(
            pred['transcription'][stem][:, :, :NUM_CLASSES[stem]],
            target['transcription'][stem][:, :, :NUM_CLASSES[stem]]
        )
        # masked velocity loss
        mask = target['transcription'][stem][:, :, :NUM_CLASSES[stem]]
        mse = ((pred['transcription'][stem][:, :, NUM_CLASSES[stem]:] -
                target['transcription'][stem][:, :, NUM_CLASSES[stem]:]) ** 2 * mask).sum() / (mask.sum() + 1e-8)
        trans_loss += bce + 2.0 * mse
    return lambda_sep * sep_loss + lambda_trans * trans_loss
```

### Phase 5: Training schedule

1. **Warm-up (5 epochs)**: freeze Demucs entirely, train only transcription
   heads. Cheap; verifies the heads can learn from bottleneck features.
2. **Unfreeze last 3 layers of Demucs (10 epochs)**: let separator
   subtly adapt to transcription.
3. **Full fine-tune (20+ epochs)**: unfreeze everything, train with joint
   loss.

### Phase 6: Inference

```python
def transcribe(audio_path):
    mix = load_audio(audio_path)
    with torch.no_grad():
        out = model(mix)
    notes = []
    for stem_name, logits in out['transcription'].items():
        # Same per-frame peak detect logic as approach 05
        notes.extend(decode_stem_to_midi(stem_name, logits))
    return sorted(notes, key=lambda n: n[0])
```

---

## Evaluation

| Metric | Target |
|--------|--------|
| F1 (transcription, e-GMD test) | ≥0.85 |
| SDR (separation, MUSDB18-HQ drums) | within 1 dB of original Demucs |
| Inference latency (3 min file, GPU) | <5s |
| Failure correlation (does sep error → trans error?) | <0.5 (so they're not fully coupled) |

---

## Estimated effort

| Phase | Time | Compute |
|-------|------|---------|
| 1: Demucs baseline | 0.5 day | GPU |
| 2: Inspect internals | 1 day | CPU |
| 3: Add heads | 1 day | GPU |
| 4: Joint loss | 0.5 day | GPU |
| 5: Train (warm-up + 30 epochs) | 5-10 days | GPU continuous |
| 6: Inference + eval | 1 day | GPU |
| **Total** | **10-15 days** | GPU-heavy (~$100-300 rented) |

---

## Escalation paths

- **If separation degrades during joint training**: lambda_sep too low.
  Increase to 5x.
- **If transcription stays bad while separation improves**: transcription
  heads need more capacity. Replace `Linear(512, N)` with a small MLP or
  per-stem mini-CRNN.
- **If joint training is unstable**: try gradient surgery
  (PCGrad/CAGrad) to handle conflicting gradients between the two tasks.
- **If you can't get GPU for this scale**: this approach is not feasible
  on CPU. Switch to approach 05 + 06.

---

## Why this is listed lower than 05/06

This approach has the highest *ceiling* but also the highest *risk*:
- Larger model = longer training cycles
- More moving parts = more places to debug
- Less established literature for drum-specific joint models

Approach 05 reuses the working separator AS-IS and trains tiny per-stem
heads. Approach 13 retrains the entire separator, accepting more risk
for a chance at better end-to-end quality.

**Recommendation**: only attempt this *after* approach 05 is working. Use
05 as the baseline to beat.
