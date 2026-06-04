# Approach 14: Diffusion-Based MIDI Generation

> Frame drum transcription as conditional generation: given audio, the
> model generates a MIDI heatmap via score-based diffusion. Novel
> research direction; lowest-confidence recommendation but potentially
> highest-quality outputs for ambiguous cases.
>
> Schema follows `00-overview.md`.

---

## Premise

The previous attempts treat transcription as deterministic regression:
given audio, produce MIDI. But drum transcription is *inherently
ambiguous* for some inputs:
- Quiet ghost notes may or may not be intended hits
- Closed vs open hi-hat is a continuous spectrum
- Reverb tails can be misread as separate hits

A diffusion model **generates a distribution of plausible MIDIs**
conditioned on the audio. Sample once → get one MIDI. Sample 100× → get
100 MIDIs that all agree on the obvious hits and disagree on the
ambiguous ones. The variance becomes a *confidence signal*.

This is recent (2024) research: see "CLAP-Diffusion-Drums" or
"AudioLDM-conditioned drum generation" lineages. The approach is not
established for transcription specifically, but the formulation is
straightforward.

---

## Architecture

```
[Audio: mel-spec, [B, T, 128]]
                │
                ▼
[Audio Encoder: pretrained AST or MERT, frozen]
                │
                ▼
[Audio embeddings: [B, T_enc, 768]]
                │
                ▼
[Diffusion U-Net, conditioned on audio embeddings via cross-attention]
                │
                ▼
[Denoised MIDI heatmap: [B, T, 20] (10 onset + 10 velocity)]
                │
                ▼
[Standard peak detection → MIDI file]
```

Training: noise the MIDI heatmap, train the U-Net to predict the noise
conditioned on audio. Standard DDPM (Ho 2020) loss.

Inference: start from pure noise, denoise iteratively (50-1000 steps),
conditioned on audio.

---

## Why this might work

1. **Calibrated uncertainty.** Sampling 10 times gives you a confidence
   per hit (10/10 = confident, 5/10 = uncertain).
2. **Distribution-matching, not point-prediction.** If two MIDIs are
   equally plausible interpretations of the audio, the model can
   generate either. The previous approaches were forced to pick one.
3. **Strong inductive bias from generative pretraining.** Diffusion
   models for MIDI exist; can warm-start from a generation checkpoint.
4. **Recent SOTA in many audio tasks.** Outperforms regression-based
   approaches for ambiguous outputs.

---

## What could go wrong

1. **Diffusion is slow.** 1000-step DDPM at inference is ~30 sec per
   3min file on a GPU. Mitigation: use DDIM with 20-50 steps; ~5x
   faster.
2. **The MIDI heatmap is a fundamentally different output domain
   than typical diffusion (images).** May need architecture
   customization.
3. **Training stability.** Diffusion training has known instabilities
   (variance schedule tuning, EMA weights, etc.). Standard tricks exist
   but add complexity.
4. **Low expected payoff over approach 05/06 for non-ambiguous cases.**
   For clean drum stems with clear hits, a regression approach is
   simpler and probably equally good.
5. **Novel — limited reference implementations.** This is research
   territory.

---

## Prerequisites

- A *working* baseline from approach 05/06 (to compare against).
- GPU with ≥24 GB VRAM (training diffusion is memory-hungry).
- Pretrained audio encoder from approach 10.
- Patience — this is the longest-tail approach.

---

## Implementation steps

### Phase 1: Build a non-diffusion baseline

Don't skip this. Approach 05/06 working at F1 ≥ 0.75 is the prerequisite
for measuring any diffusion improvement.

### Phase 2: Build the conditional diffusion U-Net

```python
import torch
from diffusers import UNet1DModel, DDPMScheduler

# Approximate: treat MIDI heatmap as 20-channel 1D signal over time
class CondDiffusionTranscriber(nn.Module):
    def __init__(self, audio_encoder, time_steps=1000):
        super().__init__()
        self.audio_encoder = audio_encoder  # frozen pretrained
        self.unet = UNet1DModel(
            in_channels=20,
            out_channels=20,
            block_out_channels=(64, 128, 256, 512),
            cross_attention_dim=768,  # match audio encoder
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=time_steps)

    def forward(self, midi_heatmap, audio):
        with torch.no_grad():
            audio_emb = self.audio_encoder(audio)
        noise = torch.randn_like(midi_heatmap)
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (audio.shape[0],))
        noisy = self.scheduler.add_noise(midi_heatmap, noise, t)
        predicted_noise = self.unet(noisy, t, encoder_hidden_states=audio_emb).sample
        return torch.nn.functional.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, audio, num_inference_steps=50):
        audio_emb = self.audio_encoder(audio)
        midi = torch.randn(audio.shape[0], 20, T)
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.scheduler.timesteps:
            predicted_noise = self.unet(midi, t, encoder_hidden_states=audio_emb).sample
            midi = self.scheduler.step(predicted_noise, t, midi).prev_sample
        return midi
```

### Phase 3: Train

```bash
python train_diffusion.py --manifest batch1_shuffled.txt --epochs 100 --device cuda
```

Long training: 5-10 days on A100.

### Phase 4: Inference with sampling

```python
samples = []
for _ in range(10):
    midi_heatmap = model.sample(audio, num_inference_steps=50)
    notes = heatmap_to_notes(midi_heatmap)
    samples.append(notes)
# Aggregate: notes appearing in ≥ 7 of 10 samples are "confident"
confident_notes = aggregate_samples(samples, threshold=7)
```

### Phase 5: Evaluate with uncertainty

```python
# Standard F1 on confident-only notes
f1_confident = mir_eval_f1(confident_notes, gt)
# Hybrid: confident + low-confidence below a threshold
f1_all = mir_eval_f1(all_notes_at_threshold_5, gt)
# Plot: F1 vs confidence threshold (should rise then plateau)
```

---

## Evaluation

| Metric | Target |
|--------|--------|
| F1 confident-only | ≥0.90 (high confidence threshold; less recall, more precision) |
| F1 all samples | ≥0.85 (matching baseline) |
| Calibration: variance correlates with errors | Spearman ρ > 0.5 |
| Inference latency (single sample) | <30s on GPU |

---

## Estimated effort

| Phase | Time | Compute |
|-------|------|---------|
| 1: Baseline | (already done) | — |
| 2: U-Net + scheduler | 3-5 days | GPU |
| 3: Train | 5-10 days | GPU continuous (~$100-300) |
| 4: Inference + sampling | 2-3 days | GPU |
| 5: Evaluate calibration | 1-2 days | CPU |
| **Total** | **15-25 days** | GPU-heavy |

---

## Escalation paths

- **If diffusion doesn't beat the baseline F1**: the deterministic nature
  of the task may make the generative formulation unnecessary. Document
  the negative result; pick a different approach.
- **If diffusion provides good uncertainty but mediocre F1**: use it as
  a *post-hoc* confidence estimator on top of an existing baseline,
  not as the primary transcriber.
- **If training is unstable**: try v-prediction parameterization
  (vs ε-prediction), classifier-free guidance, EMA weights.
- **If inference is too slow**: distill the diffusion model into a
  one-step student (consistency distillation, Song et al. 2023).

---

## Why this is the lowest-confidence recommendation

- The problem (drum transcription) is *mostly* deterministic; the
  ambiguity argument is real but not dominant.
- The compute budget is large relative to the expected quality gain.
- Reference implementations for this exact formulation are sparse.
- The simpler approaches (05, 06, 08, 09, 10) should be tried first.

**When to seriously consider this**: only after approach 05 + 06 + 10
have all been tried and the user wants either (a) better handling of
ambiguous inputs (live recording with ghost notes), or (b) a confidence
signal for downstream UI ("the model is unsure about these hits — review
them").

This document is here for completeness and to seed thought, not to be a
near-term action item.
