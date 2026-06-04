# Approach 10: Pretrained Audio Encoder + Small Classification Head

> Stop training the feature extractor from scratch. Use a frozen
> pretrained audio model (AST, wav2vec 2.0, CLAP, MERT) to produce rich
> embeddings, and bolt a tiny supervised head on top. Most successful
> 2023-2025 audio-task results follow this pattern.
>
> Schema follows `00-overview.md`.

---

## Premise

The previous attempt's Conv2d(32→64) → BiGRU(128) is a 2018-era
architecture that learns acoustic features from zero. In 2024–2025,
this is wasteful. Pretrained audio encoders trained on **millions of hours
of unlabeled audio** already capture all the low-level features (transients,
harmonic content, spectral envelopes) the task needs. Fine-tuning a small
head on top of frozen features is:
- Faster (fewer parameters to train)
- More data-efficient (the encoder already knows what audio looks like)
- More robust (the encoder has seen 10000× more audio than our 444h)

Candidate pretrained encoders (all available on HuggingFace):

| Model | Pretraining data | Params | Strengths | Weaknesses |
|-------|------------------|--------|-----------|------------|
| **AST** (Gong 2021) | AudioSet, 5000h | 86M | Mel-spec input, easy to use | General audio, not music-specific |
| **wav2vec 2.0** (Baevski 2020) | LibriSpeech 960h | 95M | Raw waveform, learned tokenization | Speech-biased |
| **HuBERT** (Hsu 2021) | LibriSpeech 60k h | 95M-300M | Better than wav2vec on transients | Speech-biased |
| **CLAP** (Wu 2023) | 633k audio-caption pairs | 90M | Audio + text understanding | Less drum-specific |
| **MERT** (Li 2023) | 160k h of music | 95M-330M | Music-specific pretraining | Less drum-specific than ideal |
| **MusicGen-Decoder** (Copet 2023) | 20k h music | 1.5B-3.3B | Music-pretrained, large | Heavy inference |

**My pick for this approach**: try AST first (most mature, easiest API),
fall back to MERT if AST's general-audio bias hurts drum specificity.

---

## Architecture

```
[Input: raw waveform or mel-spectrogram]
                │
                ▼
[Pretrained encoder, FROZEN]
  - AST: mel-spec [B, 128, 1024] → embeddings [B, 768]
  - MERT: waveform → embeddings per timestep [B, T, 768]
                │
                ▼
[Tiny trainable head]
  - For per-frame output: BiGRU(768 → 128) + Linear(256 → 10)
  - For per-event output: AvgPool → Linear(768 → 10) + Linear(768 → 1) (velocity)
                │
                ▼
[Per-frame drum probabilities or per-event class+velocity]
```

Two head variants:

### Head A: per-frame (replaces existing CRNN output)

```python
class FrozenEncoderHead(nn.Module):
    def __init__(self, encoder_output_dim=768):
        super().__init__()
        self.rnn = nn.GRU(encoder_output_dim, 128, batch_first=True, bidirectional=True)
        self.fc_onset = nn.Linear(256, 10)
        self.fc_velocity = nn.Linear(256, 10)

    def forward(self, encoded):  # [B, T_enc, 768]
        x, _ = self.rnn(encoded)
        return {
            'onset': self.fc_onset(x),         # [B, T_enc, 10] logits
            'velocity': self.fc_velocity(x),    # [B, T_enc, 10]
        }
```

Trainable params: ~500k. Encoder: 86M (frozen).

### Head B: per-event (paired with approach 06's DSP detector)

```python
class PerEventEncoderHead(nn.Module):
    def __init__(self, encoder_output_dim=768, num_classes=3):
        super().__init__()
        self.fc_class = nn.Linear(encoder_output_dim, num_classes)
        self.fc_velocity = nn.Linear(encoder_output_dim, 1)

    def forward(self, encoded):  # [B, encoder_output_dim] (avg-pooled)
        return {
            'class_logits': self.fc_class(encoded),
            'velocity': torch.sigmoid(self.fc_velocity(encoded)),
        }
```

Trainable params: ~10k per stem. Encoder: 86M (frozen, shared across all
stems).

---

## Why this should work

1. **Feature learning is solved.** AST/MERT have seen orders of magnitude
   more audio than we have. Their features beat any from-scratch
   training on data-constrained tasks.
2. **Tiny trainable head = fast to overfit-test.** A 10k-parameter head
   trains in <10 minutes on CPU. Iteration cycles are fast.
3. **Frozen encoder = no risk of breaking pretrained features.** The
   head adapts to drums; the encoder stays unchanged.
4. **The user is allowed GPUs now.** Even a single A100 hour ($1.10)
   is enough to encode the entire e-GMD dataset; thereafter all training
   is CPU-cheap on the cached embeddings.
5. **Inference compute trade-off is favorable**: encoder takes ~5s per
   3-minute file on CPU; trivial on GPU. Acceptable for non-realtime
   transcription.
6. **Robust to recording variations.** Pretrained encoders have seen
   diverse recording conditions; the model generalizes better to
   real-world drums than a from-scratch CRNN trained only on e-GMD.

---

## What could go wrong

1. **AST is general-audio (animals, machinery, music). Drum-specific
   features may be diluted.** Mitigation: try MERT (music-pretrained) as
   alternative.
2. **Per-frame output requires the encoder to return per-frame embeddings**,
   which AST does not natively support (it's a classification model).
   Mitigation: extract intermediate layer activations
   (`output_hidden_states=True` in HuggingFace), use those.
3. **Encoder input format constraints.** AST expects 10-second clips at
   16kHz. Our drum loops are arbitrary length at 44.1kHz. Mitigation:
   resample + window appropriately; documented in HuggingFace example.
4. **The "tiny head" may be too tiny.** Mitigation: increase to 1-2M
   parameter head if the small one plateaus.
5. **Caching embeddings consumes disk.** AST embedding for a 3min file is
   ~5 MB. e-GMD = 444h × 60min × 5MB / 3 ≈ 45 GB. Manageable.

---

## Prerequisites

- HuggingFace `transformers` library: `pip install transformers torchaudio`.
- GPU recommended for the encoder pass over e-GMD (could take days on CPU).
- ~50 GB scratch disk for cached embeddings.
- `03-test-prove-overfit-first.md` test harness adapted for the new head.

---

## Implementation steps

### Phase 1: Pick an encoder and verify it works

```python
# Quick sanity check
from transformers import ASTFeatureExtractor, ASTModel
import torchaudio

extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

waveform, sr = torchaudio.load("dl-1.wav")
mono = waveform.mean(dim=0, keepdim=True)
resampled = torchaudio.functional.resample(mono, sr, 16000)

inputs = extractor(resampled.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
print(f"Last hidden state shape: {outputs.last_hidden_state.shape}")
# Should be [1, T, 768] where T depends on input length
```

### Phase 2: Cache embeddings for the dataset

```bash
conda run -n drumtomidi python tools/cache_ast_embeddings.py \
    --manifest batch1_shuffled.txt --output_dir embeddings/ast/
```

Estimated time: 10-30h CPU, or 1-2h on GPU.

### Phase 3: Train head

Modify the training loop from approach 05 to load `.pt` embedding files
instead of `.wav`, and use the small head architecture.

```python
class CachedEmbeddingDataset(Dataset):
    def __init__(self, manifest, embedding_dir):
        self.entries = parse_manifest(manifest)
        self.embedding_dir = Path(embedding_dir)

    def __getitem__(self, idx):
        audio_path, midi_path = self.entries[idx]
        emb_path = self.embedding_dir / f"{audio_path.stem}.pt"
        emb = torch.load(emb_path)  # [T_enc, 768]
        notes, _ = load_midi_notes(midi_path)
        targets = build_targets(notes, emb.shape[0])
        return emb, targets
```

Train for ~20-50 epochs. With a 500k-parameter head, this is <30min
on CPU.

### Phase 4: Evaluate

Same harness as approaches 05-09.

---

## Evaluation

| Metric | Target |
|--------|--------|
| Overfit smoke test | PASS |
| F1 on e-GMD test | ≥0.85 |
| Per-class recall on rare classes | ≥0.75 |
| Robustness on out-of-domain user files | should beat approaches 05/06 by 0.05-0.10 |

---

## Estimated effort

| Phase | Time | Compute |
|-------|------|---------|
| 1: Pick + verify encoder | 0.5 day | CPU |
| 2: Cache embeddings | 1-2 days | GPU recommended |
| 3: Train head | 0.5-1 day | CPU |
| 4: Eval | 0.5 day | CPU |
| **Total** | **2.5-4 days** | GPU for caching |

---

## Variants to try

If AST doesn't work well, try in this order:
1. **MERT-95M** (music-pretrained) — best fit for drums
2. **HuBERT-Large** (general but very strong) — fallback
3. **CLAP** — useful if you want zero-shot capability
4. **EnCodec encoder** (Meta's neural audio codec) — extremely compressed
   embeddings, novel approach

For per-event (paired with approach 06):
- Use `transformers.pipeline("audio-classification")` for quick baselines
  before custom code.

---

## Escalation paths

- **If AST embeddings don't beat from-scratch baseline**: drum transients
  are too narrow-band; AST general-audio features dilute them. Switch to
  MERT or HuBERT.
- **If even pretrained encoders don't reach F1=0.85**: data quality
  (separator output, e-GMD label noise) is the bottleneck, not modeling.
  Look at approach 07 (distill from classical).
- **If you want the absolute SOTA**: graduate to approach 11 (MT3-style
  transformer trained from pretrained encoder).
