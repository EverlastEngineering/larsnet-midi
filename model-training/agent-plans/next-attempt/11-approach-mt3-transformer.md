# Approach 11: MT3-Style Token-Output Transformer

> Frame drum transcription as a sequence-to-sequence task: encoder
> reads audio, decoder emits a *token sequence* of MIDI events. This is
> the architecture behind Google's MT3 (Gardner et al., 2022), the
> current SOTA for multi-instrument music transcription. **Sidesteps the
> entire onset/threshold/peak-detection question** because the model
> outputs MIDI directly as text-like tokens.
>
> Schema follows `00-overview.md`.

---

## Premise

Every architecture in approaches 05-10 produces per-frame logits and
then post-processes them into MIDI events. That post-processing
(threshold, peak detect, snap to onset, write MIDI) is *exactly* where
the previous attempt's bugs lived (`01-critique-and-theories.md` T2).

MT3 takes a fundamentally different approach: the model is a transformer
encoder-decoder. The encoder reads a mel-spectrogram. The decoder emits a
token sequence like:

```
<program:drum> <time:120ms> <pitch:36> <vel:100>
<time:248ms> <pitch:42> <vel:60>
<time:495ms> <pitch:38> <vel:110>
<time:498ms> <pitch:42> <vel:50>
<end>
```

These tokens are then directly converted to MIDI. **There is no
threshold. There is no peak detection. There is no smear.** The model
either emits the right tokens or it doesn't, and standard
seq2seq loss handles the rest.

Reference impl: `https://github.com/magenta/mt3` (Apache 2.0). T5x
backbone (Google's variant of T5). ~30M parameters.

---

## Architecture

```
[Input: mel-spec, [B, T, 512]]
                │
                ▼
[Encoder: T5x (or any transformer encoder), 12-layer]
                │
                ▼
[Cross-attention encoder output]
                │
                ▼
[Decoder: T5x decoder, 12-layer, autoregressive]
                │
                ▼
[Token stream: <program> <time> <pitch> <velocity> ...]
                │
                ▼
[Token-to-MIDI converter]
                │
                ▼
[Output MIDI file]
```

Vocabulary:
- 256 time-shift tokens (10ms resolution × 256 = 2.56s max shift between events)
- 128 pitch tokens
- 128 velocity tokens
- Special: `<program:drum>`, `<note_on>`, `<note_off>`, `<eos>`

Total vocab size: ~600. Sequence lengths up to ~2000 tokens for a
3-minute drum loop.

---

## Why this should work

1. **Post-processing-free.** The model's output IS the MIDI. No
   threshold to tune, no peak detector to debug, no smear to balance.
2. **The hardest open problem in audio ML is exactly this formulation.**
   2022's SOTA MT3 paper, 2023's MR-MT3, and ongoing work all use
   token-output transformers. We're standing on a fertile shoulder.
3. **Pretrained T5-style backbones exist.** Can warm-start from a music
   pretrained checkpoint (e.g., MT3 checkpoint itself, or MusicGen
   decoder, or any T5x variant fine-tuned for music).
4. **Velocity is just another token.** Free; doesn't require a separate
   head.
5. **Generalizes beyond drums.** Same architecture transcribes piano,
   guitar, brass — useful if larsnet expands beyond drums later.

---

## What could go wrong

1. **Compute-heavy.** Training a transformer from scratch needs serious
   GPU. Mitigation: rent A100 spots; fine-tune from existing MT3
   checkpoint instead of training from scratch.
2. **Autoregressive decoding is slow at inference.** A 3-minute drum file
   may produce 2000 tokens; decoding one at a time on a transformer is
   ~5 seconds even on GPU. For non-realtime use, this is acceptable.
3. **Token vocabulary design choices.** Time resolution, velocity
   discretization — each is a hyperparameter that affects quality. MT3
   paper has good defaults; copy them.
4. **Data preparation is more involved.** Need to convert MIDI to token
   sequences with consistent ordering. MT3 repo has the script; adapt it.
5. **Beam search vs greedy decoding.** Greedy may collapse to silence;
   beam search adds latency. Mitigation: nucleus sampling at inference.

---

## Prerequisites

- GPU with ≥24 GB VRAM (A100, A6000, RTX 3090/4090).
- ~50 GB scratch disk for tokenized data + checkpoints.
- Python env: `pip install t5x jax flax orbax` (T5x is JAX-based)
  OR use a PyTorch port like `https://github.com/wsntxxn/MT3-pytorch`.
- e-GMD dataset.
- `03-test-prove-overfit-first.md` test harness.

---

## Implementation steps

### Phase 1: Tokenizer

```python
# model-training/tokenize_midi.py
def midi_to_tokens(midi_notes, time_resolution_ms=10, max_shift_ms=2560):
    tokens = [TOK_PROGRAM_DRUM]
    last_time = 0
    for note in sorted(midi_notes, key=lambda n: n.start_time):
        shift_ms = int((note.start_time - last_time) * 1000)
        # Encode shift as multiple shift tokens if > max
        while shift_ms > max_shift_ms:
            tokens.append(TOK_TIME_SHIFT(max_shift_ms))
            shift_ms -= max_shift_ms
        if shift_ms > 0:
            tokens.append(TOK_TIME_SHIFT(shift_ms))
        tokens.append(TOK_NOTE_ON)
        tokens.append(TOK_PITCH(note.pitch))
        tokens.append(TOK_VELOCITY(quantize_velocity(note.velocity)))
        last_time = note.start_time
    tokens.append(TOK_EOS)
    return tokens

def tokens_to_midi(token_stream, time_resolution_ms=10):
    # Inverse of midi_to_tokens
    ...
```

### Phase 2: Pick a backbone

- **Option A (JAX/T5x, official)**: clone `magenta/mt3`, follow their
  training script. Most faithful to paper.
- **Option B (PyTorch port)**: use a community port. Faster to iterate
  in PyTorch since rest of the codebase is PyTorch.
- **Option C (HuggingFace T5)**: simplest — use `T5ForConditionalGeneration`
  from transformers, adapt encoder to take spectrograms.

**Recommendation**: Option C for an MVP (1-2 days to get something
training), Option B for production (better speed/quality), Option A if
willing to learn JAX.

### Phase 3: Training data builder

For each (audio, MIDI) pair:
1. Compute mel-spec.
2. Convert MIDI to token sequence.
3. Cache as `(mel, tokens)` pair.

Chunk long files into 5-10s windows for memory.

### Phase 4: Train

```bash
# On rented A100
python train_mt3_drum.py \
    --train_manifest batch1_shuffled.txt \
    --val_manifest val1_shuffed.txt \
    --epochs 100 --batch_size 8 \
    --learning_rate 1e-4 --warmup_steps 2000 \
    --device cuda
```

Expected: 24-48 hours on A100, ~$25-50.

### Phase 5: Inference

```python
def transcribe(audio_path, model):
    mel = extract_mel(audio_path).to('cuda')
    output_ids = model.generate(
        inputs_embeds=mel,
        max_new_tokens=2000,
        num_beams=4,
        do_sample=False,
    )
    notes = tokens_to_midi(output_ids[0].tolist())
    return notes
```

### Phase 6: Evaluate

Same mir_eval harness as other approaches.

Target: F1 ≥ 0.90 (the paper reports 0.94 on MAESTRO piano; drums should
be in the same range or slightly lower).

---

## Evaluation

| Metric | Target | Notes |
|--------|--------|-------|
| F1 on e-GMD test | ≥0.90 | Highest target of any approach |
| Per-class recall (rare) | ≥0.80 | Token formulation handles imbalance well |
| Velocity correlation | ≥0.85 | Token velocity is discrete-bin, but high resolution |
| Inference latency | <30s for 3min file (GPU) | Acceptable for non-realtime |

---

## Estimated effort

| Phase | Time | Compute |
|-------|------|---------|
| 1: Tokenizer | 1 day | CPU |
| 2: Pick + adapt backbone | 2-3 days | CPU |
| 3: Data builder | 1 day | CPU |
| 4: Train | 1-3 days | GPU (~$25-100) |
| 5: Inference wrapper | 1 day | CPU |
| 6: Eval | 0.5 day | CPU |
| **Total** | **6-10 days** | GPU required |

---

## Escalation paths

- **If training diverges**: lower learning rate, increase warmup steps,
  reduce batch size.
- **If F1 below 0.7**: check tokenizer correctness — most MT3 bugs are
  in the tokenizer round-trip (encode→decode should be lossless on
  ground-truth MIDI).
- **If inference is too slow**: switch to greedy decoding or distill
  the transformer into a smaller model.
- **If you can't get GPU access**: skip this approach, use 05/06/09
  instead. MT3 is GPU-mandatory in practice.
