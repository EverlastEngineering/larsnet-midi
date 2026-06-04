# Approach 7: Distill from the Classical Pipeline

> Use the existing `stems_to_midi/` DSP pipeline as a *teacher* to label
> a much larger unlabeled corpus, then train a neural student on that
> weakly-supervised data. Trades label quality for label *quantity*.
>
> Schema follows `00-overview.md`.

---

## Premise

The previous attempt was bottlenecked by e-GMD's 444 hours of
hand-labeled audio. The classical pipeline in `stems_to_midi/` works
"well enough" on arbitrary drum recordings. Therefore:

1. Run the classical pipeline on a much larger unlabeled drum corpus
   (YouTube drum covers, free drum sample packs, the user's own work
   archive, etc.).
2. Treat the classical pipeline's MIDI output as **silver labels**
   (noisy but ~80% correct).
3. Train a neural student to imitate the teacher.
4. Optionally fine-tune the student on e-GMD's gold labels.

Self-distillation / weak supervision is well-established
(e.g., Snorkel, Self-Training, Noisy Student). It works when:
- Teacher recall is high (don't miss many hits)
- Teacher precision is OK (some false positives are fine)
- Student capacity ≥ teacher capacity (else the student just memorizes
  teacher noise)

---

## Architecture

```
[Large unlabeled drum corpus, ~1000h]
                │
                ▼ stem separation (existing)
[Per-stem audio]
                │
                ▼ classical pipeline (existing, in stems_to_midi/)
[Silver MIDI labels: ~80% accurate]
                │
                ▼ + e-GMD gold labels (~400h gold)
                │
                ▼
[Train any of the architectures from 05/08/09/10 on combined data]
                │
                ▼ optional 2nd stage: fine-tune on gold-only
[Neural transcriber, better than the teacher]
```

The student must out-generalize the teacher. This works because:
- Neural model averages over teacher errors (label noise is denoising
  during gradient descent if the noise is unbiased).
- Student sees more *kinds* of audio than e-GMD provides (different
  recording conditions, drum kits, musical styles).
- Fine-tuning on e-GMD gold labels at the end recovers precision.

---

## Why this should work

1. **Scale matters more than label quality**. Self-supervised audio
   models (wav2vec, HuBERT) achieve SOTA from unlabeled or weakly
   labeled corpora 10-100× larger than fully supervised baselines.
2. **The teacher's errors are non-random and bounded**. The classical
   pipeline misses things consistently (e.g., quiet ghost notes); the
   student sees the misses across many examples and can learn to fire
   on the same spectral patterns the teacher does NOT fire on.
3. **e-GMD is unrealistically clean**. Real-world drum stems have
   bleed, reverb, compression, microphone resonances. Training only on
   e-GMD overfits to studio conditions. Silver labels from YouTube
   covers expose the model to real-world conditions.
4. **The teacher already does post-processing**. The DSP pipeline has
   ground-tested handling of reverb continuations, sustain filtering,
   pitch classification — features hard to learn from raw labels.

---

## What could go wrong

1. **Teacher precision too low** → student overfits to teacher errors.
   Mitigation: filter silver labels by teacher confidence; only use
   "obvious" hits.
2. **Domain mismatch** → corpus is too different from inference target.
   Mitigation: use multiple corpora; include test-set-like material.
3. **No big unlabeled corpus available** → would need to collect/curate.
   Mitigation: HuggingFace has drum-related datasets; the user's own
   working folder is probably 50+ hours.
4. **Stem separator fails on the unlabeled audio** → garbage in, garbage
   out. Mitigation: pre-filter audio to drum-isolated material (track
   stems if available).

---

## Prerequisites

- Approach 05 or 06 working as a baseline (so you have a known-good
  student architecture).
- Storage: 100-500 GB for an unlabeled corpus.
- Compute: GPU strongly recommended for the larger-corpus training.
- A curated source of unlabeled drum audio.

### Candidate unlabeled corpora

| Source | Size | Quality | Cost |
|--------|------|---------|------|
| MUSDB18-HQ drums | 100 tracks | High (studio multitracks) | Free |
| BabySlakh drums | 800 tracks | Synthetic but realistic | Free |
| FMA drum-tagged | ~500h estimated | Variable | Free |
| YouTube drum covers | unlimited | Highly variable | Free + scraping work |
| User's own project archive | ~50h+? | High | Free |
| Free drum sample packs (Splice, Loopcloud, Cymatics free packs) | ~50h | High | Free |

---

## Implementation steps

### Phase 1: Curate the unlabeled corpus (1-3 days)

1. Download MUSDB18-HQ and BabySlakh drums.
2. Optionally scrape YouTube drum-cover playlists with `yt-dlp` (legal:
   for research use; respect ToS).
3. Run stem separation on everything. Save kick/snare/hihat/toms/cymbals
   stems.

### Phase 2: Run the teacher (1 day with parallelization)

```bash
for stem in unlabeled_stems/*.wav; do
    python -m stems_to_midi.transcribe --input $stem --output ${stem%.wav}.silver.mid
done
```

Filter silver labels:
- Drop entire files where teacher detected <5 hits (likely a non-drum file)
- Drop individual events with `strength < threshold` (configurable)

### Phase 3: Build training manifest

Combine:
- Gold: e-GMD with hand-labeled MIDI
- Silver: corpus with classical pipeline labels

Add a sample-weight field (gold examples weighted higher than silver).

### Phase 4: Train the student

Use the architecture from approach 05 (per-stem transcribers).
Modifications:
- Loss includes per-sample weight (gold 1.0, silver 0.3)
- Train for more epochs (silver corpus is bigger; less risk of overfitting)
- Validate ONLY on gold-labeled held-out e-GMD (silver isn't trustworthy)

### Phase 5: Fine-tune on gold-only (optional, ~10% of base training time)

Continue training with silver labels removed. This sharpens the model on
the high-quality labels.

### Phase 6: Evaluate

Same as approach 05/06: held-out e-GMD test split with mir_eval.

Expected improvement over approach 05 alone: +0.05 to +0.10 F1 if the
unlabeled corpus is sizeable and diverse.

---

## Evaluation

| Metric | Target |
|--------|--------|
| F1 on e-GMD test (vs approach 05 baseline) | +0.05 to +0.10 |
| F1 on out-of-domain test (e.g., a few of user's own files) | +0.10 to +0.20 |
| Velocity correlation | similar to baseline |

---

## Estimated effort

| Phase | Time | Compute |
|-------|------|---------|
| 1: Curate corpus | 1-3 days | mostly download |
| 2: Run teacher | 1 day | CPU, parallelizable |
| 3: Build manifest | 0.5 day | CPU |
| 4: Train student | 1-3 days | GPU recommended |
| 5: Fine-tune (optional) | 0.5 day | GPU |
| 6: Evaluate | 0.5 day | CPU |
| **Total** | **5-10 days** | mixed |

---

## Escalation paths

- **If silver labels are too noisy**: tighten teacher thresholds, or use
  the *intersection* of teacher and a second weak labeler (e.g., a
  simple energy-based onset detector).
- **If the student doesn't beat the teacher**: the student architecture
  is the ceiling. Switch to approach 08/09/10 student.
- **If domain shift dominates**: use domain-adaptation techniques
  (gradient reversal, MMD loss) to align silver and gold distributions.
