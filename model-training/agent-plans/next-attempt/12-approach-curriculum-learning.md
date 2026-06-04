# Approach 12: Curriculum Learning (Synthetic → Easy → Hard)

> Train the model on a difficulty ladder instead of dumping e-GMD on it
> all at once. Start with synthetic drum hits (zero label noise, simplest
> possible audio), then quantized loops, then human-played e-GMD, then
> real-world dirty stems. Companion approach — best applied **with** one
> of 05/06/08/09/10, not instead of.
>
> Schema follows `00-overview.md`.

---

## Premise

Neural networks trained on the hardest examples from epoch 1 often fail
to converge — the gradient signal is too noisy. Curriculum learning
(Bengio 2009) gives a smooth learning trajectory: start with what's
trivially learnable, progressively introduce harder examples.

For drum transcription, the difficulty gradient is clear:

1. **Trivial**: synthetic single hits (one drum at a time, perfect alignment, no noise)
2. **Easy**: synthetic loops (multiple drums, perfect alignment)
3. **Medium**: clean isolated drum performances (e-GMD multitracks, near-perfect MIDI alignment)
4. **Hard**: real-world drum stems with bleed, reverb, compression
5. **Hardest**: drum stems with significant musical context bleeding through

The previous attempt trained directly on level 3-4. By the curriculum
view, this is jumping into "Calculus 101" without taking arithmetic
first.

---

## Architecture

Curriculum learning is **not** an architecture — it's a training schedule.
Pair it with any model from approaches 05, 08, 09, 10, or 11.

```
                    ┌──────────────────────────────┐
                    │  Existing model architecture │
                    │  (any of approaches 05-11)   │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                Train epochs 1-5 on Level 1 (synthetic singles)
                                   │
                                   ▼
                Train epochs 6-15 on Level 2 (synthetic loops)
                                   │
                                   ▼
                Train epochs 16-30 on Level 3 (e-GMD clean)
                                   │
                                   ▼
                Train epochs 31-50 on Level 4 (real-world stems)
                                   │
                                   ▼
                Optionally cycle back through Levels 1-3 (review)
```

---

## Why this should work

1. **Smooth learning trajectory.** Each level builds confidence; the
   model never sees "from scratch" hard examples until it has a
   reasonable inductive bias.
2. **Each level is debuggable.** If Level 1 (synthetic singles) doesn't
   reach F1=0.99, the bug is in the pipeline (this is exactly the
   `03-test-prove-overfit-first.md` test as a continuous gate during
   training).
3. **Catastrophic forgetting is mitigated.** Periodic review of earlier
   levels (Level 1 every 10 epochs) keeps the model from over-specializing
   to the latest difficulty.
4. **Free data augmentation.** Synthetic data is generated on-the-fly;
   infinite training examples for the early curriculum stages.
5. **The previous attempt's smoke test data already qualifies as Level 3.**
   This curriculum is a generalization of "overfit first, then scale up."

---

## What could go wrong

1. **Synthetic data is too easy and the model never sees real noise.**
   Mitigation: don't drop synthetic completely; mix in a small fraction
   even at later curriculum levels.
2. **Domain shift between curriculum levels.** If Level 1 synthetic
   audio sounds nothing like Level 4 real audio, the model has to relearn
   features at each transition. Mitigation: use a good synthesizer
   (high-quality drum soundfont, realistic mixing) so synthetic audio
   sounds like real audio.
3. **Tuning the schedule.** When to advance from one level to the next?
   Mitigation: advance when validation F1 on the current level exceeds a
   threshold (e.g., 0.90).
4. **Compute overhead.** 5 curriculum levels × 10 epochs each = 50 epochs
   total, comparable to no-curriculum training. Mitigation: this isn't
   an overhead — it's the same total cost.

---

## Prerequisites

- A base model from any of approaches 05, 08, 09, 10, or 11.
- The synthetic data generator from `02-tooling-wishlist.md` Tool 2.
- e-GMD dataset.
- Optional: an "in-the-wild" drum corpus for Level 4 (see approach 07).
- Compute: same as the base approach.

---

## Implementation steps

### Phase 1: Build the data ladder

#### Level 1: Synthetic single hits

```python
# tools/gen_curriculum_level1.py
"""
Generate 1000 single-hit examples per drum class.
Each example: 2-second WAV with one drum hit at a random time,
random velocity, random sample bank.
"""
for drum_class in DRUM_CLASSES:
    for i in range(1000):
        wav, mid = synth_single_hit(
            drum=drum_class,
            time_in_window=random_seconds(0.3, 1.7),
            velocity=random_int(20, 127),
            soundfont=random.choice(SOUNDFONTS),
        )
        save(wav, mid, f"curriculum/level1/{drum_class}_{i:04d}")
```

#### Level 2: Synthetic loops

Use the same generator from Tool 2 with `pattern={rock, jazz, latin, hiphop}`
and `bars={1,2,4,8}`. Multiple drum classes per file, but still perfectly
aligned MIDI.

#### Level 3: e-GMD

Reuse existing manifests (`batch1_shuffled.txt`, etc.).

#### Level 4: Real-world stems

Either:
- Generate from existing user files (run separator on user's project archive)
- Use approach 07's curated unlabeled corpus
- Use the Mirex drum stem subset, BabySlakh, MUSDB18-HQ drums

### Phase 2: Curriculum scheduler

```python
class CurriculumScheduler:
    def __init__(self, levels):
        self.levels = levels  # list of (name, dataset, target_f1)
        self.current_level = 0

    def get_dataset(self):
        # Mostly current level, small fraction of earlier levels (review)
        if self.current_level == 0:
            return self.levels[0][1]
        # 70% current + 30% mixed earlier
        return WeightedMixture([
            (self.levels[self.current_level][1], 0.7),
            *[(lv[1], 0.3 / self.current_level) for lv in self.levels[:self.current_level]]
        ])

    def should_advance(self, val_f1):
        target = self.levels[self.current_level][2]
        if val_f1 >= target and self.current_level < len(self.levels) - 1:
            self.current_level += 1
            return True
        return False
```

### Phase 3: Modified training loop

```python
scheduler = CurriculumScheduler([
    ('level1', synth_singles_dataset, target_f1=0.98),
    ('level2', synth_loops_dataset, target_f1=0.95),
    ('level3', e_gmd_dataset, target_f1=0.85),
    ('level4', real_world_dataset, target_f1=0.75),
])

for epoch in range(max_epochs):
    dataset = scheduler.get_dataset()
    train_one_epoch(model, dataset)
    val_f1 = evaluate(model, val_set_for_current_level)
    print(f"Epoch {epoch}: level={scheduler.current_level} val_f1={val_f1:.3f}")
    if scheduler.should_advance(val_f1):
        print(f"ADVANCED to level {scheduler.current_level}")
```

### Phase 4: Train + evaluate

Train per the curriculum. Final evaluation on standard e-GMD test split.

---

## Evaluation

| Metric | Target |
|--------|--------|
| Level 1 (synthetic singles) | F1 ≥ 0.99 — must be near-perfect |
| Level 2 (synthetic loops) | F1 ≥ 0.95 |
| Level 3 (e-GMD) | F1 ≥ 0.85 |
| Level 4 (real-world) | F1 ≥ 0.75 |
| Final test F1 (e-GMD) | ≥0.85 (matching base approach) |
| Out-of-domain F1 | should beat no-curriculum baseline by 0.05-0.10 |

If Level 1 doesn't reach F1=0.99, **stop and debug** — this is the
`03-test-prove-overfit-first.md` failure mode, surfaced early in training.

---

## Estimated effort

| Phase | Time | Compute |
|-------|------|---------|
| 1: Build data ladder | 2-3 days | CPU |
| 2: Curriculum scheduler | 0.5 day | CPU |
| 3: Modified training loop | 0.5 day | CPU |
| 4: Train + eval | depends on base approach | base approach's compute |
| **Total overhead vs base approach** | **+3-4 days** | minimal extra compute |

---

## Escalation paths

- **If Level 1 fails**: the bug is in the pipeline (matches T1 or T3 in
  `01-critique-and-theories.md`). Run the bug-isolation grid (approach 04).
- **If transitions cause forgetting**: increase the review fraction
  (mix more earlier-level data into later epochs).
- **If Level 4 plateaus low**: domain shift dominates. Combine with
  approach 07 (distill from classical) for more in-domain training signal.
- **If overall F1 doesn't improve over no-curriculum baseline**: the
  curriculum was unnecessary for this problem; revert and use the
  simpler training.

---

## Note: pairing curriculum with approaches 05-11

Curriculum is most valuable when the model is starting from scratch with
no pretrained features (approaches 05, 08, 09). Less valuable for
approach 10 (pretrained encoder) where the encoder already has rich
features. Not applicable to approach 14 (diffusion) which has its own
training dynamics.

**Best pairing**: approach 05 (per-stem CRNN) + curriculum. The per-stem
problem has natural difficulty levels (clean single-class synthetic →
e-GMD mixed-class → real-world dirty) and the curriculum amplifies the
data-efficiency benefit of the per-stem decomposition.
