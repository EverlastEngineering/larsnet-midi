# Approach 6: Classical Onset Detection + Per-Event Classifier

> ★ **Strong 2nd recommendation.** This approach takes the
> "stop asking the neural net to do everything" principle even further
> than approach 05: don't even ask it to find onsets. Use the production
> DSP onset detector (which is calibrated and works); the neural net
> only classifies each detected event.
>
> Schema follows `00-overview.md`.

---

## Premise

Drum transcription has two sub-problems that the previous attempt
combined into one:

1. **WHEN is there a drum hit?** (onset detection — a regression problem)
2. **WHICH drum is it, and how hard?** (classification + regression)

For (1), the user already has `stems_to_midi/processing_shell.py` running
in production with calibrated thresholds, reverb continuation logic, and
per-instrument sensitivity tuning. It works.

For (2), the production code uses heuristics
(`stems_to_midi/note_classification_core.py`) that are good but not
great — they're spectral-feature decision trees, brittle to recording
conditions and drum kit variations.

**The neural net should replace (2) only.** This makes (2) a *per-event*
classification problem on a fixed-length audio crop — the easiest form
of audio classification, comparable to UrbanSound8K. A 100k-parameter
CNN handles it.

---

## Architecture

```
                  ┌─────────────────────────┐
                  │  Drum mix WAV (stereo)  │
                  └───────────┬─────────────┘
                              │
                              ▼
                ┌──────────────────────────────┐
                │ Existing stem separator      │
                │ (separate.py / MDX23C)       │
                └─────────────┬────────────────┘
                              │
                              ▼
        ┌──────────┬──────────┬──────────┬──────────┬──────────┐
        ▼          ▼          ▼          ▼          ▼
     kick      snare      hihat      toms       cymbals
        │          │          │          │          │
        ▼          ▼          ▼          ▼          ▼
   ┌─────────────────────────────────────────────────────┐
   │  Existing DSP onset detector                         │
   │  (stems_to_midi/processing_shell.py)                 │
   │  → list of (time, stem_label, intensity)             │
   └────────────────────────┬─────────────────────────────┘
                            │
                            ▼  for each onset
        ┌────────────────────────────────────────┐
        │ Crop 100ms of audio around onset       │
        │  → mel-spectrogram patch [1, 128, 9]   │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │ Per-stem classifier head               │
        │ (small CNN, ~100k params per stem)     │
        │ Output: (class_label, velocity)        │
        └────────────────┬───────────────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │  MIDI writer │
                  └──────────────┘
```

### Per-event classifier (per stem)

```python
class PerEventClassifier(nn.Module):
    """
    Classifies a single 100ms audio crop as one of N drum sub-types,
    and predicts MIDI velocity.

    Input:  [B, 1, 128, 9] (one mel-spec patch per event, ~100ms @ 11.6ms/frame)
    Output: dict with
              'class_logits': [B, N] for cross-entropy
              'velocity':     [B, 1] in [0, 1] for sigmoid+MSE
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),                      # 128→64 freq, 9→4 time
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),                      # 64→32 freq, 4→2 time
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),         # → [B, 128, 1, 1]
        )
        self.fc_class = nn.Linear(128, num_classes)
        self.fc_velocity = nn.Linear(128, 1)

    def forward(self, x):
        z = self.conv(x).flatten(1)  # [B, 128]
        return {
            'class_logits': self.fc_class(z),
            'velocity': torch.sigmoid(self.fc_velocity(z)).squeeze(-1),
        }
```

| Stem | Classes | Params per model |
|------|---------|------------------|
| kick | 1 (just velocity, no class) | ~30k |
| snare | 3 (snare, rimshot, clap) | ~100k |
| hihat | 2 (closed, open) | ~100k |
| toms | 3 (low, mid, high) | ~100k |
| cymbals | 3 (ride, crash, splash) | ~100k |
| **Total** | | **~430k** |

20× smaller than approach 05's per-stem CRNNs, because the classifier
only sees a tiny 100ms window of pre-cropped audio.

---

## Why this should work

### 1. Per-event classification is the easiest audio ML problem

UrbanSound8K (10 environmental sound classes, 4-second clips) gets ≥0.85
F1 with a 100k-parameter CNN. Our problem is *easier*: we have 100ms
clips of single-instrument drum hits in pre-separated stems. Standard
result.

### 2. Onset timing is already correct

The DSP onset detector has been tuned in production for years and has a
calibration UI for fine-tuning per-project. Its onset times match the
user's perception of when drums hit. The neural net inherits that
precision for free.

### 3. No more threshold/peak/snap-back surface area

Approach 05 still has a per-frame logit → sigmoid → threshold → peak →
snap pipeline. **Approach 6 has none of that.** The DSP detector gives
you a list of (time, stem) pairs directly. The neural net produces a
class label and a velocity number. There is no temporal post-processing.

### 4. Training data is tiny and clean

For each detected onset in e-GMD, extract a 100ms crop + look up the
ground-truth MIDI label. That gives you ~50k labeled examples per stem
(across e-GMD's 444 hours). That's *more than enough* for a 100k-param
classifier.

### 5. Per-event velocity is correlated with onset energy

The DSP onset detector already computes an "intensity" per event
(strength field in `stems_to_midi`). Use that as an INPUT feature to the
velocity regressor. The neural net only has to correct for spectral-
content-based velocity perception (a soft kick still has lots of energy;
a hard snare wire-hit can have less energy than a normal hit).

### 6. Inference is fast

Per-event classification on a 100ms crop is ~0.5ms on CPU. Even for a
3-minute song with ~2000 hits across all stems, total classifier time
< 1 second. The DSP detection is the dominant cost (already in production).

### 7. Failure is debuggable

If the cymbals classifier confuses ride and crash, you have ~10k labeled
examples to look at and a clear question: what spectral feature is being
ignored? This is the difference between a research project and a
debugging task.

---

## What could go wrong

### 1. The DSP detector misses real hits

Recall < 100% on the DSP side puts a ceiling on overall recall.
Mitigation:
- Measure DSP recall on e-GMD vs ground-truth MIDI. If it's >0.95, fine.
- If it's <0.90, lower DSP thresholds (accept more false positives) and
  let the classifier reject them with a "no-hit" class.
- Add a "rejector" output class: if classifier confidence is low for all
  drum-sub-types, treat the event as a false-positive and drop it.

### 2. The DSP detector triggers on bleed

If the kick stem has snare bleed and the DSP fires on the bleed, the
classifier sees a snare-sounding 100ms crop labeled as a kick by the
detector. Mitigation:
- Train the classifier with the SAME bleed conditions it will see at
  inference (use real separator output, not isolated ground-truth stems).
- Or: use the classifier output as a veto. If the kick classifier
  predicts "very low kick probability" on a kick-detector event, drop it.

### 3. 100ms might not be enough context

Some drums (cymbals, especially open hi-hat) have distinguishing
features in their sustain (tail), not their attack. 100ms covers the
attack. Mitigation:
- For cymbals specifically, use a longer window: 250ms or 500ms.
- Or: add a "context" feature — the DSP detector's `sustain_ms` field
  computed over a longer window.

### 4. Velocity regression on a 100ms crop is information-poor

The crop's amplitude correlates with velocity, but room reverb,
compression, and EQ all corrupt this. Mitigation:
- Use the DSP detector's pre-computed strength as an auxiliary input
  feature to the velocity regressor.
- Bonus: this means the velocity head can correct for the heuristic
  velocity estimate in `stems_to_midi/midi.py` rather than learn
  velocity from scratch.

---

## Prerequisites

- Working DSP onset detector (already exists: `stems_to_midi/`).
- Working stem separator (already exists: `separate.py`).
- `03-test-prove-overfit-first.md` test harness (adapted for per-event).
- e-GMD dataset.
- ~5 GB scratch disk per stem for cropped examples cache.
- Compute: CPU sufficient.

---

## Implementation steps

### Phase 1: Build a "labeled events" dataset

For each e-GMD multitrack:
1. Run DSP onset detector on each stem → list of (time, stem, intensity)
2. For each detected event, look up the ground-truth MIDI pitch within
   ±30ms.
3. If a match: label = (drum_class, midi_velocity).
4. If no match: label = "false_positive" (used to train the rejector).
5. Crop 100ms of audio centered on the event time, extract mel-spec.

Output: `events.npz` per stem containing
- `specs`: [N, 1, 128, 9]
- `class_labels`: [N]
- `velocity_labels`: [N]
- `dsp_intensity`: [N] (auxiliary feature)
- `is_false_positive`: [N] bool

Implementation: ~200 lines in `model-training/datasets/build_event_dataset.py`.

### Phase 2: Train per-stem classifiers

```python
# train_per_event.py
for stem in ['kick', 'snare', 'hihat', 'toms', 'cymbals']:
    data = np.load(f'events_{stem}.npz')
    train_loader, val_loader = make_splits(data)
    model = PerEventClassifier(num_classes=STEMS[stem]['num_classes'])
    train(model, train_loader, val_loader, epochs=30, lr=1e-3)
    torch.save(model.state_dict(), f'models/per_event_{stem}.pt')
```

Expected training time: 5-15 minutes per stem on CPU (the dataset is
tiny — just N × 128 × 9 floats).

### Phase 3: Wire into the inference pipeline

Modify `stems_to_midi/midi.py` (or write a new orchestrator):

```python
def transcribe_with_ml_classifier(stems_dict, midi_out_path):
    all_events = []
    for stem_name, stem_audio in stems_dict.items():
        # 1. DSP onset detection (existing)
        onsets = detect_onsets(stem_audio, stem_name)  # existing call

        # 2. Classifier per onset
        model = load_classifier(stem_name)
        for onset in onsets:
            crop = crop_audio_at(stem_audio, onset.time, window_ms=100)
            mel = audio_to_mel(crop)
            with torch.no_grad():
                out = model(mel.unsqueeze(0))
            class_idx = out['class_logits'].argmax().item()
            velocity = int(out['velocity'].item() * 127)
            midi_pitch = STEMS[stem_name]['class_to_pitch'][class_idx]
            all_events.append((onset.time, midi_pitch, velocity))

    write_midi(sorted(all_events), midi_out_path)
```

### Phase 4: Evaluate end-to-end

Same as approach 05 evaluation:
- Run on held-out e-GMD test split
- mir_eval F1 with ±50ms tolerance
- Per-class precision/recall breakdown

---

## Evaluation

| Metric | Target | Measurement |
|--------|--------|-------------|
| Per-stem classifier accuracy | ≥0.90 | held-out val set |
| Velocity regression R² | ≥0.7 | per-stem on val set |
| End-to-end MIDI F1 (±50ms) | ≥0.80 | mir_eval on e-GMD test |
| False-positive rejection rate | ≥0.85 | the rejector class accuracy |
| Inference latency for 3min file | ≤2s | wall clock |

---

## Estimated effort

| Phase | Time | Compute |
|-------|------|---------|
| Phase 1 (event dataset builder) | 1 day | CPU (10-20h DSP detection runtime) |
| Phase 2 (classifier training, 5 models) | 0.5 day | CPU |
| Phase 3 (inference orchestrator) | 0.5 day | CPU |
| Phase 4 (evaluation) | 0.5 day | CPU |
| **Total** | **2-4 days** | CPU sufficient |

---

## Comparison with approach 05

| Question | Approach 5 (stems-as-input CRNN) | Approach 6 (DSP+classifier) |
|----------|----------------------------------|-----------------------------|
| Does the NN do onset detection? | Yes (per-frame logits) | No (DSP does it) |
| Does the NN do classification? | Yes (per-frame class probs) | Yes (per-event class) |
| Does the NN do velocity? | Yes (per-frame velocity) | Yes (per-event velocity) |
| Threshold tuning required? | Yes (per-class threshold) | No |
| Model size per stem | ~2M params | ~100k params |
| Training data per stem | Continuous-time labels | Discrete events |
| Training time | 2-4h GPU per stem | <30min CPU per stem |
| Sensitivity to label timing | High (smear shape matters) | Low (DSP gives time) |
| Sensitivity to DSP quality | None | High |
| Failure mode | Misalignment, false negatives | DSP miss → permanent miss |
| Match to user's existing infra | Reuses separator | Reuses separator AND DSP detector |

**Why both 05 and 06 are recommended together**:
- 06 is faster to build and lower-risk; should reach F1 0.75 within
  3 days.
- 05 has higher ceiling because the NN learns onset timing too (in case
  DSP misses things).
- Build 06 first, ship it, then build 05 to push beyond.

---

## Escalation paths

- **If DSP recall is too low**: improve the DSP detector (it's still in
  production code; tune `stems_to_midi/processing_shell.py` configs).
  Then re-train classifier with the broader event set.
- **If classifier accuracy plateaus**: try a deeper backbone (ResNet18
  with `nn.AvgPool2d` reshape) — it's a per-image classification task,
  any standard CV backbone applies.
- **If velocity correlation is poor**: feed the DSP `strength` /
  `body_energy` features into the velocity head as auxiliary inputs.
- **If the classifier confuses similar drums (e.g., closed vs open hi-hat)**:
  extend the crop window to 250ms or 500ms to capture sustain
  characteristics.
- **If everything works but you want to push F1 above 0.90**: graduate
  to approach 08 (Onsets-and-Frames) or 09 (ADTOF) for a stronger
  per-stem onset detector replacing the DSP step.
