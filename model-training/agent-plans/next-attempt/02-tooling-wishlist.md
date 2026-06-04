# Tooling Wishlist

> What I (the agent) don't have and what would unblock progress, ranked by
> impact. The first three items are **build-it-yourself** and should be
> done before attempting any of the modeling approaches in 05–14.

---

## Bottom line

The biggest gap is not modeling capability — it is **verification
infrastructure**. The previous attempt failed in large part because there
was no falsifiable test for "the model works." Build the three test
harnesses below before doing anything else. They cost ~1 day of effort
and will save you weeks.

---

## Tier 1: Build it yourself (highest impact, can do today, no external deps)

### Tool 1: Inference-reproduction test harness

**What it is**: a pytest suite that asserts trained models can reproduce
the MIDI they were trained on (within tolerance).

**Why it matters**: this is the missing Step 8 of the original roadmap.
The previous "smoke test passing" claim measured loss descent only, not
inference quality. Build this and every claim of "the model works" becomes
falsifiable.

**Sketch**:

```python
# model-training/tests/test_overfit_reproduction.py

import pytest
import torch
from pathlib import Path
import mir_eval

# Acceptance: trained model must reproduce training MIDI within these tolerances
ONSET_TOLERANCE_S = 0.020       # 20ms (mir_eval default is 50ms)
MIN_F1 = 0.95                   # near-perfect on memorized data
MIN_VELOCITY_CORR = 0.80        # correlation of predicted vs true velocity

@pytest.fixture(scope="module")
def overfit_checkpoint(tmp_path_factory):
    """Train smoke_test.py on a 10s drum loop for 500 epochs."""
    # ... build a tiny audio + midi pair, train, return ckpt path

def test_inference_recovers_training_midi(overfit_checkpoint):
    """Run inference on the trained file; assert F1 and velocity correlation."""
    notes = run_inference(overfit_checkpoint, "fixtures/10s_loop.wav")
    gt_notes = load_midi("fixtures/10s_loop.mid")

    # use mir_eval for industry-standard onset matching
    p, r, f1 = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, est_intervals, est_pitches,
        onset_tolerance=ONSET_TOLERANCE_S, pitch_tolerance=0.1, offset_ratio=None,
    )
    assert f1 >= MIN_F1, f"Trained model can't reproduce training MIDI: F1={f1:.3f}"

    velocity_corr = np.corrcoef(gt_velocities, pred_velocities)[0, 1]
    assert velocity_corr >= MIN_VELOCITY_CORR
```

**Effort**: 1 day. **Reusability**: every approach in 05–14 must pass
this test as its first integration gate.

### Tool 2: Synthetic single-hit dataset generator

**What it is**: a script that synthesizes drum WAVs from a soundfont with
deterministic, known onset times and velocities. Output is `(synth.wav,
synth.mid)` pairs where the MIDI is *guaranteed* to match the audio
sample-accurately.

**Why it matters**:
- e-GMD MIDI labels are human-played and may be off the audio by 5–30 ms.
  Synthetic data has zero label noise.
- For debugging: if the model can't memorize "kick at 0.5s, snare at
  1.0s, kick at 1.5s, snare at 2.0s" then the bug is 100% in your code,
  not the dataset.
- For ablations: easy to vary one factor at a time (tempo, density,
  velocity dynamics, drum count, mix of similar drums).

**Sketch**:

```python
# model-training/tools/synth_drum_loop.py

import argparse
from midiutil import MIDIFile
import subprocess  # call fluidsynth

def make_loop(out_wav, out_mid, bpm=120, bars=4, pattern="rock"):
    """
    Generate synthetic drum WAV + MIDI from a pattern spec.
    pattern: "rock" = kick on 1+3, snare on 2+4, hi-hat 8ths
             "simple_kick" = just kick on every beat
             ...
    """
    midi = MIDIFile(1)
    midi.addTempo(0, 0, bpm)

    # Place hits per pattern...
    # Write MIDI file...

    # Synthesize via fluidsynth + soundfont
    subprocess.run([
        "fluidsynth", "-ni", "-g", "1.0",
        "-F", out_wav,
        "-r", "44100",
        soundfont_path,
        out_mid,
    ])
```

**Dependencies**: `fluidsynth` (brew install), `midiutil` (already a dep),
a free drum soundfont (e.g., `GeneralUser GS` or `FluidR3_GM`).

**Effort**: 0.5 days. **Output**: `tests/fixtures/synthetic_*.{wav,mid}`
covering 5–10 increasing-difficulty scenarios.

### Tool 3: mir_eval-based evaluation wrapper

**What it is**: replace the hand-rolled `compare_midi` in
`model-training/inference.py:170` with a thin wrapper around
`mir_eval.transcription`. Outputs standard precision/recall/F1
per-class and overall.

**Why it matters**: `mir_eval` is the de-facto MIR evaluation library
(BSD). Its tolerance semantics are well-defined. The current
`compare_midi` does pitch-exact comparison without a clear tolerance
policy and isn't comparable to any published paper's numbers.

**Sketch**:

```python
# model-training/tools/eval_with_mir_eval.py

import mir_eval
import pretty_midi

def evaluate(pred_midi_path, gt_midi_path, onset_tolerance=0.05):
    pred = pretty_midi.PrettyMIDI(pred_midi_path)
    gt = pretty_midi.PrettyMIDI(gt_midi_path)

    # Build (intervals, pitches) arrays for mir_eval
    pred_intervals, pred_pitches = midi_to_intervals(pred)
    gt_intervals, gt_pitches = midi_to_intervals(gt)

    metrics = mir_eval.transcription.evaluate(
        gt_intervals, gt_pitches, pred_intervals, pred_pitches,
        onset_tolerance=onset_tolerance,
    )
    # Per-class breakdown
    per_class = {}
    for class_idx in range(10):
        # filter by drum class, compute F1
        ...
    return {"overall": metrics, "per_class": per_class}
```

**Dependency**: `pip install mir_eval` (~no transitive deps, pure Python).

**Effort**: 0.5 days. **Output**: command-line tool `eval.py
ckpt.pt audio.wav truth.mid` reporting paper-style numbers.

---

## Tier 2: External tools that would help (research, debugging)

### Tool 4: Context7 MCP

**What it is**: live, version-aware documentation lookup for libraries.

**Why I want it**: the modeling approaches involve PyTorch, torchaudio,
pretty_midi, librosa, mir_eval, transformers (HuggingFace), and
potentially fairseq, magenta, or audiocraft. Without Context7 I read
source code to learn APIs, which is slow and error-prone.

**Impact**: ~2× speedup on any approach involving an unfamiliar library.

**How to add**: `opencode mcp add context7` (per opencode docs).

### Tool 5: arXiv / Papers-with-Code search

**What it is**: search papers by topic, retrieve PDFs, find linked
code repos.

**Why I want it**: I have `webfetch` (URL-targeted fetch only). I cannot
search. If the user supplies arxiv IDs I can fetch them, but I can't go
exploring on my own. Approaches 08, 09, 10, 11, 14 all benefit from
checking the current SOTA literature before locking in an architecture.

**Workaround without MCP**: user can paste arxiv URLs into chat; I'll
fetch and summarize.

### Tool 6: HuggingFace MCP / hf-hub access

**What it is**: search models, download checkpoints, load configs.

**Why I want it**: Approach 10 (pretrained audio encoder) explicitly
relies on loading pretrained AST/wav2vec2/CLAP/MERT. With MCP this is
one tool call; without it, I have to construct `from_pretrained(...)`
URLs from memory.

**Workaround**: hardcoded URLs for the 4 candidate encoders.

### Tool 7: Audio playback in chat

**Not exposed.** Would be useful for sanity-checking predicted MIDI
against ground truth without having to switch to a DAW. Workaround:
write a `playback_check.py` that renders predicted MIDI with the same
soundfont as Tool 2 and compares waveforms visually via spectrogram diff.

---

## Tier 3: Compute (now unlocked per user confirmation)

The previous attempt assumed CPU-only training. The user has confirmed
this is no longer a constraint. Options ranked by cost-effectiveness:

| Option | Cost | Setup time | When to use |
|--------|------|------------|-------------|
| Your big local CPU box | $0 | 0 min | Approaches 03, 04, 05, 06 (small models, ≤1M params) |
| **Free Kaggle GPU** (P100, 30 hr/wk) | $0 | 15 min | Approaches 07, 08, 09, 12 (medium models) |
| **Free Colab GPU** (T4, ~12 hr session) | $0 | 5 min | Quick experiments, can't run overnight |
| Lambda Labs A100 80GB | $1.10/hr | 20 min | Approaches 10, 11, 13, 14 (large models, GPU-heavy) |
| Vast.ai community A100/A6000 | $0.40-0.80/hr | 30 min | Same as above, cheaper but spottier |
| RunPod A40/A100 | $0.60-1.50/hr | 15 min | Same as above, easier UI |
| AWS/GCP/Azure GPU | $3+/hr | 30+ min | Only if you already have org credits |

**My recommendation**:
- **Local CPU** for approaches 05, 06, and the test harnesses (Tier 1).
- **Free Kaggle GPU** for 07-09 training (one model run < 12 hours typically).
- **Lambda A100 spot** for 10, 11 when you commit to one of them. Budget
  $20-100 per training run including hyperparam sweeps.

**Storage**: e-GMD is ~90GB. Renting a server means re-downloading or
syncing. Solutions:
- Sync to the rented server via `rsync -av` (slow over public internet).
- Upload to a bucket once (S3/R2), pull from there.
- Use HuggingFace Datasets which mirrors e-GMD; load via streaming.

### Practical setup script (for the rented-GPU case)

```bash
# On the rented box:
git clone https://github.com/EverlastEngineering/DrumToMIDI.git
cd DrumToMIDI && git checkout rescued
conda env create -f environment.yml -n drumtomidi
conda activate drumtomidi

# Fetch e-GMD (pick one):
# Option A: from HuggingFace (recommended, ~90GB, supports streaming)
python -c "from datasets import load_dataset; load_dataset('marsyas/gtzan')"  # placeholder; real e-GMD path differs
# Option B: from your S3 bucket
aws s3 sync s3://your-bucket/e-gmd /data/e-gmd-v1.0.0/
# Option C: rsync from your machine
rsync -avz --progress yourmac:/Volumes/1TB\ SSD\ 1/e-gmd-v1.0.0/ /data/e-gmd-v1.0.0/
```

---

## Tier 4: Library upgrades to consider

### `pretty_midi` instead of hand-rolled MIDI handling

Already a dependency for `velocity_analysis.py` and `visual_diagnostic.py`,
but the main training/inference path uses `midiutil` (write) +
`midi_shell.load_midi_file` (read). `pretty_midi` does both with a much
cleaner API and is the de-facto standard in MIR. Migrate when convenient.

### `madmom` for classical onset detection

If approach 06 needs more sophisticated onset detection than what's in
`stems_to_midi/`, `madmom` (BSD) is the standard library with multiple
algorithms (SuperFlux, CNN-based, RNN-based). Pretrained models included.

### `torchaudio.transforms` already used, but check `torchaudio.pipelines`

Has bundled pretrained models for wav2vec2, HuBERT, etc. — would simplify
approach 10 substantially.

### `lightning` or `accelerate` for training loop

The current `train.py` is hand-rolled. PyTorch Lightning / HuggingFace
Accelerate would handle:
- Gradient accumulation
- Mixed precision (fp16/bf16) — 2-3× speedup on GPU
- Multi-GPU
- Checkpoint management
- Logging (TensorBoard, W&B integration)
- Resumability

Worth migrating if any approach involves >24h training runs.

---

## Tier 5: Reference papers/repos to read before starting

> If you have arXiv access (or can be given URLs), prioritize these. I
> can fetch and summarize any of them if you paste the URL.

| # | Title | Year | Key takeaway for our problem |
|---|-------|------|------------------------------|
| 1 | **Onsets and Frames: Dual-Objective Piano Transcription** (Hawthorne et al.) | 2018 | Gated onset/frame head architecture — directly applicable. arXiv:1710.11153. |
| 2 | **ADTOF: Annotated Dataset of Drum Onsets in Polyphonic Music** (Cuisinier) | 2021 | Drum-specific benchmark + CRNN reference impl. |
| 3 | **MT3: Multi-Task Multitrack Music Transcription** (Gardner et al.) | 2022 | Token-output transformer, SOTA for multi-instrument. arXiv:2111.03017. |
| 4 | **AST: Audio Spectrogram Transformer** (Gong et al.) | 2021 | Pretrained encoder for audio tasks. arXiv:2104.01778. |
| 5 | **MERT: Acoustic Music Understanding via Self-Supervised Pretraining** | 2023 | Music-specific pretrained encoder. |
| 6 | **wav2vec 2.0** (Baevski et al.) | 2020 | Self-supervised speech encoder; works for music too. |
| 7 | **The Roland e-GMD dataset paper** | 2020 | Documentation for the dataset we're using. Important for understanding label conventions. |

---

## Summary

**Do these three things before any modeling work** (1 day total):

1. Build `model-training/tests/test_overfit_reproduction.py` (Tool 1).
2. Build `model-training/tools/synth_drum_loop.py` (Tool 2).
3. Replace `compare_midi` with mir_eval wrapper (Tool 3).

**Then**: pick an approach from 05–14 and execute it with falsifiable
goals against the new test harness.

**Compute strategy**:
- Local CPU for 03, 04, 05, 06.
- Free Kaggle / rented A100 for 07–14.

**MCP wishlist** (in priority order):
1. Context7 — doc lookup
2. arXiv search — paper research
3. HuggingFace hub — pretrained model loading

If none of these MCPs are available, ask the user to paste relevant URLs
and I can fetch them with `webfetch`.
