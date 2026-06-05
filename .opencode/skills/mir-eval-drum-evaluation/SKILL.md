---
name: mir-eval-drum-evaluation
description: Use when evaluating predicted drum MIDI against ground-truth MIDI for the model-training drum transcription project. Wraps mir_eval.transcription with drum-specific conventions (50ms onset tolerance, no offset matching, per-class breakdown across the 10 drum classes defined in model-training/config.py).
---

# Drum MIDI Evaluation Skill

Standardized evaluation for the `model-training/` drum transcription
project. Replaces the hand-rolled `compare_midi` in
`model-training/inference.py:170` with `mir_eval.transcription` (the
de-facto MIR evaluation library) so numbers are comparable to published
baselines (Onsets-and-Frames, ADTOF, MT3).

## Tolerance conventions for drums

- **Onset tolerance**: 0.05 s (50 ms) for real evaluation. Matches Vogl/ADTOF papers.
- **Onset tolerance**: 0.020 s (20 ms) for the overfit smoke test (model has seen the data; should be near-perfect).
- **Pitch tolerance**: 0.5 (exact MIDI pitch match within rounding).
- **Offset ratio**: `None`. Drums have no meaningful sustain.

## The canonical evaluation module

Create `model-training/tools/eval_with_mir_eval.py` (the path the
`/eval` command expects):

The script lives in this skill at the section below. Copy-paste it
verbatim. Dependencies: `mir_eval`, `pretty_midi`. Both are pip-installable
into the `drumtomidi` env.

## Usage

```bash
# Single file evaluation
conda run -n drumtomidi python model-training/tools/eval_with_mir_eval.py \
    --pred predicted.mid --gt ground_truth.mid

# Tight tolerance (for smoke test)
conda run -n drumtomidi python model-training/tools/eval_with_mir_eval.py \
    --pred predicted.mid --gt ground_truth.mid --tolerance 0.020

# Batch evaluation across a manifest (audio<TAB>midi per line)
conda run -n drumtomidi python model-training/tools/eval_with_mir_eval.py \
    --manifest model-training/val1_test.txt --ckpt path/to/model.ckpt
```

## Output format

Overall section: precision, recall, F1, ground-truth note count, predicted note count.

Per-class section: same metrics broken out per drum class (Kick, Snare, HHC, HHO, TomHigh, TomMid, TomLow, Crash1, Crash2, Ride). Any class with `recall=0` is flagged as a likely class-imbalance failure (Theory T4 from `01-critique-and-theories.md`).

## Anti-patterns

- **Do not use the hand-rolled `compare_midi` in `inference.py:170`** — its tolerance semantics are not paper-comparable.
- **Do not average per-file F1** — sum TP/FP/FN globally and compute F1 once (micro-F1, not macro). The script above does this correctly.
- **Do not skip the per-class breakdown** — an overall F1 of 0.7 might be 0.95 on Kick and 0.0 on rare classes. That's diagnostic.
