# e-GMD Control Group Fixtures

This directory contains **5 ground-truth .wav + .midi pairs** sampled
deterministically from the Roland e-GMD (Expanded Groove MIDI Dataset)
on `/Volumes/1TB SSD 1/e-gmd-v1.0.0/drummer1/session1/`.

## CRITICAL: these are the only files any test may use as a reference

**Test fixtures must be drawn from this directory (or another verified
ground-truth source). NEVER use as test references:**

- `*_predicted_v*_t*.mid` files in `model-training/` — these are model outputs from prior runs
- `/tmp/drumtomidi/*_pred_*.mid` — same, just in /tmp
- Any file with `_v<N>_t<threshold>` in its name
- Any file produced by a model's forward pass (drift over time, contains model's biases)

The naming convention `<input>_v{N}_t{threshold}.mid` exists in
`inference.py` for that exact reason — it is visually distinct from
ground-truth MIDI to prevent confusion. See the comment in that file
for the rationale.

## Why these specific 5 files

The sampler (`tools/sample_e_gmd_fixtures.py`) picks one file per
genre subset if possible, all with >= 100 notes, deterministically
using numpy seed 42. The selection ensures genre coverage of the
test suite.

| Genre hint | File | Notes |
|------------|------|-------|
| rock       | `202_rock-halftime_140_beat_4-4_16.{wav,midi}` | 1334 notes |
| funk       | `2_funk_80_beat_4-4_4.{wav,midi}`             |  754 notes |
| jazz       | `5_jazz-funk_116_beat_4-4_56.{wav,midi}`        | 1899 notes |
| latin      | `28_latin-samba_116_beat_4-4_17.{wav,midi}`     | 3236 notes |
| other      | `81_dance-breakbeat_170_beat_4-4_37.{wav,midi}` |  637 notes |

`other` substitutes for hiphop because e-GMD drummer1/session1 has
no hiphop samples (likely in drummer2 or later). The "other"
genre is dance-breakbeat, which is a useful high-BPM complement to
the others.

Total: ~70 MB committed (65 MB WAV + small MIDI + selection.json).
The full e-GMD has 11,352 matched pairs in drummer1/session1 alone;
we sample 5 to keep the repo slim.

## How to regenerate

If you want to re-sample (e.g. e-GMD dataset updates, or you want a
different selection):

```bash
conda run -n drumtomidi python model-training/tools/sample_e_gmd_fixtures.py
```

This rewrites `selection.json` and prints the new selection. To
actually copy the files into this directory, run the copy loop at
the bottom of the sampler script (or just rerun the same script
with `--copy` once that's added).

If the e-GMD drive is mounted at a different path on your machine,
edit `E_GMD_ROOT` at the top of the sampler.

## How tests should reference these

The smoke test `tests/test_overfit_reproduction.py` iterates over all 5
files, running the overfit-and-reproduce assertion on each. This gives
genre coverage of the regression check (a fix that breaks one genre
won't silently pass on the others).

For per-pipeline evaluation (e.g. the planned hybrid eval), use the
`eval_fixtures/` directory at the model-training root, which can
reference these same files plus user_files projects.
