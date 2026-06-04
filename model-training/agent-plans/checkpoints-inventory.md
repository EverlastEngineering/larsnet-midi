# Checkpoint Inventory (as of rescued-branch cleanup, 2026-06-04)

`model-training/models/` is **gitignored** and contains ~16 GB of training
checkpoints from multiple experiments. This file documents what's there so
a fresh agent can find their bearings without rooting through 800+ files.

## Symlink convention

- **`models/LATEST.ckpt`** → most recent training checkpoint
  (currently `train_checkpoint_v497.ckpt`). Use this as the default
  starting point for inference / inspection.

## Checkpoint series on disk

| Series                            | Count | Latest                          | Purpose                                            |
| --------------------------------- | ----- | ------------------------------- | -------------------------------------------------- |
| `train_checkpoint_v{N}.ckpt`      | 763   | `v497` (epoch 150, loss ~0.23)  | Main full-batch training run via `train.py`        |
| `train_checkpoint_v{N}_f{K}.ckpt` | (in above 763) | `v496_f100` (epoch 149, file 99) | Mid-epoch snapshots saved every 100 files          |
| `smoke_test_checkpoint_v{N}.ckpt` | 47    | `v47` (file 5, loss ~0.09)      | Single/batch overfit verification via `smoke_test.py` |
| `drum_train_v{N}_epoch{E}.ckpt`   | 1     | `v49_epoch1` (loss ~0.35)       | One-off from the abandoned `gemini_train.py` path  |
| Per-file snapshots                | 4     | `dl-1`, `31_hiphop_92...`, `39_rock-indie...`, `31_hiphop...` | Single-file overfits used as smoke targets         |
| `first.ckpt`                      | 1     | (loss ~0.011)                   | Earliest preserved single-file overfit             |

## Latest training run summary (the one that "ended in failure")

- Reached **epoch 150** over a 145-file training set.
- Best per-file train loss observed: **~0.14** (rock-halftime fills late
  in the run).
- Per-file train loss for the final epoch hovers between **0.14 and 0.23**.
- No validation loss recorded in the most recent checkpoints — training
  was running without `--val-list`.
- See last few entries of `results` field in `v497` for the file-by-file
  loss trajectory.

## Why the outcome was "failure"

Despite low training loss, real-world inference quality on the predictions
under `model-training/*_predicted_v*_t*.mid` was not usable. Root cause
was **not** isolated before the rescue branch was created. Candidate
hypotheses worth re-examining:

1. **Severe overfitting** — 145 files × 150 epochs without validation;
   training loss continued to drop but inference quality didn't follow.
2. **Loss-mask leakage** — the dampened `pos_weight` ([2.0..10.0]) may
   under-weight rare classes (HHO, TomMid, Crash2) to the point they're
   never triggered at inference time.
3. **Velocity head collapse** — the masked-MSE on the 10 velocity channels
   may be producing degenerate predictions that look fine in loss but
   render as uniform velocities.
4. **MIDI alignment drift** — the labeled hit frames assume `HOP_LENGTH=512`,
   `SAMPLE_RATE=44100`. A delta in either at inference time would silently
   misalign predicted onsets relative to ground truth.

## What to do with the disk pile

Currently: nothing. They're already gitignored. Leave them as historical
record. If disk space becomes an issue, the safe trim is:

```bash
# Keep latest 5 + every 50th + the named per-file ckpts
cd model-training/models
ls train_checkpoint_v*.ckpt | sort -V \
  | awk 'NR % 50 != 0 && NR < (n-5) { print }' n=$(ls train_checkpoint_v*.ckpt | wc -l)
# (review before piping to xargs rm)
```

Nothing in the codebase depends on the specific checkpoint version
numbers being contiguous.
