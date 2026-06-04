# Rescue Summary — model-training, 2026-06-04

## What this document is

A point-in-time map of what the `rescued` branch looks like *immediately after*
the rescue cleanup. Written for a future agent (human or LLM) coming in fresh
to re-examine why the drum-transcription model didn't converge to a useful
result.

If you are that future agent: **start here**.

## What happened (timeline, newest at bottom)

| Commit    | Label                                         | Significance                                                                                                |
| --------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `0323d46` | feat: implement deep learning pipeline        | The seed commit for the whole model-training subproject                                                     |
| `5553997` | broke files up, BCEWithLogitsLoss             | First clean split into `train_utils.py` / `io_utils.py` / `inference_core.py`. Removed sigmoid from model. |
| `20ea782` | Working for training and output (vel sucked)  | Pipeline worked end-to-end but velocity was binary                                                          |
| `4a60cc5` | **successful mtl with velocity**              | Dual-head architecture (10 onset + 10 velocity, masked-MSE) shipped per `multi-task-velocity.SHIPPED.plan.md` |
| `90fedcb` | mtl failed approach, aborting (branch)        | The quad-head approach (`mtl-quad-head.ABANDONED.plan.md`) was tried and abandoned                          |
| `1788544` | omg good                                      | Last "celebration" commit before things went sideways                                                       |
| `e7aae84` | Ugh a mess                                    | Gemini-assisted parallel implementations added (`gemini_train.py`, etc.) — eventually removed in this rescue |
| `0a6593b` | pre-training                                  | Snapshot before a big multi-day training run                                                                |
| `6a24b7d` | opencode improvements                         | Cleanup of `train.py`. Last commit before training was abandoned.                                           |
| —         | (rescue commits, this document)               | See `git log --oneline rescued` for the 7 cleanup commits                                                   |

## Final state of the experiment (the wall the user hit)

- Training ran to **epoch 150** over 145 files (see
  `checkpoints-inventory.md`).
- Training loss converged to ~0.14–0.23 per file.
- **Inference quality was not usable.** Predicted MIDI files
  (`*_predicted_v*_t*.mid`) showed degenerate or empty output across
  threshold sweeps from 0.01 to 0.6.
- No validation loss was tracked during the run.
- Root cause was never isolated; the experiment was abandoned in this
  state.

## What the rescue did (it did NOT change the model)

The rescue **preserved all experimental intent** in commit history and
cleaned the working tree so a fresh attempt starts from a defensible
baseline. Specifically:

1. **Two `rescue:` commits** capture the staged + unstaged edits that
   were in mid-flight when the user stopped, kept as separate commits
   to preserve the layering of intent.
2. **One `fix:` commit** repairs `smoke_test.py` (it was broken — its
   `setup_training` unpack was stale).
3. **One `chore:` commit** deletes failed Gemini-assisted parallel
   implementations (`gemini_train.py`, `gemini_train_utils.py`,
   `bad-file-train.py.no`, `vd.sh`). Their history remains in
   commit `e7aae84` if anyone needs to look.
4. **One `docs:` commit** moved planning documents into this directory
   with explicit `SHIPPED` / `ABANDONED` status banners.
5. **One `chore:` commit** tracks the batch/val manifest `.txt` files
   and gitignores generated artifacts (`*.mid`, `*.midi`, `*.png`,
   `models/`).
6. **One `docs:` commit** added `checkpoints-inventory.md` and a
   `models/LATEST.ckpt` symlink.

Verification: `python smoke_test.py --audio dl-1.wav --midi dl-1.mid
--epochs 1` runs cleanly end-to-end on the rescue HEAD.

## Where things stand right now (the "unpolluted baseline")

- **Architecture (`model.py`)**: 2-layer CNN → bi-GRU(256) → Linear(20).
  First 10 outputs are onset logits, last 10 are velocity regression
  outputs. No sigmoid in `forward`.
- **Loss (`MultiTaskDrumLoss` in `train_utils.py`)**:
  `BCEWithLogitsLoss(pos_weight=[2.0..10.0])` for onsets +
  `weighted masked MSE` for velocities, summed with
  `velocity_weight=2.0`.
- **Labels (`label_encoder.py`)**: causal smear `[1.0, 0.8, 0.5, 0.2]`
  on onset channels; velocity stored as `(midi_vel/127)^0.7` at the
  exact onset frame only.
- **Training (`train.py`)**: chunked at `chunk_frames=8000`, truncated
  per-file at `MAX_TRAIN_SECONDS=300`, Adam(`lr=1e-3` default but
  `1e-4` on checkpoint resume — **inconsistent**), `ReduceLROnPlateau`
  with patience 3.
- **Device**: forced to `'cpu'`. cuda/mps paths exist but are
  intentionally guarded off (see `config.py` comment) because they
  produced incorrect training behavior.

## Hypotheses worth re-examining (free real estate for the next attempt)

These were not proven; they are starting points only.

1. **Overfitting** — 150 epochs on 145 files with no validation. Run
   `python train.py --list batch1_shuffled.txt --val-list val1_shuffed.txt`
   and watch for the train/val gap.
2. **Pos-weight rebalance hurt rare classes** — the dampened weights
   `[2.0..10.0]` may starve HHO/TomMid/Crash2. The original inverse-
   frequency weights `[5.10..156.13]` are commented out in
   `train_utils.py` — A/B them with a held-out class-level F1 score.
3. **Velocity head collapse** — `velocity_analysis.py` exists for this.
   Plot the predicted-vs-ground-truth velocity scatter on a freshly
   trained model and look for the diagonal trend predicted by
   `multi-task-velocity.SHIPPED.plan.md`.
4. **MIDI alignment** — `visualizer.py` produces `alignment_check.png`.
   Confirm onset frames in labels line up with audio transients before
   blaming the model.
5. **The model is too small** — 32→64 conv + 128-unit bi-GRU is on the
   small end. Try doubling the GRU width.
6. **The label smear is wrong** — `[1.0, 0.8, 0.5, 0.2]` causal-only
   may not match the typical drum transient envelope. Try acausal
   `[0.5, 1.0, 0.8, 0.5, 0.2]` or shorter `[1.0, 0.5]`.

## Inconsistencies left in place (deliberately)

These would have been suspicious to silently fix during the rescue:

- **LR mismatch in `train.py`**: `setup_training` default is `1e-3`,
  but the checkpoint-resume branch on L217 hard-codes `lr=1e-4`. The
  rescue commit message flags this; a future iteration should decide.
- **`MAX_TRAIN_SECONDS=300` cap**: feels arbitrary. Most files are
  ~3 minutes anyway, so the cap is usually a no-op, but it changes
  training behavior on the few long files.
- **`shuffler.py`**: tiny utility, unclear if still in use given that
  pre-shuffled manifests (`batch1_shuffled.txt`, `val1_shuffed.txt`)
  are already tracked. Left alone.

## Files of interest (re-orienteering guide)

```
model-training/
├── config.py              # constants, device guard, hyperparam loaders
├── config.yaml            # tunable training defaults
├── model.py               # DrumTranscriber CRNN architecture
├── label_encoder.py       # MIDI -> 20-channel target tensor
├── train.py               # full-batch training loop
├── train_utils.py         # MultiTaskDrumLoss, setup_training, load_*
├── smoke_test.py          # single-file or small-batch overfit test
├── inference.py           # model -> MIDI prediction
├── inference_core.py      # pure inference helpers
├── visualizer.py          # alignment_check.png generator
├── velocity_analysis.py   # GT-vs-pred velocity scatter
├── visual_diagnostic.py   # diagnostic_trace.png generator
├── pulse_check.py         # quick "is the model dead?" health check
├── batch1.txt, val1*.txt  # training manifests (tab-delimited wav<TAB>midi)
├── agent-plans/
│   ├── multi-task-velocity.SHIPPED.plan.md
│   ├── mtl-quad-head.ABANDONED.plan.md
│   ├── checkpoints-inventory.md
│   └── rescue-summary-2026-06-04.md  (this file)
└── models/                # gitignored, ~16GB of checkpoints + LATEST.ckpt symlink
```

## Safety net for this rescue

- `safety/rescued-snapshot-2026-06-04` branch (local; push manually if
  you want off-machine backup) points at the pre-rescue HEAD `6a24b7d`.
- Patches of the original staged/unstaged diffs saved at
  `/var/folders/.../larsnet-rescue-2026-06-04/` (will be cleaned by OS).
