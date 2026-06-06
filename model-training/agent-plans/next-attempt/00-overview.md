# Next Attempt: Overview & Strong Recommendations

> **For a fresh agent landing here cold:** This is the entry point for the
> next round of work on `model-training/`. Read this file end-to-end, then
> pick **one** approach from the comparison table at the bottom and read
> that approach's dedicated file. Do not try to read all 14 sibling files.
>
> Before you do *anything* else, read `01-critique-and-theories.md` for the
> diagnosis of why the previous attempt failed, and `03-test-prove-overfit-first.md`
> for the one test you must run before trusting any modeling work.

## Control-group rule (CRITICAL)

**Test references MUST be ground-truth matched .wav + .midi pairs from
verified sources, NEVER model outputs.** The pattern
`<input>_v<N>_t<threshold>.mid` is the visual signal that a MIDI is a
model output, not ground truth. Specifically, the only valid test
references are:

- The committed control group at `model-training/tests/fixtures/e-gmd/`
  (5 deterministically-sampled e-GMD ground-truth pairs)
- The user's project ground truth in `user_files/`
- The full e-GMD drive at `/Volumes/1TB SSD 1/e-gmd-v1.0.0/`

The `tests/fixtures/e-gmd/README.md` documents the rule in detail.
Any test that uses `*_predicted_*.mid` or `/tmp/drumtomidi/*_pred_*.mid`
as a reference is contaminated and produces meaningless results.

This rule was added 2026-06-06 after the rescue-phase smoke test was
discovered to have persistence of prior model outputs in
`/tmp/drumtomidi/`, which could have been confused with ground truth
on subsequent runs.

---

## Repo orientation (assume you know nothing)

This repo is **larsnet** — a Python project for converting drum audio into
MIDI. It has three loosely-coupled subsystems:

1. **`stems_to_midi/`** — production classical/DSP drum transcription
   pipeline. Takes pre-separated drum stems (kick.wav, snare.wav, hihat.wav,
   toms.wav, cymbals.wav), runs onset detection + classification, emits MIDI.
   **This works.** It's the user's daily driver.

2. **The repo root** — `separate.py`, `mdx23c_*.py`, etc. — a `larsnet`-style
   stem separation system. Takes a full drum mix, splits into the 5 stems
   that `stems_to_midi/` consumes. **This also works.**

3. **`model-training/`** — an attempt to *replace* the heuristic logic in
   `stems_to_midi/` with a learned end-to-end neural model. **This is what
   failed** and what these plan documents address.

The cleanest mental model: layer 1 (separation) + layer 2 (classical
transcription) are the *baseline* that the neural attempt was meant to
improve on. Several of the approach files below propose **putting the
neural model on top of, alongside, or fed by these baselines** rather than
ignoring them.

### Key code locations (file:line)

| Concept | Where | Notes |
|---------|-------|-------|
| CRNN architecture | `model-training/model.py:13` | `DrumTranscriber`: Conv→Conv→BiGRU→Linear(20) |
| Feature extraction | `model-training/feature_extractor.py:13` | **Returns mono `[1,128,T]`** despite docstring claiming 3 channels — see critique |
| Label encoding (MIDI→target) | `model-training/label_encoder.py:51` | Causal smear `[1.0, 0.8, 0.5, 0.2]` on 10 onset channels + velocity on 10 more |
| Multi-task loss | `model-training/train_utils.py:14` | `MultiTaskDrumLoss` = BCEWithLogits(onsets) + masked MSE(velocity) |
| Training loop | `model-training/train.py:179` | Reads tab-delimited manifest, chunked at 8000 frames |
| Inference | `model-training/inference.py:36` | Loads ckpt, calls `heatmap_to_notes`, writes MIDI |
| Heatmap → MIDI | `model-training/inference_core.py:54` | Peak detection + threshold + velocity un-scale |
| Smoke / overfit test | `model-training/smoke_test.py:30` | Trains until loss drops, but **does not verify inference** |
| Classical pipeline (the baseline) | `stems_to_midi/processing_shell.py`, `stems_to_midi/midi.py` | The working alternative |
| Stem separation (upstream) | `separate.py`, `mdx23c_optimized.py` | Splits drum mix → 5 stems |
| Training data (e-GMD) | `/Volumes/1TB SSD 1/e-gmd-v1.0.0/` | Roland's Expanded Groove MIDI Dataset, ~444 hours |
| Conda env | `drumtomidi` | `conda run -n drumtomidi python ...` |

---

## The previous attempt in one paragraph

A CRNN was trained for ~150 epochs over ~145 drum-stem WAVs from e-GMD to
predict per-frame onset probabilities (10 drum classes) + per-frame
velocity regression (10 more channels). Training loss converged to ~0.17
(mean over 21,750 file-passes). When the trained checkpoint was used to
transcribe a file *from the training set*, the output MIDI was unusable.
**The model cannot reproduce data it has seen.** Per the original roadmap's
own Step 8 wording, that means there is "a fundamental bug in Step 1
(features) or Step 3 (label encoding)" — not in the modeling approach.

## Most likely cause (one finding, with high confidence)

**Channel collapse.** The roadmap specifies a 3-channel input (Left mel-spec,
Right mel-spec, Stereo-width mel-spec). The current `feature_extractor.py`
mixes stereo down to mono and returns `[1, 128, T]`. The `model.py` first
layer accepts only 1 input channel (`Conv2d(1, 32, ...)`). The
*docstring inside* `model.py` claims input is `[B, 3, 128, T]`. So an entire
input modality designed to differentiate **snare-vs-clap** (center vs wide)
and **ride-vs-crash** (position in the stereo field) is silently absent.

This is a strong but not definitive theory. See `01-critique-and-theories.md`
for the full ranked list of 7 hypotheses, each with a concrete test recipe.

---

## Strong Recommendation: the path I would take

**Step A (1–2 hours, mandatory): `03-test-prove-overfit-first.md`.**
Build a 10-second drum loop, train the existing pipeline on it for 1000
epochs, then *invoke `inference.py` on the same file* and assert the
output MIDI matches ground truth within ±20 ms and >0.8 velocity
correlation. If this fails — and the user's report says it will —
**stop and fix it before doing anything else**. The failure mode of the
test will tell you which of the 7 theories is correct.

**Step B (≤1 day, parallel with A): `02-tooling-wishlist.md` essentials.**
Build the missing test harness (`mir_eval` wrapper, synthetic single-hit
WAV generator). Cost: ~200 lines of Python. Payoff: every subsequent
attempt is *falsifiable*.

**Step C (the main bet): `05-approach-stems-as-input.md`.**
This matches the user's instinct independently arrived at. Use the
production `separate.py` stem splitter to feed *per-stem* audio to
per-stem transcribers. The hardest part of drum transcription — telling
a snare from a hihat from a tom from a cymbal — has *already been solved*
by the stem separator. The remaining problem (how many notes are in this
kick-only stem, when, how hard?) is dramatically easier. Each per-stem
model has 2–3 output classes, not 10. Convergence with thousands of
training examples per class should be trivial.

**Step D (the safety net): `06-approach-classical-onset-classify.md` in parallel with C.**
Take the existing onset detection from `stems_to_midi/processing_shell.py`.
For each detected onset, crop 100 ms of audio. Train a tiny per-event
classifier (CNN on a Mel spectrogram patch) to predict drum class and
velocity. **The neural net never has to learn "when did the hit happen?"
— that's a solved problem.** Expected size: ~100k parameters per stem.
Expected training data: 10k events per class. Expected convergence: 1
hour on CPU.

**Step E (escalation, if A–D plateau): `08-approach-onsets-and-frames-port.md`
or `09-approach-adtof-port.md`.**
Port a published reference architecture. If we can't beat a *pretrained*
ADTOF checkpoint's F1 on our test files, the problem isn't the architecture.

**Compute note:** the user has confirmed compute is **not a constraint**
— a big CPU box is available locally, and renting a GPU server (Lambda,
Vast.ai, RunPod, $0.50–$2/hr for an A100) is on the table for any
approach that benefits. This unlocks approaches 10, 11, 13, 14 which
were previously gated.

---

## Comparison table (all 10 approaches)

| # | Approach | File | Risk | Effort | Compute | Likely F1 | Why pick / why skip |
|---|----------|------|------|--------|---------|-----------|---------------------|
| 03 | **Prove-overfit-first** | `03-test-prove-overfit-first.md` | None | 0.5d | CPU | n/a | **Always do first.** Diagnoses the existing bug. |
| 04 | Bug-isolation A/B grid | `04-test-bug-isolation-grid.md` | None | 0.5d | CPU | n/a | Do if #03 fails and root cause is non-obvious. |
| 05 | **Stems-as-input ★** | `05-approach-stems-as-input.md` | Low | 3–5d | CPU | 0.85+ | **Top recommendation.** Reuses working stem separator. 5 small models instead of one big one. |
| 06 | **Classical onset + classify ★** | `06-approach-classical-onset-classify.md` | Low | 2–4d | CPU | 0.80+ | **Equally strong.** Reuses working onset detector. Tiny per-event classifier is well-posed. |
| 07 | Distill from classical | `07-approach-distill-from-classical.md` | Medium | 5–10d | CPU+GPU | 0.75+ | Use stems_to_midi as teacher on a much larger unlabeled corpus. |
| 08 | Onsets-and-Frames port | `08-approach-onsets-and-frames-port.md` | Medium | 5–7d | GPU | 0.85+ | Battle-tested. Reference impl exists. Has a published "drum" adaptation. |
| 09 | ADTOF port | `09-approach-adtof-port.md` | Low-Med | 3–5d | GPU | 0.85 (reported) | Drum-specific benchmark with pretrained ckpt. Fastest path to "known-good baseline". |
| 10 | Pretrained audio encoder | `10-approach-pretrained-audio-encoder.md` | Low | 4–6d | GPU | 0.80–0.90 | Stand on giants (AST/wav2vec2/CLAP/MERT). Heavy inference. |
| 11 | MT3-style transformer | `11-approach-mt3-transformer.md` | High | 10–20d | GPU+ | 0.90+ (SOTA) | Highest ceiling. Token output, sidesteps threshold/smear question entirely. |
| 12 | Curriculum learning | `12-approach-curriculum-learning.md` | Medium | 7–10d | CPU+GPU | 0.80+ | Synthetic → easy → hard. Best paired *with* one of 05–11. |
| 13 | Joint stems + transcribe | `13-approach-joint-stems-transcribe.md` | High | 15+d | GPU+ | unknown | Single end-to-end model. High elegance, high risk. |
| 14 | Diffusion-based | `14-approach-diffusion-based.md` | Highest | 20+d | GPU+ | unknown | Novel research direction. Lowest-confidence rec. |

### Scoring legend

- **Risk**: probability the approach fails to converge or has unforeseen blockers.
- **Effort**: solo-engineer days assuming I/O time is the bottleneck.
- **Compute**: minimum tier. CPU = your local machine. GPU = a rented A100 for the training run.
- **Likely F1**: educated guess at onset F1 on e-GMD test split. Numbers ending in "+" are author hopes; numbers labeled "(reported)" are from prior literature on similar tasks.

---

## What the previous attempt **got right** (worth keeping)

- The dataset choice (e-GMD) is the canonical academic drum dataset. Good call.
- The 10-class drum mapping is sane for general-purpose drum transcription.
- The pos_weight inverse-frequency calculation is methodologically sound.
- The decision to use BCEWithLogitsLoss (not BCELoss) was correct and stayed.
- The MIDI loader and `NoteAdapter` shim in `label_encoder.py` work well — preserve them.
- The `stems_to_midi/` and `separate.py` infrastructure is solid — every approach below leans on it.

## What the previous attempt **got wrong** (must change)

1. Silent regression from 3-channel to 1-channel input. Documented in 01.
2. Declaring the smoke test "passing" without testing the failure mode of interest (inference reproducing trained data). Documented in 01 and addressed by 03.
3. Adding a velocity head mid-stream without re-running the smoke test.
4. Running a 150-epoch full-batch training run without a validation set.
5. Hyperparameter thrash mid-run (LR, MAX_FRAMES, patience, chunk_frames all bouncing).
6. Reinventing instead of porting Onsets-and-Frames or ADTOF (both Apache/MIT).
7. Treating drum transcription as a one-shot 10-class problem instead of a multi-stage problem with an already-working classical baseline to lean on.

---

## Suggested execution order, with checkpoints

```
[ 03 ] prove-overfit-first
   │
   ├── PASS → existing pipeline can memorize; the bug is elsewhere.
   │         → run [ 04 ] bug-isolation-grid to find inference-time issue.
   │         → then proceed to [ 05 ] or [ 06 ].
   │
   └── FAIL → existing pipeline has a Step 1/3 bug (likely channel collapse).
              → fix the bug in feature_extractor.py to return [3, 128, T].
              → re-run [ 03 ]. If still fails, run [ 04 ] to bisect.
              → only then proceed to [ 05 ] or [ 06 ].

[ 05 ] OR [ 06 ] (pick one; can run in parallel if compute allows)
   │
   ├── F1 > 0.75 → ship it. The user has a working neural transcriber.
   │
   └── F1 < 0.5  → escalate to [ 09 ] (ADTOF) or [ 08 ] (Onsets-and-Frames).
                   These are battle-tested. If THEY don't beat 0.75 either,
                   the bottleneck is data quality or the stem separator,
                   not the modeling.

[ 10 ] / [ 11 ] / [ 12 ] / [ 13 ] / [ 14 ]
   Only worth attempting once one of 05/06/08/09 is at F1 > 0.75.
   These are improvements on a working baseline, not paths to one.
```

---

## How these documents are organized

```
model-training/agent-plans/next-attempt/
├── 00-overview.md                          ← you are here
├── 01-critique-and-theories.md             ← read this second
├── 02-tooling-wishlist.md                  ← ML/research tooling (mir_eval, compute, papers)
├── 02b-supercharge-the-agent.md            ← opencode-side tooling (MCPs, subagents, skills)
├── 03-test-prove-overfit-first.md          ← the mandatory baseline
├── 04-test-bug-isolation-grid.md           ← bisection tool
├── 05-approach-stems-as-input.md           ← ★ top rec
├── 06-approach-classical-onset-classify.md ← ★ 2nd rec
├── 07-approach-distill-from-classical.md
├── 08-approach-onsets-and-frames-port.md
├── 09-approach-adtof-port.md
├── 10-approach-pretrained-audio-encoder.md
├── 11-approach-mt3-transformer.md
├── 12-approach-curriculum-learning.md
├── 13-approach-joint-stems-transcribe.md
└── 14-approach-diffusion-based.md
```

Each approach file follows a fixed schema:
1. **Premise** — what problem this approach solves, in one paragraph
2. **Architecture** — diagram + key components
3. **Why this should work** — the bet, in plain terms
4. **What could go wrong** — honest risks
5. **Prerequisites** — what must be in place before you start
6. **Implementation steps** — discrete, ordered, with acceptance criteria
7. **Evaluation** — how you know it worked
8. **Estimated effort** — days, compute, dollars
9. **Escalation paths** — if it fails, what to try next

Stick to the schema when writing follow-up plans; consistency helps the
next agent skim quickly.
