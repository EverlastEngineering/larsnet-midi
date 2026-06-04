# Supercharge the Agent: opencode-Specific Tooling

> Companion to `02-tooling-wishlist.md`. That file covered **ML/research
> tooling** (mir_eval harness, synthetic data, pretrained models, compute).
> THIS file covers **opencode-specific configuration** — MCPs, subagents,
> skills, commands, plugins, permissions — that would materially improve
> what the agent (me) can do on this drum-transcription work.
>
> The user (Jason) currently has a near-empty opencode config:
> `~/.config/opencode/opencode.jsonc` is just `{"$schema": "..."}` with
> nothing else, and there is no project-level `opencode.json` /
> `.opencode/` directory in larsnet. That means there is *no friction*
> to layering in the items below — none of this would conflict with
> existing configuration.

---

## Honest self-assessment

### What I have today

| Capability | Tool | Limits I've actually hit on this project |
|------------|------|-------------------------------------------|
| Read files | `read` | None significant |
| Write/edit files | `write`, `edit` | None significant |
| Search | `glob`, `grep` | Fine for in-repo; can't search arxiv or HuggingFace |
| Run shell commands | `bash` | **120-second default timeout** — can't watch a training run |
| Fetch URLs | `webfetch` | Markdown-only output; needs user-provided URLs (can't search) |
| Spawn subagents | `task` (general/explore) | Generic; not specialized for ML tasks |
| Track multi-step work | `todowrite` | Good |
| Ask the user | `question` | Good |
| Load named workflows | `skill` | Only one is registered (`customize-opencode`) |

### What I'm missing for this specific work

| Capability gap | Project impact |
|----------------|----------------|
| **Cannot observe a long-running training** | Can't watch the bug-isolation grid (~30 min) or full training (hours). Each iteration requires me to ask the user to run + paste results. |
| **Cannot listen to audio or see PNGs** | `alignment_check.png`, `velocity_audit.png`, `diagnostic_trace.png` are key debugging tools. I can write code that generates them but cannot judge them. |
| **Cannot search arXiv / Papers-With-Code** | Approaches 08, 09, 10, 11, 14 all hinge on prior art. Without search I rely on URLs the user pastes. |
| **Cannot search HuggingFace** | Approach 10 (pretrained encoder) — I cannot programmatically discover the best encoder for drum audio. |
| **Cannot interactively explore a checkpoint** | Loading a `.ckpt`, inspecting weights, running ad-hoc forward passes — each requires a roundtrip through bash with the 120s timeout. A Jupyter MCP would collapse this 10×. |
| **No persistent memory between sessions** | I re-discover the channel-collapse bug each session unless the user re-points me at the docs. |
| **No specialized debug subagents** | When I need to inspect a checkpoint, I do it inline rather than delegating to a specialist. |
| **No project-specific commands** | `/smoke-test`, `/eval`, `/grid` would compress 3-line workflows into 1-line invocations. |
| **No auto-approved safe operations** | Every read/grep/glob outside `model-training/` may prompt for permission. Configurable away. |

The list below is ordered by **expected impact on this project** (drum
transcription rescue + re-attempt), not alphabetically.

---

## Tier S: Highest-impact additions

### 1. Subagent: `training-monitor` (HIGHEST PRIORITY)

**What it does**: handles long-running training jobs. Starts a `python
train.py ...` invocation in the background, polls its log/checkpoint
output every N seconds, and surfaces only anomalies + completion to me.

**Why it matters**: training runs for the bug-isolation grid (~30 min)
and any full training (hours-days) currently bottleneck on my 120s bash
timeout. I literally cannot watch them.

**File**: `.opencode/agent/training-monitor.md`

```markdown
---
description: Watches a long-running training job; reports anomalies (NaN, loss spike, OOM) and completion. Use when starting any training that takes >60 seconds.
mode: subagent
model: anthropic/claude-haiku-4-5
permission:
  edit: deny
  bash:
    "tail *": allow
    "ps *": allow
    "kill *": ask
    "cat */loss.log": allow
    "*": ask
---

You are a training-job watchdog. Given an invocation command + log path:

1. Verify the log path is writable
2. Start the training command in the background with output redirected
3. Poll the log file every 30 seconds:
   - Track last N loss values
   - Flag NaN or Inf immediately
   - Flag loss spikes (>2x the rolling average)
   - Flag if loss hasn't decreased in 50 consecutive epochs
4. On completion or anomaly, report:
   - Final/last loss
   - Total wall time
   - Whether checkpoint was saved
   - Any anomaly that fired
5. Do NOT make code edits or interpret model quality — only watchdog.
```

**Net effect**: I can say "start training and tell me when it's done or
broken" and get a structured report 4 hours later instead of timing out
twice and bothering the user.

### 2. MCP: Context7 (live library documentation)

**What it does**: live docs lookup for any library by version.

**Why it matters**: every modeling approach (05-14) involves library
APIs I'd otherwise have to learn by reading source. Examples this
project hits constantly:
- `torchaudio.transforms.MelSpectrogram` parameters
- `pretty_midi.PrettyMIDI` API
- `mir_eval.transcription.evaluate` signature
- `transformers.ASTModel` / `transformers.AutoFeatureExtractor`
- `diffusers.UNet1DModel`

Without Context7: I read source. With it: one tool call.

**Install**:
```json
// In ~/.config/opencode/opencode.jsonc or .opencode/opencode.json
{
  "mcp": {
    "context7": {
      "type": "local",
      "command": ["npx", "-y", "@upstash/context7-mcp"],
      "enabled": true
    }
  }
}
```

(Verify the exact npm package name from upstash docs at install time.)

**Net effect**: ~2× speedup on every approach that touches an unfamiliar
library. Removes a major class of "I think this is the API" mistakes.

### 3. MCP: HuggingFace Hub

**What it does**: search models/datasets, get config, download files.

**Why it matters**:
- Approach 09 (ADTOF) needs the pretrained checkpoint URL
- Approach 10 (pretrained encoder) needs to evaluate AST vs MERT vs HuBERT
- Approach 11 (MT3) wants the original MT3 weights
- e-GMD itself can be loaded from HF: `datasets.load_dataset("magenta/groove")`

**Install** (HF has an official MCP as of 2024):
```json
{
  "mcp": {
    "huggingface": {
      "type": "remote",
      "url": "https://huggingface.co/mcp",
      "headers": { "Authorization": "Bearer ${HF_TOKEN}" }
    }
  }
}
```

(Confirm exact URL/auth from current HF MCP docs; this is the documented
form as of early 2025.)

**Net effect**: approaches 09/10/11 become directly actionable instead
of requiring the user to research and paste model identifiers.

### 4. Skill: `drum-transcription-debug`

**What it does**: encapsulates the entire bug-bisection procedure from
`03-test-prove-overfit-first.md` into a single skill the agent can load
on demand.

**Why it matters**: the bisection procedure has 4 specific tests, each
with a precise command + interpretation. Without a skill I re-derive it
each session; with one, I follow a deterministic checklist.

**File**: `.opencode/skills/drum-transcription-debug/SKILL.md`

```markdown
---
name: drum-transcription-debug
description: Use when debugging why a trained drum transcription model fails to reproduce its training MIDI. Walks the 4-test bisection from agent-plans/next-attempt/03-test-prove-overfit-first.md.
---

# Drum Transcription Debugging Skill

The model under `model-training/` has a known failure pattern: training
loss converges low (~0.17) but inference on a training file produces
unusable MIDI. This skill walks the bisection.

## Required context first
Read these in order before running any test:
1. `model-training/agent-plans/next-attempt/01-critique-and-theories.md`
2. `model-training/agent-plans/next-attempt/03-test-prove-overfit-first.md`

## Bisection sequence

### Test 1: Does training loss actually descend?
```bash
conda run -n drumtomidi python smoke_test.py \
    --audio dl-1.wav --midi dl-1.mid --epochs 50 2>&1 | tee /tmp/smoke.log
grep "Loss:" /tmp/smoke.log | tail -5
```
- Loss stuck near 1.0 → optimizer/loss/device bug
- Loss starts <0.1 → label encoding produces mostly zeros (check build_targets)
- Loss diverges/NaN → numerical instability (check device)

### Test 2: Does the alignment visualizer show hits aligned with transients?
[see 03-test-prove-overfit-first.md Bisection 2]

### Test 3: Are inference outputs above threshold at known hits?
[see 03-test-prove-overfit-first.md Bisection 3]

### Test 4: Is the smear preventing distinct peaks?
[see 03-test-prove-overfit-first.md Bisection 4]

## Interpretation rules
- Stop at first failed test; fix root cause before continuing.
- If all 4 pass but end-to-end F1 < 0.5: bug is in `heatmap_to_notes`
  or `write_midi`, not the model.
- Always check the channel count: `python -c "from feature_extractor
  import get_input_tensor; print(get_input_tensor('dl-1.wav').shape)"`
  Expected per roadmap: [3, 128, T]. Actual today: [1, 128, T].
```

**Net effect**: any future agent (or me on a new session) with this
skill follows the right debugging order instead of guessing.

### 5. Skill: `mir-eval-drum-evaluation`

**What it does**: encapsulates the evaluation protocol — how to convert
MIDI to mir_eval intervals/pitches, what tolerance to use, how to read
per-class F1.

**File**: `.opencode/skills/mir-eval-drum-evaluation/SKILL.md`

```markdown
---
name: mir-eval-drum-evaluation
description: Use when evaluating predicted drum MIDI against ground truth. Wraps mir_eval.transcription with drum-specific conventions (50ms onset tolerance, no offset matching, per-class breakdown).
---

# Drum MIDI Evaluation Skill

## Standard tolerance for drums
- Onset tolerance: 0.05 seconds (50ms). This matches Vogl/ADTOF papers.
- Pitch tolerance: 0.5 (exact MIDI pitch match within rounding).
- Offset ratio: None (drums have no meaningful sustain).

## The canonical evaluation snippet

```python
import mir_eval
import pretty_midi
import numpy as np

def midi_to_arrays(pm):
    intervals, pitches = [], []
    for inst in pm.instruments:
        for note in inst.notes:
            intervals.append([note.start, note.end])
            pitches.append(float(note.pitch))
    if not intervals:
        return np.zeros((0, 2)), np.zeros((0,))
    return np.array(intervals), np.array(pitches)

def eval_drum_transcription(pred_midi_path, gt_midi_path):
    pred = pretty_midi.PrettyMIDI(pred_midi_path)
    gt = pretty_midi.PrettyMIDI(gt_midi_path)
    p_int, p_pit = midi_to_arrays(pred)
    g_int, g_pit = midi_to_arrays(gt)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        g_int, g_pit, p_int, p_pit,
        onset_tolerance=0.05,
        pitch_tolerance=0.5,
        offset_ratio=None,
    )
    return {'precision': p, 'recall': r, 'f1': f,
            'gt_count': len(g_int), 'pred_count': len(p_int)}
```

## Per-class breakdown
Always compute per-class F1 in addition to overall. The 10 drum classes
in this project map per `model-training/config.py:INDEX_TO_MIDI`. A
trained model with overall F1=0.7 might be F1=0.95 on Kick but F1=0.0
on HHO/TomMid/Crash2 — that's a class-imbalance failure, not a model
failure.

## Anti-patterns
- Do not use the hand-rolled `compare_midi` in `inference.py:170` —
  it has known buggy tolerance semantics.
- Do not aggregate F1 across files by averaging — sum TP/FP/FN
  globally and compute F1 once (micro-F1, not macro).
```

**Net effect**: every evaluation across approaches 05-14 uses the same
convention. Numbers are comparable to published baselines.

### 6. Commands: `/smoke`, `/eval`, `/grid`, `/train-stem`

**What they do**: compress common multi-step workflows into single
invocations.

**File**: add to `.opencode/opencode.json` or `~/.config/opencode/opencode.jsonc`:

```json
{
  "command": {
    "smoke": {
      "description": "Run the prove-overfit-first smoke test on dl-1 and report PASS/FAIL.",
      "prompt": "Run `conda run -n drumtomidi pytest model-training/tests/test_overfit_reproduction.py -v -s`. If it fails, follow the bisection procedure in 03-test-prove-overfit-first.md."
    },
    "eval": {
      "description": "Evaluate a checkpoint against e-GMD test split using mir_eval.",
      "prompt": "Use the mir-eval-drum-evaluation skill. Run `conda run -n drumtomidi python model-training/tools/eval_with_mir_eval.py --ckpt $CKPT --manifest model-training/val1_test.txt` and report overall + per-class F1."
    },
    "grid": {
      "description": "Run the 2x2x2 bug-isolation grid on dl-1.",
      "prompt": "Execute the plan in `model-training/agent-plans/next-attempt/04-test-bug-isolation-grid.md`. Run all 8 configurations, tabulate results, identify the winning config."
    },
    "train-stem": {
      "description": "Train one per-stem transcriber per approach 05. Args: stem name (kick|snare|hihat|toms|cymbals).",
      "prompt": "Follow approach 05 (model-training/agent-plans/next-attempt/05-approach-stems-as-input.md). Spawn the training-monitor subagent to watch the run. Report final per-stem F1."
    }
  }
}
```

**Net effect**: workflows that currently require me to recall a 5-step
procedure become "/smoke", "/eval $CKPT", "/grid", "/train-stem kick".

---

## Tier A: High-impact additions

### 7. Subagent: `paper-summarizer`

**What it does**: given an arxiv URL or PDF link, fetches it (via
`webfetch`), extracts the architecture diagram + table of numbers +
hyperparameter table, returns a structured summary.

**Why it matters**: approaches 08 (Onsets-and-Frames), 09 (ADTOF), 11
(MT3) all hinge on faithful porting of published architectures. A
specialized summarizer ensures I extract the *exact* hyperparameters
the paper used, not approximations.

**File**: `.opencode/agent/paper-summarizer.md`

```markdown
---
description: Given an arXiv or paper URL, produces a structured summary focused on (architecture, hyperparameters, reported metrics, training recipe). Use when porting a published model.
mode: subagent
model: anthropic/claude-sonnet-4-6
permission:
  edit: deny
  webfetch: allow
  read: allow
  bash: deny
---

You produce structured paper summaries for reproducible porting work.

Output format (markdown):

## Paper
- Title, authors, year, venue
- arXiv ID + URL

## Architecture
- Layer-by-layer (with shapes, param counts)
- Any non-standard ops with citation

## Loss
- Exact formulation
- Weighting between terms

## Training Recipe
- Optimizer + learning rate schedule
- Batch size
- Epochs / steps
- Augmentations
- Validation cadence

## Reported Metrics
- Dataset + split
- Headline F1/precision/recall
- Any ablation numbers

## Data Pipeline
- Sample rate
- Window/hop
- Mel bins
- Label encoding

## Caveats / Replication Notes
- What the paper omits
- Known reproducibility issues from follow-up work
- Compatible/incompatible licenses for our codebase

Be terse. Cite exact section/figure numbers.
```

### 8. Subagent: `dataset-statistician`

**What it does**: given a training manifest (`batch1.txt`), produces a
full statistical report: per-class counts, per-file duration histogram,
audio loudness distribution, MIDI density distribution, train/val
overlap detection.

**Why it matters**: Theory T5 (overfitting) and T4 (class imbalance)
both need empirical answers. I currently have to write one-off scripts;
a subagent encapsulates them.

**File**: `.opencode/agent/dataset-statistician.md`

```markdown
---
description: Audits a training manifest. Reports per-class hit counts, file duration histogram, audio loudness, MIDI density, val/train leak detection. Use before any training run on a new dataset.
mode: subagent
model: anthropic/claude-sonnet-4-6
permission:
  edit: allow   # creates scratch reports
  bash: allow
  read: allow
---

Given a manifest path (tab-delimited audio<TAB>midi per line):

1. Count files; report total duration (hours)
2. For each file, load MIDI and tally per-pitch hit counts
3. Aggregate per-class (using model-training/config.py:MIDI_TO_INDEX)
4. Report:
   - Per-class total + percentage
   - Class imbalance ratio (max/min)
   - File-duration histogram (10s, 30s, 1min, 3min, 5min+ buckets)
   - Empty-MIDI files (likely corrupt)
   - MIDI/audio length mismatch (>10% diff = suspicious)
5. If a val manifest is also provided:
   - Detect train/val filename overlap
   - Detect train/val content duplication (hash audio files)
6. Save report to /tmp/dataset_stats_<manifest_name>.md

Output: path to report + a 5-line summary.
```

### 9. Subagent: `checkpoint-inspector`

**What it does**: given a `.ckpt` file path, loads it, reports the
architecture it expects, weight statistics (NaN check, mean/std per
layer), training metadata (epoch, loss, file_idx), and runs a forward
pass on a known input to verify it produces non-degenerate output.

**Why it matters**: 99 checkpoints currently exist in
`model-training/models/`. Manually inspecting one takes ~20 lines of
Python each time. A subagent batches this.

### 10. Plugin: `pre-commit-overfit-test`

**What it does**: opencode plugin hook that runs the overfit smoke
test before commits that touch `model-training/*.py`.

**Why it matters**: prevents the *exact failure pattern* that motivated
this whole rescue (silent regression of the smoke test). If anyone
touches `feature_extractor.py` and breaks the 3-channel intent again,
the commit fails.

**File**: `.opencode/plugin/pre-commit-overfit-test.ts`

```typescript
import type { Plugin } from "@opencode-ai/plugin"

export default (async ({ $, project }) => {
  return {
    "tool.execute.before": async (input, output) => {
      // Trigger on bash commands that look like git commits
      if (input.tool !== 'bash') return
      const cmd = output.args?.command ?? ''
      if (!cmd.match(/git\s+commit/)) return

      // Check if any model-training/*.py files are staged
      const staged = await $`git diff --cached --name-only`
      const touched = staged.text().split('\n').filter(f =>
        f.startsWith('model-training/') && f.endsWith('.py')
      )
      if (touched.length === 0) return

      // Run the overfit smoke test
      console.log(`[hook] model-training touched, running overfit test...`)
      const result = await $`conda run -n drumtomidi pytest model-training/tests/test_overfit_reproduction.py -x --tb=short`
      if (result.exitCode !== 0) {
        throw new Error(`Overfit smoke test FAILED. Commit blocked. Fix the test or use --no-verify intentionally.`)
      }
    }
  }
}) satisfies Plugin
```

### 11. Permission rules to reduce friction

**What it does**: pre-approve safe operations so I don't pause for
permission on every read inside `model-training/`.

**File**: `.opencode/opencode.json`

```json
{
  "$schema": "https://opencode.ai/config.json",
  "permission": {
    "read": "allow",
    "glob": "allow",
    "grep": "allow",
    "list": "allow",
    "edit": {
      "model-training/**": "allow",
      "model-training/models/**": "deny",
      "*": "ask"
    },
    "bash": {
      "git status": "allow",
      "git diff *": "allow",
      "git log *": "allow",
      "ls *": "allow",
      "conda run -n drumtomidi python -c *": "allow",
      "conda run -n drumtomidi pytest model-training/*": "allow",
      "rm *": "deny",
      "rm -rf *": "deny",
      "git push *": "ask",
      "*": "ask"
    },
    "external_directory": {
      "/Volumes/1TB SSD 1/e-gmd-v1.0.0/**": "allow",
      "/var/folders/**/opencode/**": "allow",
      "*": "ask"
    }
  }
}
```

**Net effect**: I stop asking for read/grep permission inside
`model-training/`, stop asking to read e-GMD audio for inspection,
but still confirm before `rm`, `git push`, or edits outside
`model-training/`.

---

## Tier B: Useful, lower priority

### 12. MCP: Sequential Thinking / Memory

The `@modelcontextprotocol/server-sequential-thinking` MCP gives a
scratchpad for multi-step reasoning. The `server-memory` MCP gives
key-value persistence across sessions.

For this project: marginal. The detailed plan files in
`agent-plans/next-attempt/` already serve as persistent memory.

### 13. MCP: Filesystem (server-filesystem)

Adds batch file operations (copy, move, recursive ops). Useful if I
need to reorganize `model-training/models/` (e.g., move 99 checkpoints
into per-epoch subdirectories). Lower priority because I rarely need it.

### 14. MCP: Git (server-git)

Richer git inspection (semantic diff, blame, log queries with filters).
Marginal because vanilla `git log` via bash already works.

### 15. Skill: `pytorch-training-loop-best-practices`

A skill that documents the gotchas of PyTorch training loops applicable
to *all* approaches:
- Set `torch.manual_seed`, `torch.cuda.manual_seed_all` for reproducibility
- Use `model.train()` / `model.eval()` toggles
- Clip gradients (`torch.nn.utils.clip_grad_norm_`) to prevent explosion
- Validate with `torch.no_grad()` to save memory
- Save optimizer state alongside model state for resumability
- Use `pin_memory=True` in DataLoader when moving to GPU

Useful but generic; lower priority than drum-specific skills above.

### 16. Subagent: `model-zoo-explorer`

Given a task description (e.g., "drum onset detection"), searches
HuggingFace + Papers-With-Code for recent SOTA. Requires HuggingFace
MCP (item #3) to be functional first.

---

## Tier C: Things tooling alone can't fix

These are limitations of the agent paradigm that no MCP/skill/subagent
fully solves. Worth being explicit about.

### 17. I cannot judge audio quality

Predicting MIDI from drum stems is fundamentally an *auditory* judgment
problem. "Does this predicted MIDI sound right?" can only be answered
by the user listening, or by metrics that approximate hearing
(F1 is one such approximation, but it doesn't capture timing precision
or velocity nuance the way ears do).

**Workaround**: pair every quantitative metric with a workflow that
renders the predicted MIDI to audio via fluidsynth + a drum soundfont,
so the user can quickly A/B against ground truth.

### 18. I cannot see PNG outputs

`visualizer.py` produces `alignment_check.png`. `visual_diagnostic.py`
produces `diagnostic_trace.png`. `velocity_analysis.py` produces
`velocity_audit.png`. These are the primary debug tools for the project.
I can write the code but cannot interpret the images.

**Workarounds**:
- Convert PNGs to ASCII art / text summaries (poor fidelity)
- Compute numerical statistics that the images are designed to surface
  (e.g., correlation between predicted-onset frames and ground-truth-
  onset frames — replaces "look at the alignment plot")
- Pair every PNG-generating script with a `.txt` companion that emits
  the key numerical features

### 19. I cannot run continuous training over days

Even with a `training-monitor` subagent, opencode sessions terminate.
Hours-long training runs require backgrounding via `tmux`/`screen`/
`systemd` on the user's machine or on rented compute.

**Workaround**: the `training-monitor` subagent gets a `--detach` mode
that ssh's into a remote host, starts the job in tmux, registers a
webhook on completion. (Requires SSH MCP or careful permission setup.)

### 20. I cannot dynamically adjust the strategy mid-training

When loss starts diverging at epoch 47, a human ML engineer might
lower LR, change batch size, restart. I can only do this in a fresh
session with explicit user request — not autonomously during a run.

**Workaround**: build the adaptive logic into `train.py` itself
(ReduceLROnPlateau already does some of this; gradient clipping with
adaptive bounds would extend it).

---

## Recommended adoption order

| Order | Item | Effort to add | Project impact |
|-------|------|---------------|----------------|
| 1 | Permission rules (#11) | 5 min | High — frictionless inside model-training/ |
| 2 | Skill `drum-transcription-debug` (#4) | 30 min | High — codifies the bisection workflow |
| 3 | Skill `mir-eval-drum-evaluation` (#5) | 20 min | High — every approach uses it |
| 4 | Commands `/smoke /eval /grid /train-stem` (#6) | 30 min | High — workflow compression |
| 5 | Subagent `training-monitor` (#1) | 1 hour | Highest — fixes the 120s timeout blocker |
| 6 | MCP: Context7 (#2) | 15 min | High — every modeling approach benefits |
| 7 | MCP: HuggingFace (#3) | 30 min | High — required by approaches 09/10/11 |
| 8 | Subagent `paper-summarizer` (#7) | 1 hour | Medium — only needed when porting |
| 9 | Subagent `dataset-statistician` (#8) | 2 hours | Medium — one-time + every new dataset |
| 10 | Subagent `checkpoint-inspector` (#9) | 1 hour | Medium — needed during debug |
| 11 | Plugin pre-commit hook (#10) | 1 hour | High value, low usage |
| 12+ | Tier B items | varies | Marginal |

**Total time to install items 1-7 (the high-impact set): ~3 hours.**
**Expected payoff: 30-50% faster iteration on every modeling approach.**

---

## How to install the high-impact set (concrete steps)

I can do all of these in this session given build mode. If you want me
to, say "set up the high-impact tooling" and I'll:

1. Create `.opencode/` directory + `opencode.json` (just for this project)
2. Add the permission rules (#11)
3. Create the two skills (`drum-transcription-debug`,
   `mir-eval-drum-evaluation`)
4. Create the `training-monitor` subagent
5. Add the 4 commands (`/smoke`, `/eval`, `/grid`, `/train-stem`)
6. Verify each is loaded by running `opencode config show` (or
   equivalent) — note that the user will need to restart opencode for
   config changes to take effect
7. Document in this file the actual file paths created

For the MCPs (Context7 + HuggingFace), I can write the JSON but the
user will need to:
1. Install Node/npx if not already installed (for Context7)
2. Obtain a HuggingFace token (free, `huggingface-cli login` or env var)
3. Restart opencode

---

## What about the 02-tooling-wishlist.md items?

That file (in this same directory) covered:
- Tier 1: 3 test harnesses you build yourself (mir_eval wrapper,
  synthetic data, inference reproduction test)
- Tier 2: external tools to add (Context7, arXiv, HuggingFace) —
  duplicated and expanded in this file
- Tier 3: compute options (Kaggle, Lambda, etc.)
- Tier 4: library upgrades (pretty_midi, madmom, lightning)
- Tier 5: papers to read

**Read both.** They are complementary:
- `02-tooling-wishlist.md`: what the user/developer needs in the
  *project* (libraries, data, compute)
- `02b-supercharge-the-agent.md` (this file): what the *agent* needs in
  its opencode configuration to be effective on this project

The agent-side tooling matters because it determines how much of the
project tooling I can actually use without dragging the user into every
loop.
