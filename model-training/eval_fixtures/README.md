# eval_fixtures/ — Self-contained evaluation fixtures for hybrid pipeline

This directory is the **self-contained, reproducible** material for the
planned hybrid (classical + neural) evaluation pipeline. It's
intentionally empty in the repo; the fixtures live in three
locations and this manifest is the index.

## Control-group rule (CRITICAL)

**Never use as a test reference:**
- `*_predicted_*.mid` files anywhere in the repo
- `model-training/<name>_v*_t*.mid` (model outputs from prior runs)
- `/tmp/drumtomidi/*_pred_*.mid` (overfit smoke test outputs)
- Any file produced by a model's forward pass (drift over time)

**Always use as a test reference:**
- The ground-truth MIDI files listed below
- Their matched `.wav` or pre-separated `.wav` stem files

## Fixture index

### Tier 1: e-GMD ground truth (also in `tests/fixtures/e-gmd/`)

5 paired .wav + .midi files, committed to git at
`tests/fixtures/e-gmd/`. Use for quick iteration during development.

- `tests/fixtures/e-gmd/202_rock-halftime_140_beat_4-4_16.{wav,midi}` — rock, 1334 notes over 380s
- `tests/fixtures/e-gmd/2_funk_80_beat_4-4_4.{wav,midi}`             — funk, 754 notes
- `tests/fixtures/e-gmd/5_jazz-funk_116_beat_4-4_56.{wav,midi}`        — jazz, 1899 notes
- `tests/fixtures/e-gmd/28_latin-samba_116_beat_4-4_17.{wav,midi}`     — latin, 3236 notes
- `tests/fixtures/e-gmd/81_dance-breakbeat_170_beat_4-4_37.{wav,midi}` — dance, 637 notes

The MIDI is ground truth. The WAV is the e-GMD mix (drums-only,
no other instruments). Full e-GMD metadata:
`/Volumes/1TB SSD 1/e-gmd-v1.0.0/drummer1/session1/`.

### Tier 2: user_files projects (NOT committed; on disk only)

5 user-authored projects. **NOT committed to git** because the stems
directory alone is ~700MB per project. They live at:

```
user_files/3 - Metallica_Cyanide_Drums/
user_files/old/16 - 01_Taylor_Swift_The_Fate_of_Ophelia_Drums/
user_files/old/17 - magic_man_part/
user_files/old/21 - Yes_Owner_of_a_Lonely_Heart_Drums/
user_files/old/22 - AC_DC_Thunderstruck_Drums/
```

Each project has:
- `stems/<name>-{kick,snare,hihat,toms,cymbals}.wav` — pre-separated by MDX23C
- `midi/<name>.mid` — ground truth MIDI authored by the user

The MIDI is **the user's ground truth** (not a model output). The
stems are derived (MDX23C separator output) but are legitimate
inputs to the per-stem pipeline.

### Tier 3: External e-GMD (NOT committed; on the drive)

The full e-GMD dataset at `/Volumes/1TB SSD 1/e-gmd-v1.0.0/` has
~11,000 matched .wav + .midi pairs in drummer1/session1/ alone. Use
for full-scale evaluation after the pipeline is validated on Tier 1
and Tier 2. The sampler `model-training/tools/sample_e_gmd_fixtures.py`
is the canonical way to deterministically select subsets.

## What goes in THIS directory (eval_fixtures/)

Nothing for the default install — the directory is a marker for git
plus this README. Eval code that consumes fixtures should reference
the Tier 1/2/3 paths directly, not this directory.

If you want a reproducible offline evaluation, you can populate this
directory with copies of the Tier 1 fixtures:
```bash
cp -r model-training/tests/fixtures/e-gmd/* model-training/eval_fixtures/
```

But for now, all callers should reference the canonical paths.

## Related

- `tests/fixtures/e-gmd/README.md` — fixture selection criteria and sampling code
- `model-training/tools/sample_e_gmd_fixtures.py` — the sampler
- `model-training/agent-plans/next-attempt/00-overview.md` — overall plan
- `model-training/agent-plans/next-attempt/01-critique-and-theories.md` — failure mode analysis (T1 channel collapse, T4 pos_weight)
