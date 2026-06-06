"""
Deterministic sampler for e-GMD control-group fixtures.

Picks 5 matched .wav + .midi pairs from /Volumes/1TB SSD 1/e-gmd-v1.0.0/drummer1/session1/
using a fixed numpy seed so the selection is reproducible.

Selection criteria:
  - one file per genre subset if possible (rock, funk, jazz, latin, hiphop)
  - non-trivial note count (>= 100 notes in the MIDI)
  - deterministic via numpy random with seed 42

Output:
  - tests/fixtures/e-gmd/selection.json: machine-readable list of selected files
  - prints the chosen files so the README can include them

Run: conda run -n drumtomidi python tools/sample_e_gmd_fixtures.py
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pretty_midi

E_GMD_ROOT = Path("/Volumes/1TB SSD 1/e-gmd-v1.0.0/drummer1/session1")
SEED = 42
TARGET_COUNT = 5
MIN_NOTES = 100
GENRES = ["rock", "funk", "jazz", "latin", "hiphop"]  # preferred order; one of each if possible


def note_count(midi_path: Path) -> int:
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        return sum(len(i.notes) for i in pm.instruments)
    except Exception:
        return 0


def main():
    if not E_GMD_ROOT.exists():
        print(f"ERROR: e-GMD root not found: {E_GMD_ROOT}", file=sys.stderr)
        print("Set E_GMD_ROOT env var or pass path as arg if your drive is different.",
              file=sys.stderr)
        sys.exit(1)

    # Enumerate all matched pairs
    all_midi = sorted(E_GMD_ROOT.glob("*.midi"))
    pairs = []
    for midi in all_midi:
        wav = midi.with_suffix(".wav")
        if not wav.exists():
            continue
        n = note_count(midi)
        if n < MIN_NOTES:
            continue
        pairs.append({"wav": str(wav), "midi": str(midi), "notes": n,
                       "genre_hint": _guess_genre(midi.name)})

    print(f"Found {len(pairs)} candidate pairs with >= {MIN_NOTES} notes", file=sys.stderr)

    # Bucket by genre hint
    by_genre = {g: [] for g in GENRES}
    for p in pairs:
        if p["genre_hint"] in by_genre:
            by_genre[p["genre_hint"]].append(p)

    # Strategy: pick one from each genre deterministically, fill if needed
    rng = np.random.default_rng(SEED)
    selected = []
    for g in GENRES:
        bucket = by_genre[g]
        if not bucket:
            print(f"  WARNING: no candidates in genre={g}", file=sys.stderr)
            continue
        pick_idx = int(rng.integers(0, len(bucket)))
        selected.append(bucket[pick_idx])
    # Fill if we have fewer than TARGET_COUNT
    if len(selected) < TARGET_COUNT:
        remaining = [p for p in pairs if p not in selected]
        rng.shuffle(remaining)
        for p in remaining:
            if len(selected) >= TARGET_COUNT:
                break
            if p not in selected:
                selected.append(p)

    selected = selected[:TARGET_COUNT]

    # Write selection.json
    out = {
        "seed": SEED,
        "min_notes": MIN_NOTES,
        "candidates": len(pairs),
        "selected": [
            {"wav": Path(p["wav"]).name, "midi": Path(p["midi"]).name,
             "notes": p["notes"], "genre_hint": p["genre_hint"]}
            for p in selected
        ],
    }
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "e-gmd"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    sel_path = fixtures_dir / "selection.json"
    sel_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {sel_path}")

    # Also print to stdout for the README
    print("\n=== SELECTED CONTROL GROUP FIXTURES ===")
    for s in out["selected"]:
        print(f"  {s['genre_hint']:7s} | {s['notes']:5d} notes | {s['wav']}")


def _guess_genre(filename: str) -> str:
    """e-GMD filenames embed genre-prefix as '_<genre>_<bpm>_...'.
    Genres are often hyphenated (e.g. rock-halftime, latin-brazilian-baiao),
    so match on the prefix before the first underscore-after-genre."""
    for g in GENRES:
        # _rock_ matches in _rock_86_ but not in _rock-halftime_
        if f"_{g}_" in filename:
            return g
        # Also match hyphenated: _rock-halftime_ matches "rock" prefix
        if f"_{g}-" in filename:
            return g
    return "other"


if __name__ == "__main__":
    main()
