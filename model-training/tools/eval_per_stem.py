"""
Per-stem MIDI evaluation.

Standard mir_eval requires exact pitch matches, but the per-stem
architecture emits one canonical pitch per stem (e.g. all hihat hits
become MIDI 42, even if the original was 46=open or 26=pedal). So we
need a stem-level evaluation that ignores pitch mismatches within a
stem and asks: "did we detect a hit in this stem at the right time?"

For each stem, find matching predicted-to-gt note pairs within ±50ms
tolerance. Predicted pitch is collapsed to its stem; gt pitch is
collapsed to its stem. Then compute precision/recall/F1 per stem.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pretty_midi

MT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(MT_DIR))

from datasets.per_stem import STEM_PITCHES, STEM_CANONICAL_PITCH  # noqa: E402


# Reverse map: MIDI pitch -> stem
PITCH_TO_STEM = {}
for stem, pitches in STEM_PITCHES.items():
    for p in pitches:
        PITCH_TO_STEM[p] = stem


def midi_to_stem_notes(midi_path: str) -> list:
    """
    Return list of (time_seconds, stem_name) tuples.
    Velocity discarded; pitch mapped to stem.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    out = []
    for inst in pm.instruments:
        for note in inst.notes:
            stem = PITCH_TO_STEM.get(note.pitch)
            if stem is None:
                continue  # not a drum note (e.g. pitch 27 anchor)
            out.append((note.start, stem))
    out.sort()
    return out


def per_stem_f1(
    pred_notes: list,  # [(time, stem), ...]
    gt_notes: list,    # [(time, stem), ...]
    tolerance_s: float = 0.05,
) -> dict:
    """
    Per-stem precision/recall/F1.
    Pred and gt notes have stem labels (not pitches).
    """
    results = {}
    for stem in STEM_PITCHES:
        pred_times = [t for t, s in pred_notes if s == stem]
        gt_times = [t for t, s in gt_notes if s == stem]

        pred_arr = np.array(pred_times) if pred_times else np.zeros(0)
        gt_arr = np.array(gt_times) if gt_times else np.zeros(0)

        if len(pred_arr) == 0 and len(gt_arr) == 0:
            continue
        if len(pred_arr) == 0:
            results[stem] = {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                             "pred_count": 0, "gt_count": len(gt_arr)}
            continue
        if len(gt_arr) == 0:
            results[stem] = {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                             "pred_count": len(pred_arr), "gt_count": 0}
            continue

        # Greedy matching: for each pred, find nearest unmatched gt in same stem
        matched_pred = set()
        matched_gt = set()
        for i, pt in enumerate(pred_arr):
            best_d = tolerance_s + 1
            best_j = -1
            for j, gt in enumerate(gt_arr):
                if j in matched_gt:
                    continue
                d = abs(pt - gt)
                if d <= tolerance_s and d < best_d:
                    best_d = d
                    best_j = j
            if best_j >= 0:
                matched_pred.add(i)
                matched_gt.add(best_j)

        tp = len(matched_pred)
        fp = len(pred_arr) - tp
        fn = len(gt_arr) - tp
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f = 2 * p * r / (p + r) if (p + r) else 0
        results[stem] = {"precision": p, "recall": r, "f1": f,
                         "pred_count": len(pred_arr), "gt_count": len(gt_arr),
                         "tp": tp, "fp": fp, "fn": fn}
    return results


def evaluate(pred_midi: str, gt_midi: str, tolerance_s: float = 0.05) -> None:
    pred_notes = midi_to_stem_notes(pred_midi)
    gt_notes = midi_to_stem_notes(gt_midi)
    print(f"\n=== Per-stem evaluation (onset tolerance {tolerance_s*1000:.0f}ms) ===")
    print(f"Pred notes (mapped to stem): {len(pred_notes)}")
    print(f"GT   notes (mapped to stem): {len(gt_notes)}")

    results = per_stem_f1(pred_notes, gt_notes, tolerance_s)
    print(f"\n{'Stem':<8} {'P':>6} {'R':>6} {'F1':>6} {'Pred':>5} {'GT':>5} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 60)
    for stem, r in results.items():
        print(f"{stem:<8} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} "
              f"{r['pred_count']:>5} {r['gt_count']:>5} {r.get('tp',0):>5} "
              f"{r.get('fp',0):>5} {r.get('fn',0):>5}")

    if results:
        f1s = [r["f1"] for r in results.values() if r["gt_count"] > 0]
        if f1s:
            print(f"\n  Macro F1 (over present stems): {sum(f1s)/len(f1s):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()
    evaluate(args.pred, args.gt, args.tolerance)
