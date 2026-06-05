"""
Drum MIDI evaluation using mir_eval.transcription.

Standardized evaluator for the model-training drum transcription
project. Replaces the hand-rolled comparator in inference.py with
the canonical mir_eval implementation, so numbers are paper-comparable.

Usage:
    python tools/eval_with_mir_eval.py --pred predicted.mid --gt truth.mid
    python tools/eval_with_mir_eval.py --pred pred.mid --gt truth.mid --tolerance 0.020

See: .opencode/skills/mir-eval-drum-evaluation/SKILL.md
"""

import argparse
from pathlib import Path

import numpy as np
import mir_eval
import pretty_midi


INDEX_TO_MIDI = {0: 36, 1: 38, 2: 42, 3: 46, 4: 50, 5: 47,
                 6: 43, 7: 49, 8: 57, 9: 51}
INDEX_TO_NAME = {0: 'Kick', 1: 'Snare', 2: 'HHC', 3: 'HHO',
                 4: 'TomHigh', 5: 'TomMid', 6: 'TomLow',
                 7: 'Crash1', 8: 'Crash2', 9: 'Ride'}
PITCH_ALIASES = {35: 36, 22: 42, 44: 42, 26: 46, 48: 50, 45: 47,
                 58: 43, 41: 43, 55: 49, 52: 57, 53: 51, 59: 51,
                 37: 38, 40: 38, 39: 38}
ONSET_TOLERANCE_S = 0.05
PITCH_TOLERANCE = 0.5


def midi_to_arrays(pm, apply_aliases=True):
    intervals, pitches = [], []
    for inst in pm.instruments:
        for note in inst.notes:
            p = note.pitch
            if apply_aliases:
                p = PITCH_ALIASES.get(p, p)
            intervals.append([note.start, note.end])
            pitches.append(float(p))
    if not intervals:
        return np.zeros((0, 2)), np.zeros((0,))
    return np.array(intervals), np.array(pitches)


def evaluate(pred_path, gt_path, onset_tolerance=ONSET_TOLERANCE_S):
    pred = pretty_midi.PrettyMIDI(str(pred_path))
    gt = pretty_midi.PrettyMIDI(str(gt_path))
    p_int, p_pit = midi_to_arrays(pred)
    g_int, g_pit = midi_to_arrays(gt)

    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        g_int, g_pit, p_int, p_pit,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=PITCH_TOLERANCE,
        offset_ratio=None,
    )

    per_class = {}
    for cls_idx, midi_pitch in INDEX_TO_MIDI.items():
        g_mask = g_pit == float(midi_pitch)
        p_mask = p_pit == float(midi_pitch)
        if not g_mask.any() and not p_mask.any():
            continue
        cp, cr, cf, _ = mir_eval.transcription.precision_recall_f1_overlap(
            g_int[g_mask], g_pit[g_mask],
            p_int[p_mask], p_pit[p_mask],
            onset_tolerance=onset_tolerance,
            pitch_tolerance=PITCH_TOLERANCE,
            offset_ratio=None,
        )
        per_class[INDEX_TO_NAME[cls_idx]] = {
            'precision': cp, 'recall': cr, 'f1': cf,
            'gt_count': int(g_mask.sum()),
            'pred_count': int(p_mask.sum()),
        }
    return {
        'overall': {'precision': p, 'recall': r, 'f1': f,
                    'gt_count': len(g_int), 'pred_count': len(p_int)},
        'per_class': per_class,
    }


def print_report(result, onset_tolerance):
    o = result['overall']
    print(f"\n=== Overall (onset_tolerance={onset_tolerance*1000:.0f}ms) ===")
    print(f"  Precision : {o['precision']:.3f}")
    print(f"  Recall    : {o['recall']:.3f}")
    print(f"  F1        : {o['f1']:.3f}")
    print(f"  GT notes  : {o['gt_count']}")
    print(f"  Pred notes: {o['pred_count']}")

    print(f"\n=== Per-class ===")
    print(f"  {'Class':<10} {'P':>6} {'R':>6} {'F1':>6} {'GT':>6} {'Pred':>6}")
    print(f"  {'-'*46}")
    for name in INDEX_TO_NAME.values():
        if name not in result['per_class']:
            continue
        c = result['per_class'][name]
        flag = '  ⚠' if c['recall'] == 0 and c['gt_count'] > 0 else ''
        print(f"  {name:<10} {c['precision']:>6.3f} {c['recall']:>6.3f} {c['f1']:>6.3f} {c['gt_count']:>6} {c['pred_count']:>6}{flag}")

    rare_zero = [n for n, c in result['per_class'].items()
                 if c['recall'] == 0 and c['gt_count'] > 0]
    if rare_zero:
        print(f"\n  ⚠ Classes with recall=0 (likely class-imbalance T4): {', '.join(rare_zero)}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='Predicted MIDI file path')
    ap.add_argument('--gt', required=True, help='Ground-truth MIDI file path')
    ap.add_argument('--tolerance', type=float, default=ONSET_TOLERANCE_S,
                    help='Onset tolerance in seconds (default 0.05)')
    args = ap.parse_args()

    result = evaluate(args.pred, args.gt, onset_tolerance=args.tolerance)
    print_report(result, args.tolerance)
