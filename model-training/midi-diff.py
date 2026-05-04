"""
midi-diff.py — Compare two MIDI files and show differences.

Usage:
    python midi-diff.py output.mid truth.mid
    python midi-diff.py output.mid truth.mid --output /tmp/diff.png
    python midi-diff.py output.mid truth.mid --tolerance 0.1

Output:
    - Text table to stdout: time, class, velocity, diff
    - Visual timeline plot saved to PNG (default: visualizer/midi_diff.png)
"""

import argparse
from collections import Counter
from pathlib import Path

from inference_core import group_notes
from io_utils import get_models_dir


# Note name lookup including aliases
NOTE_NAMES = {
    # Canonical (primary MIDI) - alphabetical by name
    36: 'Kick', 35: 'Kick',
    38: 'Snare', 40: 'Snare', 37: 'Snare', 39: 'Snare',
    42: 'HHC', 44: 'HHC', 22: 'HHC',
    46: 'HHO', 26: 'HHO',
    49: 'Crash1', 55: 'Crash1',
    57: 'Crash2', 52: 'Crash2',
    51: 'Ride', 53: 'Ride', 59: 'Ride',
    50: 'TomHigh', 48: 'TomHigh',
    47: 'TomMid', 45: 'TomMid',
    43: 'TomLow', 58: 'TomLow', 41: 'TomLow',
}

# Pitch aliases for canonical comparison
PITCH_ALIASES = {22: 42, 26: 46, 35: 36, 44: 42, 55: 49}


def load_midi_notes(path: str):
    """Load notes from a MIDI file, returning (time, pitch) list."""
    from midi_shell import load_midi_file
    from midi_core import extract_midi_notes_from_tracks, build_tempo_map_from_tracks
    
    midi_file = load_midi_file(path)
    tempo_map = build_tempo_map_from_tracks(midi_file.tracks, midi_file.ticks_per_beat)
    notes, _ = extract_midi_notes_from_tracks(
        midi_file.tracks, midi_file.ticks_per_beat, tempo_map
    )
    return [(n.time, n.midi_note) for n in notes]


def compute_diff(gen_path: str, gt_path: str, time_tolerance: float = 0.05):
    """
    Compute diff between two MIDI files.
    
    Returns:
        dict with metrics and per-note details
    """
    gen_raw = load_midi_notes(gen_path)
    gt_raw = load_midi_notes(gt_path)
    
    # Group notes
    gen_grouped = group_notes([(t, p, 100) for t, p in gen_raw], time_tolerance)
    gt_grouped = group_notes([(t, p, 100) for t, p in gt_raw], time_tolerance)
    
    # Match with tolerance
    matched_gen = set()
    matched_gt = set()
    
    for gi, (g_start, g_end, g_pitch) in enumerate(gen_grouped):
        for gi2, (gt_start, gt_end, gt_pitch) in enumerate(gt_grouped):
            if abs(g_start - gt_start) <= time_tolerance and g_pitch == gt_pitch:
                matched_gen.add(gi)
                matched_gt.add(gi2)
                break
    
    tp = len(matched_gt)
    fp = len(gen_grouped) - len(matched_gen)
    fn = len(gt_grouped) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Per-class breakdown
    gt_by_class = Counter()
    for start, end, pitch in gt_grouped:
        name = NOTE_NAMES.get(pitch, f'Note({pitch})')
        gt_by_class[name] += 1
    
    gen_by_class = Counter()
    for start, end, pitch in gen_grouped:
        name = NOTE_NAMES.get(pitch, f'Note({pitch})')
        gen_by_class[name] += 1
    
    all_classes = sorted(set(gt_by_class.keys()) | set(gen_by_class.keys()))
    breakdown = {name: {'gt': gt_by_class.get(name, 0), 'gen': gen_by_class.get(name, 0)} for name in all_classes}
    
    # Per-note details for first 20 matches
    tp_details = []
    for gi, (g_start, g_end, g_pitch) in enumerate(gen_grouped):
        if gi in matched_gen and len(tp_details) < 20:
            for gi2, (gt_start, gt_end, gt_pitch) in enumerate(gt_grouped):
                if gi2 in matched_gt and abs(g_start - gt_start) <= time_tolerance and g_pitch == gt_pitch:
                    name = NOTE_NAMES.get(g_pitch, f'Note({g_pitch})')
                    tp_details.append({
                        'name': name,
                        'pitch': g_pitch,
                        'gt': gt_start,
                        'gen': g_start,
                        'offset': gt_start - g_start,
                    })
                    break
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'generated_count': len(gen_grouped),
        'ground_truth_count': len(gt_grouped),
        'note_breakdown': breakdown,
        'tp_details': tp_details,
        'gen_grouped': gen_grouped,
        'gt_grouped': gt_grouped,
    }


def print_text_diff(metrics: dict):
    """Print a text table diff to stdout."""
    print("\n" + "=" * 50)
    print("MIDI Comparison Results")
    print("=" * 50)
    print(f"  Precision:  {metrics['precision']:.3f}")
    print(f"  Recall:     {metrics['recall']:.3f}")
    print(f"  F1:         {metrics['f1']:.3f}")
    print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
    print(f"  Generated:  {metrics['generated_count']}")
    print(f"  Ground truth: {metrics['ground_truth_count']}")
    
    print(f"\n  Per-note breakdown:")
    print(f"  {'Note':<12} {'GT':>6} {'Gen':>6} {'Diff':>7}")
    print(f"  {'-' * 33}")
    for name, data in sorted(metrics['note_breakdown'].items()):
        diff = data['gen'] - data['gt']
        sign = '+' if diff > 0 else ''
        print(f"  {name:<12} {data['gt']:>6} {data['gen']:>6} {sign}{diff:>6}")
    
    if metrics.get('tp_details'):
        print(f"\n  First {len(metrics['tp_details'])} True Positives:")
        print(f"  {'Note':<12} {'GT':>8} {'Gen':>8} {'Offset':>8}")
        print(f"  {'-' * 40}")
        for tp in metrics['tp_details']:
            print(f"  {tp['name']:<12} {tp['gt']:>8.4f} {tp['gen']:>8.4f} {tp['offset']:>+8.4f}")


def plot_timeline(metrics: dict, output_path: Path = None):
    """
    Plot overlaid timeline of generated vs ground truth notes.
    Saves to PNG.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    if output_path is None:
        output_path = get_models_dir().parent / "visualizer" / "midi_diff.png"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    gt_grouped = metrics['gt_grouped']
    gen_grouped = metrics['gen_grouped']
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Plot ground truth in blue
    for start, end, pitch in gt_grouped:
        name = NOTE_NAMES.get(pitch, f'{pitch}')
        ax.barh(name, end - start, left=start, height=0.4, color='blue', alpha=0.5, align='center')
    
    # Plot generated in red (offset for visibility)
    for start, end, pitch in gen_grouped:
        name = NOTE_NAMES.get(pitch, f'{pitch}')
        ax.barh(name, end - start, left=start, height=0.3, color='red', alpha=0.5, align='center')
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Drum Class')
    ax.set_title('MIDI Comparison: Blue=GT, Red=Generated')
    ax.legend(['Ground Truth', 'Generated'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nTimeline plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare two MIDI files')
    parser.add_argument('generated', help='Generated MIDI file')
    parser.add_argument('ground_truth', help='Ground truth MIDI file')
    parser.add_argument('--output', '-o', help='Output PNG path for timeline plot')
    parser.add_argument('--tolerance', '-t', type=float, default=0.05,
                        help='Time tolerance in seconds (default: 0.05)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip visual plot generation')
    
    args = parser.parse_args()
    
    if not Path(args.generated).exists():
        print(f"ERROR: Generated file not found: {args.generated}")
        exit(1)
    if not Path(args.ground_truth).exists():
        print(f"ERROR: Ground truth file not found: {args.ground_truth}")
        exit(1)
    
    print(f"Comparing:")
    print(f"  Generated:      {args.generated}")
    print(f"  Ground truth:   {args.ground_truth}")
    print(f"  Tolerance:      {args.tolerance}s")
    
    metrics = compute_diff(args.generated, args.ground_truth, time_tolerance=args.tolerance)
    
    print_text_diff(metrics)
    
    if not args.no_plot:
        plot_timeline(metrics, output_path=args.output)
