"""
Inference Post-Processor - Step 7 of Deep Learning Roadmap

Converts neural heatmap output back into a standard MIDI file.

Usage:
    python inference.py dl-1.wav --output output.mid
    
    python inference.py dl-1.wav --output output.mid --threshold 0.5
    
    python inference.py dl-1.wav --output output.mid --checkpoint models/smoke_test.ckpt
    
    python inference.py dl-1.wav --output output.mid --compare dl-1.mid
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import torch

from config import DEVICE, get_inference_config, INDEX_TO_MIDI, INDEX_TO_NAME
from feature_extractor import get_input_tensor
from inference_core import (
    heatmap_to_notes,
    group_notes,
    write_midi,
    prediction_stats,
    notes_by_class,
)
from model import DrumTranscriber


def run_inference(
    audio_path: str,
    output_path: str = None,
    checkpoint_path: str = None,
    threshold: float = None,
    device: str = None,
    compare_path: str = None,
) -> list:
    """
    Run inference on audio and optionally compare to ground truth.
    
    Returns:
        List of (time_seconds, midi_note, velocity) tuples
    """
    if threshold is None:
        threshold = get_inference_config().get('threshold', 0.8)
    
    if device is None:
        device = DEVICE
    
    print(f"Using device: {device}")
    
    # Auto-generate output path if not provided
    if output_path is None:
        audio_name = Path(audio_path).stem
        output_path = f"{Path(audio_path).parent}/{audio_name}_predicted.mid"

    output_path = Path(output_path)
    base = output_path.stem
    base_clean = re.sub(r'_t[0-9.]+$', '', base)
    ext = output_path.suffix
    parent = output_path.parent

    # CRITICAL: the output naming convention is <stem>_v<N>_t<threshold>.mid.
    # This suffix is the visual signal that distinguishes MODEL OUTPUT
    # from GROUND-TRUTH MIDI. Any test or evaluation that wants to use
    # a MIDI file as a reference MUST verify it does NOT match this
    # pattern; otherwise the test is contaminated. See
    # tests/fixtures/e-gmd/README.md for the control-group rule.
    counter = 1
    while True:
        import glob
        pattern = f"{base_clean}_v{counter}_t*"
        existing = glob.glob(str(parent / pattern))
        if existing or (parent / f"{base_clean}_v{counter}{ext}").exists():
            counter += 1
        else:
            break

    output_path = parent / f"{base_clean}_v{counter}_t{threshold}{ext}"
    
    print(f"Loading audio: {audio_path}")
    spec = get_input_tensor(audio_path)
    spec = spec.unsqueeze(0).to(device)
    print(f"  Input shape: {spec.shape}")
    
    # Load model
    model = DrumTranscriber().to(device)
    
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No checkpoint — producing random output.")
        print("Run smoke_test.py first to train and save a model.")
    
    # Run inference
    print(f"\nRunning inference with threshold={threshold}...")
    model.eval()
    with torch.no_grad():
        prediction = model(spec)
    
    print(f"  Prediction shape: {prediction.shape}")
    
    # Debug: per-class stats
    stats = prediction_stats(prediction)
    print(f"  Per-class prediction stats (max/mean):")
    for name, s in stats.items():
        print(f"    {name:<10}: max={s['max']:.4f}, mean={s['mean']:.4f}")
    
    # Convert to MIDI notes
    notes = heatmap_to_notes(prediction, threshold=threshold)
    print(f"  Detected {len(notes)} note events")
    
    # Per-class breakdown
    counts = notes_by_class(notes)
    print(f"\n  Per-class note counts:")
    print(f"  {'Class':<10} {'MIDI':>5} {'Count':>6}")
    print(f"  {'-'*22}")
    for class_idx in range(10):
        name = INDEX_TO_NAME[class_idx]
        midi = INDEX_TO_MIDI[class_idx]
        cnt = counts.get(name, 0)
        print(f"  {name:<10} {midi:>5} {cnt:>6}")
    
    # Extract tempo from ground truth if available
    bpm = 120.0
    if compare_path:
        try:
            from midi_shell import load_midi_file
            gt_midi = load_midi_file(compare_path)
            for track in gt_midi.tracks:
                for msg in track:
                    if msg.type == 'set_tempo':
                        bpm = 60_000_000 / msg.tempo
                        break
                break
        except Exception:
            pass
    
    # Write MIDI
    print(f"\nWriting MIDI: {output_path} (tempo: {bpm:.1f} BPM)")
    write_midi(notes, output_path, bpm=bpm)
    
    # Compare if ground truth provided
    if compare_path:
        print(f"\nComparing to ground truth: {compare_path}")
        metrics = compare_midi(notes, compare_path)
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1:        {metrics['f1']:.3f}")
        print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
        print(f"  Generated: {metrics['generated_count']}, Ground truth: {metrics['ground_truth_count']}")
        print(f"\n  Per-note breakdown:")
        print(f"  {'Note':<10} {'GT':>6} {'Gen':>6} {'Diff':>6}")
        print(f"  {'-'*30}")
        for name, data in sorted(metrics['note_breakdown'].items()):
            diff = data['gen'] - data['gt']
            sign = '+' if diff > 0 else ''
            print(f"  {name:<10} {data['gt']:>6} {data['gen']:>6} {sign}{diff:>5}")
        if metrics.get('tp_details'):
            print(f"\n  First 10 True Positives (GT time, Gen time, offset):")
            for tp in metrics['tp_details'][:10]:
                print(f"    {tp['name']:<10} GT {tp['gt']:.4f}s, Gen {tp['gen']:.4f}s, Offset: {tp['offset']:+.4f}s")
    
    return notes


def compare_midi(generated_notes: list, ground_truth_path: str, time_tolerance: float = 0.05) -> dict:
    """
    Compare generated notes against ground truth MIDI.
    """
    from midi_shell import load_midi_file
    from midi_core import extract_midi_notes_from_tracks, build_tempo_map_from_tracks
    
    midi_file = load_midi_file(ground_truth_path)
    tempo_map = build_tempo_map_from_tracks(midi_file.tracks, midi_file.ticks_per_beat)
    gt_notes, _ = extract_midi_notes_from_tracks(
        midi_file.tracks, midi_file.ticks_per_beat, tempo_map
    )
    
    raw_pitch_counts = Counter(n.midi_note for n in gt_notes)
    print(f"\n  [DEBUG] Raw GT pitch counts: {dict(sorted(raw_pitch_counts.items()))}")
    
    gt_raw = [(n.time, n.midi_note) for n in gt_notes]
    gt_grouped = group_notes([(t, p, 100) for t, p in gt_raw], time_tolerance)
    
    gen_grouped = group_notes(generated_notes, time_tolerance)
    
    pitch_aliases = {22: 42, 26: 46, 35: 36, 44: 42, 55: 49}
    
    note_names = {
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
    
    gt_set = set()
    for start, end, pitch in gt_grouped:
        t = start
        canonical = pitch_aliases.get(pitch, pitch)
        gt_set.add((t, canonical))
    
    gen_set = set()
    for start, end, pitch in gen_grouped:
        t = start
        gen_set.add((t, pitch))
    
    matched_gt = set()
    matched_gen = set()
    
    for gi, (g_start, g_end, g_pitch) in enumerate(gen_grouped):
        for gi2, (gt_start, gt_end, gt_pitch) in enumerate(gt_grouped):
            time_diff = abs(g_start - gt_start)
            if time_diff <= time_tolerance and g_pitch == gt_pitch:
                matched_gen.add(gi)
                matched_gt.add(gi2)
                break
    
    true_positives = len(matched_gt)
    false_positives = len(gen_grouped) - len(matched_gen)
    false_negatives = len(gt_grouped) - len(matched_gt)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    tp_details = []
    for gi, (g_start, g_end, g_pitch) in enumerate(gen_grouped):
        if gi in matched_gen:
            for gi2, (gt_start, gt_end, gt_pitch) in enumerate(gt_grouped):
                if gi2 in matched_gt and abs(g_start - gt_start) <= time_tolerance and g_pitch == gt_pitch:
                    name = note_names.get(g_pitch, f'Note({g_pitch})')
                    offset = gt_start - g_start
                    if len(tp_details) < 10:
                        tp_details.append({'name': name, 'pitch': g_pitch, 'gt': gt_start, 'gen': g_start, 'offset': offset})
                    break
    
    gt_by_class = Counter()
    for start, end, pitch in gt_grouped:
        name = note_names.get(pitch, f'Note({pitch})')
        gt_by_class[name] += 1
    
    gen_by_class = Counter()
    for start, end, pitch in gen_grouped:
        name = note_names.get(pitch, f'Note({pitch})')
        gen_by_class[name] += 1
    
    all_classes = sorted(set(gt_by_class.keys()) | set(gen_by_class.keys()))
    breakdown = {}
    for name in all_classes:
        breakdown[name] = {'gt': gt_by_class.get(name, 0), 'gen': gen_by_class.get(name, 0)}
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'generated_count': len(gen_grouped),
        'ground_truth_count': len(gt_grouped),
        'note_breakdown': breakdown,
        'tp_details': tp_details,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DrumToMIDI Inference Post-Processor')
    parser.add_argument('audio', nargs='?', help='Input audio file (.wav)')
    parser.add_argument('--output', '-o', help='Output MIDI file path')
    parser.add_argument('--checkpoint', '-c', help='Model checkpoint path')
    parser.add_argument('--threshold', '-t', type=float, default=None)
    parser.add_argument('--device', '-d', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--compare', help='Ground truth MIDI to compare against')
    parser.add_argument('--list', '-l', help='Batch file: audio.wav\\tmidi.mid\\t[output.mid]')
    
    args = parser.parse_args()
    
    if args.list:
        if not Path(args.list).exists():
            print(f"ERROR: List file not found: {args.list}")
            sys.exit(1)
        
        with open(args.list, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                if len(parts) < 2:
                    print(f"  Line {line_num}: Malformed, skipping")
                    continue
                
                audio_path = parts[0].strip()
                midi_path = parts[1].strip()
                output_path = parts[2].strip() if len(parts) > 2 else None
                
                print(f"\n=== File {line_num}: {audio_path} ===")
                
                if not Path(audio_path).exists():
                    print(f"  ERROR: Audio not found: {audio_path}")
                    continue
                
                compare_path = midi_path if Path(midi_path).exists() else None
                
                run_inference(
                    audio_path=audio_path,
                    output_path=output_path,
                    checkpoint_path=args.checkpoint,
                    threshold=args.threshold,
                    device=args.device,
                    compare_path=compare_path,
                )
        print(f"\n=== Batch complete ===")
        sys.exit(0)
    
    if not args.audio:
        parser.print_help()
        print("\n  Supply audio file, or use --list/-l for batch processing")
        sys.exit(1)
    
    run_inference(
        audio_path=args.audio,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        device=args.device,
        compare_path=args.compare,
    )
