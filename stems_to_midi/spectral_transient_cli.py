#!/usr/bin/env python
"""CLI: detect onsets in an audio file using the spectral transient
method (per-band power profile with max/median ratio peak-picking).

Usage:
    python -m stems_to_midi.spectral_transient_cli <wav>
    python -m stems_to_midi.spectral_transient_cli <wav> --out events.json
    python -m stems_to_midi.spectral_transient_cli <wav> --min-band-ratio 3.0

Compares to the existing energy-based detector if --compare flag is
passed (requires the toms analysis.json in the same project dir).
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from stems_to_midi.spectral_transient_core import (
    DEFAULT_BANDS,
    SpectralTransientConfig,
    detect_spectral_transients,
)


def _maybe_compare_to_existing(wav_path: Path, spectral_events):
    """If an analysis.json with events_configured exists in the same
    project, compute the match stats and return a printable summary.
    """
    # Walk up from the wav. The project root is the dir whose
    # parent name is "user_files".
    cur = wav_path.parent
    project_root = None
    for _ in range(5):
        if cur.parent.name == "user_files":
            project_root = cur
            break
        cur = cur.parent
    if project_root is None:
        return None
    # Look for any analysis.json under midi/ or analysis/
    for sub in ("midi", "analysis"):
        sub_dir = project_root / sub
        if not sub_dir.is_dir():
            continue
        candidates = list(sub_dir.glob("*.analysis.json"))
        if candidates:
            with open(candidates[0]) as f:
                ana = json.load(f)
            configured = ana.get("events_configured") or ana.get("events") or []
            return _compare(spectral_events, configured, candidates[0])
    return None


def _compare(spectral_events, existing_events, source_label):
    """For each spectral event, find the nearest existing event. Print
    match stats. Both event lists are dicts with time_sec or time keys.
    """
    def t_of(e):
        return e.get("time_sec", e.get("time", 0.0))

    if not spectral_events or not existing_events:
        return None

    spec_times = np.array([t_of(e) for e in spectral_events])
    exist_times = np.array([t_of(e) for e in existing_events])
    exist_status = [e.get("status", e.get("classification", "?")) for e in existing_events]

    # For each spectral event, find nearest existing event
    matches = []
    for i, st in enumerate(spec_times):
        j = int(np.argmin(np.abs(exist_times - st)))
        diff_ms = (exist_times[j] - st) * 1000.0
        matches.append((st, exist_times[j], exist_status[j], diff_ms))

    return {
        "source_analysis": str(source_label),
        "n_spectral": len(spectral_events),
        "n_existing": len(existing_events),
        "n_matched_within_30ms": sum(1 for m in matches if abs(m[3]) < 30),
        "n_matched_within_50ms": sum(1 for m in matches if abs(m[3]) < 50),
        "n_matched_within_100ms": sum(1 for m in matches if abs(m[3]) < 100),
        "n_existing_unmatched": len(existing_events) - sum(1 for m in matches if abs(m[3]) < 100),
        "median_diff_ms": float(np.median([abs(m[3]) for m in matches])),
        "matches": [
            {
                "spectral_time_sec": float(s),
                "existing_time_sec": float(e),
                "existing_status": st,
                "diff_ms": float(d),
            }
            for (s, e, st, d) in matches
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("wav", help="Path to audio file (wav, flac, etc.)")
    parser.add_argument("--out", help="Write events to JSON file")
    parser.add_argument("--n-fft", type=int, default=1024, help="FFT size (default 1024)")
    parser.add_argument("--hop", type=int, default=256, help="Hop size (default 256)")
    parser.add_argument("--min-band-ratio", type=float, default=2.0,
                        help="Min top/median band ratio to count as a hit (default 2.0)")
    parser.add_argument("--min-spacing-ms", type=float, default=100.0,
                        help="Min peak spacing (default 100ms)")
    parser.add_argument("--prominence", type=float, default=2.0,
                        help="Min prominence in band-ratio units (default 2.0)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare to existing energy-based detector events")
    args = parser.parse_args()

    wav_path = Path(args.wav)
    if not wav_path.exists():
        print(f"error: {wav_path} not found", file=sys.stderr)
        sys.exit(1)

    y, sr = sf.read(str(wav_path), always_2d=True)
    y = y.mean(axis=1)

    cfg = SpectralTransientConfig(
        n_fft=args.n_fft, hop=args.hop,
        bands=DEFAULT_BANDS,
        min_band_ratio=args.min_band_ratio,
        min_peak_spacing_ms=args.min_spacing_ms,
        prominence=args.prominence,
    )
    events, debug = detect_spectral_transients(y, sr, config=cfg)

    print(f"Detected {len(events)} spectral transient events in {wav_path}")
    print(f"  sr={sr}, duration={len(y)/sr:.2f}s, config={cfg}")
    for e in events:
        bp_str = ", ".join(f"{x:.2e}" for x in e.band_powers)
        print(f"    t={e.time_sec:7.3f}s  top=B{e.band_max_idx}  "
              f"ratio={e.band_max_ratio:7.2f}  bp=[{bp_str}]")

    if args.compare:
        cmp = _maybe_compare_to_existing(wav_path, [asdict(e) for e in events])
        if cmp is None:
            print("\nNo existing analysis.json found in project dir for comparison.")
        else:
            print(f"\nComparison to {cmp['source_analysis']}:")
            print(f"  spectral events: {cmp['n_spectral']}, existing events: {cmp['n_existing']}")
            print(f"  matched <30ms:   {cmp['n_matched_within_30ms']}")
            print(f"  matched <50ms:   {cmp['n_matched_within_50ms']}")
            print(f"  matched <100ms:  {cmp['n_matched_within_100ms']}")
            print(f"  existing unmatched (>100ms from any spectral): {cmp['n_existing_unmatched']}")
            print(f"  median |diff|:   {cmp['median_diff_ms']:.1f}ms")

    if args.out:
        out = {
            "source": str(wav_path),
            "sr": sr,
            "config": asdict(cfg),
            "events": [asdict(e) for e in events],
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=list)
        print(f"\nWrote {len(events)} events to {args.out}")


if __name__ == "__main__":
    main()
