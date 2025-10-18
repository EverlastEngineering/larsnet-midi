#!/usr/bin/env python3
"""
Debug script to analyze specific kick onset times and visualize what the spectral analysis sees.
This helps diagnose why certain kicks have very low energy values.
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_audio(path, sr=44100):
    """Load audio file."""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def analyze_kick_at_time(audio, sr, time_sec, window_sec=0.01, context_sec=0.2):
    """
    Analyze a specific kick at a given time with detailed diagnostics.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        time_sec: Time of the kick onset in seconds
        window_sec: Analysis window size (default 50ms)
        context_sec: Context window for visualization (default 200ms)
    """
    onset_sample = int(time_sec * sr)
    
    # Extract analysis window (what the algorithm sees)
    window_samples = int(window_sec * sr)
    analysis_segment = audio[onset_sample:onset_sample + window_samples]
    
    # Extract context window (for visualization)
    context_samples = int(context_sec * sr)
    context_start = max(0, onset_sample - context_samples // 2)
    context_end = min(len(audio), onset_sample + context_samples // 2)
    context_segment = audio[context_start:context_end]
    
    print(f"\n{'='*80}")
    print(f"ANALYZING KICK AT {time_sec:.3f} seconds")
    print(f"{'='*80}")
    
    # 1. Time-domain analysis
    print(f"\nTime Domain Analysis:")
    print(f"  Onset sample: {onset_sample}")
    print(f"  Analysis window: {window_sec*1000:.1f}ms ({window_samples} samples)")
    print(f"  Peak amplitude in window: {np.max(np.abs(analysis_segment)):.4f}")
    print(f"  RMS in window: {np.sqrt(np.mean(analysis_segment**2)):.4f}")
    
    # Find the actual peak in the context window
    peak_sample_rel = np.argmax(np.abs(context_segment))
    peak_sample_abs = context_start + peak_sample_rel
    peak_time = peak_sample_abs / sr
    time_diff_ms = (peak_time - time_sec) * 1000
    
    print(f"\n  Peak in context window:")
    print(f"    Location: {peak_time:.3f}s (sample {peak_sample_abs})")
    print(f"    Time difference from onset: {time_diff_ms:+.1f}ms")
    print(f"    Peak amplitude: {np.abs(context_segment[peak_sample_rel]):.4f}")
    
    if abs(time_diff_ms) > 10:
        print(f"  ⚠️  WARNING: Peak is {abs(time_diff_ms):.1f}ms away from onset!")
        print(f"      This suggests onset detection may be off.")
    
    # 2. Spectral analysis
    if len(analysis_segment) < 100:
        print(f"\n  ✗ ERROR: Analysis segment too short ({len(analysis_segment)} samples)")
        return
    
    fft = np.fft.rfft(analysis_segment)
    freqs = np.fft.rfftfreq(len(analysis_segment), 1/sr)
    magnitude = np.abs(fft)
    
    # Calculate energies in kick frequency ranges
    ranges = {
        'Fundamental (40-80Hz)': (40, 80),
        'Body (80-250Hz)': (80, 250),
        'Attack (1500-5000Hz)': (1500, 5000),
    }
    
    print(f"\nSpectral Energy Analysis:")
    energies = {}
    for name, (min_hz, max_hz) in ranges.items():
        mask = (freqs >= min_hz) & (freqs < max_hz)
        energy = np.sum(magnitude[mask])
        energies[name] = energy
        print(f"  {name:30s}: {energy:8.1f}")
    
    # Calculate geomean
    fund_e = energies['Fundamental (40-80Hz)']
    body_e = energies['Body (80-250Hz)']
    attack_e = energies['Attack (1500-5000Hz)']
    geomean = np.cbrt(fund_e * body_e * attack_e)
    
    print(f"\n  3-way GeoMean: {geomean:.1f}")
    if geomean < 70:
        print(f"  ✗ REJECTED (threshold: 70.0)")
    else:
        print(f"  ✓ KEPT (threshold: 70.0)")
    
    # 3. Spectral distribution
    total_energy = np.sum(magnitude)
    print(f"\nSpectral Distribution:")
    for name, energy in energies.items():
        pct = (energy / total_energy * 100) if total_energy > 0 else 0
        print(f"  {name:30s}: {pct:5.1f}%")
    
    # Find dominant frequency
    peak_freq_idx = np.argmax(magnitude)
    peak_freq = freqs[peak_freq_idx]
    print(f"\n  Dominant frequency: {peak_freq:.1f} Hz")
    
    # 4. Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Time domain with context
    ax1 = axes[0]
    context_time = np.arange(len(context_segment)) / sr + context_start / sr
    ax1.plot(context_time, context_segment, 'b-', linewidth=0.5, label='Audio')
    
    # Mark the onset
    ax1.axvline(time_sec, color='r', linestyle='--', linewidth=2, label=f'Onset Detection ({time_sec:.3f}s)')
    
    # Mark the analysis window
    analysis_end = time_sec + window_sec
    ax1.axvspan(time_sec, analysis_end, alpha=0.2, color='yellow', label=f'Analysis Window ({window_sec*1000:.0f}ms)')
    
    # Mark the peak
    ax1.axvline(peak_time, color='g', linestyle=':', linewidth=2, label=f'Peak ({peak_time:.3f}s)')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Time Domain: Kick at {time_sec:.3f}s (GeoMean={geomean:.1f})')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Spectrum
    ax2 = axes[1]
    ax2.plot(freqs, magnitude, 'b-', linewidth=1)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Frequency Spectrum (Analysis Window)')
    ax2.set_xlim(0, 8000)
    
    # Mark frequency ranges
    for name, (min_hz, max_hz) in ranges.items():
        ax2.axvspan(min_hz, max_hz, alpha=0.15, label=name)
    
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Energy bars
    ax3 = axes[2]
    names = list(ranges.keys())
    values = [energies[name] for name in names]
    colors = ['blue', 'green', 'red']
    bars = ax3.bar(names, values, color=colors, alpha=0.6)
    
    # Add geomean line
    ax3.axhline(geomean, color='black', linestyle='--', linewidth=2, label=f'GeoMean={geomean:.1f}')
    ax3.axhline(70, color='orange', linestyle=':', linewidth=2, label='Threshold=70')
    
    ax3.set_ylabel('Energy')
    ax3.set_title('Spectral Energy by Frequency Range')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Rotate x labels
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    plt.tight_layout()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Debug kick onset detection and spectral analysis')
    parser.add_argument('audio_file', help='Path to kick audio file (separated stem)')
    parser.add_argument('times', nargs='+', type=float, help='Onset times to analyze (in seconds)')
    parser.add_argument('--window', type=float, default=0.05, help='Analysis window size in seconds (default: 0.05)')
    parser.add_argument('--context', type=float, default=0.2, help='Context window for visualization (default: 0.2)')
    parser.add_argument('--output', help='Output directory for plots (default: show interactive)')
    
    args = parser.parse_args()
    
    # Load audio
    print(f"Loading audio: {args.audio_file}")
    audio = load_audio(args.audio_file)
    sr = 44100
    
    print(f"Audio loaded: {len(audio)} samples, {len(audio)/sr:.2f} seconds")
    
    # Analyze each time
    for i, time_sec in enumerate(args.times):
        fig = analyze_kick_at_time(audio, sr, time_sec, args.window, args.context)
        
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f'kick_debug_{time_sec:.3f}s.png'
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"\nSaved plot to: {output_file}")
            plt.close(fig)
        else:
            plt.show()


if __name__ == '__main__':
    main()
