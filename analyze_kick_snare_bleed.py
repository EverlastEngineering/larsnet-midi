#!/usr/bin/env python3
"""
Analyze snare bleed in kick track.
Strategy: Detect when snares hit, then check if there's suspicious snare-like energy in the kick track at those times.
"""

import argparse
import numpy as np
import librosa
import scipy.signal
from pathlib import Path
import matplotlib.pyplot as plt


def load_audio(path, sr=44100):
    """Load audio file."""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def detect_onsets(audio, sr, instrument_type='general'):
    """
    Detect onset times using librosa.
    Returns onset times in seconds.
    """
    # More aggressive onset detection for drums
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
    
    # Different thresholds for different instruments
    if instrument_type == 'kick':
        # Kicks are loud and clear
        delta = 0.3
    elif instrument_type == 'snare':
        # Snares can vary more
        delta = 0.25
    else:
        delta = 0.2
    
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        delta=delta,
        wait=int(0.05 * sr / hop_length)  # 50ms minimum between hits
    )
    
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    return onset_times


def extract_spectral_features(audio, sr, time_points, window_ms=50):
    """
    Extract spectral features at specific time points.
    window_ms: analysis window in milliseconds
    """
    window_samples = int(window_ms * sr / 1000)
    n_fft = 2048
    
    features = {
        'rms': [],
        'spectral_centroid': [],
        'low_energy': [],      # 0-200 Hz (kick fundamental)
        'low_mid_energy': [],  # 200-500 Hz (kick body)
        'mid_energy': [],      # 500-2000 Hz (snare body)
        'high_mid_energy': [], # 2000-5000 Hz (snare crack)
        'high_energy': [],     # 5000+ Hz (snare sizzle/cymbals)
        'zcr': [],             # zero crossing rate
    }
    
    for t in time_points:
        # Extract window around this time point
        center_sample = int(t * sr)
        start = max(0, center_sample - window_samples // 2)
        end = min(len(audio), center_sample + window_samples // 2)
        window = audio[start:end]
        
        if len(window) < window_samples // 2:
            continue
            
        # RMS energy
        rms = np.sqrt(np.mean(window ** 2))
        features['rms'].append(rms)
        
        # Zero crossing rate (higher for snare than kick)
        zcr = np.sum(np.abs(np.diff(np.sign(window)))) / (2 * len(window))
        features['zcr'].append(zcr)
        
        # Spectral analysis
        spectrum = np.abs(librosa.stft(window, n_fft=n_fft))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # Spectral centroid
        if spectrum.size > 0:
            S = np.sum(spectrum, axis=1)
            centroid = np.sum(freqs * S) / (np.sum(S) + 1e-10)
            features['spectral_centroid'].append(centroid)
        else:
            features['spectral_centroid'].append(0)
        
        # Energy in frequency bands
        def band_energy(f_low, f_high):
            mask = (freqs >= f_low) & (freqs < f_high)
            return np.sum(spectrum[mask] ** 2)
        
        features['low_energy'].append(band_energy(0, 200))
        features['low_mid_energy'].append(band_energy(200, 500))
        features['mid_energy'].append(band_energy(500, 2000))
        features['high_mid_energy'].append(band_energy(2000, 5000))
        features['high_energy'].append(band_energy(5000, sr/2))
    
    # Convert to numpy arrays
    for key in features:
        features[key] = np.array(features[key])
    
    return features


def find_snare_only_times(snare_times, kick_times, tolerance_ms=30):
    """
    Find snare hits that DON'T coincide with kick hits.
    These are the clearest indicators of snare bleed in kick track.
    """
    tolerance_s = tolerance_ms / 1000.0
    snare_only = []
    
    for st in snare_times:
        # Check if any kick hit is within tolerance
        if not np.any(np.abs(kick_times - st) < tolerance_s):
            snare_only.append(st)
    
    return np.array(snare_only)


def analyze_bleed(kick_path, snare_path, sr=44100):
    """Main analysis function."""
    
    print("\nLoading audio files...")
    kick_audio = load_audio(kick_path, sr)
    snare_audio = load_audio(snare_path, sr)
    
    print("Detecting onsets...")
    kick_times = detect_onsets(kick_audio, sr, 'kick')
    snare_times = detect_onsets(snare_audio, sr, 'snare')
    
    print(f"  Kick onsets:  {len(kick_times)}")
    print(f"  Snare onsets: {len(snare_times)}")
    
    # Find snare-only times (no coinciding kick)
    snare_only_times = find_snare_only_times(snare_times, kick_times, tolerance_ms=30)
    print(f"  Snare-only hits (no coinciding kick): {len(snare_only_times)}")
    
    print("\nAnalyzing spectral content...")
    
    # Extract features from kick track at different time points
    kick_at_snare_times = extract_spectral_features(kick_audio, sr, snare_only_times)
    kick_at_kick_times = extract_spectral_features(kick_audio, sr, kick_times)
    
    # Also analyze the actual snare track at snare times (for comparison)
    snare_at_snare_times = extract_spectral_features(snare_audio, sr, snare_only_times)
    
    # Print results
    print("\n" + "="*80)
    print("SNARE BLEED ANALYSIS")
    print("="*80)
    print("\nQUESTION: Is there snare energy in the kick track when snares hit?")
    print(f"\nAnalyzing {len(snare_only_times)} snare-only hits (excluding kick+snare coincidences)")
    
    print("\n" + "="*80)
    print("1. ENERGY ANALYSIS")
    print("="*80)
    
    print("\nKick track RMS at snare-only times:")
    print(f"  Mean:   {np.mean(kick_at_snare_times['rms']):.6f}")
    print(f"  Median: {np.median(kick_at_snare_times['rms']):.6f}")
    print(f"  Max:    {np.max(kick_at_snare_times['rms']):.6f}")
    
    print("\nKick track RMS at kick times (baseline):")
    print(f"  Mean:   {np.mean(kick_at_kick_times['rms']):.6f}")
    print(f"  Median: {np.median(kick_at_kick_times['rms']):.6f}")
    
    energy_ratio = np.mean(kick_at_snare_times['rms']) / np.mean(kick_at_kick_times['rms'])
    print(f"\n  Energy ratio (snare-times / kick-times): {energy_ratio:.2f}")
    if energy_ratio > 0.3:
        print(f"  ⚠️  SIGNIFICANT BLEED: Kick track has {energy_ratio:.1%} of normal energy at snare times!")
    elif energy_ratio > 0.1:
        print(f"  ⚠️  MODERATE BLEED: Kick track has {energy_ratio:.1%} of normal energy at snare times")
    else:
        print(f"  ✓  Minimal bleed: Only {energy_ratio:.1%} of normal energy at snare times")
    
    print("\n" + "="*80)
    print("2. FREQUENCY CONTENT ANALYSIS")
    print("="*80)
    
    def print_freq_bands(features, label):
        total = (features['low_energy'] + features['low_mid_energy'] + 
                features['mid_energy'] + features['high_mid_energy'] + features['high_energy'])
        print(f"\n{label}:")
        print(f"  Low (0-200 Hz, kick fundamental):     {np.mean(features['low_energy']):.2e}  ({np.mean(features['low_energy']/total)*100:.1f}%)")
        print(f"  Low-mid (200-500 Hz, kick body):      {np.mean(features['low_mid_energy']):.2e}  ({np.mean(features['low_mid_energy']/total)*100:.1f}%)")
        print(f"  Mid (500-2000 Hz, snare body):        {np.mean(features['mid_energy']):.2e}  ({np.mean(features['mid_energy']/total)*100:.1f}%)")
        print(f"  High-mid (2-5 kHz, snare crack):      {np.mean(features['high_mid_energy']):.2e}  ({np.mean(features['high_mid_energy']/total)*100:.1f}%)")
        print(f"  High (5+ kHz, snare sizzle):          {np.mean(features['high_energy']):.2e}  ({np.mean(features['high_energy']/total)*100:.1f}%)")
        print(f"  Spectral centroid: {np.mean(features['spectral_centroid']):.1f} Hz")
        print(f"  Zero crossing rate: {np.mean(features['zcr']):.4f}")
    
    print_freq_bands(kick_at_snare_times, "Kick track at snare-only times")
    print_freq_bands(kick_at_kick_times, "Kick track at kick times (baseline)")
    print_freq_bands(snare_at_snare_times, "Actual snare track at snare times (reference)")
    
    print("\n" + "="*80)
    print("3. SNARE SIGNATURE DETECTION")
    print("="*80)
    
    # Calculate ratios that distinguish snare from kick
    kick_snare_mid = np.mean(kick_at_snare_times['mid_energy'])
    kick_snare_low = np.mean(kick_at_snare_times['low_energy'])
    kick_kick_mid = np.mean(kick_at_kick_times['mid_energy'])
    kick_kick_low = np.mean(kick_at_kick_times['low_energy'])
    
    snare_mid_ratio_at_snare = kick_snare_mid / (kick_snare_low + 1e-10)
    snare_mid_ratio_at_kick = kick_kick_mid / (kick_kick_low + 1e-10)
    
    print(f"\nMid/Low frequency ratio in kick track:")
    print(f"  At snare times: {snare_mid_ratio_at_snare:.4f}")
    print(f"  At kick times:  {snare_mid_ratio_at_kick:.4f}")
    print(f"  Ratio:          {snare_mid_ratio_at_snare / snare_mid_ratio_at_kick:.2f}x")
    
    if snare_mid_ratio_at_snare > snare_mid_ratio_at_kick * 1.5:
        print(f"  ⚠️  SNARE SIGNATURE DETECTED: Mid frequencies are {snare_mid_ratio_at_snare / snare_mid_ratio_at_kick:.1f}x higher at snare times!")
    
    # Zero crossing rate comparison
    zcr_at_snare = np.mean(kick_at_snare_times['zcr'])
    zcr_at_kick = np.mean(kick_at_kick_times['zcr'])
    print(f"\nZero crossing rate in kick track:")
    print(f"  At snare times: {zcr_at_snare:.4f}")
    print(f"  At kick times:  {zcr_at_kick:.4f}")
    if zcr_at_snare > zcr_at_kick * 1.2:
        print(f"  ⚠️  Higher ZCR at snare times suggests snare bleed (snare is 'noisier' than kick)")
    
    # Spectral centroid comparison
    centroid_at_snare = np.mean(kick_at_snare_times['spectral_centroid'])
    centroid_at_kick = np.mean(kick_at_kick_times['spectral_centroid'])
    print(f"\nSpectral centroid in kick track:")
    print(f"  At snare times: {centroid_at_snare:.1f} Hz")
    print(f"  At kick times:  {centroid_at_kick:.1f} Hz")
    if centroid_at_snare < centroid_at_kick * 0.9:
        print(f"  ⚠️  LOWER centroid at snare times - unexpected! May indicate issue with detection")
    elif centroid_at_snare > centroid_at_kick * 1.1:
        print(f"  ⚠️  Higher centroid at snare times suggests brighter/snare-like content")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    bleed_indicators = 0
    if energy_ratio > 0.2:
        bleed_indicators += 1
        print(f"✓ Significant energy present at snare times ({energy_ratio:.1%})")
    if snare_mid_ratio_at_snare > snare_mid_ratio_at_kick * 1.3:
        bleed_indicators += 1
        print(f"✓ Elevated mid-frequency content at snare times")
    if zcr_at_snare > zcr_at_kick * 1.15:
        bleed_indicators += 1
        print(f"✓ Higher zero crossing rate at snare times")
    
    print(f"\nBleed indicators found: {bleed_indicators}/3")
    if bleed_indicators >= 2:
        print("\n⚠️  STRONG EVIDENCE OF SNARE BLEED IN KICK TRACK")
        print("Recommendation: Apply sidechain compression from snare to kick")
    elif bleed_indicators == 1:
        print("\n⚠️  MODERATE EVIDENCE OF SNARE BLEED")
        print("Recommendation: Consider light sidechain compression")
    else:
        print("\n✓  MINIMAL OR NO SNARE BLEED DETECTED")
        print("The kick track appears clean")
    
    return {
        'kick_times': kick_times,
        'snare_times': snare_times,
        'snare_only_times': snare_only_times,
        'kick_at_snare': kick_at_snare_times,
        'kick_at_kick': kick_at_kick_times,
        'energy_ratio': energy_ratio,
        'bleed_indicators': bleed_indicators
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze snare bleed in kick track by comparing spectral content at snare hit times'
    )
    parser.add_argument('-k', '--kick', required=True, help='Path to kick track WAV file')
    parser.add_argument('-s', '--snare', required=True, help='Path to snare track WAV file')
    parser.add_argument('--sr', type=int, default=44100, help='Sample rate (default: 44100)')
    
    args = parser.parse_args()
    
    results = analyze_bleed(args.kick, args.snare, args.sr)
    
    print("\n✓ Analysis complete!\n")


if __name__ == '__main__':
    main()