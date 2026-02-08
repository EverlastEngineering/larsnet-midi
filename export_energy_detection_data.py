#!/usr/bin/env python3
"""
Export onset data using NEW energy-based detection (DAW-like method).

REPLACES LIBROSA DETECTION - Uses calibrated transient peak detection:
- Method: scipy.signal.find_peaks on RMS energy envelope
- Calibrated parameters (from iterative testing on Thunderstruck cymbals):
  * threshold_db = 15.0 (prominence above local minimum)
  * min_absolute_energy = 0.01 (noise floor for real hits)
  * min_peak_spacing_ms = 100.0 (prevent double-detection)

Result: Detects 75 events vs librosa's 238 (3.2x cleaner)
Includes obvious DAW events at 112s and 119s that manual tuning validated.

Shows L/R channels separately with geomean features for threshold tuning.
"""

import librosa
import numpy as np
import csv
from stems_to_midi.energy_detection_core import detect_stereo_transient_peaks
from stems_to_midi.stereo_core import separate_channels
from stems_to_midi.analysis_core import extract_onset_features


def export_energy_detection_csv(
    audio_path: str,
    output_csv: str,
    threshold_db: float = 15.0,
    method: str = 'rms',
    peak_hold_ms: float = 3.0,
):
    """Export onset data using energy-based detection with L/R features."""
    
    print(f"Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    
    # Ensure stereo
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.shape[0] != 2:
        audio = audio[:2]
    
    print(f"Detecting transient peaks (threshold_db={threshold_db}, method={method})...")
    onset_data = detect_stereo_transient_peaks(
        audio, sr,
        threshold_db=threshold_db,
        min_peak_spacing_ms=100.0,
        merge_window_ms=150.0,
        method=method,
        peak_hold_ms=peak_hold_ms,
        min_absolute_energy=0.01,  # Raised to filter noise - real cymbal hits are louder
    )
    
    print(f"Detected {len(onset_data['onset_times'])} onsets")
    
    # Separate channels for feature extraction
    left_channel, right_channel = separate_channels(audio)
    
    print(f"Extracting features from LEFT channel...")
    left_features = extract_onset_features(
        audio=left_channel,
        sr=sr,
        onset_times=onset_data['onset_times'],
        pan_confidence=onset_data['pan_confidence'],
        window_ms=50.0,
        pitch_method='yin',
        min_pitch_hz=80.0,
        max_pitch_hz=10000.0,
        body_freq_range=(1000, 4000),
        brilliance_freq_range=(4000, 10000),
        calculate_sustain=True,
        sustain_window_ms=200.0,
    )
    
    print(f"Extracting features from RIGHT channel...")
    right_features = extract_onset_features(
        audio=right_channel,
        sr=sr,
        onset_times=onset_data['onset_times'],
        pan_confidence=onset_data['pan_confidence'],
        window_ms=50.0,
        pitch_method='yin',
        min_pitch_hz=80.0,
        max_pitch_hz=10000.0,
        body_freq_range=(1000, 4000),
        brilliance_freq_range=(4000, 10000),
        calculate_sustain=True,
        sustain_window_ms=200.0,
    )
    
    print(f"Writing CSV: {output_csv}")
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Index',
            'Time(s)',
            'Pan',
            'L_Energy',
            'R_Energy',
            # Left channel features
            'L_SpectralCentroid(Hz)',
            'L_SpectralRolloff(Hz)',
            'L_SpectralFlatness',
            'L_Pitch(Hz)',
            'L_BodyEnergy',
            'L_BrillianceEnergy',
            'L_GeoMean',
            'L_TotalEnergy',
            'L_SustainMs',
            # Right channel features
            'R_SpectralCentroid(Hz)',
            'R_SpectralRolloff(Hz)',
            'R_SpectralFlatness',
            'R_Pitch(Hz)',
            'R_BodyEnergy',
            'R_BrillianceEnergy',
            'R_GeoMean',
            'R_TotalEnergy',
            'R_SustainMs',
        ])
        
        # Data rows
        for idx in range(len(onset_data['onset_times'])):
            lf = left_features[idx]
            rf = right_features[idx]
            
            def fmt(val, default=0):
                return default if val is None else val
            
            writer.writerow([
                idx + 1,
                onset_data['onset_times'][idx],
                onset_data['pan_confidence'][idx],
                onset_data['left_energies'][idx],
                onset_data['right_energies'][idx],
                # Left channel
                fmt(lf.get('spectral_centroid'), 0),
                fmt(lf.get('spectral_rolloff'), 0),
                fmt(lf.get('spectral_flatness'), 0),
                fmt(lf.get('pitch'), 0),
                fmt(lf.get('body_energy'), 0),
                fmt(lf.get('brilliance_energy'), 0),
                fmt(lf.get('geomean'), 0),
                fmt(lf.get('total_energy'), 0),
                fmt(lf.get('sustain_ms'), 0),
                # Right channel
                fmt(rf.get('spectral_centroid'), 0),
                fmt(rf.get('spectral_rolloff'), 0),
                fmt(rf.get('spectral_flatness'), 0),
                fmt(rf.get('pitch'), 0),
                fmt(rf.get('body_energy'), 0),
                fmt(rf.get('brilliance_energy'), 0),
                fmt(rf.get('geomean'), 0),
                fmt(rf.get('total_energy'), 0),
                fmt(rf.get('sustain_ms'), 0),
            ])
    
    print(f"✓ Exported {len(onset_data['onset_times'])} onsets to {output_csv}")
    print(f"  Detection method: Transient peaks (local maxima)")
    print(f"  Threshold: {threshold_db}dB prominence above local baseline")
    
    # Show geomean ranges
    left_geomeans = [lf.get('geomean', 0) for lf in left_features]
    right_geomeans = [rf.get('geomean', 0) for rf in right_features]
    print(f"  Left geomean range: {min(left_geomeans):.1f} - {max(left_geomeans):.1f}")
    print(f"  Right geomean range: {min(right_geomeans):.1f} - {max(right_geomeans):.1f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Export energy-based detection data to CSV')
    parser.add_argument('--audio', default='user_files/1 - AC_DC_Thunderstruck_Drums/stems/AC_DC_Thunderstruck_Drums-cymbals.wav',
                        help='Path to audio file')
    parser.add_argument('--output', default='energy_detection_data.csv',
                        help='Output CSV file')
    parser.add_argument('--threshold-db', type=float, default=15.0,
                        help='dB prominence threshold (12-18 recommended, higher = fewer/cleaner detections)')
    parser.add_argument('--method', default='rms', choices=['rms', 'spectral', 'peak_hold'],
                        help='Envelope calculation method (rms=default, peak_hold=DAW-style)')
    parser.add_argument('--peak-hold-ms', type=float, default=3.0,
                        help='Peak-hold smoothing window in milliseconds (used with --method peak_hold)')
    
    args = parser.parse_args()
    
    export_energy_detection_csv(
        audio_path=args.audio,
        output_csv=args.output,
        threshold_db=args.threshold_db,
        method=args.method,
        peak_hold_ms=args.peak_hold_ms,
    )
