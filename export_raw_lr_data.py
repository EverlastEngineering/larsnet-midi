#!/usr/bin/env python3
"""
Export RAW L/R channel data separately with ultra-high sensitivity.
Shows ALL detected onsets with separate features for left and right channels.
"""

import librosa
import numpy as np
import csv
from stems_to_midi.stereo_core import detect_dual_channel_onsets, separate_channels
from stems_to_midi.analysis_core import extract_onset_features


def export_raw_lr_csv(
    audio_path: str,
    output_csv: str,
    threshold: float = 1.0,  # Ultra sensitive - see EVERYTHING
):
    """Export raw L/R onset data with separate channel features."""
    
    print(f"Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    
    # Ensure stereo
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.shape[0] != 2:
        audio = audio[:2]
    
    print(f"Detecting onsets with ultra-high sensitivity (threshold={threshold})...")
    onset_data = detect_dual_channel_onsets(
        audio,
        sr,
        merge_window_ms=100,
        threshold=threshold,
        delta=0.01,
        wait=3,
        hop_length=512,
    )
    
    print(f"Detected {len(onset_data['onset_times'])} onsets")
    
    # Separate channels
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
        
        # Header - L/R split
        writer.writerow([
            'Index',
            'Time(s)',
            'Pan',
            'L_Strength',
            'R_Strength',
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
                onset_data['left_strengths'][idx],
                onset_data['right_strengths'][idx],
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
    print(f"  Left strength range: {min(onset_data['left_strengths']):.3f} - {max(onset_data['left_strengths']):.3f}")
    print(f"  Right strength range: {min(onset_data['right_strengths']):.3f} - {max(onset_data['right_strengths']):.3f}")
    
    # Show geomean ranges
    left_geomeans = [lf.get('geomean', 0) for lf in left_features]
    right_geomeans = [rf.get('geomean', 0) for rf in right_features]
    print(f"  Left geomean range: {min(left_geomeans):.1f} - {max(left_geomeans):.1f}")
    print(f"  Right geomean range: {min(right_geomeans):.1f} - {max(right_geomeans):.1f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Export raw L/R channel data to CSV')
    parser.add_argument('--audio', default='user_files/1 - AC_DC_Thunderstruck_Drums/stems/AC_DC_Thunderstruck_Drums-cymbals.wav',
                        help='Path to audio file')
    parser.add_argument('--output', default='raw_lr_data.csv',
                        help='Output CSV file')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Onset detection threshold (lower = more sensitive, use 1.0-2.0 for raw data)')
    
    args = parser.parse_args()
    
    export_raw_lr_csv(
        audio_path=args.audio,
        output_csv=args.output,
        threshold=args.threshold,
    )
