import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from model import DrumTranscriber
from train_utils import load_audio
from config import DEVICE, SAMPLE_RATE, HOP_LENGTH
import os

# Optional dependencies for advanced features
try:
    import pretty_midi
    HAS_PRETTY_MIDI = True
except ImportError:
    HAS_PRETTY_MIDI = False

try:
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def generate_visual_report(model_path, audio_path, midi_path=None, output_png="diagnostic_trace.png"):
    """
    Generates a visual report showing all 10 detected drum classes with equal visual priority.
    """
    # 1. Load Model
    model = DrumTranscriber()
    ckpt = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    # 2. Process Audio
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    audio_tensor = load_audio(audio_path).to(DEVICE)
    
    if audio_tensor.dim() == 3:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    with torch.no_grad():
        logits = model(audio_tensor)
        # Apply Sigmoid here because model returns raw logits for BCEWithLogitsLoss
        # probs shape is [Time, 20] — channels 0-9 are onset, 10-19 are velocity
        probs = torch.sigmoid(logits).cpu().numpy()
        if probs.ndim == 3:
            probs = probs[0]
        # Only plot onset channels (first 10) for probability visualization
        probs = probs[:, :10]
    num_classes = probs.shape[1]

    # 3. Plotting Setup
    plt.figure(figsize=(15, 25)) 
    duration = min(60, len(y)/SAMPLE_RATE)
    time_axis = np.linspace(0, len(y)/SAMPLE_RATE, num=len(probs))
    mask = time_axis <= duration
    time_zoom = time_axis[mask]
    probs_zoom = probs[mask]
    
    # Use a 10-color categorical palette
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Subplot 1: Waveform
    plt.subplot(5, 1, 1)
    librosa.display.waveshow(y[:int(duration * SAMPLE_RATE)], sr=SAMPLE_RATE, alpha=0.5)
    plt.title(f"1. Audio Waveform (First {duration}s)")
    plt.xlim(0, duration)

    # Subplot 2: Spectrogram
    plt.subplot(5, 1, 2)
    S = librosa.feature.melspectrogram(y=y[:int(duration * SAMPLE_RATE)], sr=SAMPLE_RATE, n_mels=128, hop_length=HOP_LENGTH)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel')
    plt.title("2. Mel-Spectrogram (Check Low-End for Kicks)")
    plt.xlim(0, duration)

    # Subplot 3: Raw Probabilities (All 10 Classes Equal)
    plt.subplot(5, 1, 3)
    for i in range(num_classes):
        plt.plot(time_zoom, probs_zoom[:, i], label=f"C{i}", color=colors[i], alpha=0.8, linewidth=1.5)
            
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.2)
    plt.title(f"3. Raw Probability (Full 10-Class Comparison)")
    plt.ylabel("Prob")
    plt.xlim(0, duration)
    plt.legend(loc='upper right', fontsize='x-small', ncol=5)

    # Subplot 4: Signal Delta (Onset Intensity)
    plt.subplot(5, 1, 4)
    for i in range(num_classes):
        delta = np.diff(probs_zoom[:, i], prepend=0)
        delta = np.maximum(0, delta)
        plt.plot(time_zoom, delta, color=colors[i], alpha=0.7, linewidth=1, label=f"C{i}")
    plt.title("4. Signal Delta (Rising Edge Detection)")
    plt.xlim(0, duration)

    # Subplot 5: Recall Audit & Peak View
    plt.subplot(5, 1, 5)
    
    # Track detected frames at 0.5 threshold
    detections = (probs_zoom > 0.5)
    
    if midi_path and HAS_PRETTY_MIDI:
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            all_gt_times = []
            for inst in midi_data.instruments:
                for note in inst.notes:
                    if note.start <= duration:
                        all_gt_times.append(note.start)
            
            for gt_t in all_gt_times:
                idx = np.abs(time_zoom - gt_t).argmin()
                # Check all classes for a hit near this GT time
                is_detected = np.any(detections[max(0, idx-2):idx+3, :]) 
                line_color = 'green' if is_detected else 'red'
                plt.axvline(x=gt_t, color=line_color, linestyle='-', alpha=0.2, linewidth=0.5)
        except Exception as e:
            print(f"MIDI Diagnostic Error: {e}")

    for i in range(num_classes):
        p = probs_zoom[:, i]
        # Plot subtle traces in the background of the scatter
        plt.plot(time_zoom, p, alpha=0.1, color=colors[i]) 
        
        if HAS_SCIPY:
            # SENSITIVITY BOOST: Show peaks above 0.05 to see "Whisper" hits
            peaks, _ = find_peaks(p, height=0.05, distance=10) 
            if len(peaks) > 0:
                plt.scatter(time_zoom[peaks], p[peaks], s=10, color=colors[i], alpha=0.6, label=f"C{i}" if i < 10 else None)

    plt.title("5. Full Recall Scan (Sensitivity 0.05, Red=Missed GT, Green=Found)")
    plt.xlabel("Time (seconds)")
    plt.xlim(0, duration)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_png)
    print(f"Diagnostic saved: {output_png}")

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "latest_model.pth"
    audio_path = sys.argv[2] if len(sys.argv) > 2 else "39_rock-indie_63_beat_4-4_10.wav"
    midi_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    if os.path.exists(model_path) and os.path.exists(audio_path):
         generate_visual_report(model_path, audio_path, midi_path)