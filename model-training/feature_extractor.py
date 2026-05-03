"""
Feature Extractor - 3-Channel Mel-Spectrogram Generator

Transforms raw 1D audio into a 2D feature set with 3 channels:
- Left spectrogram
- Right spectrogram  
- Stereo Width (L-R)

Dimensions: [3, 128, Time_Steps]
"""

import torch
import torchaudio
import torchaudio.transforms as T


def get_input_tensor(audio_path: str, sample_rate: int = 44100) -> torch.Tensor:
    """
    Converts a stereo wav file into a 3-channel Mel-Spectrogram.
    
    Args:
        audio_path: Path to stereo WAV file
        sample_rate: Audio sample rate (default 44100)
    
    Returns:
        Tensor of shape [3, 128, Time_Steps]
        Channel 0: Left mel-spec (dB)
        Channel 1: Right mel-spec (dB)
        Channel 2: Stereo width (L-R) mel-spec (dB)
    """
    # Load audio: waveform shape is [channels, samples]
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if needed
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    
    # Initialize the MelSpectrogram transform
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )
    
    # Convert to decibels for neural network stability
    amplitude_to_db = T.AmplitudeToDB()
    
    # Extract Left and Right channels
    spec_l = amplitude_to_db(mel_transform(waveform[0:1]))  # [1, 128, Time]
    spec_r = amplitude_to_db(mel_transform(waveform[1:2]))  # [1, 128, Time]
    
    # Feature 3: Stereo Width (The "Clap vs Snare" engine)
    # Calculated as the energy of the 'Side' signal (L-R)
    side = waveform[0:1] - waveform[1:2]
    spec_width = amplitude_to_db(mel_transform(side))  # [1, 128, Time]
    
    # Stack into a single 3D tensor: [Channels, Freq, Time]
    return torch.cat([spec_l, spec_r, spec_width], dim=0)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python feature_extractor.py <audio.wav>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    print(f"Loading: {audio_path}")
    
    tensor = get_input_tensor(audio_path)
    print(f"Output shape: {tensor.shape}")
    print(f"Expected: [3, 128, Time_Steps]")
    print(f"Data type: {tensor.dtype}")
    print(f"Value range: [{tensor.min():.2f}, {tensor.max():.2f}] dB")