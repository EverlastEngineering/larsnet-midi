"""
Feature Extractor - Mono Mel-Spectrogram Generator

Transforms raw 1D audio into a mono mel-spectrogram for simplicity.
Dimensions: [1, 128, Time_Steps]
"""

import torch
import torchaudio
import torchaudio.transforms as T


def get_input_tensor(audio_path: str, sample_rate: int = 44100) -> torch.Tensor:
    """
    Converts audio to mono mel-spectrogram.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Audio sample rate (default 44100)
    
    Returns:
        Tensor of shape [1, 128, Time_Steps]
    """
    # Load audio: waveform shape is [channels, samples]
    waveform, sr = torchaudio.load(audio_path)
    
    # Handle mono by duplicating to stereo
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)  # [2, samples]
    
    # Normalize audio to max amplitude 1.0
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
    
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
    
    # Mix stereo to mono
    mono = waveform.mean(dim=0, keepdim=True)  # [1, samples]
    spec = amplitude_to_db(mel_transform(mono))  # [1, 128, Time]
    
    return spec


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