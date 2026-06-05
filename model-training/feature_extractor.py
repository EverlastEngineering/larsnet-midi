"""
Feature Extractor - 3-Channel Mel-Spectrogram Generator

Transforms raw 1D audio into a 3-channel mel-spectrogram, preserving
stereo spatial information that is critical for distinguishing e.g.
centered snares from wide hand-claps.

Dimensions: [3, 128, Time_Steps]
Channels:
    0 = Left-channel mel-spec (dB)
    1 = Right-channel mel-spec (dB)
    2 = Stereo-width mel-spec (dB), computed from the (L-R) side signal

Per model-training/Deep Learning Roadmap.md §1.
"""

import torch
import torchaudio
import torchaudio.transforms as T


def get_input_tensor(audio_path: str, sample_rate: int = 44100) -> torch.Tensor:
    """
    Converts stereo audio to a 3-channel mel-spectrogram.

    Args:
        audio_path: Path to audio file
        sample_rate: Audio sample rate (default 44100)

    Returns:
        Tensor of shape [3, 128, Time_Steps]
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
        hop_length=512,
    )

    # Convert to decibels for neural network stability
    amplitude_to_db = T.AmplitudeToDB()

    # Channel 0: Left mel-spec
    spec_l = amplitude_to_db(mel_transform(waveform[0:1]))  # [1, 128, T]
    # Channel 1: Right mel-spec
    spec_r = amplitude_to_db(mel_transform(waveform[1:2]))  # [1, 128, T]
    # Channel 2: Stereo-width mel-spec from the (L-R) side signal
    side = waveform[0:1] - waveform[1:2]
    spec_w = amplitude_to_db(mel_transform(side))  # [1, 128, T]

    return torch.cat([spec_l, spec_r, spec_w], dim=0)  # [3, 128, T]


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