"""
Audio Analysis Utilities

Pure helper functions for audio signal processing.

Functions:
- ensure_mono: Convert stereo to mono
"""

import numpy as np


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert stereo audio to mono by averaging channels.

    Args:
        audio: Audio signal (mono or stereo)

    Returns:
        Mono audio signal
    """
    if audio.ndim == 2:
        return np.mean(audio, axis=1)
    return audio
