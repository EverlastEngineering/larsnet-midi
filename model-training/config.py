"""
Config.py — Shared constants and device detection for model-training.

Loads hyperparameters from config.yaml.
"""

import sys
from pathlib import Path
import torch

# Add parent workspace to path for device_shell import
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

# Paths
MODEL_TRAINING_DIR = Path(__file__).parent
MODELS_DIR = MODEL_TRAINING_DIR / "models"
CONFIG_PATH = MODEL_TRAINING_DIR / "config.yaml"

# Audio processing constants (shared across all modules)
HOP_LENGTH = 512
SAMPLE_RATE = 44100
SECONDS_PER_FRAME = HOP_LENGTH / SAMPLE_RATE

# Drum class mappings
INDEX_TO_MIDI = {
    0: 36, 1: 38, 2: 42, 3: 46, 4: 50, 5: 47,
    6: 43, 7: 49, 8: 57, 9: 51
}

MIDI_TO_INDEX = {v: k for k, v in INDEX_TO_MIDI.items()}

INDEX_TO_NAME = {
    0: 'Kick', 1: 'Snare', 2: 'HHC', 3: 'HHO',
    4: 'TomHigh', 5: 'TomMid', 6: 'TomLow',
    7: 'Crash1', 8: 'Crash2', 9: 'Ride'
}

LABEL_NAMES = list(INDEX_TO_NAME.values())

# MTL training constants (overridden by config.yaml if present)
VELOCITY_WEIGHT = 2.0
SMEAR_THRESHOLD = 0.2

# Load config.yaml once at module load
_config = None

def _load_config():
    global _config
    if _config is None:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                _config = yaml.safe_load(f)
        else:
            _config = {}
    return _config

def get_training_config():
    """Return the training section of config.yaml."""
    return _load_config().get('training', {})

def get_inference_config():
    """Return the inference section of config.yaml."""
    return _load_config().get('inference', {})

def get_learning_rate():
    """Return learning rate from config.yaml (default 0.001)."""
    return get_training_config().get('learning_rate', 0.001)

def get_chunk_frames():
    """Return chunk_frames from config.yaml (default 2000)."""
    return get_training_config().get('chunk_frames', 2000)

def get_velocity_weight():
    """Return velocity_weight from config.yaml (default 2.0)."""
    return get_training_config().get('velocity_weight', 2.0)

# Global device — INTENTIONALLY FORCED TO CPU.
#
# The cuda/mps code paths below appear correct but fail at runtime during
# training (silent NaNs / wrong-device tensors when crossing the MultiTaskDrumLoss
# boundary on Apple MPS, untested on CUDA). Until those failures are debugged,
# _DEVICE_CACHE is pre-seeded to 'cpu' so the `is None` branch in get_device()
# never runs. Leave the auto-detect function in place as a documented escape
# hatch: set `_DEVICE_CACHE = None` to re-enable it.
_DEVICE_CACHE = 'cpu'  # set to None to re-enable cuda/mps auto-detect

def get_device() -> str:
    """Auto-detect the best available device (cuda > mps > cpu).

    DORMANT: see _DEVICE_CACHE comment above. Currently always returns 'cpu'
    because cuda/mps paths produced incorrect training behavior.
    """
    global _DEVICE_CACHE
    if _DEVICE_CACHE is None:
        if torch.cuda.is_available():
            _DEVICE_CACHE = 'cuda'
        elif torch.backends.mps.is_available():
            _DEVICE_CACHE = 'mps'
        else:
            _DEVICE_CACHE = 'cpu'
    return _DEVICE_CACHE

DEVICE = get_device()


def get_models_dir() -> Path:
    """Return the models directory, creating it if needed."""
    MODELS_DIR.mkdir(exist_ok=True)
    return MODELS_DIR
