"""
Configuration and data structures for stems-to-MIDI conversion.

This module provides configuration loading and data classes used throughout
the stems-to-MIDI processing pipeline.

Architecture: Part of the Imperative Shell
- Handles I/O (YAML file loading)
- Provides data structures for coordination
"""

from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
import yaml


__all__ = ['load_config', 'DrumMapping']


def load_config(config_path: Optional[Path] = None) -> Dict:
    """
    Load MIDI conversion configuration from YAML file.
    
    Args:
        config_path: Path to config file (defaults to midiconfig.yaml in project root)
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if config_path is None:
        # Config files are in the project root, not in the package
        config_path = Path(__file__).parent.parent / 'midiconfig.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


@dataclass
class DrumMapping:
    """MIDI note mapping for drum sounds (General MIDI standard)."""
    kick: int = 36          # C1 - Acoustic Bass Drum
    snare: int = 38         # D1 - Acoustic Snare
    tom_low: int = 45       # A1 - Low Tom
    tom_mid: int = 47       # B1 - Mid Tom
    tom_high: int = 50      # D2 - High Tom
    hihat_closed: int = 42  # F#1 - Closed Hi-Hat
    hihat_open: int = 46    # A#1 - Open Hi-Hat
    crash: int = 49         # C#2 - Crash Cymbal 1
    ride: int = 51          # D#2 - Ride Cymbal 1
    
    # Aliases for convenience
    @property
    def hihat(self) -> int:
        """Alias for closed hi-hat (default)."""
        return self.hihat_closed
    
    @property
    def cymbals(self) -> int:
        """Alias for crash cymbal (default cymbal)."""
        return self.crash
    
    @property
    def toms(self) -> int:
        """Alias for mid tom (default tom)."""
        return self.tom_mid
    
    @property
    def handclap(self) -> int:
        """Handclap detected from hihat bleed."""
        return 39  # D#1 - Hand Clap
