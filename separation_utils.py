"""
Shared utilities for drum separation with optional post-processing EQ.
"""
from larsnet import LarsNet
from pathlib import Path
from typing import Union, Optional, Dict
import soundfile as sf
import torch
import torchaudio.functional as F
import yaml


def load_eq_config(config_path: Union[str, Path] = "eq.yaml") -> Dict:
    """Load EQ configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"EQ config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def apply_frequency_cleanup(
    waveform: torch.Tensor, 
    sr: int, 
    stem_type: str,
    eq_config: Optional[Dict] = None
) -> torch.Tensor:
    """
    Apply frequency-specific cleanup to reduce bleed between stems.
    
    Args:
        waveform: Audio tensor (channels, samples)
        sr: Sample rate
        stem_type: Type of stem ('kick', 'snare', 'toms', 'hihat', 'cymbals')
        eq_config: EQ configuration dict (if None, will load from eq.yaml)
    
    Returns:
        Processed waveform
    """
    if eq_config is None:
        eq_config = load_eq_config()
    
    # Ensure we're working with the right shape
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # Get stem-specific EQ settings
    if stem_type not in eq_config:
        return waveform
    
    stem_eq = eq_config[stem_type]
    
    # Apply high-pass filter if configured
    if 'highpass' in stem_eq:
        waveform = F.highpass_biquad(waveform, sr, stem_eq['highpass'])
    
    # Apply low-pass filter if configured
    if 'lowpass' in stem_eq:
        waveform = F.lowpass_biquad(waveform, sr, stem_eq['lowpass'])
    
    return waveform


def process_stems(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    wiener_exponent: Optional[float],
    device: str,
    apply_eq: bool = False,
    eq_config_path: str = "eq.yaml",
    verbose: bool = True
):
    """
    Separate drums with optional post-processing EQ to reduce bleed.
    
    Args:
        input_dir: Directory containing drum mixes
        output_dir: Directory to save separated stems
        wiener_exponent: Wiener filter exponent (None to disable)
        device: 'cpu' or 'cuda'
        apply_eq: Whether to apply frequency cleanup
        eq_config_path: Path to EQ configuration file
        verbose: Whether to print progress information
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise RuntimeError(f'{input_dir} was not found.')

    if wiener_exponent is not None and wiener_exponent <= 0:
        raise ValueError(f'α-Wiener filter exponent should be positive.')

    # Load EQ config if needed
    eq_config = None
    if apply_eq:
        eq_config = load_eq_config(eq_config_path)

    if verbose:
        print(f"Initializing LarsNet...")
        print(f"  Wiener filter: {'Enabled (α=' + str(wiener_exponent) + ')' if wiener_exponent else 'Disabled'}")
        print(f"  Post-processing EQ: {'Enabled' if apply_eq else 'Disabled'}")
        print(f"  Device: {device}")
    
    larsnet = LarsNet(
        wiener_filter=wiener_exponent is not None,
        wiener_exponent=wiener_exponent,
        device=device,
        config="config.yaml",
    )

    for mixture in input_dir.rglob("*.wav"):
        if verbose:
            print(f"\nProcessing: {mixture.name}")
        
        stems = larsnet(mixture)

        for stem, waveform in stems.items():
            # Apply frequency cleanup if enabled
            if apply_eq:
                if verbose:
                    print(f"  Applying EQ cleanup to {stem}...")
                waveform = apply_frequency_cleanup(waveform, larsnet.sr, stem, eq_config)
            
            # Save
            save_path = output_dir.joinpath(mixture.stem, f'{mixture.stem}-{stem}.wav')
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to numpy for saving
            waveform_np = waveform.cpu().numpy()
            if waveform_np.ndim == 1:
                waveform_np = waveform_np.reshape(-1, 1)
            else:
                waveform_np = waveform_np.T
            
            sf.write(save_path, waveform_np, larsnet.sr)
            if verbose:
                print(f"  Saved: {save_path}")
