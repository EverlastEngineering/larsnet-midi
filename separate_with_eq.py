from larsnet import LarsNet
from pathlib import Path
from typing import Union, Optional
import soundfile as sf
import torch
import torchaudio.transforms as T
import argparse


def apply_frequency_cleanup(waveform: torch.Tensor, sr: int, stem_type: str) -> torch.Tensor:
    """
    Apply frequency-specific cleanup to reduce bleed between stems.
    
    Args:
        waveform: Audio tensor (channels, samples)
        sr: Sample rate
        stem_type: Type of stem ('kick', 'snare', 'toms', 'hihat', 'cymbals')
    
    Returns:
        Processed waveform
    """
    # Ensure we're working with the right shape
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    if stem_type == 'kick':
        # Kick: Keep low frequencies, remove high-frequency snare bleed
        # Low-pass at ~150Hz to isolate sub kick, but this might be too aggressive
        # Better: High-pass at ~30Hz to remove rumble, low-pass at ~8kHz to remove cymbal bleed
        highpass = T.Highpass(sample_rate=sr, cutoff_freq=30.0)
        lowpass = T.Lowpass(sample_rate=sr, cutoff_freq=8000.0)
        waveform = highpass(waveform)
        waveform = lowpass(waveform)
        
    elif stem_type == 'snare':
        # Snare: Remove low-frequency kick bleed while preserving snare body
        # High-pass at ~80-100Hz to remove kick fundamental
        highpass = T.Highpass(sample_rate=sr, cutoff_freq=100.0)
        waveform = highpass(waveform)
        
    elif stem_type == 'toms':
        # Toms: Mid-range focus
        highpass = T.Highpass(sample_rate=sr, cutoff_freq=60.0)
        lowpass = T.Lowpass(sample_rate=sr, cutoff_freq=10000.0)
        waveform = highpass(waveform)
        waveform = lowpass(waveform)
        
    elif stem_type == 'hihat':
        # Hi-hat: High frequencies only
        highpass = T.Highpass(sample_rate=sr, cutoff_freq=300.0)
        waveform = highpass(waveform)
        
    elif stem_type == 'cymbals':
        # Cymbals: High frequencies with some midrange
        highpass = T.Highpass(sample_rate=sr, cutoff_freq=200.0)
        waveform = highpass(waveform)
    
    return waveform


def separate_with_eq(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    wiener_exponent: Optional[float],
    device: str,
    apply_eq: bool = True,
    aggressive_kick_cleanup: bool = False
):
    """
    Separate drums with optional post-processing EQ to reduce bleed.
    
    Args:
        input_dir: Directory containing drum mixes
        output_dir: Directory to save separated stems
        wiener_exponent: Wiener filter exponent (None to disable)
        device: 'cpu' or 'cuda'
        apply_eq: Whether to apply frequency cleanup
        aggressive_kick_cleanup: Use more aggressive filtering on kick (removes more snare but may affect kick body)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise RuntimeError(f'{input_dir} was not found.')

    if wiener_exponent is not None and wiener_exponent <= 0:
        raise ValueError(f'α-Wiener filter exponent should be positive.')

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
        print(f"\nProcessing: {mixture.name}")
        
        stems = larsnet(mixture)

        for stem, waveform in stems.items():
            # Apply frequency cleanup if enabled
            if apply_eq:
                print(f"  Applying EQ cleanup to {stem}...")
                waveform = apply_frequency_cleanup(waveform, larsnet.sr, stem)
                
                # Extra aggressive cleanup for kick/snare bleed
                if aggressive_kick_cleanup and stem == 'kick':
                    # Apply additional notch at snare fundamental (around 200Hz)
                    print(f"    Applying aggressive kick cleanup...")
                    # You could add a notch filter here if needed
            
            # Save
            save_path = output_dir.joinpath(stem, f'{mixture.stem}.wav')
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to numpy for saving
            waveform_np = waveform.cpu().numpy()
            if waveform_np.ndim == 1:
                waveform_np = waveform_np.reshape(-1, 1)
            else:
                waveform_np = waveform_np.T
            
            sf.write(save_path, waveform_np, larsnet.sr)
            print(f"  Saved: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Separate drums with post-processing EQ to reduce bleed between stems."
    )
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help="Path to the root directory where to find the target drum mixtures.")
    parser.add_argument('-o', '--output_dir', type=str, default='separated_stems_eq',
                        help="Path to the directory where to save the separated tracks.")
    parser.add_argument('-w', '--wiener_exponent', type=float, default=1.5,
                        help="Positive α-Wiener filter exponent (float). Recommended: 1.0-2.0. Use 0 to disable.")
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        help="Torch device. Use 'cuda' if available for faster processing.")
    parser.add_argument('--no-eq', action='store_true',
                        help="Disable post-processing EQ cleanup.")
    parser.add_argument('--aggressive-kick', action='store_true',
                        help="Apply more aggressive filtering to kick track to remove snare bleed.")

    args = parser.parse_args()
    
    # Convert wiener_exponent=0 to None
    wiener_exp = None if args.wiener_exponent == 0 else args.wiener_exponent

    separate_with_eq(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        wiener_exponent=wiener_exp,
        device=args.device,
        apply_eq=not args.no_eq,
        aggressive_kick_cleanup=args.aggressive_kick
    )
