"""                                                                                                                                                                                                                       
Shared utilities for drum separation with optional post-processing EQ.
"""                                                                                                                                                                                                                       
from larsnet import LarsNet
from pathlib import Path
from typing import Union, Optional, Dict
import soundfile as sf # type: ignore
import torch # type: ignore
import torchaudio # type: ignore
import torchaudio.functional as F # type: ignore
import yaml # type: ignore
import numpy as np
from mdx23c_utils import load_mdx23c_checkpoint, get_checkpoint_hyperparameters

# Try to import optimized MDX processor
try:
    from mdx23c_optimized import OptimizedMDX23CProcessor
    MDX_OPTIMIZED_AVAILABLE = True
except ImportError:
    MDX_OPTIMIZED_AVAILABLE = False
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
    eq_config: Optional[Dict] = None,
    chunk_size: int = 44100 * 30  # 30 seconds per chunk
) -> torch.Tensor:
    """
    Apply frequency-specific cleanup to reduce bleed between stems.
    
    Processes audio in chunks to avoid memory issues with biquad filters on large buffers.
    
    Args:
        waveform: Audio tensor (channels, samples)
        sr: Sample rate
        stem_type: Type of stem ('kick', 'snare', 'toms', 'hihat', 'cymbals')
        eq_config: EQ configuration dict (if None, will load from eq.yaml)
        chunk_size: Number of samples to process at once (default: 30s)
    
    Returns:
        Processed waveform
    """
    try:
        if eq_config is None:
            eq_config = load_eq_config()
        
        # Ensure we're working with the right shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Get stem-specific EQ settings
        if stem_type not in eq_config:
            print(f"  Warning: No EQ config for {stem_type}, skipping")
            return waveform
        
        stem_eq = eq_config[stem_type]
        
        # Validate filter parameters
        if 'highpass' in stem_eq:
            cutoff = stem_eq['highpass']
            if cutoff <= 0 or cutoff >= sr / 2:
                raise ValueError(f"Invalid highpass cutoff {cutoff} Hz (must be 0 < cutoff < {sr/2} Hz)")
        
        if 'lowpass' in stem_eq:
            cutoff = stem_eq['lowpass']
            if cutoff <= 0 or cutoff >= sr / 2:
                raise ValueError(f"Invalid lowpass cutoff {cutoff} Hz (must be 0 < cutoff < {sr/2} Hz)")
        
        # Process in chunks to avoid memory issues with large buffers
        channels, num_samples = waveform.shape
        
        if num_samples <= chunk_size:
            # Small enough to process in one go
            return _apply_filters_to_chunk(waveform, sr, stem_type, stem_eq)
        
        # Process in chunks - simple non-overlapping approach to maintain exact length
        processed_chunks = []
        
        num_chunks = (num_samples + chunk_size - 1) // chunk_size
        print(f"  Processing {stem_type} in {num_chunks} chunks ({num_samples} samples)")
        
        for i in range(0, num_samples, chunk_size):
            end = min(i + chunk_size, num_samples)
            chunk = waveform[:, i:end]
            processed_chunk = _apply_filters_to_chunk(chunk, sr, stem_type, stem_eq)
            processed_chunks.append(processed_chunk)
        
        result = torch.cat(processed_chunks, dim=1)
        
        # Verify output length matches input length
        assert result.shape[1] == num_samples, f"Output length mismatch: {result.shape[1]} != {num_samples}"
        
        return result
        
    except Exception as e:
        print(f"  ERROR in apply_frequency_cleanup for {stem_type}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


def _apply_filters_to_chunk(
    chunk: torch.Tensor,
    sr: int,
    stem_type: str,
    stem_eq: Dict
) -> torch.Tensor:
    """
    Apply highpass/lowpass filters to a single audio chunk.
    
    Args:
        chunk: Audio chunk (channels, samples)
        sr: Sample rate
        stem_type: Stem type for logging
        stem_eq: EQ config dict with 'highpass' and/or 'lowpass' keys
    
    Returns:
        Filtered chunk
    """
    # Apply high-pass filter if configured
    if 'highpass' in stem_eq:
        cutoff = stem_eq['highpass']
        chunk = F.highpass_biquad(chunk, sr, cutoff)
    
    # Apply low-pass filter if configured
    if 'lowpass' in stem_eq:
        cutoff = stem_eq['lowpass']
        chunk = F.lowpass_biquad(chunk, sr, cutoff)
    
    return chunk


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
        print(f"Status Update: Saving Stems...")
        for stem, waveform in stems.items():
            try:
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
                    
            except Exception as e:
                print(f"ERROR processing {stem}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                raise


def _process_with_mdx23c(
    audio_file: Path,
    model: torch.nn.Module,
    chunk_size: int,
    target_sr: int,
    instruments: list,
    overlap: int,
    device: str,
    verbose: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Process audio file with MDX23C model using overlap-add for long files.
    
    Args:
        audio_file: Path to audio file
        model: Loaded MDX23C model
        chunk_size: Samples per chunk
        target_sr: Target sample rate
        instruments: List of instrument names in order
        overlap: Overlap value (2-50), controls hop_length = chunk_size / overlap
        device: Processing device
        verbose: Print progress
        
    Returns:
        Dict mapping stem names to waveforms
    """
    # Load audio
    waveform, sr = torchaudio.load(str(audio_file))
    
    # Resample if needed
    if sr != target_sr:
        if verbose:
            print(f"  Resampling from {sr}Hz to {target_sr}Hz...")
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    # Convert to stereo
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    
    # Add batch dimension
    waveform = waveform.unsqueeze(0).to(device)  # (1, 2, time)
    
    total_length = waveform.shape[-1]
    
    if total_length <= chunk_size:
        # Short enough to process in one chunk
        if verbose:
            print(f"  Processing audio ({total_length / target_sr:.1f}s)...")
        
        # Pad to chunk size
        if total_length < chunk_size:
            pad_size = chunk_size - total_length
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        
        with torch.no_grad():
            output = model(waveform)  # (1, instruments, 2, time)
        
        # Trim padding
        output = output[:, :, :, :total_length]
    else:
        # Process with overlap-add
        hop_length = chunk_size // overlap
        num_chunks = (total_length - chunk_size) // hop_length + 1
        
        if verbose:
            overlap_pct = ((chunk_size - hop_length) / chunk_size) * 100
            print(f"  Processing {total_length / target_sr:.1f}s audio in {num_chunks} chunks (overlap={overlap}, {overlap_pct:.1f}%)...")
        
        # Initialize output buffer
        output = torch.zeros(1, 5, 2, total_length, device=device)
        overlap_count = torch.zeros(total_length, device=device)
        
        with torch.no_grad():
            for i in range(num_chunks):
                start = i * hop_length
                end = min(start + chunk_size, total_length)
                
                # Extract chunk
                chunk = waveform[:, :, start:end]
                
                # Pad last chunk if needed
                if chunk.shape[-1] < chunk_size:
                    pad_size = chunk_size - chunk.shape[-1]
                    chunk = torch.nn.functional.pad(chunk, (0, pad_size))
                
                # Process chunk
                chunk_output = model(chunk)
                
                # Add to output buffer
                actual_length = min(chunk_size, total_length - start)
                output[:, :, :, start:start+actual_length] += chunk_output[:, :, :, :actual_length]
                overlap_count[start:start+actual_length] += 1
                
                if verbose and ((i + 1) % 5 == 0 or i == num_chunks - 1):
                    progress = int(15 + ((i + 1) / num_chunks) * 75)
                    print(f"Progress: {progress}% (chunk {i+1}/{num_chunks})")
        
        # Average overlapping regions
        output = output / overlap_count.view(1, 1, 1, -1)
    
    # Convert to dict with instrument names
    stems = {}
    for i, instrument in enumerate(instruments):
        stems[instrument] = output[0, i]  # (2, time)
    
    return stems


def process_stems_for_project(
    project_dir: Path,
    stems_dir: Path,
    config_path: Union[str, Path],
    model: str = 'mdx23c',
    overlap: int = 8,
    wiener_exponent: Optional[float] = None,
    device: str = 'cpu',
    apply_eq: bool = False,
    verbose: bool = True
):
    """
    Separate drums for a project using project-specific configuration.
    
    This is the project-aware version of process_stems. It:
    - Finds audio files in the project directory
    - Uses project-specific config
    - Outputs to project/stems/ directory
    
    Args:
        project_dir: Path to project directory
        stems_dir: Path to stems output directory (project/stems/)
        config_path: Path to config.yaml (project-specific or root)
        model: Separation model ('mdx23c' or 'larsnet')
        overlap: Overlap value for MDX23C (2-50, higher=better quality but slower)
        wiener_exponent: Wiener filter exponent (None to disable, LarsNet only)
        device: 'cpu' or 'cuda'
        apply_eq: Whether to apply frequency cleanup
        verbose: Whether to print progress information
    """
    project_dir = Path(project_dir)
    stems_dir = Path(stems_dir)
    config_path = Path(config_path)
    
    if not project_dir.exists():
        raise RuntimeError(f'Project directory not found: {project_dir}')
    
    if not config_path.exists():
        raise RuntimeError(f'Config file not found: {config_path}')
    
    if wiener_exponent is not None and wiener_exponent <= 0:
        raise ValueError(f'α-Wiener filter exponent should be positive.')
    
    # Find audio files in project root
    audio_files = [f for f in project_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in {'.wav', '.mp3', '.flac', '.aiff', '.aif'}]
    
    if not audio_files:
        raise RuntimeError(f'No audio files found in {project_dir}')
    
    # Load EQ config if needed
    eq_config = None
    if apply_eq:
        eq_config_path = project_dir / "eq.yaml"
        if not eq_config_path.exists():
            eq_config_path = Path("eq.yaml")  # Fall back to root
        if eq_config_path.exists():
            eq_config = load_eq_config(eq_config_path)
        else:
            print("Warning: EQ requested but eq.yaml not found, skipping EQ")
            apply_eq = False
    
    if verbose:
        print(f"Initializing {model.upper()} separation model...")
        if model == 'larsnet':
            print(f"  Config: {config_path}")
            print(f"  Wiener filter: {'Enabled (α=' + str(wiener_exponent) + ')' if wiener_exponent else 'Disabled'}")
        print(f"  Post-processing EQ: {'Enabled' if apply_eq else 'Disabled'}")
        print(f"  Device: {device}")
    
    print("Progress: 0%")
    
    if model == 'mdx23c':
        # Load MDX23C model
        mdx_checkpoint = Path("mdx_models/drumsep_5stems_mdx23c_jarredou.ckpt")
        mdx_config = Path("mdx_models/config_mdx23c.yaml")
        
        if not mdx_checkpoint.exists():
            raise RuntimeError(f"MDX23C checkpoint not found: {mdx_checkpoint}")
        
        # Use optimized processor if available
        if MDX_OPTIMIZED_AVAILABLE:
            # Determine batch size based on device and overlap
            if device == "cuda":
                # GPU: use larger batches
                batch_size = min(8, max(2, 16 // overlap))
            else:
                # CPU: smaller batches to avoid memory issues
                batch_size = min(4, max(1, 8 // overlap))
            
            separator = OptimizedMDX23CProcessor(
                checkpoint_path=str(mdx_checkpoint),
                config_path=str(mdx_config),
                device=device,
                batch_size=batch_size,
                use_fp16=(device == "cuda"),
                optimize_for_inference=True
            )
            target_sr = separator.target_sr
            instruments = separator.instruments
            
            if verbose:
                print(f"  Model: MDX23C (Optimized with batch_size={batch_size})")
                print(f"  Chunk size: {separator.chunk_size} samples (~{separator.chunk_size/target_sr:.1f}s)")
                print(f"  Target SR: {target_sr} Hz")
                print(f"  Overlap: {overlap} (hop={separator.chunk_size//overlap} samples)")
                if device == "cuda":
                    print(f"  Mixed Precision: Enabled (fp16)")
        else:
            # Fallback to original implementation
            separator = load_mdx23c_checkpoint(mdx_checkpoint, mdx_config, device=device)
            mdx_params = get_checkpoint_hyperparameters(mdx_checkpoint, mdx_config)
            chunk_size = mdx_params['audio']['chunk_size']
            target_sr = mdx_params['audio']['sample_rate']
            # Get instruments from config and map 'hh' to 'hihat' for consistency
            config_instruments = mdx_params['training']['instruments']
            instruments = [inst if inst != 'hh' else 'hihat' for inst in config_instruments]
            
            if verbose:
                print(f"  Model: MDX23C (TFC_TDF_net)")
                print(f"  Chunk size: {chunk_size} samples (~{chunk_size/target_sr:.1f}s)")
                print(f"  Target SR: {target_sr} Hz")
                print(f"  Overlap: {overlap} (hop={chunk_size//overlap} samples)")
    else:
        # Load LarsNet model
        separator = LarsNet(
            wiener_filter=wiener_exponent is not None,
            wiener_exponent=wiener_exponent,
            device=device,
            config=str(config_path),
        )
        target_sr = separator.sr
        instruments = None  # LarsNet returns dict with stem names
    
    print("Progress: 15%")
    
    # Create stems directory
    stems_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each audio file
    for audio_file in audio_files:
        if verbose:
            print(f"\nProcessing: {audio_file.name}")
        
        if model == 'mdx23c':
            # MDX23C processing
            if MDX_OPTIMIZED_AVAILABLE and isinstance(separator, OptimizedMDX23CProcessor):
                # Use optimized processor
                stems = separator.process_audio(
                    str(audio_file), 
                    overlap=overlap, 
                    verbose=verbose
                )
            else:
                # Fallback to original implementation
                mdx_params = get_checkpoint_hyperparameters(mdx_checkpoint, mdx_config)
                chunk_size = mdx_params['audio']['chunk_size']
                stems = _process_with_mdx23c(
                    audio_file, separator, chunk_size, target_sr, instruments, overlap, device, verbose
                )
        else:
            # LarsNet processing
            stems = separator(audio_file)
        
        # After all stems are separated: 15% (init) + 75% (5 stems * 15%) = 90%
        print("Progress: 90%")
        
        # Saving phase: 90-100%
        total_stems = len(stems)
        print(f"Status Update: Saving Stems...")
        for stem_idx, (stem, waveform) in enumerate(stems.items(), 1):
            # Apply frequency cleanup if enabled
            if apply_eq:
                if verbose:
                    print(f"  Applying EQ cleanup to {stem}...")
                waveform = apply_frequency_cleanup(waveform, target_sr, stem, eq_config)
            
            # Save to stems directory
            save_path = stems_dir / f'{audio_file.stem}-{stem}.wav'
            
            # Convert to numpy for saving
            waveform_np = waveform.cpu().numpy()
            if waveform_np.ndim == 1:
                waveform_np = waveform_np.reshape(-1, 1)
            else:
                waveform_np = waveform_np.T
            
            sf.write(save_path, waveform_np, target_sr)
            if verbose:
                print(f"  Saved: {save_path}")
            
            # Progress: saving stems (90-100%)
            save_progress = int(90 + (stem_idx / total_stems) * 10)
            print(f"Progress: {save_progress}%")
