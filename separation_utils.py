"""
Shared utilities for drum separation.
"""
from pathlib import Path
from typing import Union, Optional, Dict
import soundfile as sf # type: ignore
import torch # type: ignore
import numpy as np
from mdx23c_utils import load_mdx23c_checkpoint, get_checkpoint_hyperparameters
from device_utils import detect_best_device

# Try to import optimized MDX processor
try:
    from mdx23c_optimized import OptimizedMDX23CProcessor
    MDX_OPTIMIZED_AVAILABLE = True
except ImportError:
    MDX_OPTIMIZED_AVAILABLE = False


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
    batch_size: Optional[int] = None,
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
        model: Separation model ('mdx23c' currently supported, extensible for future models)
        overlap: Overlap value for MDX23C (2-50, higher=better quality but slower)
        wiener_exponent: Reserved for future model use (not used by MDX23C)
        device: 'cpu', 'cuda', or 'mps'
        batch_size: Batch size for MDX23C (None=auto-detect)
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
    
    if verbose:
        print(f"Initializing {model.upper()} separation model...")
        print(f"  Device: {device}")
    
    print("Progress: 0%")
    print("Progress: 5%")
    
    if model == 'mdx23c':
        # Load MDX23C model
        mdx_checkpoint = Path("mdx_models/drumsep_5stems_mdx23c_jarredou.ckpt")
        mdx_config = Path("mdx_models/config_mdx23c.yaml")
        
        if not mdx_checkpoint.exists():
            raise RuntimeError(f"MDX23C checkpoint not found: {mdx_checkpoint}")
        
        # Use optimized processor if available
        if MDX_OPTIMIZED_AVAILABLE:
            # Determine batch size based on device and overlap (unless overridden)
            if batch_size is None:
                # MPS has different memory characteristics than CUDA
                if device == "cuda":
                    # CUDA: use larger batches
                    batch_size = min(8, max(2, 16 // overlap))
                elif device == "mps":
                    # MPS: Always use batch_size=1 for optimal performance
                    # Benchmarks show batch_size=1 is faster than 2 or 4 even at low overlap
                    batch_size = 1
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
            print("Progress: 10%")
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
        # Future: Add other model support here
        raise ValueError(f"Unsupported model: {model}. Currently only 'mdx23c' is supported.")
    
    print("Progress: 15%")
    
    # Create stems directory
    stems_dir.mkdir(parents=True, exist_ok=True)
    print("Progress: 16%")
    
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
            # Future: Add other model processing here
            raise ValueError(f"Unsupported model: {model}")
        
        # After all stems are separated: model inference reports 16-90%
        print("Progress: 91%")
        
        # Saving phase: 91-100%
        total_stems = len(stems)
        print(f"Status Update: Saving Stems...")
        for stem_idx, (stem, waveform) in enumerate(stems.items(), 1):
            # Save to stems directory
            save_path = stems_dir / f'{audio_file.stem}-{stem}.wav'
            
            # Convert to numpy for saving (ensure float32 for soundfile compatibility)
            if waveform.dtype == torch.float16:
                waveform = waveform.to(torch.float32)
            waveform_np = waveform.cpu().numpy()
            if waveform_np.ndim == 1:
                waveform_np = waveform_np.reshape(-1, 1)
            else:
                waveform_np = waveform_np.T
            
            sf.write(save_path, waveform_np, target_sr)
            if verbose:
                print(f"  Saved: {save_path}")
            
            # Progress: saving stems (91-100%)
            save_progress = int(91 + (stem_idx / total_stems) * 9)
            print(f"Progress: {save_progress}%")
