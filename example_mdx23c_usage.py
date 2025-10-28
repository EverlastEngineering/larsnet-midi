"""
Example script showing how to use the MDX23C model for drum separation.

This demonstrates:
1. Loading the model
2. Processing audio in chunks
3. Extracting individual stems (kick, snare, toms, hihat, cymbals)
"""
import torch
import torchaudio
import numpy as np
from pathlib import Path
from mdx23c_utils import load_mdx23c_checkpoint, get_checkpoint_hyperparameters


def separate_drums(
    audio_path: str,
    output_dir: str,
    checkpoint_path: str = "mdx_models/drumsep_5stems_mdx23c_jarredou.ckpt",
    config_path: str = "mdx_models/config_mdx23c.yaml",
    device: str = "cpu",
):
    """
    Separate drums from an audio file into 5 stems.
    
    Args:
        audio_path: Path to input audio file
        output_dir: Directory to save separated stems
        checkpoint_path: Path to MDX23C checkpoint
        config_path: Path to config YAML
        device: Device to run on ('cpu' or 'cuda')
    """
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = load_mdx23c_checkpoint(checkpoint_path, config_path, device=device)
    params = get_checkpoint_hyperparameters(checkpoint_path, config_path)
    
    # Get audio parameters
    chunk_size = params['audio']['chunk_size']
    target_sr = params['audio']['sample_rate']
    hop_length = chunk_size // 2  # 50% overlap
    
    print(f"Model config: {target_sr}Hz, chunk_size={chunk_size} samples")
    
    # Load audio
    print(f"Loading audio from {audio_path}...")
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if needed
    if sr != target_sr:
        print(f"Resampling from {sr}Hz to {target_sr}Hz...")
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    # Convert to stereo if mono
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]  # Take first 2 channels
    
    # Add batch dimension
    waveform = waveform.unsqueeze(0).to(device)  # (1, 2, time)
    
    # Process in chunks with overlap
    print(f"Processing audio in chunks...")
    total_length = waveform.shape[-1]
    num_chunks = (total_length - chunk_size) // hop_length + 1
    
    # Initialize output buffer (batch, instruments, channels, time)
    # Note: config uses 'hh' but we map to 'hihat' for consistency
    instruments = ['kick', 'snare', 'toms', 'hihat', 'cymbals']
    output = torch.zeros(1, 5, 2, total_length, device=device)
    overlap_count = torch.zeros(total_length, device=device)
    
    with torch.no_grad():
        for i in range(num_chunks):
            start = i * hop_length
            end = min(start + chunk_size, total_length)
            
            # Pad last chunk if needed
            chunk = waveform[:, :, start:end]
            if chunk.shape[-1] < chunk_size:
                pad_size = chunk_size - chunk.shape[-1]
                chunk = torch.nn.functional.pad(chunk, (0, pad_size))
            
            # Process chunk
            chunk_output = model(chunk)  # (1, 5, 2, chunk_size)
            
            # Add to output buffer with overlap
            actual_length = min(chunk_size, total_length - start)
            output[:, :, :, start:start+actual_length] += chunk_output[:, :, :, :actual_length]
            overlap_count[start:start+actual_length] += 1
            
            if (i + 1) % 10 == 0 or i == num_chunks - 1:
                print(f"  Processed {i+1}/{num_chunks} chunks")
    
    # Average overlapping regions
    output = output / overlap_count.view(1, 1, 1, -1)
    
    # Save stems
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving stems to {output_dir}...")
    for i, instrument in enumerate(instruments):
        stem = output[0, i].cpu()  # (2, time)
        output_path = output_dir / f"{instrument}.wav"
        torchaudio.save(str(output_path), stem, target_sr)
        print(f"  Saved {instrument}.wav")
    
    print("\n✅ Separation complete!")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Separate drums using MDX23C model"
    )
    parser.add_argument(
        "input",
        help="Input audio file path"
    )
    parser.add_argument(
        "-o", "--output",
        default="output_stems",
        help="Output directory for stems (default: output_stems)"
    )
    parser.add_argument(
        "--checkpoint",
        default="mdx_models/drumsep_5stems_mdx23c_jarredou.ckpt",
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--config",
        default="mdx_models/config_mdx23c.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on"
    )
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1
    
    separate_drums(
        args.input,
        args.output,
        args.checkpoint,
        args.config,
        args.device,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
