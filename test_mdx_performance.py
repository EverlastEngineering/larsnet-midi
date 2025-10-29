#!/usr/bin/env python3
"""
Test script to compare MDX23C performance: original vs optimized.

This benchmarks both implementations to quantify speed improvements.
"""
import time
import argparse
from pathlib import Path
import torch
import torchaudio
import numpy as np

from mdx23c_utils import load_mdx23c_checkpoint, get_checkpoint_hyperparameters
from mdx23c_optimized import OptimizedMDX23CProcessor
from separation_utils import _process_with_mdx23c


def benchmark_original(audio_path: str, overlap: int = 8, device: str = "cpu"):
    """Benchmark original MDX23C implementation."""
    print("\n" + "="*60)
    print("BENCHMARKING ORIGINAL MDX23C")
    print("="*60)
    
    # Load model
    mdx_checkpoint = Path("mdx_models/drumsep_5stems_mdx23c_jarredou.ckpt")
    mdx_config = Path("mdx_models/config_mdx23c.yaml")
    
    if not mdx_checkpoint.exists():
        raise RuntimeError(f"MDX23C checkpoint not found: {mdx_checkpoint}")
    
    print("Loading model...")
    model = load_mdx23c_checkpoint(mdx_checkpoint, mdx_config, device=device)
    params = get_checkpoint_hyperparameters(mdx_checkpoint, mdx_config)
    chunk_size = params['audio']['chunk_size']
    target_sr = params['audio']['sample_rate']
    
    # Map instruments
    config_instruments = params['training']['instruments']
    instruments = [inst if inst != 'hh' else 'hihat' for inst in config_instruments]
    
    # Load audio info
    waveform, sr = torchaudio.load(audio_path)
    duration = waveform.shape[-1] / sr
    
    print(f"Audio duration: {duration:.1f} seconds")
    print(f"Overlap: {overlap} ({100*(1-1/overlap):.1f}% overlap)")
    print(f"Device: {device}")
    
    # Warm-up run (important for MPS to pre-compile kernels)
    print("\nWarm-up run...")
    _ = _process_with_mdx23c(
        Path(audio_path), model, chunk_size, target_sr, instruments, 
        overlap, device, verbose=False
    )
    
    # Clear cache
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    
    # Timed runs
    times = []
    num_runs = 3
    print(f"\nPerforming {num_runs} timed runs...")
    
    for i in range(num_runs):
        # Synchronize before timing for accurate MPS benchmarks
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        _ = _process_with_mdx23c(
            Path(audio_path), model, chunk_size, target_sr, instruments,
            overlap, device, verbose=False
        )
        
        # Synchronize after to ensure completion
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        elapsed = end - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f} seconds")
        
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    rtf = avg_time / duration
    
    print(f"\nResults:")
    print(f"  Average time: {avg_time:.2f} ± {std_time:.2f} seconds")
    print(f"  Real-time factor: {rtf:.2f}x")
    
    return {
        "implementation": "original",
        "overlap": overlap,
        "device": device,
        "avg_time": avg_time,
        "std_time": std_time,
        "rtf": rtf,
        "duration": duration
    }


def benchmark_optimized(
    audio_path: str, 
    overlap: int = 8, 
    batch_size: int = 4,
    device: str = "cpu"
):
    """Benchmark optimized MDX23C implementation."""
    print("\n" + "="*60)
    print("BENCHMARKING OPTIMIZED MDX23C")
    print("="*60)
    
    # Load processor
    print("Loading optimized processor...")
    processor = OptimizedMDX23CProcessor(
        device=device,
        batch_size=batch_size,
        use_fp16=(device == "cuda" or device == "mps"),
        optimize_for_inference=True
    )
    
    # Load audio info
    waveform, sr = torchaudio.load(audio_path)
    duration = waveform.shape[-1] / sr
    
    print(f"Audio duration: {duration:.1f} seconds")
    print(f"Overlap: {overlap} ({100*(1-1/overlap):.1f}% overlap)")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    if device == "cuda" or device == "mps":
        print(f"Mixed precision: Enabled (fp16)")
    
    # Warm-up run (important for MPS to pre-compile kernels)
    print("\nWarm-up run...")
    _ = processor.process_audio(audio_path, overlap=overlap, verbose=False)
    
    # Clear cache
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    
    # Timed runs
    times = []
    num_runs = 3
    print(f"\nPerforming {num_runs} timed runs...")
    
    for i in range(num_runs):
        # Synchronize before timing for accurate MPS benchmarks
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        _ = processor.process_audio(audio_path, overlap=overlap, verbose=False)
        
        # Synchronize after to ensure completion
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        elapsed = end - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f} seconds")
        
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    rtf = avg_time / duration
    
    print(f"\nResults:")
    print(f"  Average time: {avg_time:.2f} ± {std_time:.2f} seconds")
    print(f"  Real-time factor: {rtf:.2f}x")
    
    return {
        "implementation": "optimized",
        "overlap": overlap,
        "batch_size": batch_size,
        "device": device,
        "avg_time": avg_time,
        "std_time": std_time,
        "rtf": rtf,
        "duration": duration
    }


def compare_quality(audio_path: str, overlap: int = 8, device: str = "cpu"):
    """Compare output quality between implementations."""
    print("\n" + "="*60)
    print("COMPARING OUTPUT QUALITY")
    print("="*60)
    
    # Process with original
    print("\nProcessing with original implementation...")
    mdx_checkpoint = Path("mdx_models/drumsep_5stems_mdx23c_jarredou.ckpt")
    mdx_config = Path("mdx_models/config_mdx23c.yaml")
    
    model = load_mdx23c_checkpoint(mdx_checkpoint, mdx_config, device=device)
    params = get_checkpoint_hyperparameters(mdx_checkpoint, mdx_config)
    chunk_size = params['audio']['chunk_size']
    target_sr = params['audio']['sample_rate']
    config_instruments = params['training']['instruments']
    instruments = [inst if inst != 'hh' else 'hihat' for inst in config_instruments]
    
    stems_original = _process_with_mdx23c(
        Path(audio_path), model, chunk_size, target_sr, instruments,
        overlap, device, verbose=False
    )
    
    # Process with optimized
    print("Processing with optimized implementation...")
    processor = OptimizedMDX23CProcessor(
        device=device,
        batch_size=4,
        use_fp16=False,  # Disable fp16 for quality comparison
        optimize_for_inference=True
    )
    
    stems_optimized = processor.process_audio(
        audio_path, overlap=overlap, verbose=False
    )
    
    # Compare outputs
    print("\nComparing outputs:")
    max_diffs = []
    
    for instrument in instruments:
        orig = stems_original[instrument].cpu().numpy()
        opt = stems_optimized[instrument].cpu().numpy()
        
        # Handle potential shape mismatches
        min_len = min(orig.shape[-1], opt.shape[-1])
        orig = orig[..., :min_len]
        opt = opt[..., :min_len]
        
        # Check for NaN or Inf
        if np.any(np.isnan(orig)) or np.any(np.isnan(opt)):
            print(f"  {instrument}: WARNING - NaN values detected")
            max_diffs.append(float('inf'))
            continue
        
        # Compute difference
        diff = np.abs(orig - opt)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # Compute SNR
        signal_power = np.mean(orig**2)
        noise_power = np.mean(diff**2)
        if noise_power > 0 and signal_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = float('inf') if noise_power == 0 else 0
        
        max_diffs.append(max_diff)
        
        print(f"  {instrument}:")
        print(f"    Max difference: {max_diff:.6f}")
        print(f"    Mean difference: {mean_diff:.6f}")
        print(f"    SNR: {snr:.1f} dB")
    
    overall_max_diff = max(max_diffs)
    
    if overall_max_diff < 1e-5:
        print("\n✅ Outputs are identical (within numerical precision)")
    elif overall_max_diff < 1e-3:
        print("\n✅ Outputs are very similar (negligible differences)")
    else:
        print(f"\n⚠️ Outputs have noticeable differences (max: {overall_max_diff:.6f})")
    
    return overall_max_diff < 1e-3


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MDX23C performance: original vs optimized"
    )
    parser.add_argument(
        "input",
        help="Input audio file path"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=8,
        help="Overlap factor (default: 8)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for optimized version (default: 4)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Processing device"
    )
    parser.add_argument(
        "--quality-check",
        action="store_true",
        help="Also compare output quality"
    )
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Run benchmarks
    results = []
    
    # Benchmark original
    try:
        result_original = benchmark_original(
            args.input, args.overlap, args.device
        )
        results.append(result_original)
    except Exception as e:
        print(f"Error benchmarking original: {e}")
        result_original = None
    
    # Benchmark optimized
    try:
        result_optimized = benchmark_optimized(
            args.input, args.overlap, args.batch_size, args.device
        )
        results.append(result_optimized)
    except Exception as e:
        print(f"Error benchmarking optimized: {e}")
        result_optimized = None
    
    # Summary
    if result_original and result_optimized:
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        
        speedup = result_original["avg_time"] / result_optimized["avg_time"]
        
        print(f"Original implementation:")
        print(f"  Time: {result_original['avg_time']:.2f}s (RTF: {result_original['rtf']:.2f}x)")
        
        print(f"\nOptimized implementation (batch_size={args.batch_size}):")
        print(f"  Time: {result_optimized['avg_time']:.2f}s (RTF: {result_optimized['rtf']:.2f}x)")
        
        print(f"\n🚀 Speedup: {speedup:.2f}x faster!")
        
        if result_optimized["rtf"] < 1.0:
            print("✅ Faster than real-time!")
        elif result_optimized["rtf"] < 2.0:
            print("✅ Near real-time performance")
    
    # Quality check
    if args.quality_check:
        quality_ok = compare_quality(args.input, args.overlap, args.device)
        if not quality_ok:
            print("\n⚠️ Warning: Quality differences detected")
    
    return 0


if __name__ == "__main__":
    exit(main())