"""
Optimized MDX23C processing with batch support and performance improvements.

Key optimizations:
1. Batch processing of multiple chunks simultaneously
2. Reduced memory allocations with buffer reuse
3. Configurable overlap with quality/speed tradeoffs
4. Optional mixed precision support
5. Optimized STFT operations
"""
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging
from contextlib import contextmanager
import time

from mdx23c_utils import load_mdx23c_checkpoint, get_checkpoint_hyperparameters
from device_utils import detect_best_device, validate_device

logger = logging.getLogger(__name__)


@contextmanager
def timer(name: str):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        logger.info(f"{name} took {end - start:.2f} seconds")


class OptimizedMDX23CProcessor:
    """
    Optimized MDX23C processor with batching and performance improvements.
    
    Args:
        checkpoint_path: Path to MDX23C checkpoint
        config_path: Path to config YAML
        device: Processing device ('cpu' or 'cuda')
        batch_size: Number of chunks to process simultaneously
        use_fp16: Use mixed precision for faster inference (GPU only)
        optimize_for_inference: Apply inference-specific optimizations
    """
    
    def __init__(
        self,
        checkpoint_path: str = "mdx_models/drumsep_5stems_mdx23c_jarredou.ckpt",
        config_path: str = "mdx_models/config_mdx23c.yaml",
        device: Optional[str] = None,
        batch_size: int = 4,
        use_fp16: bool = False,
        optimize_for_inference: bool = True,
    ):
        # Auto-detect device if not specified
        if device is None:
            device = detect_best_device(verbose=True)
        else:
            device = validate_device(device, fallback=True)
        
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and device == "cuda"
        
        # Load model
        logger.info(f"Loading MDX23C model from {checkpoint_path}")
        self.model = load_mdx23c_checkpoint(checkpoint_path, config_path, device=device)
        
        # Get parameters
        self.params = get_checkpoint_hyperparameters(checkpoint_path, config_path)
        self.chunk_size = self.params['audio']['chunk_size']
        self.target_sr = self.params['audio']['sample_rate']
        
        # Map 'hh' to 'hihat' for consistency
        config_instruments = self.params['training']['instruments']
        self.instruments = [inst if inst != 'hh' else 'hihat' for inst in config_instruments]
        
        # Optimize model for inference
        if optimize_for_inference:
            self._optimize_model()
        
        # Pre-allocate buffers for batch processing
        self._init_buffers()
        
        logger.info(f"Initialized with batch_size={batch_size}, fp16={self.use_fp16}")
    
    def _optimize_model(self):
        """Apply inference-specific optimizations to the model."""
        self.model.eval()
        
        # Disable gradient computation globally for this model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Enable cudnn benchmarking for optimal convolution algorithms
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        
        # Try to compile model with torch.compile if available (PyTorch 2.0+)
        # Note: torch.compile requires a C++ compiler which may not be available in containers
        if hasattr(torch, 'compile') and False:  # Disabled by default due to compiler requirements
            try:
                logger.info("Compiling model with torch.compile for faster inference")
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as e:
                logger.warning(f"torch.compile failed (may not be supported): {e}")
    
    def _init_buffers(self):
        """Pre-allocate reusable buffers for batch processing."""
        # Input buffer for batched chunks
        self.input_buffer = torch.zeros(
            self.batch_size, 2, self.chunk_size,
            device=self.device,
            dtype=torch.float16 if self.use_fp16 else torch.float32
        )
        
        # Output buffer for batched results  
        self.output_buffer = torch.zeros(
            self.batch_size, 5, 2, self.chunk_size,
            device=self.device,
            dtype=torch.float16 if self.use_fp16 else torch.float32
        )
    
    def process_audio(
        self,
        audio_path: str,
        overlap: int = 4,
        verbose: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Process audio file with optimized batching.
        
        Args:
            audio_path: Path to input audio file
            overlap: Overlap factor (2-50), lower = faster but lower quality
            verbose: Print progress information
            
        Returns:
            Dictionary mapping instrument names to waveforms
        """
        with timer("Total processing"):
            # Load and prepare audio
            with timer("Audio loading"):
                waveform = self._load_and_prepare_audio(audio_path, verbose)
            
            # Process with batching
            with timer("Model inference"):
                if self.use_fp16:
                    with torch.cuda.amp.autocast():
                        output = self._process_batched(waveform, overlap, verbose)
                else:
                    output = self._process_batched(waveform, overlap, verbose)
            
            # Convert to instrument dict
            stems = {}
            for i, instrument in enumerate(self.instruments):
                stems[instrument] = output[0, i].cpu()  # (2, time)
            
            return stems
    
    def _load_and_prepare_audio(
        self,
        audio_path: str,
        verbose: bool = True
    ) -> torch.Tensor:
        """Load and prepare audio for processing."""
        if verbose:
            print(f"Loading audio from {audio_path}")
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.target_sr:
            if verbose:
                print(f"  Resampling from {sr}Hz to {self.target_sr}Hz")
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        
        # Convert to stereo
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]
        
        # Move to device and add batch dimension
        waveform = waveform.unsqueeze(0).to(self.device)
        
        # Convert to fp16 if using mixed precision
        if self.use_fp16:
            waveform = waveform.half()
        
        return waveform
    
    def _process_batched(
        self,
        waveform: torch.Tensor,
        overlap: int,
        verbose: bool
    ) -> torch.Tensor:
        """Process audio with batched chunks."""
        total_length = waveform.shape[-1]
        
        if total_length <= self.chunk_size:
            # Single chunk - process directly
            return self._process_single_chunk(waveform, total_length)
        
        # Multi-chunk with batching
        hop_length = self.chunk_size // overlap
        num_chunks = (total_length - self.chunk_size) // hop_length + 1
        
        if verbose:
            overlap_pct = ((self.chunk_size - hop_length) / self.chunk_size) * 100
            print(f"  Processing {total_length/self.target_sr:.1f}s audio")
            print(f"  {num_chunks} chunks, batch_size={self.batch_size}, overlap={overlap} ({overlap_pct:.1f}%)")
        
        # Initialize output accumulator
        dtype = torch.float16 if self.use_fp16 else torch.float32
        output = torch.zeros(1, 5, 2, total_length, device=self.device, dtype=dtype)
        overlap_count = torch.zeros(total_length, device=self.device, dtype=dtype)
        
        # Process in batches
        with torch.no_grad():
            for batch_start in range(0, num_chunks, self.batch_size):
                batch_end = min(batch_start + self.batch_size, num_chunks)
                batch_size = batch_end - batch_start
                
                # Prepare batch
                for i in range(batch_size):
                    chunk_idx = batch_start + i
                    start = chunk_idx * hop_length
                    end = min(start + self.chunk_size, total_length)
                    
                    # Extract chunk into buffer
                    chunk_data = waveform[:, :, start:end]
                    self.input_buffer[i, :, :chunk_data.shape[-1]] = chunk_data[0]
                    
                    # Pad if needed
                    if chunk_data.shape[-1] < self.chunk_size:
                        self.input_buffer[i, :, chunk_data.shape[-1]:] = 0
                
                # Process batch (only the filled portion)
                if batch_size < self.batch_size:
                    batch_input = self.input_buffer[:batch_size]
                else:
                    batch_input = self.input_buffer
                
                batch_output = self.model(batch_input)
                
                # Accumulate results
                for i in range(batch_size):
                    chunk_idx = batch_start + i
                    start = chunk_idx * hop_length
                    actual_length = min(self.chunk_size, total_length - start)
                    
                    output[:, :, :, start:start+actual_length] += batch_output[i, :, :, :actual_length]
                    overlap_count[start:start+actual_length] += 1
                
                if verbose and ((batch_end % 10 == 0) or batch_end == num_chunks):
                    progress = (batch_end / num_chunks) * 100
                    print(f"  Progress: {progress:.1f}% ({batch_end}/{num_chunks} chunks)")
        
        # Average overlapping regions
        output = output / overlap_count.view(1, 1, 1, -1)
        
        return output
    
    def _process_single_chunk(
        self,
        waveform: torch.Tensor,
        total_length: int
    ) -> torch.Tensor:
        """Process a single chunk that fits in memory."""
        # Pad to chunk size if needed
        if total_length < self.chunk_size:
            pad_size = self.chunk_size - total_length
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        
        with torch.no_grad():
            output = self.model(waveform)
        
        # Trim padding
        return output[:, :, :, :total_length]
    
    def export_onnx(
        self,
        output_path: str = "mdx23c_optimized.onnx",
        opset_version: int = 14,
    ):
        """
        Export model to ONNX format for faster inference.
        
        Args:
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
        """
        logger.info(f"Exporting model to ONNX: {output_path}")
        
        # Create dummy input
        dummy_input = torch.randn(1, 2, self.chunk_size, device=self.device)
        
        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"ONNX model saved to {output_path}")


def benchmark_performance(
    audio_path: str,
    overlaps: List[int] = [2, 4, 6, 8],
    batch_sizes: List[int] = [1, 2, 4, 8],
    device: str = "cpu",
):
    """
    Benchmark performance with different settings.
    
    Args:
        audio_path: Path to test audio file
        overlaps: List of overlap values to test
        batch_sizes: List of batch sizes to test
        device: Device to test on
    """
    print("\n" + "="*60)
    print("MDX23C Performance Benchmark")
    print("="*60)
    print(f"Audio file: {audio_path}")
    print(f"Device: {device}")
    
    # Load audio once to get duration
    waveform, sr = torchaudio.load(audio_path)
    duration = waveform.shape[-1] / sr
    print(f"Duration: {duration:.1f} seconds")
    print("\n")
    
    results = []
    
    for batch_size in batch_sizes:
        for overlap in overlaps:
            print(f"Testing: batch_size={batch_size}, overlap={overlap}")
            
            # Initialize processor
            processor = OptimizedMDX23CProcessor(
                device=device,
                batch_size=batch_size,
                use_fp16=(device == "cuda"),
                optimize_for_inference=True
            )
            
            # Warm-up run
            processor.process_audio(audio_path, overlap=overlap, verbose=False)
            
            # Timed run
            start_time = time.perf_counter()
            processor.process_audio(audio_path, overlap=overlap, verbose=False)
            end_time = time.perf_counter()
            
            process_time = end_time - start_time
            real_time_factor = process_time / duration
            
            results.append({
                'batch_size': batch_size,
                'overlap': overlap,
                'time': process_time,
                'rtf': real_time_factor
            })
            
            print(f"  Time: {process_time:.2f}s (RTF: {real_time_factor:.2f}x)")
            
            # Clear cache
            if device == "cuda":
                torch.cuda.empty_cache()
    
    # Print summary
    print("\n" + "="*60)
    print("Summary (sorted by speed):")
    print("="*60)
    
    results.sort(key=lambda x: x['time'])
    
    print(f"{'Config':<30} {'Time (s)':<10} {'RTF':<10} {'Speedup':<10}")
    print("-"*60)
    
    baseline_time = results[-1]['time']  # Slowest as baseline
    
    for r in results:
        config = f"batch={r['batch_size']}, overlap={r['overlap']}"
        speedup = baseline_time / r['time']
        print(f"{config:<30} {r['time']:<10.2f} {r['rtf']:<10.2f} {speedup:<10.2f}x")


def main():
    """Example usage and benchmarking."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Optimized MDX23C drum separation with batching"
    )
    parser.add_argument(
        "input",
        help="Input audio file path"
    )
    parser.add_argument(
        "-o", "--output",
        default="output_stems",
        help="Output directory for stems"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=4,
        help="Overlap factor (2-50), lower=faster (default: 4)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for processing (default: 4)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Processing device"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark"
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export model to ONNX format"
    )
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    if args.benchmark:
        # Run benchmark
        benchmark_performance(
            args.input,
            overlaps=[2, 4, 6, 8],
            batch_sizes=[1, 2, 4, 8] if args.device == "cuda" else [1, 2, 4],
            device=args.device
        )
    else:
        # Normal processing
        processor = OptimizedMDX23CProcessor(
            device=args.device,
            batch_size=args.batch_size,
            use_fp16=(args.device == "cuda"),
            optimize_for_inference=True
        )
        
        if args.export_onnx:
            processor.export_onnx()
            print("ONNX model exported successfully")
            return 0
        
        # Process audio
        print(f"Processing {args.input}")
        print(f"Settings: overlap={args.overlap}, batch_size={args.batch_size}")
        
        stems = processor.process_audio(
            args.input,
            overlap=args.overlap,
            verbose=True
        )
        
        # Save stems
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving stems to {output_dir}")
        for instrument, waveform in stems.items():
            output_path = output_dir / f"{instrument}.wav"
            torchaudio.save(str(output_path), waveform, processor.target_sr)
            print(f"  Saved {instrument}.wav")
        
        print("\n✅ Separation complete!")
    
    return 0


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    exit(main())