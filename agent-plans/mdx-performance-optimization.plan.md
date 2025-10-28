# MDX Performance Optimization Plan

## Problem Statement
The MDX23C model implementation in this codebase is significantly slower than UVR (Ultimate Vocal Remover) despite producing identical quality output. The model processes drum audio to separate it into 5 stems (kick, snare, hihat, toms, cymbals).

## Current Implementation Analysis

### Model Architecture
- **Model**: TFC_TDF_net (Time-Frequency/Channel - Time-Domain/Frequency)
- **Checkpoint**: drumsep_5stems_mdx23c_jarredou.ckpt
- **Input**: Stereo audio at 44.1kHz
- **Output**: 5 stereo stems
- **Chunk Size**: 523,776 samples (~11.9 seconds at 44.1kHz)

### Processing Flow
1. Audio loaded and resampled to 44.1kHz if needed
2. Converted to stereo (2 channels)
3. Split into overlapping chunks (default overlap=8, meaning hop_length = chunk_size/8)
4. Each chunk processed through model
5. Overlap-add to combine chunks
6. Output averaged by overlap count

### Current Performance Issues

#### 1. **Inefficient Chunking Strategy**
- Default overlap=8 means each sample is processed 8 times
- Hop length = 523,776 / 8 = 65,472 samples (~1.48 seconds)
- For a 3-minute song: ~122 chunks to process
- No batch processing of chunks

#### 2. **Memory Management**
- Each chunk allocated separately
- No reuse of intermediate tensors
- Overlap buffer allocated for full song length upfront

#### 3. **STFT Implementation**
- Custom STFT class with MPS fallback to CPU
- Potential inefficiency in frequency domain conversions
- No caching of window functions

#### 4. **Missing Optimizations**
- No batch processing of multiple chunks
- No ONNX export for faster inference
- No mixed precision (fp16) usage
- No cudnn.benchmark tuning for CUDA
- Sequential processing only

## Optimization Strategy

### Phase 1: Immediate Optimizations (No Model Changes)

#### 1.1 Batch Processing
- Process multiple chunks in a single forward pass
- Optimal batch size based on available VRAM/RAM

#### 1.2 Reduce Default Overlap
- Test quality with overlap=4 (75% overlap) vs overlap=8 (87.5%)
- Provide quality vs speed tradeoffs to users

#### 1.3 Optimize Memory Usage
- Pre-allocate reusable buffers
- Use in-place operations where possible
- Clear gradients explicitly (even with no_grad)

### Phase 2: Advanced Optimizations

#### 2.1 ONNX Export
- Export model to ONNX for inference
- Use ONNXRuntime with optimization providers
- Benchmark against PyTorch

#### 2.2 Mixed Precision
- Enable fp16/bf16 for supported hardware
- Use autocast for automatic precision management

#### 2.3 Optimize STFT Operations
- Use native PyTorch STFT optimizations
- Cache window functions
- Investigate using convolution-based STFT

#### 2.4 Parallel Processing
- Process independent frequency bands in parallel
- Use multiprocessing for multi-file batches

### Phase 3: Architecture Optimizations

#### 3.1 Model Quantization
- INT8 quantization for CPU inference
- Dynamic quantization for weights

#### 3.2 Pruning
- Identify and remove redundant connections
- Structured pruning for speed gains

#### 3.3 Knowledge Distillation
- Train smaller student model
- Maintain quality with faster inference

## Testing & Validation

### Benchmarks to Track
1. **Inference Time**: Seconds per minute of audio
2. **Memory Usage**: Peak RAM/VRAM usage
3. **Quality Metrics**: SDR/SIR/SAR compared to current
4. **Real-time Factor**: Processing speed vs playback speed

### Test Cases
1. Short clip (30 seconds)
2. Standard song (3-4 minutes) 
3. Long mix (10+ minutes)
4. Various sample rates and channel configs

### Quality Validation
- A/B testing with current implementation
- Null test to verify identical output with overlap=8
- Listening tests for lower overlap values

## Success Criteria

### Minimum Goals
- 2x speedup with identical quality (same overlap)
- 4x speedup with acceptable quality loss (<1dB SDR)
- Documentation of all speed/quality tradeoffs

### Stretch Goals
- Real-time processing on GPU (1x real-time factor)
- 10x speedup on CPU with quantization
- ONNX model with 5x speedup

## Risk Mitigation

### Risks
1. **Quality Degradation**: Lower overlap or optimizations affect output
   - Mitigation: Extensive A/B testing, keep original path available
   
2. **Hardware Compatibility**: Optimizations don't work on all systems
   - Mitigation: Fallback paths, runtime detection
   
3. **Maintenance Burden**: Complex optimizations hard to maintain
   - Mitigation: Clean abstractions, comprehensive tests

## Implementation Order

1. Batch processing (highest impact, lowest risk)
2. Reduce default overlap with documentation
3. Memory optimizations
4. ONNX export
5. Mixed precision
6. Advanced optimizations based on profiling

## Notes on UVR Differences

UVR likely uses:
- Batch processing by default
- Optimized overlap (possibly adaptive)
- ONNX or TorchScript exports
- Hardware-specific optimizations
- Efficient memory management
- Possible custom CUDA kernels

Investigation needed:
- Profile UVR's actual processing pipeline
- Check their default parameters
- Examine their batching strategy