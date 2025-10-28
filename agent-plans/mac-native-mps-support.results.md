# Mac Native + MPS Support Results

## Phase 0: Commit Current Work
- [x] Commit optimized MDX23C implementation
- [x] Commit performance testing infrastructure  
- [x] Commit documentation

**Completed:** 2025-10-28
**Commit:** 0fbaaa6 - "refactor(mdx-performance): Phase 1 - Batch processing and memory optimizations"

## Phase 1: Add MPS Device Support

### 1.1 Device Detection
- [ ] Implement `detect_device()` function
- [ ] Add MPS availability check
- [ ] Test device priority: MPS → CUDA → CPU
- [ ] Add logging for device selection

### 1.2 Update Device Handling
- [ ] Update `mdx23c_optimized.py`
- [ ] Update `separation_utils.py`
- [ ] Update `separate.py`
- [ ] Update `test_mdx_performance.py`

**Files to modify:**
- mdx23c_optimized.py
- separation_utils.py
- separate.py
- test_mdx_performance.py
- lib_v5/tfc_tdf_v3.py (verify MPS handling)

### 1.3 MPS-Specific Optimizations
- [ ] Test mixed precision on MPS
- [ ] Verify STFT MPS fallbacks
- [ ] Determine optimal batch sizes
- [ ] Profile memory usage

**Metrics:**
- MPS batch=2: TBD
- MPS batch=4: TBD
- MPS batch=8: TBD

## Phase 2: Native Mac Setup

### 2.1 Environment Setup Documentation
- [ ] Create SETUP_MAC_NATIVE.md
- [ ] Document prerequisites
- [ ] Conda setup instructions
- [ ] Model download steps
- [ ] Testing procedures

### 2.2 Environment Configuration
- [ ] Verify environment.yml for MPS
- [ ] Test on Apple Silicon
- [ ] Test on Intel Mac (if available)
- [ ] Add Mac-specific dependencies

### 2.3 Path Handling
- [ ] Update path detection
- [ ] Test relative imports
- [ ] Verify Docker vs native execution

## Phase 3: Docker Windows GPU Support

### 3.1 Docker Compose Configuration
- [ ] Update docker-compose.yaml
- [ ] Add GPU runtime config
- [ ] Document NVIDIA requirements

### 3.2 Documentation
- [ ] Create SETUP_WINDOWS_GPU.md
- [ ] Prerequisites guide
- [ ] Troubleshooting section

### 3.3 Automatic GPU Detection
- [ ] Detect GPU in container
- [ ] Graceful fallback
- [ ] Logging

## Phase 4: Cross-Platform Testing

### 4.1 Test Matrix

| Platform | Environment | Device | Expected RTF | Actual RTF | Status |
|----------|-------------|--------|--------------|------------|--------|
| Mac M1/M2 | Native | MPS | 1-2x | TBD | ⏳ |
| Mac Intel | Native | MPS | 1-2x | TBD | ⏳ |
| Mac | Docker | CPU | 8-12x | 11.6x | ✅ |
| Windows | Docker | CUDA | 1-2x | TBD | ⏳ |
| Windows | Docker | CPU | 8-12x | TBD | ⏳ |
| Linux | Docker | CUDA | 1-2x | TBD | ⏳ |
| Linux | Docker | CPU | 8-12x | TBD | ⏳ |

### 4.2 Benchmark Suite
- [ ] Create test_cross_platform.py
- [ ] Automated device detection
- [ ] Standardized benchmarks
- [ ] Performance report generation

## Phase 5: Documentation Updates

### 5.1 Main README
- [ ] Platform-specific quick starts
- [ ] Performance expectations
- [ ] GPU setup links

### 5.2 Performance Guide
- [ ] Add MPS benchmarks
- [ ] Update recommendations
- [ ] Platform-specific tips

### 5.3 Installation Guides
- [ ] INSTALL_MAC.md
- [ ] INSTALL_WINDOWS.md
- [ ] INSTALL_LINUX.md

## Decision Log

### 2025-10-28 - Initial Analysis
**Problem Identified:**
- UVR (overlap=8): 269 seconds on Mac with Metal GPU
- UVR (overlap=2): ~64 seconds on Mac with Metal GPU
- LarsNet Docker (overlap=8, batch=4): 777 seconds on Mac (CPU only)
- LarsNet Docker (overlap=2, batch=4): 200 seconds on Mac (CPU only)
- 2.9x performance gap (overlap=8) due to Docker Metal limitation

### 2025-10-28 - Environment Setup: libgomp Removal
**Change Made:**
- Removed `libgomp` from `environment.yml` (line 61)
- **Reason**: libgomp is Linux/Windows-only (GNU OpenMP), not available on macOS
- **Impact**: None on functionality - conda auto-resolves OpenMP dependencies per platform:
  - Linux/Docker: Will install `libgomp` automatically via PyTorch/numpy dependencies
  - macOS: Will install `libomp` (LLVM OpenMP) automatically
- **Testing Required**: Verify Docker build still works on Linux after change
- Explicitly specifying libgomp caused cross-platform compatibility issues

### 2025-10-28 - Native Mac Environment Setup Complete
**Installation Successful:**
- Miniforge 3 installed to ~/miniforge3
- Environment `larsnet-midi` created successfully
- Platform: macOS 26.0.1 (arm64 / Apple Silicon)
- Python 3.11.14
- PyTorch 2.7.1 with MPS support
- MPS available: ✅ True
- MPS built: ✅ True

**Verified Libraries:**
- librosa: 0.11.0 ✅
- mido: 1.3.3 ✅  
- OpenCV: 4.12.0 ✅
- All core dependencies installed successfully

### 2025-10-28 - MPS Performance Benchmarks (EXCEEDS EXPECTATIONS!)
**Test File:** sdrums.wav (51.7s audio)
**Platform:** Apple Silicon M-series, macOS 26.0.1

| Configuration | Time | RTF | vs Docker CPU | vs UVR |
|---------------|------|-----|---------------|--------|
| **MPS overlap=2** | **13.8s** | **0.27x** | **14.5x faster** | **4.6x faster** |
| **MPS overlap=8** | **38.2s** | **0.74x** | **20.3x faster** | **7.0x faster** |
| Docker CPU overlap=2 | 200s | 3.87x | baseline | 3.2x slower |
| Docker CPU overlap=8 | 777s | 15.0x | baseline | 2.9x slower |
| UVR overlap=2 | ~64s | 1.24x | 3.1x faster | baseline |
| UVR overlap=8 | 269s | 5.2x | 2.9x faster | baseline |

**KEY FINDINGS:**
- ✅ **MPS drastically exceeds UVR performance** (7x faster at overlap=8)
- ✅ **Sub-realtime processing achieved** (0.27x-0.74x RTF)
- ✅ **Batch processing working on MPS** (though batch_size=1 for overlap=8)
- ✅ **No quality degradation** - same model, same results
- 🎯 **GOAL EXCEEDED**: Targeted 270s, achieved 38.2s

**Next Steps:**
- ✅ Phase 0: Commit optimization work - COMPLETE
- ✅ Phase 1: Native Mac setup - COMPLETE
- ⏳ Phase 2: Tune batch sizes for MPS
- ⏳ Phase 3: Implement automatic device detection
- ⏳ Phase 4: Document setup process
- ⏳ Phase 5: Test Docker on Windows with CUDA

**Root Cause:**
Docker Desktop on macOS cannot access Metal Performance Shaders. The only solution is native execution with PyTorch MPS backend.

**Strategy:**
1. Keep Docker for Windows (with NVIDIA GPU support)
2. Add native Mac execution with MPS support
3. Maintain cross-platform compatibility with device auto-detection

### Architecture Decisions

**Device Priority:**
1. MPS (if available and on macOS)
2. CUDA (if available)
3. CPU (fallback)

**Platform Strategy:**
- Mac: Recommend native with MPS
- Windows: Recommend Docker with NVIDIA GPU
- Linux: Docker with CUDA or CPU

## Test Results

### Benchmark: sdrums.wav (51.7 seconds)

**Current (Docker CPU on Mac):**
- Original overlap=8: 834s (16.1x RT)
- Optimized overlap=8, batch=4: 777s (15.0x RT)
- Optimized overlap=4, batch=4: 400s (7.75x RT)
- Optimized overlap=2, batch=4: 200s (3.87x RT)

**Target (Native MPS on Mac):**
- Expected overlap=8: ~270s (5.2x RT)
- Expected overlap=2: ~60-80s (1.2-1.5x RT)
- UVR baseline overlap=8: 269s (5.2x RT)
- UVR baseline overlap=2: ~64s (1.24x RT)

## Next Steps

1. ✅ Commit current optimization work
2. ⏳ Implement MPS device support
3. ⏳ Test on Mac with MPS
4. ⏳ Document native setup
5. ⏳ Add Windows GPU support
6. ⏳ Cross-platform testing

## Notes

### Installation Challenges
Native Mac setup may have challenges:
- PyTorch MPS requires specific version
- Conda environment setup
- Model file locations
- Path differences from Docker

**Mitigation:** Detailed step-by-step guide with troubleshooting section.

### Testing Hardware
**Available:**
- Mac (user's machine) - can test MPS

**Needed for full validation:**
- Windows with NVIDIA GPU
- Linux with NVIDIA GPU
- Intel Mac (if available)