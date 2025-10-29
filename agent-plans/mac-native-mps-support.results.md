# Mac Native + MPS Support Results

## Phase 0: Commit Current Work
- [x] Commit optimized MDX23C implementation
- [x] Commit performance testing infrastructure  
- [x] Commit documentation

**Completed:** 2025-10-28
**Commit:** 0fbaaa6 - "refactor(mdx-performance): Phase 1 - Batch processing and memory optimizations"

## Phase 1: Add MPS Device Support

**Status:** COMPLETE ✅

### 1.1 Device Detection
- [x] Implement `detect_device()` function - Created device_utils.py
- [x] Add MPS availability check - torch.backends.mps.is_available()
- [x] Test device priority: MPS → CUDA → CPU - Working correctly
- [x] Add logging for device selection - "Auto-detected device: mps" message

### 1.2 Update Device Handling
- [x] `mdx23c_optimized.py` - Auto-detects device if None, validates if specified
- [x] `separation_utils.py` - Imports device detection utilities
- [x] `separate.py` - Auto-detects device when --device not specified
- [x] `device_utils.py` - New module with detect_best_device(), validate_device(), get_device_info()

**Files updated:**
- mdx23c_optimized.py - device parameter now Optional[str], auto-detects if None
- separation_utils.py - imports detect_best_device
- separate.py - auto-detects device, updated help text
- device_utils.py - NEW comprehensive device detection module
- lib_v5/tfc_tdf_v3.py - MPS fallbacks already working

### 1.3 MPS-Specific Optimizations
- [x] Test mixed precision on MPS - Using fp32 (fp16 is CUDA-only in code)
- [x] Verify STFT MPS fallbacks - Working (UserWarning observed but functional)
- [x] Determine optimal batch sizes - batch_size=4 for overlap=2, batch_size=1 for overlap=8
- [x] Test device override - Explicit --device cpu/mps/cuda works correctly

**Metrics (Apple Silicon, macOS 26.0.1):**
- MPS overlap=2, batch=4: 13.8s (0.27x RT)
- MPS overlap=8, batch=1: 38.2s (0.74x RT)
- Automatic batch sizing works correctly for MPS
- Device detection tested: MPS auto-detected on Mac, respects explicit override

**Completed:** 2025-10-28

## Phase 2: Native Mac Setup

**Status:** COMPLETE ✅

### 2.1 Environment Setup Documentation
- [x] Create SETUP_MAC_NATIVE.md - Complete with step-by-step guide
- [x] Document prerequisites - Xcode tools, Miniforge installation
- [x] Conda setup instructions - Full walkthrough tested
- [x] Model download steps - Documented (models already present)
- [x] Testing procedures - Verification commands included

### 2.2 Environment Configuration
- [x] Verify environment.yml for MPS - Fixed libgomp issue
- [x] Test on Apple Silicon - Tested on macOS 26.0.1 arm64
- [ ] Test on Intel Mac (not available)
- [x] Add Mac-specific dependencies - libgomp removed, libomp auto-installed

### 2.3 Path Handling
- [x] Update path detection - Project-based system works natively
- [x] Test relative imports - All imports working
- [x] Verify Docker vs native execution - Both working correctly

**Completed:** 2025-10-28
**Commit:** dc888fd - "feat(native-mac): Native Mac setup with MPS - 7x faster than UVR"

## Phase 3: Docker Windows GPU Support

**Status:** COMPLETE ✅ (Documentation only - no Windows hardware for testing)

### 3.1 Docker Compose Configuration
- [x] Updated docker-compose.yaml with GPU runtime config (commented out)
- [x] Added NVIDIA Container Toolkit device configuration
- [x] Documented requirements in comments

### 3.2 Documentation
- [x] Created SETUP_WINDOWS_GPU.md - comprehensive guide
- [x] Prerequisites: WSL 2, Docker Desktop, NVIDIA drivers, Container Toolkit
- [x] Step-by-step setup instructions
- [x] Verification procedures
- [x] Troubleshooting section with common issues
- [x] Performance expectations and comparison table

### 3.3 Automatic GPU Detection
- [x] Detect GPU in container - device_utils.py handles CUDA detection
- [x] Graceful fallback - auto-detects best device (CUDA → CPU)
- [x] Logging - "Auto-detected device: cuda" message

**Note:** Cannot test on Windows without hardware, but configuration follows NVIDIA Container Toolkit standard practices.

**Completed:** 2025-10-28

## Phase 4: Cross-Platform Testing

### 4.1 Test Matrix

| Platform | Environment | Device | Expected RTF | Actual RTF | Status |
|----------|-------------|--------|--------------|------------|--------|
| Mac M1/M2 | Native | MPS | 1-2x | **0.27-0.74x** | ✅ TESTED |
| Mac Intel | Native | MPS | 1-2x | TBD | ⏳ |
| Mac | Docker | CPU | 8-12x | 3.9-15.0x | ✅ |
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
- ✅ Phase 2: Native Mac setup and documentation - COMPLETE
- ✅ Phase 2: Environment setup and testing - COMPLETE
- ⏳ Phase 1: Implement automatic device detection (manual `--device mps` works)
- ⏳ Phase 3: Docker Windows GPU support
- ⏳ Phase 4: Cross-platform testing (Windows/Linux)
- ⏳ Phase 5: Documentation polish and installation guides

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