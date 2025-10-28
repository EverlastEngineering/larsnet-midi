# Mac Native + MPS Support Plan

## Problem Statement

UVR achieves ~64 seconds processing time on Mac using Metal GPU acceleration, while our Docker implementation takes ~600 seconds (10x slower). Docker Desktop on macOS **cannot access Metal Performance Shaders (MPS)**, forcing CPU-only execution.

### Key Findings
- **UVR on Mac**: 64s for 51.7s audio (1.24x real-time) using Metal
- **LarsNet Docker on Mac**: 600s for 51.7s audio (11.6x real-time) using CPU only
- **Root cause**: Docker isolation prevents Metal GPU access
- **CPU usage difference**: UVR shows lower CPU usage, confirming GPU acceleration

## Goals

1. **Enable native execution on Mac** with Metal GPU acceleration via PyTorch MPS backend
2. **Maintain Docker on Windows** with NVIDIA GPU support
3. **Achieve UVR-comparable performance** on Mac (~1-2x real-time)
4. **Preserve cross-platform compatibility** with graceful fallbacks

## Platform-Specific Requirements

### macOS (Native Execution)
- **GPU**: Metal Performance Shaders (MPS) via PyTorch
- **Setup**: Conda environment outside Docker
- **Target**: 60-120 seconds for 51.7s audio
- **Device string**: `mps`

### Windows (Docker Execution)
- **GPU**: NVIDIA CUDA via Docker with `--gpus all`
- **Setup**: Docker Desktop + NVIDIA Container Toolkit + WSL2
- **Target**: Similar to Mac with equivalent GPU
- **Device string**: `cuda`

### Fallback (All Platforms)
- **GPU**: None
- **Setup**: Docker or native
- **Device string**: `cpu`

## Implementation Phases

### Phase 0: Commit Current Work
- [x] Commit optimized MDX23C implementation
- [x] Commit performance testing infrastructure
- [x] Commit documentation

### Phase 1: Add MPS Device Support

#### 1.1 Device Detection
- Add `detect_device()` function that returns best available device:
  1. Try MPS (if available and on Mac)
  2. Try CUDA (if available)
  3. Fallback to CPU
- Validate MPS availability with `torch.backends.mps.is_available()`

#### 1.2 Update Device Handling
Files to modify:
- `mdx23c_optimized.py`: Add MPS to device choices
- `separation_utils.py`: Auto-detect device when not specified
- `separate.py`: Add `mps` as device option
- `test_mdx_performance.py`: Add MPS benchmarking

#### 1.3 MPS-Specific Optimizations
- Test mixed precision support on MPS (may differ from CUDA)
- Handle MPS fallbacks in STFT operations (lib_v5/tfc_tdf_v3.py already has MPS→CPU fallback)
- Optimize batch sizes for Metal architecture

### Phase 2: Native Mac Setup

#### 2.1 Environment Setup Documentation
Create `SETUP_MAC_NATIVE.md`:
- Prerequisites (Xcode Command Line Tools, Homebrew)
- Conda installation
- Environment creation from `environment.yml`
- Model download instructions
- Testing native setup

#### 2.2 Environment Configuration
- Ensure `environment.yml` includes MPS-compatible PyTorch
- Add Mac-specific dependencies if needed
- Test on both Intel and Apple Silicon Macs

#### 2.3 Path Handling
- Update scripts to handle both Docker paths (`/app/...`) and native paths
- Add workspace root detection
- Ensure relative imports work in both contexts

### Phase 3: Docker Windows GPU Support

#### 3.1 Docker Compose Configuration
Update `docker-compose.yaml`:
- Add GPU runtime configuration
- Document NVIDIA Container Toolkit requirements
- Add conditional GPU allocation

#### 3.2 Documentation
Create `SETUP_WINDOWS_GPU.md`:
- Prerequisites (NVIDIA drivers, WSL2, Docker Desktop)
- NVIDIA Container Toolkit installation
- GPU testing and validation
- Troubleshooting common issues

#### 3.3 Automatic GPU Detection in Docker
- Detect if GPU is available in container
- Fall back to CPU gracefully if GPU unavailable
- Log device selection for transparency

### Phase 4: Cross-Platform Testing

#### 4.1 Test Matrix
| Platform | Environment | Device | Expected RTF | Status |
|----------|-------------|--------|--------------|--------|
| Mac M1 | Native | MPS | 1-2x | To test |
| Mac Intel | Native | MPS | 1-2x | To test |
| Mac | Docker | CPU | 8-12x | ✅ Tested |
| Windows | Docker | CUDA | 1-2x | To test |
| Windows | Docker | CPU | 8-12x | To test |
| Linux | Docker | CUDA | 1-2x | To test |
| Linux | Docker | CPU | 8-12x | To test |

#### 4.2 Benchmark Suite
Create `test_cross_platform.py`:
- Automated device detection
- Standardized benchmarking across platforms
- Performance comparison report
- Quality validation across devices

### Phase 5: Documentation Updates

#### 5.1 Main README
Update `README.md`:
- Quick start for Mac native vs Docker
- Platform-specific instructions
- Performance expectations by platform
- GPU setup links

#### 5.2 Performance Guide
Update `MDX23C_PERFORMANCE.md`:
- Add MPS performance benchmarks
- Update device recommendations
- Platform-specific optimization tips
- Troubleshooting by platform

#### 5.3 Installation Guides
- `INSTALL_MAC.md`: Native setup with MPS
- `INSTALL_WINDOWS.md`: Docker with GPU support
- `INSTALL_LINUX.md`: Docker with GPU support

## Technical Details

### PyTorch MPS Backend
```python
# Device detection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

### Known MPS Limitations
- Mixed precision support may differ from CUDA
- Some ops fall back to CPU automatically
- Memory management differs from CUDA
- First inference may be slower (compilation)

### Batch Size Tuning
- **CUDA**: 4-8 chunks typical
- **MPS**: TBD (test 2-8 range)
- **CPU**: 2-4 chunks typical

## Testing Strategy

### Manual Testing
1. Set up native Mac environment
2. Run benchmark with MPS: `python test_mdx_performance.py --device mps`
3. Compare to UVR performance
4. Test all scripts with MPS device
5. Verify quality matches CPU/CUDA output

### Automated Testing
- Add pytest fixtures for device detection
- Mock device availability for CI
- Test device selection logic
- Validate graceful fallbacks

## Success Criteria

### Minimum Requirements
- ✅ MPS device support implemented
- ✅ Native Mac setup documented
- ✅ Performance within 2x of UVR on same hardware
- ✅ All existing functionality works with MPS
- ✅ Graceful fallback when MPS unavailable

### Stretch Goals
- Native Mac setup script (automated)
- Performance profiling tool
- Automatic batch size tuning per device
- WebUI support for native execution

## Risk Mitigation

### Risk: MPS Performance Worse Than Expected
- **Mitigation**: Profile and optimize MPS-specific bottlenecks
- **Fallback**: Document CPU performance and recommend accordingly
- **Investigation**: Compare with UVR's Metal implementation

### Risk: Breaking Changes for Docker Users
- **Mitigation**: Maintain backward compatibility
- **Testing**: Comprehensive Docker testing on all platforms
- **Documentation**: Clear migration guide if needed

### Risk: Complex Setup Barrier
- **Mitigation**: Detailed step-by-step instructions
- **Automation**: Setup scripts where possible
- **Support**: Troubleshooting guides for common issues

## Implementation Order

1. **Commit current work** (optimization code)
2. **Add MPS device detection and support**
3. **Test MPS performance on Mac**
4. **Document native Mac setup**
5. **Add Docker GPU support for Windows**
6. **Cross-platform testing and benchmarking**
7. **Update all documentation**

## Timeline Estimate

- Phase 0 (Commit): Immediate
- Phase 1 (MPS Support): 2-4 hours
- Phase 2 (Mac Native Setup): 2-3 hours
- Phase 3 (Windows GPU): 1-2 hours
- Phase 4 (Testing): 3-4 hours
- Phase 5 (Documentation): 2-3 hours
- **Total**: 10-16 hours

## Notes

### UVR Implementation Research
If needed, examine UVR's approach to:
- Metal/MPS integration
- Device selection logic
- Batch processing with GPU
- Memory management

### Future Optimizations
After MPS is working:
- Custom Metal kernels (if needed)
- CoreML conversion for deployment
- Optimized STFT for Metal
- Shared memory optimizations