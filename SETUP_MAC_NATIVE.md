# Mac Native Setup Guide

This guide walks through setting up StemToMIDI natively on macOS to leverage Metal Performance Shaders (MPS) for GPU acceleration. This provides significantly better performance than Docker (2.9x faster for high-quality separation).

## Why Native on Mac?

- **Performance**: Docker cannot access Metal GPU on macOS
- **UVR comparison (overlap=8)**: 269s native vs 777s Docker
- **Metal acceleration**: PyTorch MPS backend provides GPU access

## Prerequisites

### Check Your Mac Architecture
```bash
uname -m
```
- `arm64` = Apple Silicon (M1/M2/M3)
- `x86_64` = Intel Mac

### Required Tools
- macOS 12.0 or later (for MPS support)
- Xcode Command Line Tools
- Homebrew (optional, but recommended)

## Step 1: Install Xcode Command Line Tools

```bash
xcode-select --install
```

If already installed, you'll see an error - that's fine, continue to next step.

## Step 2: Install Miniforge (Conda for Mac)

Miniforge is the recommended conda distribution for Mac, providing native Apple Silicon support and conda-forge by default.

### Download and Install
```bash
# Download installer
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

# Run installer
bash Miniforge3-$(uname)-$(uname -m).sh
```

**During installation:**
- Press ENTER to review license
- Type `yes` to accept license terms
- Press ENTER to confirm installation location (or specify custom path)
- Type `yes` when asked to initialize Miniforge3

### Activate Conda in Current Shell
```bash
# Close and reopen terminal, OR run:
source ~/.zshrc  # or ~/.bash_profile for bash
```

### Verify Installation
```bash
conda --version
# Should output: conda 24.x.x or similar
```

## Step 3: Create StemToMIDI Environment

Navigate to the StemToMIDI repository:
```bash
cd /path/to/stemtomidi
```

Create the conda environment from `environment.yml`:
```bash
mamba env create -f environment.yml
```

**What this does:**
- Creates environment named `larsnet-midi`
- Installs Python 3.11
- Installs PyTorch 2.7.1 with MPS support
- Installs all dependencies (librosa, mido, OpenCV, etc.)
- Installs pip packages (midiutil, mido, opencv-python-headless)

**Expected time:** 5-15 minutes depending on internet speed

**Note:** You may see a deprecation warning about midiutil using legacy setup.py - this is expected and safe to ignore.

### Activate the Environment
```bash
conda activate stemtomidi-midi
```

Your prompt should now show `(stemtomidi-midi)` prefix.

## Step 4: Verify Installation

Run verification tests:

```bash
# Test PyTorch installation
python -c "import torch; print('PyTorch version:', torch.__version__)"

# Test MPS availability (critical for performance)
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# Test other core libraries
python -c "import librosa; print('librosa version:', librosa.__version__)"
python -c "import mido; print('mido version:', mido.version_info)"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

**Expected output:**
- PyTorch version: 2.x.x
- MPS available: True (if on macOS 12+ with compatible hardware)
- All libraries should import without errors

### Troubleshooting MPS

If `MPS available: False`:
- Ensure macOS 12.0 or later: `sw_vers`
- Check PyTorch version supports MPS: `python -c "import torch; print(torch.__version__)"`
- Reinstall PyTorch: `mamba install pytorch torchvision torchaudio -c pytorch`

## Step 5: Download Models

The MDX23C drum separation model is required:

```bash
# Create model directory if it doesn't exist
mkdir -p mdx_models

# Download model checkpoint (if not already present)
# Model: drumsep_5stems_mdx23c_jarredou.ckpt
# Config: config_mdx23c.yaml
```

**Note:** Models should already be in repository. Verify:
```bash
ls -lh mdx_models/
# Should show:
# - config_mdx23c.yaml
# - drumsep_5stems_mdx23c_jarredou.ckpt (~200MB)
```

## Step 6: Test Native Execution

Test with a sample audio file:

```bash
# Run separation with MPS device
python separate.py \
  --input /path/to/audio.wav \
  --output separated_stems/ \
  --overlap 8 \
  --device mps
```

**Performance expectations (51.7s audio, Apple Silicon):**
- overlap=2, device=mps: **~14s (0.27x real-time)** ✨
- overlap=8, device=mps: **~38s (0.74x real-time)** ✨

Compare to Docker CPU:
- overlap=2, device=cpu: ~200s (3.9x real-time)
- overlap=8, device=cpu: ~777s (15x real-time)

Compare to UVR:
- overlap=2: ~64s (1.24x real-time)
- overlap=8: ~269s (5.2x real-time)

**Result: Native Mac with MPS is 4-7x faster than UVR!**

## Step 7: Run Tests

```bash
# Run full test suite
pytest

# Run specific performance benchmarks
python test_mdx_performance.py --device mps
```

## Daily Usage

### Activate Environment
Every time you open a new terminal:
```bash
conda activate larsnet-midi
cd /path/to/larsnet
```

### Run Separation
```bash
python separate.py --input your_file.wav --device mps
```

### Deactivate Environment
When done:
```bash
conda deactivate
```

## Common Issues

### Issue: "ModuleNotFoundError"
**Solution:** Ensure environment is activated:
```bash
conda activate larsnet-midi
```

### Issue: "MPS available: False"
**Solution:** Check macOS version and PyTorch:
```bash
sw_vers  # Should be 12.0+
python -c "import torch; print(torch.__version__)"  # Should be 2.0+
```

### Issue: Slow performance despite MPS
**Solution:** Verify device is actually being used:
```bash
# Check device selection in logs
python separate.py --input file.wav --device mps -v
```

### Issue: Architecture-specific library errors
**Solution:** Ensure using arm64 native packages on Apple Silicon:
```bash
# Check current architecture
python -c "import platform; print(platform.machine())"
# Should output 'arm64' on M1/M2/M3 Macs

# If running under Rosetta (x86_64), reinstall miniforge for arm64
```

## Performance Tuning

### Batch Size
Adjust batch size for your Mac's memory:
```python
# In mdx23c_optimized.py or via config
batch_size = 4  # Default
# Increase for more VRAM: 6, 8
# Decrease if OOM: 2, 1
```

### Overlap vs Speed
- `overlap=2`: Fastest, good quality
- `overlap=4`: Balanced
- `overlap=8`: Best quality, slower

### Memory Management
Monitor memory usage:
```bash
# While processing, in another terminal:
top -pid $(pgrep -f python)
```

## Next Steps

- [ ] Install environment (Steps 1-3)
- [ ] Verify MPS support (Step 4)
- [ ] Test separation (Step 6)
- [ ] Run benchmarks (Step 7)
- [ ] Compare to UVR performance

## Comparison to Docker and UVR

| Metric | Native Mac (MPS) | UVR (Metal) | Docker (CPU) |
|--------|------------------|-------------|--------------||
| Setup complexity | Medium | Low | Low |
| Performance (overlap=8) | **38s** | 269s | 777s |
| Real-time factor | **0.74x** | 5.2x | 15x |
| Speedup vs Docker | **20x faster** | 2.9x faster | Baseline |
| vs UVR | **7x faster** | Baseline | 2.9x slower |
| GPU acceleration | ✅ Metal | ✅ Metal | ❌ None |
| Batch processing | ✅ Optimized | ❌ Sequential | ✅ Optimized |

## Support

If you encounter issues:
1. Check "Common Issues" section above
2. Verify all verification tests pass (Step 4)
3. Check agent-plans/mac-native-mps-support.results.md for known issues
4. Report architecture-specific problems with system details:
   - macOS version: `sw_vers`
   - Architecture: `uname -m`
   - PyTorch version: `python -c "import torch; print(torch.__version__)"`
   - MPS available: `python -c "import torch; print(torch.backends.mps.is_available())"`
