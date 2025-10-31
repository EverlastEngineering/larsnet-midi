# Dependency Management

This document explains DrumToMIDI's dependency management strategy and provides guidance for contributors.

## Philosophy

We use **Conda** (via `environment.yml`) for dependency management because:

1. **Cross-platform reproducibility**: Works consistently across Linux (x86_64, aarch64), macOS (Intel, Apple Silicon), and Docker
2. **System-level dependencies**: Conda manages native libraries (BLAS, LAPACK, FFmpeg) that pip cannot
3. **PyTorch platform handling**: Conda handles platform-specific PyTorch builds (CPU/CUDA/MPS) better than pip
4. **Audio library dependencies**: librosa, soundfile, and related packages have complex native dependencies

## Complete Dependency List

### Production Dependencies

| Package | Version | Purpose | Source |
|---------|---------|---------|--------|
| **python** | 3.11 | Interpreter | conda |
| **numpy** | ≥1.24.0 | Array operations | conda |
| **scipy** | ≥1.10.0 | Signal processing | conda |
| **pytorch** | ≥2.0.0 | Deep learning | conda |
| **torchaudio** | ≥2.0.0 | Audio ML | conda |
| **librosa** | ≥0.10.0 | Audio analysis | conda |
| **soundfile** | ≥0.12.0 | Audio I/O | conda |
| **pyyaml** | ≥6.0 | Config files | conda |
| **tqdm** | ≥4.60.0 | Progress bars | conda |
| **midiutil** | ≥1.2.1 | MIDI creation | pip |
| **mido** | ≥1.3.0 | MIDI reading | pip |
| **opencv-python-headless** | ≥4.8.0 | Video rendering | pip |
| **scikit-learn** | ≥1.3.0 | Tom pitch clustering (optional) | conda |
| **pytest** | ≥7.0.0 | Testing | conda |

### System Libraries (Managed by Conda)

- **libopenblas**: Optimized BLAS implementation
- **libgomp**: OpenMP for parallel processing
- **zlib, bzip2, xz**: Compression libraries

## Usage by Module

### Root Scripts

**separation_utils.py**
- torch, torchaudio
- pyyaml
- soundfile
- tqdm

**separate.py, separate_with_eq.py**
- separation_utils (above dependencies)

**stems_to_midi.py**
- librosa
- soundfile
- midiutil
- mido
- pyyaml

**render_midi_to_video.py**
- mido
- opencv-python-headless (cv2)
- numpy

**sidechain_cleanup.py**
- numpy
- soundfile
- scipy

### stems_to_midi Package

**detection.py**
- librosa
- numpy
- scipy

**helpers.py**
- numpy
- scipy
- scikit-learn (optional, with fallback)

**learning.py**
- librosa
- soundfile
- numpy
- pyyaml
- mido

**processor.py**
- soundfile
- numpy

**midi.py**
- midiutil
- mido

**config.py**
- pyyaml

**test_*.py**
- pytest
- All above dependencies

## Platform-Specific Notes

### Linux (x86_64, aarch64)
- All dependencies work out of the box
- For GPU: Install CUDA-enabled PyTorch separately

### macOS (Intel, Apple Silicon)
- MPS (Metal Performance Shaders) support available for Apple Silicon
- opencv-python-headless works better than opencv-python on ARM

### Docker
- Uses miniforge3 base image
- Environment automatically activated in container
- Headless OpenCV avoids GUI dependencies

## Why Not pip?

We evaluated using `requirements.txt` but chose Conda because:

1. **PyTorch complexity**: Pip requires users to manually select CPU/CUDA/ROCm variants
2. **Audio libraries**: librosa has complex dependencies (FFmpeg, libsndfile) that conda handles automatically
3. **BLAS/LAPACK**: Conda ensures optimized linear algebra libraries are correctly linked
4. **Consistency**: One tool for all dependencies reduces environment-related bugs

## For Contributors

### Adding a New Dependency

1. Check if available on conda-forge: `conda search -c conda-forge package_name`
2. If yes, add to `dependencies:` section in `environment.yml`
3. If no, add to `pip:` section
4. Document in this file
5. Update Dockerfile verification if critical

### Testing Environment

```bash
# From scratch
conda env remove -n DrumToMIDI-midi
```

```bash
conda activate DrumToMIDI-midi
conda env create -f environment.yml
conda activate larsnet

# Run tests
pytest stems_to_midi/

# Test separation
python separate.py -i input/ -o output/
```

### Reproducible Environments with conda-lock

**What is conda-lock?**

conda-lock generates platform-specific lockfiles from `environment.yml` that pin exact package versions and checksums. This ensures identical environments across machines and over time, similar to `package-lock.json` for npm or `Pipfile.lock` for pipenv.

**When to use it:**

- CI/CD pipelines requiring exact reproducibility
- Deploying to production environments
- Sharing exact environments with collaborators
- Preventing dependency drift over time

**Installation:**

```bash
conda install -c conda-forge conda-lock
```

**Generating lockfiles:**

```bash
# Generate unified lockfile for multiple platforms
conda-lock -f environment.yml -p linux-64 -p linux-aarch64

# Note: Cannot generate osx-arm64 locks from Linux due to platform-specific 
# libraries like libgomp. Run from macOS to include osx-arm64.
```

This creates `conda-lock.yml` containing pinned versions for all specified platforms.

**Using lockfiles:**

```bash
# Install from lockfile
conda-lock install -n DrumToMIDI-midi conda-lock.yml

# Update specific package
conda-lock lock --lockfile conda-lock.yml --update numpy

# Regenerate from updated environment.yml
conda-lock -f environment.yml --lockfile conda-lock.yml
```

**Current state:**

The repository includes `conda-lock.yml` with locks for linux-64 and linux-aarch64. macOS users should regenerate locks on their platform if exact reproducibility is needed.

## Deprecated Files

The following files are **deprecated** and should **not be used**:

- ❌ `requirements.txt.deprecated.txt`
- ❌ `requirements_midi.txt` (consolidated into environment.yml)
- ❌ `requirements_visualization.txt` (consolidated into environment.yml)
- ❌ `pyproject.toml.deprecated.toml`
- ❌ `base.txt.deprecated.txt`
- ❌ `clean.txt.deprecated.txt`

These are kept for historical reference but should not be maintained or used for new installations.

## Optional: GPU Support

### NVIDIA CUDA

```bash
# After creating environment, install CUDA PyTorch
conda activate larsnet
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### AMD ROCm

```bash
# After creating environment, install ROCm PyTorch
conda activate larsnet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

### Apple MPS (Metal)

Already included with default PyTorch installation on macOS.

## Troubleshooting

### Slow conda install
Use mamba instead:
```bash
conda install -c conda-forge mamba
mamba env create -f environment.yml
```

### Package conflicts
Clear conda cache and retry:
```bash
conda clean --all
conda env create -f environment.yml
```

### Missing sklearn
scikit-learn is optional. The code has fallback for tom pitch classification.

### OpenCV not found
Make sure opencv-python-headless is installed via pip (not opencv-python from conda).

## Questions?

If you encounter dependency issues:
1. Check your platform is supported (Linux, macOS)
2. Verify conda/mamba version: `conda --version` (should be ≥23.0)
3. Try with fresh environment
4. Open an issue with your platform and conda list output
