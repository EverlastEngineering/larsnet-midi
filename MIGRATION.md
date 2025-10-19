# Migration Guide: Moving to Unified environment.yml

If you previously set up LarsNet using `requirements.txt`, `requirements_midi.txt`, or `requirements_visualization.txt`, follow this guide to migrate to the new unified `environment.yml` setup.

## Why Migrate?

The new setup provides:
- ✅ **Better cross-platform support** (Linux x86_64/aarch64, macOS Intel/ARM)
- ✅ **Simplified dependency management** (one file, not three)
- ✅ **Proper handling of native dependencies** (BLAS, FFmpeg, etc.)
- ✅ **No more version conflicts** between requirements files
- ✅ **Docker compatibility** out of the box

## Quick Migration (Recommended)

### Step 1: Remove Old Environment

If using pip/virtualenv:
```bash
deactivate  # Exit current virtualenv
rm -rf venv/ .venv/  # Remove old virtual environment
```

If using old conda environment:
```bash
conda deactivate
conda env remove -n myenv-arm  # Or whatever your old env was named
```

### Step 2: Install Conda/Mamba

If you don't have conda:
```bash
# Download Miniforge (includes mamba)
# For macOS ARM:
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
bash Miniforge3-MacOSX-arm64.sh

# For macOS Intel:
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh"
bash Miniforge3-MacOSX-x86_64.sh

# For Linux:
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh"
bash Miniforge3-Linux-$(uname -m).sh
```

### Step 3: Create New Environment

```bash
# Update to latest code
git pull origin main

# Create environment
mamba env create -f environment.yml
# OR: conda env create -f environment.yml

# Activate
conda activate larsnet
```

### Step 4: Verify Installation

```bash
# Check Python packages
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import librosa; print('librosa:', librosa.__version__)"
python -c "import mido; print('mido:', mido.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"

# Run tests
pytest stems_to_midi/

# Test separation (if you have pretrained models)
python separate.py -i input/ -o output/
```

Done! 🎉

## Gradual Migration (If You Need Both)

If you need to keep the old setup temporarily:

### Keep Both Environments

```bash
# Install conda if needed (see above)

# Create new environment with different name
mamba env create -f environment.yml -n larsnet-new

# You can now switch between them:
conda activate larsnet-new  # New setup
conda activate myenv-arm    # Old setup (if conda)
# OR
source venv/bin/activate    # Old setup (if pip)
```

### Test Side-by-Side

```bash
# Test with new environment
conda activate larsnet-new
python separate.py -i test_audio/ -o output_new/

# Compare with old environment
conda activate myenv-arm  # or source venv/bin/activate
python separate.py -i test_audio/ -o output_old/

# Compare results
diff -r output_new/ output_old/
```

### Once Satisfied, Remove Old Environment

```bash
conda env remove -n myenv-arm
# OR
rm -rf venv/
```

## Troubleshooting Migration Issues

### "ModuleNotFoundError" after migration

Make sure you activated the new environment:
```bash
conda activate larsnet
which python  # Should point to conda environment
```

### Old environment variables interfering

Clean shell environment:
```bash
# Edit ~/.bashrc or ~/.zshrc and remove old PYTHONPATH
unset PYTHONPATH
```

### Docker container still using old environment

Rebuild the container:
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Conda/Mamba is slow

Use mamba instead of conda:
```bash
conda install -c conda-forge mamba -y
mamba env create -f environment.yml
```

Or use libmamba solver:
```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

### Package version conflicts

This shouldn't happen with the new setup, but if it does:
```bash
conda clean --all
mamba env create -f environment.yml
```

### GPU PyTorch doesn't work after migration

Install CUDA/ROCm PyTorch separately:
```bash
conda activate larsnet

# For NVIDIA CUDA:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# For AMD ROCm:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

## What Changed?

### Before (Multiple Files)

```
requirements.txt.deprecated.txt     ❌ (incomplete, old)
requirements_midi.txt               ❌ (MIDI-specific)
requirements_visualization.txt      ❌ (Video-specific)
environment.yml                     ❌ (incomplete, old name)
```

Problems:
- Inconsistent version requirements
- Missing dependencies (opencv, scikit-learn, pytest)
- Confusing which file to use when
- Version conflicts between files

### After (Single File)

```
environment.yml                     ✅ (complete, all dependencies)
```

Benefits:
- One source of truth
- All dependencies with proper versions
- Clear documentation in comments
- Platform-specific builds handled by conda

## Updating Your Scripts

No code changes needed! All imports remain the same:

```python
# These all work exactly as before
import torch
import librosa
import mido
from stems_to_midi import process_stem_to_midi
```

## CI/CD Updates

If you have CI/CD pipelines, update them:

### Old GitHub Actions

```yaml
# OLD - Don't use
- name: Install dependencies
  run: |
    pip install -r requirements_midi.txt
    pip install -r requirements_visualization.txt
```

### New GitHub Actions

```yaml
# NEW - Use this
- name: Setup Miniforge
  uses: conda-incubator/setup-miniconda@v2
  with:
    miniforge-version: latest
    environment-file: environment.yml
    activate-environment: larsnet

- name: Verify installation
  run: |
    conda activate larsnet
    pytest stems_to_midi/
```

## Dockerfile Updates

The new Dockerfile already uses the correct setup. If you customized it:

### Old Pattern
```dockerfile
# OLD
RUN pip install -r requirements_midi.txt
```

### New Pattern
```dockerfile
# NEW
COPY environment.yml .
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "larsnet", "/bin/bash", "-c"]
```

## FAQ

**Q: Can I still use pip?**  
A: Not recommended. The environment.yml approach handles native dependencies better and ensures cross-platform compatibility.

**Q: What about pyproject.toml?**  
A: The old pyproject.toml is deprecated. We don't package LarsNet as a pip-installable library currently.

**Q: I'm getting "environment already exists" error**  
A: Remove the old environment first: `conda env remove -n larsnet` or use a different name.

**Q: Why is scikit-learn optional?**  
A: It improves tom pitch classification, but the code has a fallback using numpy percentiles if sklearn is missing.

**Q: Do I need to reinstall the pretrained models?**  
A: No, models are separate from the environment. Just point to the same directory.

**Q: Can I contribute changes to environment.yml?**  
A: Yes! But test on multiple platforms first. See `DEPENDENCIES.md` for guidelines.

## Need Help?

If migration fails:
1. Save the error message
2. Run `conda list` and save the output  
3. Note your platform: `uname -a`
4. Open an issue with this information

The maintainers can help debug platform-specific issues.
