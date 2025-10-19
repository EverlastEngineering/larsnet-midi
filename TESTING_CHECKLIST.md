# Dependency Migration Testing Checklist

Use this checklist to verify the new `environment.yml` setup works correctly on your platform.

## Pre-Migration Backup

- [ ] Save current environment package list:
  ```bash
  conda list > old_environment_backup.txt
  # OR
  pip freeze > old_requirements_backup.txt
  ```

- [ ] Test current setup still works:
  ```bash
  python separate.py --help
  python stems_to_midi.py --help
  ```

## Fresh Installation Test

### Step 1: Environment Creation

- [ ] Remove old environment (if exists):
  ```bash
  conda env remove -n myenv-arm
  conda env remove -n larsnet
  ```

- [ ] Create new environment:
  ```bash
  mamba env create -f environment.yml
  # Record time taken: _________
  ```

- [ ] Activate environment:
  ```bash
  conda activate larsnet
  which python
  # Should point to: ...envs/larsnet/bin/python
  ```

### Step 2: Package Verification

- [ ] Verify Python version:
  ```bash
  python --version
  # Expected: Python 3.11.x
  ```

- [ ] Test core imports:
  ```bash
  python -c "import numpy; print('numpy:', numpy.__version__)"
  python -c "import scipy; print('scipy:', scipy.__version__)"
  python -c "import torch; print('torch:', torch.__version__)"
  python -c "import torchaudio; print('torchaudio:', torchaudio.__version__)"
  ```

- [ ] Test audio processing imports:
  ```bash
  python -c "import librosa; print('librosa:', librosa.__version__)"
  python -c "import soundfile; print('soundfile:', soundfile.__version__)"
  ```

- [ ] Test MIDI imports:
  ```bash
  python -c "import mido; print('mido:', mido.__version__)"
  python -c "import midiutil; print('midiutil:', midiutil.__version__)"
  ```

- [ ] Test visualization imports:
  ```bash
  python -c "import cv2; print('opencv:', cv2.__version__)"
  ```

- [ ] Test optional imports:
  ```bash
  python -c "import sklearn; print('sklearn:', sklearn.__version__)"
  python -c "import pytest; print('pytest:', pytest.__version__)"
  ```

- [ ] Test all imports together:
  ```bash
  python -c "import torch, torchaudio, numpy, scipy, librosa, soundfile, mido, midiutil, cv2, sklearn, pytest; print('✅ All imports successful')"
  ```

### Step 3: Module Imports

- [ ] Test main modules:
  ```bash
  python -c "from larsnet import LarsNet; print('✅ larsnet')"
  python -c "from unet import UNet; print('✅ unet')"
  python -c "from separation_utils import process_stems; print('✅ separation_utils')"
  ```

- [ ] Test stems_to_midi package:
  ```bash
  python -c "from stems_to_midi import load_config; print('✅ config')"
  python -c "from stems_to_midi import create_midi_file; print('✅ midi')"
  python -c "from stems_to_midi import process_stem_to_midi; print('✅ processor')"
  python -c "from stems_to_midi.detection import detect_onsets; print('✅ detection')"
  python -c "from stems_to_midi.helpers import ensure_mono; print('✅ helpers')"
  python -c "from stems_to_midi.learning import learn_threshold_from_midi; print('✅ learning')"
  ```

### Step 4: Unit Tests

- [ ] Run all tests:
  ```bash
  pytest stems_to_midi/ -v
  # Record results: ___ passed, ___ failed
  ```

- [ ] Run specific test modules:
  ```bash
  pytest stems_to_midi/test_helpers.py -v
  pytest stems_to_midi/test_detection.py -v
  pytest stems_to_midi/test_learning.py -v
  pytest stems_to_midi/test_stems_to_midi.py -v
  ```

- [ ] Check test coverage (optional):
  ```bash
  pytest stems_to_midi/ --cov=stems_to_midi --cov-report=html
  ```

### Step 5: Functional Tests

- [ ] Test help commands:
  ```bash
  python separate.py --help
  python separate_with_eq.py --help
  python stems_to_midi.py --help
  python render_midi_to_video.py --help
  python sidechain_cleanup.py --help
  ```

- [ ] Test separation (if you have pretrained models):
  ```bash
  # Create test input
  python test.py  # Or use your test audio
  
  # Run separation
  python separate.py -i input/ -o output_test/
  
  # Verify output files exist
  ls output_test/
  ```

- [ ] Test MIDI conversion (if you have separated stems):
  ```bash
  python stems_to_midi.py -i output_test/ -o midi_test/
  
  # Verify MIDI files created
  ls midi_test/
  ```

- [ ] Test video rendering (if you have MIDI files):
  ```bash
  python render_midi_to_video.py -i midi_test/drums.mid -o test_video.mp4
  
  # Verify video created
  ls -lh test_video.mp4
  ```

### Step 6: Platform-Specific Tests

#### macOS Only
- [ ] Test MPS (Metal) availability:
  ```bash
  python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
  ```

#### Linux Only
- [ ] Check CUDA availability (if GPU present):
  ```bash
  python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
  ```

#### Docker
- [ ] Build Docker image:
  ```bash
  docker-compose build
  ```

- [ ] Start container:
  ```bash
  docker-compose up -d
  ```

- [ ] Enter container and test:
  ```bash
  docker exec -it larsnet-midi /bin/bash
  python -c "import torch, librosa, mido, cv2; print('✅ Docker imports OK')"
  pytest stems_to_midi/
  exit
  ```

- [ ] Stop container:
  ```bash
  docker-compose down
  ```

## Performance Tests

- [ ] Benchmark separation speed:
  ```bash
  time python separate.py -i input/ -o output_benchmark/
  # Record time: _________
  ```

- [ ] Benchmark MIDI conversion speed:
  ```bash
  time python stems_to_midi.py -i output_benchmark/ -o midi_benchmark/
  # Record time: _________
  ```

- [ ] Compare with old environment (if available):
  ```
  Old environment time: _________
  New environment time: _________
  Difference: _________
  ```

## Regression Tests

- [ ] Compare output with known-good results:
  ```bash
  # Separate with new environment
  python separate.py -i test_audio/ -o output_new/
  
  # Compare with reference output (if available)
  diff -r output_new/ output_reference/
  ```

- [ ] Verify MIDI output consistency:
  ```bash
  # Convert with new environment
  python stems_to_midi.py -i stems_test/ -o midi_new/
  
  # Compare with reference (if available)
  diff midi_new/drums.mid midi_reference/drums.mid
  ```

## Documentation Tests

- [ ] README setup instructions work:
  - [ ] Conda/Mamba section
  - [ ] Docker section
  - [ ] Separation examples

- [ ] MIGRATION.md steps work:
  - [ ] Migration instructions
  - [ ] Troubleshooting tips

- [ ] DEPENDENCIES.md is accurate:
  - [ ] All listed packages installed
  - [ ] Version requirements met

## Edge Cases

- [ ] Test with missing optional dependencies:
  ```bash
  # Remove scikit-learn temporarily
  pip uninstall -y scikit-learn
  
  # Test tom pitch detection still works (fallback)
  python -c "from stems_to_midi.helpers import classify_tom_pitch; import numpy as np; classify_tom_pitch(np.array([100, 200, 300]), 'tom')"
  
  # Reinstall
  conda install -c conda-forge scikit-learn
  ```

- [ ] Test with very large audio file (if available):
  ```bash
  python separate.py -i large_audio/ -o output_large/
  ```

- [ ] Test with unusual file formats:
  ```bash
  # Test with different sample rates, bit depths
  ```

## Platform-Specific Issues

### Record Your Results

**Platform**: [ ] macOS ARM  [ ] macOS Intel  [ ] Linux x86_64  [ ] Linux ARM64  [ ] Docker

**Conda version**: `conda --version` = _____________

**Mamba version**: `mamba --version` = _____________

**Environment creation time**: _____________

**Total environment size**: `du -sh ~/miniforge3/envs/larsnet` = _____________

**Issues encountered**:
```
(List any problems, warnings, or errors)
```

**All tests passed**: [ ] Yes  [ ] No

**Ready for production**: [ ] Yes  [ ] No

## Cleanup

- [ ] Remove test outputs:
  ```bash
  rm -rf output_test/ midi_test/ test_video.mp4
  rm -rf output_benchmark/ midi_benchmark/
  ```

- [ ] Keep environment for use:
  ```bash
  conda activate larsnet
  # Ready to use!
  ```

## Reporting Issues

If tests fail, collect this information:

1. **Platform details**:
   ```bash
   uname -a
   conda --version
   conda list > conda_list.txt
   ```

2. **Error messages**: Copy full stack traces

3. **Test that failed**: Which checkbox above?

4. **Environment info**:
   ```bash
   conda env export > environment_actual.yml
   ```

5. **Open issue** on GitHub with:
   - Platform details
   - Error messages
   - Steps to reproduce
   - conda_list.txt
   - environment_actual.yml

## Sign-Off

**Tested by**: _______________

**Date**: _______________

**Platform**: _______________

**Result**: [ ] ✅ All tests passed  [ ] ⚠️ Some issues  [ ] ❌ Major problems

**Notes**:
```
(Any additional observations)
```

---

**This checklist ensures the dependency migration is working correctly on your platform.**
