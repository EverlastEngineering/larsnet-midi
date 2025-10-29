# MDX23C Model System Guide

This guide explains how to use the MDX23C model system for drum separation in this project.

## Overview

The MDX23C system uses the TFC-TDF neural network architecture to separate drum audio into 5 stems:
- **Kick** - Bass drum
- **Snare** - Snare drum  
- **Toms** - Tom drums
- **Hihat** - Hi-hat cymbals
- **Cymbals** - Crash and ride cymbals

## Model Files

The system consists of:
- `mdx_models/drumsep_5stems_mdx23c_jarredou.ckpt` - Pre-trained model weights
- `mdx_models/config_mdx23c.yaml` - Model configuration (audio params, architecture)
- `lib_v5/tfc_tdf_v3.py` - TFC_TDF_net architecture implementation
- `mdx23c_utils.py` - Utility functions for loading and using models

## Quick Start

### Basic Usage

Separate drums from an audio file:

```bash
python example_mdx23c_usage.py input_audio.wav -o output_directory
```

This will create 5 WAV files in `output_directory/`:
- `kick.wav`
- `snare.wav`
- `toms.wav`
- `hihat.wav`
- `cymbals.wav`

### Command Line Options

```bash
python example_mdx23c_usage.py input.wav \
  --output output_dir \
  --checkpoint mdx_models/drumsep_5stems_mdx23c_jarredou.ckpt \
  --config mdx_models/config_mdx23c.yaml \
  --device cpu  # or 'cuda' for GPU
```

## Using with separate.py

The MDX23C model is now the default in `separate.py`:

```bash
# Use MDX23C with default settings (overlap=4)
python separate.py 1

# Use custom overlap (2-50, higher=better quality but slower)
python separate.py 1 --overlap 2   # Fastest (50% overlap)
python separate.py 1 --overlap 4   # Default - good balance (75% overlap)
python separate.py 1 --overlap 8   # High quality (87.5% overlap) 
python separate.py 1 --overlap 16  # Maximum quality (93.75% overlap)

# MDX23C is the default model
# Future: python separate.py 1 --model <other_model>
```

### Overlap Parameter

The `--overlap` parameter (2-50) controls how many times each audio sample is processed:

- **overlap=2**: 50% overlap, fastest, minimal quality improvement
- **overlap=4**: 75% overlap, 2x slower than overlap=2
- **overlap=8**: 87.5% overlap, **default**, matches UVR quality
- **overlap=16**: 93.75% overlap, very high quality, 8x slower than overlap=2
- **overlap=50**: 98% overlap, maximum quality, 25x slower than overlap=2

Higher overlap values produce smoother, more artifact-free results by averaging more predictions for each sample.

## Python API

### Loading a Model

```python
from mdx23c_utils import load_mdx23c_checkpoint

# Load model (config is auto-detected if in same directory as checkpoint)
model = load_mdx23c_checkpoint(
    "mdx_models/drumsep_5stems_mdx23c_jarredou.ckpt",
    device="cpu"
)

# With explicit config path
model = load_mdx23c_checkpoint(
    "mdx_models/drumsep_5stems_mdx23c_jarredou.ckpt",
    config_path="mdx_models/config_mdx23c.yaml",
    device="cpu"
)
```

### Processing Audio

```python
import torch
import torchaudio
from mdx23c_utils import load_mdx23c_checkpoint, get_checkpoint_hyperparameters

# Load model and config
checkpoint_path = "mdx_models/drumsep_5stems_mdx23c_jarredou.ckpt"
model = load_mdx23c_checkpoint(checkpoint_path, device="cpu")
params = get_checkpoint_hyperparameters(checkpoint_path)

# Get required chunk size
chunk_size = params['audio']['chunk_size']  # 523776 samples (~11.9 seconds)

# Load and prepare audio
waveform, sr = torchaudio.load("drums.wav")

# Resample to 44100 Hz if needed
if sr != 44100:
    resampler = torchaudio.transforms.Resample(sr, 44100)
    waveform = resampler(waveform)

# Ensure stereo (2 channels)
if waveform.shape[0] == 1:
    waveform = waveform.repeat(2, 1)
elif waveform.shape[0] > 2:
    waveform = waveform[:2]

# Add batch dimension: (channels, time) -> (batch, channels, time)
waveform = waveform.unsqueeze(0)

# Process (input must be exactly chunk_size samples or chunked manually)
with torch.no_grad():
    output = model(waveform)

# Output shape: (batch, instruments, channels, time)
# output[0, 0] = kick (stereo)
# output[0, 1] = snare (stereo)
# output[0, 2] = toms (stereo)
# output[0, 3] = hihat (stereo)
# output[0, 4] = cymbals (stereo)

# Save individual stems
instruments = ['kick', 'snare', 'toms', 'hihat', 'cymbals']
for i, name in enumerate(instruments):
    stem = output[0, i]  # Extract stem (channels, time)
    torchaudio.save(f"{name}.wav", stem, 44100)
```

### Inspecting Model Configuration

```python
from mdx23c_utils import get_checkpoint_hyperparameters

params = get_checkpoint_hyperparameters(
    "mdx_models/drumsep_5stems_mdx23c_jarredou.ckpt"
)

print(f"Sample rate: {params['audio']['sample_rate']}")
print(f"Chunk size: {params['audio']['chunk_size']}")
print(f"FFT size: {params['audio']['n_fft']}")
print(f"Hop length: {params['audio']['hop_length']}")
print(f"Instruments: {params['training']['instruments']}")
```

## Model Architecture

The model uses the TFC_TDF_net architecture:

- **Input**: Stereo audio at 44.1 kHz
- **STFT**: 2048 FFT size, 512 hop length
- **Encoder-Decoder**: 5 scales with skip connections
- **Output**: 5 stereo stems (kick, snare, toms, hihat, cymbals)

### Configuration Parameters

From `config_mdx23c.yaml`:

```yaml
audio:
  sample_rate: 44100
  n_fft: 2048
  hop_length: 512
  chunk_size: 523776  # ~11.9 seconds
  num_channels: 2

model:
  num_channels: 128
  num_scales: 5
  num_blocks_per_scale: 2
  growth: 128
  bottleneck_factor: 4
  num_subbands: 4
  act: gelu
  norm: InstanceNorm
```

## Processing Long Audio Files

For files longer than the chunk size (~11.9 seconds), use overlapping chunks:

```python
chunk_size = 523776
hop_length = chunk_size // 2  # 50% overlap

# Process in chunks (see example_mdx23c_usage.py for full implementation)
for start in range(0, audio_length, hop_length):
    end = min(start + chunk_size, audio_length)
    chunk = audio[:, :, start:end]
    
    # Pad if needed
    if chunk.shape[-1] < chunk_size:
        chunk = torch.nn.functional.pad(chunk, (0, chunk_size - chunk.shape[-1]))
    
    chunk_output = model(chunk)
    # Accumulate with overlap handling...
```

See `example_mdx23c_usage.py` for a complete implementation with overlap-add.

## Testing

Run the test suite:

```bash
pytest test_mdx23c_utils.py -v
```

Run manual smoke test:

```bash
python test_mdx23c_utils.py
```

## Supported Formats

### Checkpoint Formats

The loader supports two checkpoint formats:

1. **Modern format** (used by this repo): Raw state dict + separate YAML config
   - Checkpoint: Just model weights
   - Config: `config_mdx23c.yaml` with all hyperparameters
   - Architecture: `TFC_TDF_net`

2. **Legacy format**: PyTorch Lightning checkpoint with embedded hyperparameters
   - Checkpoint: Includes `hyper_parameters` dict
   - Architecture: `ConvTDFNet`
   - Requires: `pytorch_lightning` library

## Error Handling

Common issues and solutions:

### "Checkpoint not found"
- Verify the checkpoint file exists
- Use absolute paths or ensure working directory is correct

### "Config file not found"
- Config must be in same directory as checkpoint, or
- Provide explicit `config_path` parameter

### "Sizes of tensors must match"
- Input audio must be exactly `chunk_size` samples long
- Pad or chunk longer audio appropriately
- Use overlap-add for continuous processing

### "No module named 'pytorch_lightning'"
- Only needed for legacy checkpoint format
- Modern checkpoints don't require pytorch_lightning

## Performance Tips

1. **Use GPU**: Set `device="cuda"` for much faster processing
2. **Batch processing**: Process multiple chunks in parallel if GPU memory allows
3. **Chunk overlap**: 50% overlap provides good quality/speed tradeoff
4. **Half precision**: Use `model.half()` for faster inference on modern GPUs

## Further Reading

- [MDX Challenge Results](https://www.aicrowd.com/challenges/sound-demixing-challenge-2023)
- [TFC-TDF Architecture Paper](https://arxiv.org/abs/2109.03600)
- [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui) - Original implementation
