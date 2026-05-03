# Welcome to DrumToMIDI

DrumToMIDI is an audio-to-MIDI conversion system for drum tracks using deep learning separation, spectral analysis, and temporal detection.

## Quick Links

- [Architecture Overview](ARCH_OVERVIEW.md) - System context and user workflows
- [Container Architecture](ARCH_CONTAINERS.md) - Application components
- [Coding Patterns](ARCH_PATTERNS.md) - Functional core / imperative shell
- [Adding Features](ADDING_FEATURES.md) - Checklist for adding new features
- [File Inventory](FILE_INVENTORY.md) - All source files

## Key Capabilities

- **5-stem separation**: Kick, snare, hi-hat, toms, cymbals using MDX23C
- **GPU acceleration**: CUDA (Windows/Linux), Metal/MPS (Mac native)
- **Sidechain cleanup**: Reduces bleed between stems
- **Adaptive detection**: Energy thresholds, spectral analysis
- **Learning mode**: Calibrate thresholds from labeled data
- **Rock Band visualization**: Falling notes with waveforms

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| ML | PyTorch 2.7 |
| Audio | LibROSA |
| Rendering | ModernGL, OpenCV |
| Web | Flask |
| MIDI | Mido, MidiUtil |

## Getting Started

1. **Web UI**: Run `python -m webui.app` and open browser
2. **CLI**: See individual scripts in project root
3. **Development**: See [ADDING_FEATURES.md](ADDING_FEATURES.md)

## Documentation Status

This documentation is built with MkDocs with API reference generated from Python docstrings.

### Building the Docs

```bash
# Install mkdocs (if not in environment)
pip install mkdocs mkdocs-material

# Generate API docs from docstrings
python new_docs/docs/generate_api_docs.py

# Generate file inventory
python new_docs/docs/generate_inventory.py

# Serve locally
mkdocs serve

# Build
mkdocs build
```

See [ARCH_PATTERNS.md](ARCH_PATTERNS.md#documentation) for more details.
