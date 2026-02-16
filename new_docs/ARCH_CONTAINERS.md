# Container Architecture

Shows the major application containers and how they interact.

## Container Diagram

```mermaid
C4Container
    title Container Diagram - DrumToMIDI

    Person(user, "User", "Musician/Producer")

    Container_Boundary(app, "DrumToMIDI Application") {
        Container(webui, "Web UI", "Flask + JavaScript", "Browser-based interface for all operations")
        Container(cli, "CLI Tools", "Python scripts", "Command-line tools for batch processing")
        Container(separation, "Separation Engine", "PyTorch + MDX23C", "Separates drums into 5 stems")
        Container(detection, "Detection Engine", "LibROSA + NumPy", "Detects drum hits in stems")
        Container(midi, "MIDI Engine", "Mido + MidiUtil", "Converts detections to MIDI")
        Container(rendering, "Rendering Engine", "ModernGL/OpenCV", "Creates Rock Band-style videos")
        Container(project, "Project Manager", "Python", "Manages project state and files")
    }

    ContainerDb(files, "File System", "Audio, MIDI, Video", "Stores user projects and outputs")
    ContainerDb(models, "Model Storage", "Git LFS", "Pre-trained neural network weights")

    Rel(user, webui, "Uses", "HTTPS")
    Rel(user, cli, "Runs", "Shell")

    Rel(webui, project, "Manages projects", "Python API")
    Rel(cli, project, "Manages projects", "Python API")

    Rel(project, separation, "Requests separation", "Function call")
    Rel(project, detection, "Requests detection", "Function call")
    Rel(project, midi, "Requests MIDI", "Function call")
    Rel(project, rendering, "Requests video", "Function call")

    Rel(separation, models, "Loads weights", "File I/O")
    Rel(project, files, "Reads/writes", "File I/O")
    Rel(separation, files, "Writes stems", "File I/O")
    Rel(detection, files, "Reads stems", "File I/O")
    Rel(midi, files, "Writes MIDI", "File I/O")
    Rel(rendering, files, "Reads MIDI, writes video", "File I/O")
```

## Container Responsibilities

### Web UI (`webui/`)
- **Role**: HTTP API + browser interface
- **Tech**: Flask, JavaScript, HTML/CSS
- **Responsibilities**:
  - Project management (create, list, delete)
  - File uploads
  - Operation orchestration (separate, convert, render)
  - Job status tracking
  - Configuration management
  - File downloads

### CLI Tools (Root scripts)
- **Role**: Command-line batch processing
- **Scripts**:
  - `separate.py` - Stem separation
  - `sidechain_cleanup.py` - Bleed reduction
  - `stems_to_midi_cli.py` - MIDI conversion
  - `render_midi_to_video.py` - Video rendering
- **Responsibilities**:
  - Single-operation focus
  - Progress reporting
  - Error handling

### Separation Engine (`separation_shell.py`, `mdx23c_*`)
- **Role**: Audio source separation
- **Tech**: PyTorch, MDX23C model, torchaudio
- **Responsibilities**:
  - Load pre-trained model
  - Split audio into 5 stems
  - GPU acceleration (CUDA/MPS)
  - Chunk-based processing

### Detection Engine (`stems_to_midi/processing_shell.py`)
- **Role**: Drum hit detection
- **Tech**: LibROSA, NumPy, SciPy
- **Responsibilities**:
  - Onset detection (energy thresholds)
  - Spectral analysis (frequency bins)
  - Hi-hat state classification (open/closed)
  - Tom pitch estimation
  - Velocity estimation

### MIDI Engine (`stems_to_midi/midi.py`)
- **Role**: MIDI file generation
- **Tech**: Mido, MidiUtil
- **Responsibilities**:
  - Convert detections to MIDI events
  - Map drums to standard GM notes
  - Apply timing quantization
  - Write MIDI files

### Rendering Engine (`moderngl_renderer/`)
- **Role**: Video visualization
- **Tech**: ModernGL (GPU), OpenCV (fallback)
- **Responsibilities**:
  - Parse MIDI events
  - Render falling notes (Rock Band style)
  - Overlay waveforms
  - GPU-accelerated compositing
  - MP4 encoding

### Project Manager (`project_manager.py`)
- **Role**: State management
- **Tech**: Pure Python, YAML
- **Responsibilities**:
  - Project folder structure
  - File path resolution
  - Configuration persistence
  - Orchestration helpers

## Deployment Containers

### Docker Container
```
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
- Conda environment (drumtomidi)
- All Python dependencies
- GPU support (NVIDIA only)
- Volume mount: /app
- Port: 5000 (Web UI)
```

### Native Installation
```
- Miniforge/Conda environment
- PyTorch with MPS (Mac) or CUDA (Windows/Linux)
- Direct GPU access
- Better performance than Docker
```

## Related Documentation

- [ARCH_OVERVIEW.md](ARCH_OVERVIEW.md) - System context
- [ARCH_PATTERNS.md](ARCH_PATTERNS.md) - Coding patterns
- [ADDING_FEATURES.md](ADDING_FEATURES.md) - Feature checklist
