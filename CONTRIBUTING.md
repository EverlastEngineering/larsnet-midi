# Contributing to LarsNet

Thank you for your interest in contributing to LarsNet! This guide will help you get started with development, testing, and submitting contributions.

## Development Setup

### Prerequisites

- **Conda or Mamba** (required for dependency management)
- **Git** for version control
- **Docker** (optional, for containerized development)

### Quick Start

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/larsnet-midi.git
   cd larsnet-midi
   ```

2. **Create development environment:**
   ```bash
   # Using mamba (faster)
   mamba env create -f environment.yml
   
   # OR using conda
   conda env create -f environment.yml
   
   # Activate environment
   conda activate larsnet
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch, librosa, mido; print('✅ All core packages imported')"
   ```

### Using Docker

For a consistent development environment:

```bash
# Build and start container
docker-compose up -d

# Enter container
docker exec -it larsnet-midi /bin/bash

# Run tests inside container
cd /app && pytest
```

## Architecture Overview

LarsNet follows the **Functional Core, Imperative Shell (FCIS)** pattern:

### Functional Core (Pure Logic)
- Located in `stems_to_midi/helpers.py`
- Pure functions with no side effects
- Fully deterministic and testable
- Examples: `calculate_spectral_energies()`, `should_keep_onset()`

### Imperative Shell (I/O & Coordination)
- Main entry points: `stems_to_midi.py`, `separate.py`
- Handles file operations, printing, user interaction
- Coordinates functional core for workflows

### Key Principles

1. **Separation of Concerns**: Logic in functional core, I/O in shell
2. **Configuration-Driven**: Use YAML configs instead of hardcoded values
3. **Test Before Refactor**: Write tests to cover existing behavior first
4. **Single Source of Truth**: Avoid code duplication

See the [architecture section in README.md](README.md#code-architecture-) for details.

## Testing

### Running Tests

```bash
# All tests with coverage
pytest --cov=stems_to_midi --cov-report=html

# Specific test file
pytest stems_to_midi/test_helpers.py

# Specific test function
pytest stems_to_midi/test_helpers.py::test_ensure_mono

# View coverage report
open htmlcov/index.html
```

### Docker Testing

```bash
# Run tests inside container
docker exec -it larsnet-midi bash -c "cd /app && pytest"

# Or use the convenience script
docker exec -it larsnet-midi bash /app/run_tests.sh
```

### Writing Tests

**Test Pure Functions (Functional Core):**
```python
def test_calculate_geomean():
    """Test geometric mean calculation."""
    result = calculate_geomean(100.0, 400.0)
    assert abs(result - 200.0) < 0.01  # √(100 × 400) = 200
```

**Test I/O Functions (Integration):**
```python
def test_process_stem(tmp_path, sample_audio):
    """Test full processing pipeline."""
    audio_path = tmp_path / "test.wav"
    sf.write(str(audio_path), sample_audio, 44100)
    
    events = process_stem_to_midi(audio_path, config={...})
    assert len(events) > 0
```

### Test Coverage Goals

- **Functional Core**: Aim for 90%+ coverage
- **Overall**: Maintain 80%+ coverage
- All new functions should have tests

## Code Standards

### Style Guidelines

- **PEP 8**: Follow Python style guide
- **Type Hints**: Use type hints for function signatures
- **Docstrings**: Document all public functions
- **Config Over Code**: Move magic numbers to `midiconfig.yaml`

**Example:**
```python
def calculate_spectral_energies(
    audio: np.ndarray,
    sr: int,
    start_idx: int,
    freq_ranges: List[Tuple[float, float]]
) -> Dict[str, float]:
    """
    Calculate energy in specified frequency ranges.
    
    Args:
        audio: Audio samples (mono)
        sr: Sample rate in Hz
        start_idx: Starting sample index
        freq_ranges: List of (min_freq, max_freq) tuples
        
    Returns:
        Dict mapping range index to energy value
    """
    # Implementation...
```

### FCIS Pattern Checklist

When adding new functionality:

- [ ] Pure logic goes in `helpers.py` (functional core)
- [ ] I/O operations stay in shell modules
- [ ] Configuration loaded from YAML, not hardcoded
- [ ] Tests written for pure functions
- [ ] Integration tests for I/O if needed

## Git Workflow

### Branch Naming

- `feature/short-description` - New features
- `fix/short-description` - Bug fixes
- `refactor/short-description` - Code refactoring
- `docs/short-description` - Documentation updates

### Commit Messages

Follow conventional commit format:

```
<type>(<scope>): <short summary>

<optional body>

<optional footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `docs`: Documentation changes
- `chore`: Maintenance tasks

**Examples:**
```
feat(stems_to_midi): add statistical outlier filtering for kick

Implements two-pass filtering to catch snare bleed in kick channel.
Uses population statistics to identify spectral outliers.

Closes #42
```

```
fix(midi): correct timing offset application

Move timing_offset from detection to MIDI creation to preserve
spectral analysis accuracy.
```

### Pull Request Process

1. **Create feature branch:**
   ```bash
   git checkout -b feature/your-feature
   ```

2. **Make changes and test:**
   ```bash
   pytest
   ```

3. **Commit with descriptive messages:**
   ```bash
   git add .
   git commit -m "feat(module): add feature description"
   ```

4. **Push and create PR:**
   ```bash
   git push origin feature/your-feature
   ```

5. **PR Description should include:**
   - What changed and why
   - How to test the changes
   - Any breaking changes
   - Related issues/PRs

### Before Submitting PR

- [ ] All tests pass (`pytest`)
- [ ] Coverage maintained or improved
- [ ] Code follows style guidelines
- [ ] Documentation updated if needed
- [ ] Commit messages are clear
- [ ] No merge conflicts with main

## Refactoring Guidelines

For significant refactoring work:

1. **Create plan file**: `<module>.plan.md` documenting approach, phases, and success criteria
2. **Create results file**: `<module>.results.md` tracking progress with checkboxes and metrics
3. **Do NOT edit plan** after creation (it represents original intent)
4. **Update results** as work progresses
5. **Commit at phase boundaries** with descriptive messages

See `docs/archive/` for examples of completed refactoring plans.

## Documentation

### User Documentation

- **README.md**: Project overview, quick start, architecture
- **STEMS_TO_MIDI_GUIDE.md**: User guide for MIDI conversion
- **MIDI_VISUALIZATION_GUIDE.md**: Video rendering guide
- **LEARNING_MODE.md**: Threshold calibration guide
- **TODO.md**: Active development tasks

### Writing Style

- Present tense, active voice
- Keep it factual and concise
- No speculation or assumptions
- Update in place (not like a changelog)
- See `.github/instructions/writing-documentation.instructions.md`

## Getting Help

- **Issues**: Check existing issues or create new one
- **Discussions**: For questions and ideas
- **Architecture**: Review `README.md` and archived plans in `docs/archive/`

## Project Structure

```
larsnet/
├── stems_to_midi/          # MIDI conversion package
│   ├── helpers.py          # Functional core (pure functions)
│   ├── detection.py        # Onset detection
│   ├── processor.py        # Audio processing pipeline
│   ├── midi.py             # MIDI file operations
│   ├── learning.py         # Threshold learning
│   └── test_*.py           # Tests
├── separate.py             # Source separation script
├── larsnet.py              # LarsNet model
├── unet.py                 # U-Net architecture
├── render_midi_to_video.py # Video visualization
├── midiconfig.yaml         # MIDI conversion config
├── environment.yml         # Conda dependencies
├── docs/
│   └── archive/            # Historical refactoring plans
└── .github/
    └── instructions/       # AI coding assistant instructions
```

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

**Thank you for contributing to LarsNet! 🥁🎵**
