# Adding New Features

This guide covers what to do when adding a new feature to DrumToMIDI.

## Quick Checklist

When adding a new feature, work through this checklist:

- [ ] Add to appropriate module
- [ ] Add type hints
- [ ] Add Google-style docstrings
- [ ] Add `__all__` export if new module
- [ ] Add unit tests
- [ ] Add integration test if applicable
- [ ] Update ARCH_OVERVIEW.md if new container
- [ ] Update ARCH_CONTAINERS.md if new container

## Detailed Steps

### 1. Code (Functional Core + Shell)

Determine if your feature is a **core** (pure logic) or **shell** (I/O):

| If your code... | Then it's a... | Place in... |
|-----------------|----------------|-------------|
| Does math, transforms data, no file/network I/O | Core | `*_core.py` |
| Reads/writes files, calls APIs, uses GPU | Shell | `*_shell.py` |

**Core Example:**
```python
# analysis_core.py
def calculate_rms(audio: np.ndarray) -> float:
    """Calculate RMS energy of audio signal.

    Args:
        audio: Audio samples

    Returns:
        RMS energy value
    """
    return float(np.sqrt(np.mean(audio ** 2)))
```

**Shell Example:**
```python
# processing_shell.py
def process_audio(audio_path: str) -> List[Detection]:
    audio, sr = librosa.load(audio_path)  # I/O
    rms = calculate_rms(audio)           # Core
    detections = detect_onsets(audio, sr) # Core
    return detections
```

### 2. Type Hints

Add type hints to all functions:

```python
# Good
def detect_onsets(audio: np.ndarray, sr: int, threshold: float) -> List[Dict[str, float]]:
    ...

# Needs work
def detect_onsets(audio, sr, threshold):
    ...
```

### 3. Docstrings

Use Google-style docstrings:

```python
def function_name(param: Type) -> ReturnType:
    """Brief one-liner description.

    Longer description if needed.

    Args:
        param: What this parameter does

    Returns:
        What this function returns

    Raises:
        ValueError: When this happens
    """
```

### 4. Module Exports

If creating a new module, add `__init__.py` with `__all__`:

```python
# stems_to_midi/__init__.py
"""Stems to MIDI conversion package."""

from .processing_shell import process_stem_to_midi
from .config import DrumMapping, Config

__all__ = [
    'process_stem_to_midi',
    'DrumMapping',
    'Config',
]
```

### 5. Tests

**Unit tests for cores:**
```python
# test_analysis_core.py
def test_calculate_rms():
    audio = np.array([0.0, 1.0, -1.0, 0.0])
    rms = calculate_rms(audio)
    assert rms == 0.5  # sqrt(mean([0,1,1,0]))
```

**Integration tests for shells:**
```python
# test_integration.py
@pytest.mark.slow
def test_process_audio_creates_detections():
    result = process_audio('test.wav')
    assert len(result) > 0
    assert all('time' in d for d in result)
```

### 6. Architecture Updates

**When to update ARCH docs:**

| Change | Update |
|--------|--------|
| New container (e.g., new engine) | ARCH_OVERVIEW + ARCH_CONTAINERS |
| New pattern or naming convention | ARCH_PATTERNS |
| New module (auto-generated) | FILE_INVENTORY (run script) |

**When NOT to update:**
- Adding new functions (docstrings sufficient)
- Adding new tests
- Bug fixes
- Performance improvements

## What NOT to Document

- Individual function signatures (auto-generated via mkdocstrings)
- Internal implementation details (in-code docs sufficient)
- ARCH_COMPONENTS.md (deprecated)

## Common Scenarios

### Adding a New Detection Algorithm

1. Create `*_core.py` for pure algorithm
2. Create `*_shell.py` for I/O orchestration
3. Add tests for core
4. Add integration test for shell
5. Update config schema if needed

### Adding a New Web UI Feature

1. Add JS to `webui/static/js/`
2. Add JSDoc comments
3. Add Python endpoint to `webui/api/`
4. Add unit tests for endpoint

### Adding Configuration

1. Add to `midiconfig.yaml` schema
2. Add type hints in `config.py`
3. Document in existing guide (e.g., MIDI_YAML_SETTINGS.md)

## Troubleshooting

### "I don't know where to put this code"

Ask: Does this code do I/O (file, network, GPU)?
- Yes → Shell (`*_shell.py`)
- No → Core (`*_core.py`)

### "Do I need to update documentation?"

Ask: Did you add a new container or change system boundaries?
- Yes → Update ARCH_OVERVIEW and ARCH_CONTAINERS
- No → Docstrings are sufficient

### "How do I test this?"

Ask: Is it pure logic or I/O?
- Pure logic → Unit test (fast, no mocks)
- I/O → Integration test (slower, real files)

## Related Documentation

- [ARCH_PATTERNS.md](ARCH_PATTERNS.md) - Coding patterns
- [ARCH_CONTAINERS.md](ARCH_CONTAINERS.md) - Container reference
- [FILE_INVENTORY.md](FILE_INVENTORY.md) - Source file list
