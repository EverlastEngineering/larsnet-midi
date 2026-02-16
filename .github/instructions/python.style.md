---
applyTo: "**/*.py"
---

# Python Style Guide

Requirements for Python code in this project.

## Required

### Type Hints

All functions must have type hints:

```python
# Good
def process_audio(audio_path: str) -> List[Dict[str, float]]:
    ...

# Bad
def process_audio(audio_path):
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def function_name(param: Type) -> ReturnType:
    """Brief one-liner description.

    Additional details if needed.

    Args:
        param: Description of parameter

    Returns:
        Description of return value

    Raises:
        ValueError: When this happens
    """
```

### Module Exports

All modules must define `__all__`:

```python
# my_module.py
from .core import core_function
from .types import MyType

__all__ = [
    'core_function',
    'MyType',
]
```

## Recommended

### Architecture Labels

Add architecture type to module docstrings:

```python
"""
Module Name — Brief description.

Architecture: Functional Core
- Pure logic, no side effects
- No file I/O, network, or GPU calls
"""
```

Or for shells:

```python
"""
Module Name — Brief description.

Architecture: Imperative Shell
- Orchestrates I/O operations
- Calls functional cores
"""
```

### Immutable Types for Data

Use frozen dataclasses for data structures:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Detection:
    time: float
    velocity: int
    note: str
```

## Linting

Run ruff before committing:

```bash
ruff check .
```

## Testing

- Functional cores: Unit tests (fast, no mocks)
- Imperative shells: Integration tests
- Target: 80% coverage for cores

See [.github/instructions/how-to-perform-testing.instructions.md](how-to-perform-testing.instructions.md)
