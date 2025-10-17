```instructions
---
applyTo: '**/*.py'
---
When testing, once you're sure the docker container is running and stable, use the following commands to run tests inside the container:

## Quick Test Commands

**Run all tests with coverage (recommended):**
```bash
docker exec -it larsnet-larsnet_env-1 bash /app/run_tests.sh
```

**Run specific test files:**
```bash
docker exec -it larsnet-larsnet_env-1 bash -c "cd /app && pytest stems_to_midi/test_helpers.py -v"
```

**Run with coverage report:**
```bash
docker exec -it larsnet-larsnet_env-1 bash -c "cd /app && pytest stems_to_midi --cov=stems_to_midi --cov-report=term-missing --cov-report=html"
```

## Test File Locations

All test files are located in the `stems_to_midi/` package:
- `stems_to_midi/test_stems_to_midi.py` - Integration tests for the full workflow
- `stems_to_midi/test_helpers.py` - Unit tests for helper functions
- `stems_to_midi/test_learning.py` - Tests for threshold learning module
- `stems_to_midi/test_detection.py` - Tests for onset and pitch detection

## Test Coverage

The project maintains >90% test coverage on the functional core modules:
- `helpers.py`: 94% coverage (pure functional code)
- `learning.py`: 99% coverage (pure functional code)
- `detection.py`: 95% coverage (mostly pure functional code)
- `midi.py`: 100% coverage
- `config.py`: 94% coverage
- `processor.py`: 60% coverage (imperative shell - acceptable)

Coverage reports are generated in `htmlcov/` directory.

```
