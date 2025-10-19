---
applyTo: '**'
---
When testing, once you're sure the docker container is running and stable, use the following commands to run tests inside the container:

## Quick Test Commands

**Run all tests with coverage (recommended):**
```bash
docker exec -it larsnet-midi bash -c "cd /app && pytest"
```

Find test files by searching for `import pytest`.

## Test Coverage

Attempt to maintain coverage on functional core code.