---
applyTo: '**/*.py'
---
When testing, once you're sure the docker container is running and stable, use the following command to run tests inside the container:

For the `stems_to_midi` module tests, run:
```bash
docker exec -it larsnet-dev-container bash -c "cd /app && pytest stems_to_midi/test_stems_to_midi.py stems_to_midi/test_helpers.py -v
```
