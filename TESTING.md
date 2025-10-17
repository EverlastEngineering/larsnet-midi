# How to test

Run:

```bash
# Run all tests easily:
docker exec -it larsnet-larsnet_env-1 bash /app/run_tests.sh

# Or inside the container:
./run_tests.sh

# View coverage report:
open htmlcov/index.html
```