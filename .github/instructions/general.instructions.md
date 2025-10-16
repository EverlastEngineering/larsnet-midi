---
applyTo: '**/*.py, **/*.yaml'
---
Use functional core, imperative shell to separate logic from side effects.
Before refactoring, write tests to cover existing behavior.
Use configuration files for parameters instead of hardcoding values.

For significant refactoring work:
1. Create an immutable plan file: `<module>.plan.md` documenting the approach, phases, risks, and success criteria.
2. Create a mutable results file: `<module>.results.md` tracking completion of each phase with checkboxes, metrics, and decision log.
3. Do NOT edit the plan file after creation - it represents original intent.
4. Update the results file as work progresses to track actual outcomes vs plan.
5. Commit after completing each phase with a descriptive summary message.

Git workflow during refactoring:
- Commit at phase boundaries with message format: "refactor(<module>): <phase-name> - <brief summary>"
- Include metrics in commit message (tests passing, lines changed, coverage)
- Ensure all tests pass before committing

Update these instructions when critical fundamentals around architecture or best practices changes.