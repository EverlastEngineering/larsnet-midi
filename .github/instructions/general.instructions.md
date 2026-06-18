---
applyTo: '**'
---
Be very critical of code quality, approaches, and architecture.
Use functional core, imperative shell to separate logic from side effects.
Before refactoring, write tests to cover existing behavior.
Use configuration files for parameters instead of hardcoding values.
Define explicit type contracts using dataclasses (like `midi_types.py`) for data shared between modules.
You may edit your own instruction files, and there's an instruction file for that!
Always validate edit results: lint/compile errors reported by tools indicate actual syntax failures requiring immediate correction.
Review the writing-documentation.instructions.md for guidance before writing documentation.
Review the how-to-perform-testing.instructions.md when dealing with tests.
Let user know you've read instructions for confidence in your actions.

Documentation and error handling are first-class concerns, not afterthoughts:
- Write documentation as you build each feature, not at the end
- Document how things work alongside implementation
- Include error handling and user feedback from the start
- Good documentation enables seamless handoffs during AI-driven development
- Documentation serves as reference during testing and debugging

For significant refactoring work:
1. Create an immutable plan file: `./agent-plans/<module>.plan.md` documenting the approach, phases, risks, and success criteria.
2. Create a mutable results file: `./agent-plans/<module>.results.md` tracking completion of each phase with checkboxes, metrics, and decision log.
3. Do NOT edit the plan file after creation - it represents original intent.
4. Update the results file as work progresses to track actual outcomes vs plan.
5. Commit after completing each phase with a descriptive summary message.

Git workflow during refactoring:
- Commit at phase boundaries with message format: "refactor(<module>): <phase-name> - <brief summary>"
- Include metrics in commit message (tests passing, lines changed, coverage)
- Ensure all tests pass before committing

Cross-task commit hygiene (2026-06-18):
- Before starting what appears to be a NEW task or feature, check whether the working tree already contains uncommitted edits from a prior task. If it does, ask the user explicitly: "I see <N> files with uncommitted changes from the previous work. Want me to commit them (in one or more logical commits) before we start on this new task?" Do NOT silently bundle prior work into the new task's commit — keep logical changes in their own commits.
- If a single request touches multiple distinct logical concerns (e.g. a detector change AND a new config flag), split into separate commits — one per concern — unless the user explicitly says to combine them.
- When in doubt about whether the working tree's modifications are intentional (e.g. .vscode/settings.json noise from test runs, formatter artifacts), `git diff` first and drop anything that wasn't a deliberate edit before staging.

Update these instructions when critical fundamentals around architecture or best practices changes.