---
applyTo: '**'
---
Be very critical of code quality, approaches, and architecture.
Use functional core, imperative shell to separate logic from side effects.
Before refactoring, write tests to cover existing behavior.
Use configuration files for parameters instead of hardcoding values.
You may edit your own instruction files, and there's an instruction file for that!
Review the writing-documentation.instructions.md for guidance before writing documentation.
Review the how-to-perform-testing.instructions.md when dealing with tests.
Let user know you've read instructions for confidence in your actions.
When you complete a task, simply say "Done!" - do not provide summaries, explanations, or lists of changes made. The user can see what you did through the tool calls and test results.
Do not make summary documents after completing tasks.

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

Update these instructions when critical fundamentals around architecture or best practices changes.