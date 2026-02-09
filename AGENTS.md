IMPORTANT:

When you complete a task, simply say "Done!" - do not provide summaries, explanations, or lists of changes made. The user can see what you did through the tool calls and test results.
Do not make summary documents after completing tasks.
Before taking action that the user might want to pause, just stop and ask to continue. Any blank response there would be taken as permission to proceed.

## Feature & Bug Tracking

Track features and bugs in `agent-plans/feature-tracking.md` (and bugs in `agent-plans/bug-tracking.md`). When the user reports a bug or requests a feature, add it to the appropriate tracking file before starting work. Update status as items are completed.

## Architecture Documentation Maintenance

When making significant changes to the codebase structure, update the relevant ARCH_*.md files in `docs/`:

- **docs/ARCH_C1_OVERVIEW.md**: Add new user workflows, update system context diagram
- **docs/ARCH_C2_CONTAINERS.md**: Add/remove major application components
- **docs/ARCH_C3_COMPONENTS.md**: Update package structure, module dependencies, coverage stats
- **docs/ARCH_DATA_FLOW.md**: Modify data transformation stages if pipeline changes
- **docs/ARCH_LAYERS.md**: Document new functional cores or imperative shells
- **docs/ARCH_FILES.md**: Update file/folder listings when structure changes

**Note**: All architecture documentation (ARCH_*.md) lives in docs/ alongside user guides.

**When to update**:
- Adding new packages or major modules
- Restructuring code organization
- Changing architectural patterns
- After significant refactoring work
- When coverage numbers shift significantly

**How to update**:
- Prefer incremental updates over full rewrites
- Update mermaid diagrams when relationships change
- Keep file listings current (consider running `tree` or similar)
- Verify cross-references between ARCH_*.md files remain valid

**Note**: These files are designed for LLM context windows - keep them focused and scannable.