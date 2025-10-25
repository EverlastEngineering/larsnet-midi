---
applyTo: '**'
---

When bugs are identified during development or reported by users, track them in a running list at `./agent-plans/bug-tracking.md`.

## Bug Entry Format

```markdown
## Bug: [Short descriptive title]
- **Status**: Open | Fixed
- **Priority**: High | Medium | Low
- **Description**: Clear description of the issue
- **Steps to Reproduce**: (if applicable)
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Fixed in Commit**: (commit hash when resolved)
```

## Workflow

1. When a bug is reported, add it to the bug tracking file immediately
2. Set status to "Open" and assign appropriate priority
3. When fixed, update status to "Fixed" and reference the commit
4. Keep fixed bugs in the file for historical reference
5. Periodically archive old fixed bugs to a separate section

## Regression Protocol

When user indicates functionality previously worked but now fails (temporal language: "no longer", "stopped working", "now broken", "used to work"), treat as regression rather than new defect:

1. **Immediate**: Examine recent changesets via `get_changed_files` (unstaged/staged) or `git log -p -n 5` (committed diffs)
2. **Identify**: Locate modifications intersecting with reported failure domain
3. **Bisect**: Systematically A/B test suspect changesets to isolate causative diff
4. **Resolve**: Apply minimal reversion or correction to restore prior behavior
5. **Validate**: Confirm fix restores functionality without introducing new regressions

Skip exploratory reimplementation; regressions indicate proximate causation in recent changesets.

## Guidelines

- Be specific and factual in bug descriptions
- Include reproduction steps when possible
- Link to relevant code or files
- Update status promptly when bugs are resolved
- Group related bugs together
