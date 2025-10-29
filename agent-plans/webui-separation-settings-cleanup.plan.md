# WebUI Separation Settings Cleanup Plan

## Objective
Remove comparison mode feature and update device selection to include MPS support.

## Phases

### Phase 1: Remove Comparison Mode
- Remove comparison mode UI elements from index.html
- Remove comparison mode logic from operations.js
- Remove `/api/compare` and `/api/delete-comparison` endpoints
- Remove comparison-related functions from operations.py
- Remove `compare_separation_configs.py` and related files
- Remove comparison tests
- Clean up project.py comparison file discovery

### Phase 2: Add MPS Device Support
- Add MPS option to device dropdown
- Update device validation in operations.py
- Test MPS selection with actual separation

### Phase 3: Review EQ Cleanup Setting
- Assess current effectiveness of EQ cleanup
- Determine if setting should be removed, kept, or modified
- Document decision rationale

## Success Criteria
- Comparison mode completely removed from codebase
- MPS device option available and functional
- No broken references to comparison functionality
- All tests pass

## Risks
- Breaking existing workflows that depend on comparison feature
- Need to ensure MPS validation works correctly in API

## Notes
- EQ cleanup decision deferred to Phase 3 pending user input
- Wiener filter kept as-is per user request
