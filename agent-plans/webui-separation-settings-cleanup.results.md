# WebUI Separation Settings Cleanup Results

## Phase 1: Remove Comparison Mode
- [x] Remove comparison mode UI from index.html
- [x] Remove comparison mode JS logic from operations.js
- [x] Remove API endpoints from operations.py (/compare, /delete-comparison)
- [x] Remove run_comparison function
- [x] Delete compare_separation_configs.py
- [x] Delete comparison_configs.example.yaml
- [x] Delete test_compare_separation_configs.py
- [x] Delete SEPARATION_COMPARISON_GUIDE.md
- [x] Delete webui/test_comparison_api.py (found after initial commit)
- [x] Remove comparison file discovery from projects.py
- [x] Update comparison references in downloads.py docstrings
- [x] Remove compare/deleteComparison methods from api.js
- [x] Update README.md to remove comparison guide references

## Phase 2: Add MPS Device Support
- [x] Add MPS option to device dropdown in index.html
- [x] Update device validation in operations.py separate endpoint
- [x] Update device description text to include MPS
- [ ] Test MPS device selection end-to-end

## Phase 3: Review EQ Cleanup Setting
- [ ] Assess EQ cleanup effectiveness
- [ ] Make removal/keep decision
- [ ] Update documentation

## Decision Log
- Starting cleanup per user request
- User wants comparison mode completely removed
- MPS support needed (currently missing)
- EQ cleanup needs critical review
- Wiener filter stays (user confirmed)
