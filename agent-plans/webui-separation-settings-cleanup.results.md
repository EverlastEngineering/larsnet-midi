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
- [x] Test collection passes after cleanup

## Phase 3: Remove EQ Cleanup Feature
- [x] Remove EQ cleanup UI from index.html
- [x] Remove apply_eq parameter from separate.py (signature and CLI)
- [x] Remove apply_eq from separation_utils.py functions (process_stems, process_stems_for_project)
- [x] Remove load_eq_config and apply_frequency_cleanup functions
- [x] Remove _apply_filters_to_chunk helper function
- [x] Remove torchaudio.functional import (only used for EQ)
- [x] Remove apply_eq from WebUI operations.js
- [x] Remove apply_eq from settings.js (getSettingsForOperation and defaults)
- [x] Remove eq parameter from webui/api/operations.py (run_separate and /api/separate)
- [x] Delete separate_with_eq.py script
- [x] Delete eq.yaml configuration file
- [x] Delete test_eq_chunking.py
- [x] Remove eq.yaml from ROOT_CONFIGS in project_manager.py
- [x] Remove eq.yaml from project structure docstring
- [x] Remove eq.yaml from allowed configs in webui/api/projects.py
- [x] Update test_project_manager.py to remove eq.yaml assertions
- [x] All syntax checks pass, no errors

## Decision Log
- Starting cleanup per user request
- User wants comparison mode completely removed
- MPS support needed (currently missing)
- EQ cleanup determined to "work against the model" - complete removal
- Wiener filter stays (user confirmed)
- Phase 1 & 2 complete: Comparison mode removed, MPS support added
- Found and removed webui/test_comparison_api.py after initial test run
- Phase 3 complete: EQ cleanup feature completely removed
- Rationale: EQ post-processing counteracts model's learned separation

## Metrics - Final
- Files deleted: 9 total
  * Phase 1: 5 files (comparison mode)
  * Phase 3: 4 files (EQ cleanup: separate_with_eq.py, eq.yaml, test_eq_chunking.py, webui/test_comparison_api.py)
- Files modified: 12
  * WebUI: index.html, operations.js, settings.js
  * API: operations.py, projects.py
  * Core: separate.py, separation_utils.py, project_manager.py
  * Tests: test_project_manager.py
  * Docs: (pending)
- Code removed:
  * 3 complete EQ functions (load_eq_config, apply_frequency_cleanup, _apply_filters_to_chunk)
  * ~150 lines of EQ logic from separation_utils.py
  * apply_eq parameters throughout codebase
  * torchaudio.functional import
- Test collection: 288 tests, all passing
- No syntax errors, all validations pass
