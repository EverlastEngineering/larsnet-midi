# ## Plan Overview
Refactor file management system from `-i/-o` flags to unified `user_files/` project-based structure with per-project YAML configs.

**Created**: 2025-10-19  
**Updated**: 2025-10-19 (removed backward compatibility, added project-specific configs)  
**Status**: Planning Complete, Ready to BeginManagement Refactoring - Results Tracker

## Plan Overview
Refactor file management system from `-i/-o` flags to unified `user_files/` project-based structure.

**Created**: 2025-10-19  
**Status**: Planning Complete, Awaiting Approval

---

## Phase Completion

### Phase 1: Core Infrastructure ✅
**Status**: Complete  
**Started**: 2025-10-19  
**Completed**: 2025-10-19

**Tasks**:
- [x] Create `user_files/` directory with `.gitkeep`
- [x] Update `.gitignore` to include `user_files/`
- [x] Create `project_manager.py` module
  - [x] `discover_projects()` function
  - [x] `find_loose_files()` function
  - [x] `create_project()` function (with config copying)
  - [x] `get_project_by_number()` function
  - [x] `select_project()` function
  - [x] `update_project_metadata()` function
  - [x] `get_project_config()` function (with fallback to root)
  - [x] `copy_configs_to_project()` function
- [x] Write comprehensive tests for project manager
- [x] Document API in docstrings

**Metrics**:
- Tests written: 30
- Tests passing: 30/30 (100%)
- Lines of code added: ~900 (project_manager.py: ~550, tests: ~350)
- Code coverage: Complete for functional core

**Notes**:
- All pure functions tested without file I/O
- Imperative shell tested with temporary directories
- Integration test covers complete workflow
- Clean functional core / imperative shell separation
- 100% test pass rate in 0.10s

---

### Phase 2: Refactor separate.py ✅
**Status**: Complete  
**Started**: 2025-10-19  
**Completed**: 2025-10-19

**Tasks**:
- [x] Remove all `-i/-o` argument parsing
- [x] Add project detection logic
- [x] Support parameterless invocation
- [x] Support `separate.py <number>` syntax
- [x] Implement file moving into project structure (via create_project)
- [x] Load config from project folder
- [x] Update project metadata after separation
- [x] Update tests
- [ ] Update LARSNET.md documentation (deferred to Phase 6)

**Metrics**:
- Tests written: 2
- Tests passing: 32/32 (100%)
- Lines changed: ~150 (separate.py: ~120, separation_utils.py: +90, tests: +60)

**Notes**:
- Removed legacy -i/-o flags completely
- Added process_stems_for_project() to separation_utils.py
- Auto-detects loose files and creates projects
- Uses project-specific config.yaml with fallback to root
- Updates project status after successful separation
- Clean error messages guide users

---

### Phase 3: Refactor stems_to_midi.py ✅
**Status**: Complete  
**Started**: 2025-10-19  
**Completed**: 2025-10-19

**Tasks**:
- [x] Remove all `-i/-o` argument parsing
- [x] Add project detection
- [x] Auto-detect `cleaned/` vs `stems/` folder within project
- [x] Output MIDI to `midi/` subfolder
- [x] Load midiconfig.yaml from project folder
- [x] Update project metadata after MIDI generation
- [x] Update tests (reuses existing tests)
- [x] Fix bug: removed undefined audio_files_to_process reference
- [x] Fix bug: added tempo loading from config when None
- [ ] Update STEMS_TO_MIDI_GUIDE.md (deferred to Phase 6)

**Metrics**:
- Tests written: 0 new (existing tests still passing)
- Tests passing: 188/188 (100%)
- Lines changed: ~200
- Bug fixes: 2

**Notes**:
- Removed legacy -i/-o flags completely
- Auto-detects cleaned/ or stems/ in project structure
- Uses project-specific midiconfig.yaml with fallback to root
- Loads tempo from config['midi']['default_tempo'] when None
- Updates project status after MIDI generation
- Clean error messages for missing stems/config

---

### Phase 4: Refactor sidechain_cleanup.py ✅
**Status**: Complete  
**Started**: 2025-10-19  
**Completed**: 2025-10-19

**Tasks**:
- [x] Remove all `-i/-o` argument parsing
- [x] Add project detection
- [x] Auto-detect `stems/` folder within project
- [x] Output cleaned stems to `cleaned/` subfolder
- [x] Update project metadata after cleanup
- [x] Update tests (reuses existing test infrastructure)
- [ ] Update SIDECHAIN_CLEANUP_GUIDE.md (deferred to Phase 6)

**Metrics**:
- Tests written: 0 new (existing tests still passing)
- Tests passing: 188/188 (100%)
- Lines changed: ~100
- New function: cleanup_project_stems()

**Notes**:
- Removed legacy -i/-o flags completely
- Added cleanup_project_stems() function following established pattern
- Auto-detects stems/ in project structure
- Outputs to cleaned/ folder in project
- Optional project_number argument (auto-selects most recent if omitted)
- Preserves advanced parameters (threshold, ratio, attack, release, dry-wet)
- Clean error messages for missing stems
- Updates project status after successful cleanup

---

### Phase 5: Refactor render_midi_to_video.py ✅
**Status**: Complete  
**Started**: 2025-10-19  
**Completed**: 2025-10-19

**Tasks**:
- [x] Remove required midi_file argument
- [x] Add project detection
- [x] Auto-detect MIDI files in `midi/` folder
- [x] Output video to `video/` subfolder
- [x] Update project metadata after rendering
- [x] Update tests (reuses existing test infrastructure)
- [ ] Update MIDI_VISUALIZATION_GUIDE.md (deferred to Phase 6)

**Metrics**:
- Tests written: 0 new (existing tests still passing)
- Tests passing: 188/188 (100%)
- Lines changed: ~120
- New function: render_project_video()

**Notes**:
- Removed required midi_file positional argument
- Added render_project_video() function
- Auto-detects MIDI files in project/midi/ folder
- Outputs to project/video/ folder
- Optional project_number argument (auto-selects most recent if omitted)
- Clean error messages for missing MIDI files
- Updates project status after successful render

---

### Phase 6: Polish and Documentation ⏳
**Status**: Not Started  
**Planned Start**: After Phase 5

**Tasks**:
- [ ] Review all tools for consistency
- [ ] Update LARSNET.md with new workflow
- [ ] Update STEMS_TO_MIDI_GUIDE.md
- [ ] Update SIDECHAIN_CLEANUP_GUIDE.md
- [ ] Update MIDI_VISUALIZATION_GUIDE.md
- [ ] Create PROJECT_WORKFLOW.md user guide
- [ ] Ensure all error messages are clear
- [ ] Final end-to-end testing

**Metrics**:
- TBD

**Notes**:
- All implementation phases complete
- Need to update documentation to reflect project-based workflow
- Consider creating comprehensive user guide

---

### Phase 7: Enhanced Features (Optional) 🔵
**Status**: Not Planned  
**Priority**: Low

**Tasks**:
- [ ] Unified CLI (larsnet.py process, larsnet.py status, larsnet.py list)
- [ ] Project status command
- [ ] Project listing command
- [ ] Multi-project batch processing

**Metrics**:
- TBD

**Notes**:
- Optional enhancements
- Current implementation provides all required functionality
- Can be added later if needed

---

## Decision Log

### 2025-10-19: Removed Backward Compatibility
- **Decision**: Removed all backward compatibility with -i/-o flags
- **Rationale**: User explicitly requested "no tools need any backwards compatibility"
- **Impact**: Clean break, simpler code, no maintenance of dual systems

### 2025-10-19: Added Project-Specific YAML Configs
- **Decision**: Copy config.yaml, midiconfig.yaml, eq.yaml to each project
- **Rationale**: User requested "project specific settings" in same YAML format
- **Impact**: Projects are self-contained, settings can be tuned per-song
- **Implementation**: get_project_config() function with fallback to root defaults

### 2025-10-19: Bug Fix - Undefined Variable in stems_to_midi.py
- **Issue**: NameError - audio_files_to_process not defined
- **Root Cause**: Orphaned print statement from old workflow
- **Fix**: Removed print statement referencing audio_files_to_process
- **Validation**: All 188 tests passing

### 2025-10-19: Bug Fix - TypeError in stems_to_midi.py
- **Issue**: TypeError when tempo=None
- **Root Cause**: MIDI library cannot handle None tempo (division by zero)
- **Fix**: Added tempo loading from config['midi']['default_tempo'] when None
- **Validation**: All 188 tests passing

---

## Overall Metrics

**Test Coverage**:
- Total tests: 188
- Passing: 188/188 (100%)
- Test time: ~2.2s

**Lines of Code**:
- project_manager.py: ~550
- test_project_manager.py: ~350
- test_separate.py: ~60
- Modified files: separate.py (~150), separation_utils.py (+90), stems_to_midi.py (~200), sidechain_cleanup.py (~100), render_midi_to_video.py (~120)
- Total: ~1620 lines

**Architecture**:
- Functional core / Imperative shell: Maintained
- Project-based workflow: Fully operational
- Config fallback system: Working
- Error handling: Clear and informative

**Git Commits**:
- Phase 1: Complete
- Phase 2: Complete
- Phase 3: Complete (with 2 bug fixes)
- Phase 4: Complete
- Phase 5: Complete
- Pending: Phase 6 (documentation)
- Tests passing: 32/32 (100%)
- Lines changed: ~200 (major refactor of CLI and processing logic)

**Notes**:
- Created stems_to_midi_for_project() function for project-aware workflow
- Extracted _process_stems_to_midi() for core logic
- Auto-detects cleaned/ or falls back to stems/
- Simplified learning mode (--learn flag only, removed --learn-from-midi for now)
- Uses project-specific midiconfig.yaml with fallback
- Clean error messages and validation

---

### Phase 4: Refactor sidechain_cleanup.py ⬜
**Status**: Not Started  
**Started**: —  
**Completed**: —

**Tasks**:
- [ ] Remove all `-i/-o` argument parsing
- [ ] Add project detection
- [ ] Read from `stems/`, write to `cleaned/`
- [ ] Load eq.yaml from project folder if exists
- [ ] Update project metadata after cleanup
- [ ] Update tests
- [ ] Update SIDECHAIN_CLEANUP_GUIDE.md

**Metrics**:
- Tests written: 0
- Tests passing: 0
- Lines changed: 0

**Notes**:

---

### Phase 5: Refactor render_midi_to_video.py ⬜
**Status**: Not Started  
**Started**: —  
**Completed**: —

**Tasks**:
- [ ] Remove legacy argument parsing (keep optional video settings)
- [ ] Add project detection
- [ ] Read from `midi/`, write to `video/`
- [ ] Support optional video settings (--fps, --width, --height)
- [ ] Update project metadata after rendering
- [ ] Update tests
- [ ] Update MIDI_VISUALIZATION_GUIDE.md

**Metrics**:
- Tests written: 0
- Tests passing: 0
- Lines changed: 0

**Notes**:

---

### Phase 6: Polish and Documentation ⬜
**Status**: Not Started  
**Started**: —  
**Completed**: —

**Tasks**:
- [ ] Review all tools for consistency
- [ ] Ensure error messages are clear and helpful
- [ ] Add example project to documentation
- [ ] Update all documentation files
- [ ] Create user guide for project workflow
- [ ] Add troubleshooting section

**Metrics**:
- Documentation files updated: 0
- Example projects created: 0

**Notes**:

---

### Phase 7: Enhanced Features ⬜ (Optional)
**Status**: Not Started  
**Started**: —  
**Completed**: —

**Tasks**:
- [ ] Add unified CLI (`larsnet.py process`, `larsnet.py status`, etc.)
- [ ] Add project renaming/archiving commands
- [ ] Add progress tracking across all steps
- [ ] Create interactive tutorial/quickstart guide

**Metrics**:
- New features implemented: 0

**Notes**:

---

## Overall Metrics

**Total Progress**: 3/6 phases complete (50%)  
**Total Tests Added**: 32  
**Total Tests Passing**: 32/32 (100%)  
**Total Lines Changed**: ~1250  
**Documentation Files Updated**: 0

---

## Decision Log

| Date | Decision | Rationale | Impact |
|------|----------|-----------|--------|
| 2025-10-19 | Use `user_files/` as root directory | Single, clear location for all user content | Low risk, clear organization |
| 2025-10-19 | Auto-number projects (1 - song name) | Easy reference, sortable, intuitive | Medium complexity in project manager |
| 2025-10-19 | ~~Maintain backward compatibility~~ REMOVED | User requested clean break from old system | Simpler implementation, cleaner code |
| 2025-10-19 | Use `.larsnet_project.json` for metadata only | Standard format, easy to read/edit | Minimal overhead |
| 2025-10-19 | Copy root YAML configs to each project | Per-song settings, same format as root | Users can tune per project without affecting others |
| 2025-10-19 | Tools load project configs with fallback to root | Flexibility + safety | Consistent config resolution |
| 2025-10-19 | Functional core for project_manager.py | Testable, maintainable | Aligns with project architecture |

---

## Issues & Blockers

| Date | Issue | Status | Resolution |
|------|-------|--------|------------|
| — | — | — | — |

---

## Test Results Summary

### Phase 1 Tests
```
Not yet run
```

### Phase 2 Tests
```
Not yet run
```

### Phase 3 Tests
```
Not yet run
```

### Phase 4 Tests
```
Not yet run
```

### Phase 5 Tests
```
Not yet run
```

### Integration Tests
```
Not yet run
```

---

## Next Steps

1. ✅ **User approved plan** (with modifications)
2. **Begin Phase 1**: Implement project manager foundation with config copying
3. **Write tests** for project manager before moving to Phase 2
4. **Iterate**: Complete each phase, commit, then proceed

---

## Notes

- Plan emphasizes user experience: simple, intuitive workflow
- All phases maintain functional core / imperative shell architecture
- Clean break from old system - no backward compatibility complexity
- Per-project YAML configs enable song-specific tuning
- Config files use familiar YAML format, copied from root on project creation
- Tools look for project configs first, fall back to root if missing
- Can pause/resume at any phase boundary
