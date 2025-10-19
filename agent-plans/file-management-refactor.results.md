# ## Plan Overview
Refactor file management system from `-i/-o` flags to unified `user-files/` project-based structure with per-project YAML configs.

**Created**: 2025-10-19  
**Updated**: 2025-10-19 (removed backward compatibility, added project-specific configs)  
**Status**: Planning Complete, Ready to BeginManagement Refactoring - Results Tracker

## Plan Overview
Refactor file management system from `-i/-o` flags to unified `user-files/` project-based structure.

**Created**: 2025-10-19  
**Status**: Planning Complete, Awaiting Approval

---

## Phase Completion

### Phase 1: Core Infrastructure ✅
**Status**: Complete  
**Started**: 2025-10-19  
**Completed**: 2025-10-19

**Tasks**:
- [x] Create `user-files/` directory with `.gitkeep`
- [x] Update `.gitignore` to include `user-files/`
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

### Phase 3: Refactor stems_to_midi.py ⬜
**Status**: Not Started  
**Started**: —  
**Completed**: —

**Tasks**:
- [ ] Remove all `-i/-o` argument parsing
- [ ] Add project detection
- [ ] Auto-detect `cleaned/` vs `stems/` folder within project
- [ ] Output MIDI to `midi/` subfolder
- [ ] Load midiconfig.yaml from project folder
- [ ] Update project metadata after MIDI generation
- [ ] Update tests
- [ ] Update STEMS_TO_MIDI_GUIDE.md

**Metrics**:
- Tests written: 0
- Tests passing: 0
- Lines changed: 0

**Notes**:

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

**Total Progress**: 2/6 phases complete (33%)  
**Total Tests Added**: 32  
**Total Tests Passing**: 32/32 (100%)  
**Total Lines Changed**: ~1050  
**Documentation Files Updated**: 0

---

## Decision Log

| Date | Decision | Rationale | Impact |
|------|----------|-----------|--------|
| 2025-10-19 | Use `user-files/` as root directory | Single, clear location for all user content | Low risk, clear organization |
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
