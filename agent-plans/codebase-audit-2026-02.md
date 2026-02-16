# Codebase Analysis and Tracking

This document tracks the current state of the codebase, documentation accuracy, and areas needing attention.

---

## Summary Statistics

| Metric | Value | Verified |
|--------|-------|----------|
| Total Python files (excl. archive/lib_v5) | 128 | ✅ |
| Test files | 50+ | |
| Total tests collected | 960 (980 discovered, 20 deselected) | ✅ |
| ARCH documentation files | 6 | |
| Agent plan files | 84 | |

---

## 1. Architectural Documentation Status

### Current ARCH Files ✅

| File | Status | Last Updated | Accuracy |
|------|--------|--------------|----------|
| ARCH_C1_OVERVIEW.md | ⚠️ Needs Update | Jan 18 2026 | Partial - missing new pipelines |
| ARCH_C2_CONTAINERS.md | ⚠️ Needs Update | Jan 18 2026 | Partial - missing new modules |
| ARCH_C3_COMPONENTS.md | ⚠️ Needs Update | Jan 18 2026 | Outdated - many new files |
| ARCH_DATA_FLOW.md | ⚠️ Needs Update | Jan 19 2026 | Partial - missing rebuild pipeline |
| ARCH_LAYERS.md | ⚠️ Needs Update | Jan 18 2026 | Partial - missing new cores |
| ARCH_FILES.md | ⚠️ Needs Update | Jan 19 2026 | Incomplete - missing many files |

### Issues Found

1. **Missing documented modules:**
   - `stems_to_midi/analysis_core.py` (2631 lines - MAJOR)
   - `stems_to_midi/processing_shell.py` (1204 lines)
   - `stems_to_midi/energy_detection_core.py` (713 lines)
   - `stems_to_midi/note_classification_core.py` (757 lines)
   - `stems_to_midi/rebuild_core.py` (639 lines)
   - `stems_to_midi/stereo_core.py` (558 lines)
   - `stems_to_midi/clustering_core.py` (310 lines)
   - `stems_to_midi/optimization_core.py` (465 lines)
   - `webui/yaml_config_core.py` (NEW - not in docs)
   - `webui/settings_schema.py` (33024 bytes - NEW)
   - `webui/api/job_status.py` (232341 bytes - NEW)
   - `webui/api/operations.py` (NEW)
   - `webui/api/settings.py` (NEW)

2. **Incorrect/outdated coverage data in ARCH_C3_COMPONENTS.md:**
   - Shows `detection.py` at 91% but file is now `detection_shell.py`
   - Shows `helpers.py` at 66% but file is now `analysis_core.py`
   - Shows `processor.py` at 65% but file is now `processing_shell.py`
   - Missing all new test files in stems_to_midi/

3. **Two-pass architecture not documented:**
   - Pass 1: Detect onsets, compute features (analysis_core.py)
   - Pass 2: Classify notes (note_classification_core.py)
   - Rebuild pipeline (rebuild_core.py, rebuild_shell.py)

---

## 2. Documentation Files

### User Guides (docs/)

| File | Status | Notes |
|------|--------|-------|
| DEPENDENCIES.md | ✅ Current | |
| ARCHIVED_FEATURES.md | ✅ Current | |
| ALTERNATE_AUDIO_FEATURE.md | ⚠️ May be stale | |
| CPU_THREADING_FIX.md | ✅ Current | |
| DETECTION_OUTPUT_CONTRACT.md | ✅ Current | |
| LEARNING_MODE.md | ✅ Current | |
| midi-yaml-settings.md | ✅ Current | Large (27KB) |
| midi-yaml-settings-suggestions.md | ✅ Current | |
| deprecations.md | ✅ Current | |

### Documentation Gaps

- No guide for rebuild pipeline
- No guide for clustering/threshold optimization
- No guide for stereo processing
- No guide for note classification

---

## 3. Functional Core vs Imperative Shell

### Current Distribution

#### Functional Cores (Pure Logic) ✅

| Module | Lines | Coverage | Status |
|--------|-------|----------|--------|
| `midi_types.py` | 119 | 95% | ✅ Documented |
| `midi_core.py` | 108 | ~78% | ✅ Documented |
| `midi_render_core.py` | 122 | 100% | ✅ Documented |
| `sidechain_core.py` | 76 | 100% | ✅ Documented |
| `render_video_core.py` | 99 | 100% | ✅ Documented |
| `stems_to_midi/analysis_core.py` | 2631 | HIGH | ⚠️ NEW - Not documented |
| `stems_to_midi/note_classification_core.py` | 757 | HIGH | ⚠️ NEW - Not documented |
| `stems_to_midi/energy_detection_core.py` | 713 | HIGH | ⚠️ NEW - Not documented |
| `stems_to_midi/rebuild_core.py` | 639 | HIGH | ⚠️ NEW - Not documented |
| `stems_to_midi/stereo_core.py` | 558 | ? | ⚠️ NEW - Not documented |
| `stems_to_midi/clustering_core.py` | 310 | ? | ⚠️ NEW - Not documented |
| `stems_to_midi/midi.py` | 70 | 100% | ✅ Documented |
| `stems_to_midi/learning.py` | 166 | 95% | ✅ Documented |
| `moderngl_renderer/core.py` | 185 | 100% | ✅ Documented |
| `moderngl_renderer/animation.py` | 119 | 98% | ✅ Documented |
| `moderngl_renderer/midi_animation.py` | 121 | 94% | ✅ Documented |
| `moderngl_renderer/midi_video_core.py` | 104 | 58% | ✅ Documented |

#### Imperative Shells (I/O, GPU, Orchestration)

| Module | Lines | Coverage | Notes |
|--------|-------|----------|-------|
| `separation_shell.py` | ~142 | 8% | |
| `device_shell.py` | 118 | 8% | |
| `sidechain_shell.py` | ~143 | 19% | |
| `render_midi_video_shell.py` | 573 | 15% | |
| `stems_to_midi/processing_shell.py` | 1204 | ? | NEW - Main orchestrator |
| `stems_to_midi/detection_shell.py` | ? | ? | NEW |
| `stems_to_midi/energy_detection_shell.py` | ? | ? | NEW |
| `stems_to_midi/rebuild_shell.py` | ? | ? | NEW |
| `moderngl_renderer/shell.py` | ~49082 bytes | 62% | |
| `moderngl_renderer/midi_video_shell.py` | ~18213 bytes | 9% | |
| `moderngl_renderer/text_overlay_shell.py` | ~5890 bytes | 12% | |

### Pattern Compliance: GOOD

The core/shell pattern is well-established and followed in new code:
- Clear docstrings stating "Functional Core" or "Imperative Shell"
- Cores import from other cores, shells import from cores
- Shells handle I/O, cores are pure

---

## 4. Test Coverage Assessment

### Coverage Summary

| Module | Lines Covered | Notes |
|--------|---------------|-------|
| midi_types.py | 119 | 95% |
| midi_render_core.py | 122 | 100% |
| moderngl_renderer/core.py | 185 | 100% |
| sidechain_core.py | 76 | 100% |
| render_video_core.py | 99 | 100% |
| stems_to_midi/analysis_core.py | 653 | HIGH |
| stems_to_midi/midi.py | 70 | 100% |
| stems_to_midi/learning.py | 166 | 95% |
| moderngl_renderer/animation.py | 119 | 98% |
| moderngl_renderer/midi_animation.py | 121 | 94% |
| project_manager.py | 187 | 68% |
| midi_core.py | 108 | ~78% |

### Test Files Not in Coverage (excluded by .coveragerc)

- All webui/ tests (excluded)
- Shell modules (excluded)
- CLI scripts (excluded)
- lib_v5/ (excluded)
- archive/ (excluded)

### Test Stats

- Total tests: 960 (selected after deselections)
- Well-covered cores: 16+
- Integration tests: Multiple (test_integration.py, etc.)

---

## 5. Deprecated/Leftover Files

### Research/Analysis Files (can be deleted or moved to archive)

| File | Size | Purpose | Recommendation |
|------|------|---------|----------------|
| `*.csv` (7 files) | ~1MB total | Analysis artifacts | DELETE or move to archive |
| `*.log` (2 files) | ~500KB | Debug logs | DELETE |
| `comparison_*.txt` | ~400KB | Test results | DELETE |
| `detection_comparison_results.txt` | ~82KB | Analysis output | DELETE |
| `stem_comparison_results.txt` | ~376KB | Analysis output | DELETE |

### Runtime Files That Should Be Gitignored

| File/Dir | Status | Action |
|----------|--------|--------|
| `user_files/` | Has runtime data | Already has .gitkeep |
| `user_files/1 - .../` | Project data | Add to .gitignore |
| `user_files/old/` | Old project | DELETE or archive |
| `user_files/test.wav` | Test file | DELETE |

### Duplicate/Conflicting Files

| File | Issue |
|------|-------|
| `midiconfig.normalized.yaml` | Duplicate of midiconfig.yaml? |
| `midiconfig_calibrated.yaml` | May be deprecated by normalized |

### Archive Directory

Already exists with:
- `archive/debugging/` - Research scripts
- `archive/demos/` - Demo scripts
- `archive/examples/` - Example scripts
- `archive/benchmarks/` - Benchmark scripts

**These should be kept as-is or expanded.**

---

## 6. LLM Challenges

### What LLMs Struggle With

1. **Large code files:** `analysis_core.py` (2631 lines) is too large for efficient context

2. **Missing type hints:** Some modules lack comprehensive type annotations

3. **Implicit dependencies:** Configuration dicts passed around without clear schemas

4. **Test data:** Cannot run tests without proper audio files (tests need fixtures)

5. **Coverage gaps in documentation:**
   - New modules not in ARCH files
   - Coverage percentages outdated
   - Missing pipeline documentation

6. **Complex configuration:**
   - midiconfig.yaml has 19908 bytes
   - settings_schema.py has 33024 bytes
   - Hard to understand all options without exploration

7. **Git state issues:**
   - .DS_Store files everywhere
   - __pycache__ scattered
   - user_files with project data

### Recommendations for Better LLM Support

1. **Split large files:**
   - `analysis_core.py` → split into `spectral_analysis.py`, `onset_detection.py`, etc.

2. **Add module-level docstrings:**
   - Every module should have a 1-line summary
   - Clear functional core / shell designation

3. **Create configuration schema docs:**
   - Auto-generate from Pydantic models
   - Document all midiconfig.yaml options

4. **Add more fixtures:**
   - Synthetic audio for tests
   - Smaller test files

5. **Clean up git state:**
   - Add .gitignore entries
   - Remove .DS_Store
   - Clear __pycache__

6. **Create quick-start docs:**
   - "For LLM developers: Start here"
   - Key entry points
   - Testing commands

---

## 7. Action Items

### Immediate (low effort)

- [ ] Delete log files (*.log)
- [ ] Delete analysis CSV files (or move to archive)
- [ ] Delete comparison result txt files
- [ ] Clean up user_files/old/
- [ ] Remove .DS_Store from git

### Short-term (medium effort)

- [ ] Update ARCH_C3_COMPONENTS.md with new modules
- [ ] Update ARCH_LAYERS.md with new cores
- [ ] Update ARCH_FILES.md with complete file list
- [ ] Add rebuild pipeline docs
- [ ] Document note classification system

### Long-term (high effort)

- [ ] Split analysis_core.py into smaller modules
- [ ] Split processing_shell.py if possible
- [ ] Create comprehensive configuration guide
- [ ] Add more type hints throughout

---

*Generated: 2026-02-15*
*Verified: 2026-02-15 (repo scan)*
*For: DrumToMIDI codebase audit*

---

## Reproduction Commands

These commands were used to verify the counts above:

```bash
# Count Python files (excl. archive/lib_v5)
find . -name "*.py" -not -path "./archive/*" -not -path "./lib_v5/*" | wc -l

# Top 20 Python files by line count
find . -name "*.py" -not -path "./archive/*" -not -path "./lib_v5/*" -exec wc -l {} + | sort -rn | head -n 20

# Count tests collected
python -m pytest --collect-only -q
```
