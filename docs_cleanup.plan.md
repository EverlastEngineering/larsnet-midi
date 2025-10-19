# Documentation Cleanup Plan

## Executive Summary

The repository contains 30+ markdown files, many of which are summary documents from completed refactoring work. These are not appropriate for an open-source repository and need to be consolidated or removed. This plan organizes the cleanup to retain essential documentation while removing redundant summary files.

## Current State Analysis

### Documentation Categories

**Core Documentation (Keep & Improve):**
- README.md - Main project documentation
- STEMS_TO_MIDI_GUIDE.md - User guide for MIDI conversion
- DEPENDENCIES.md - Dependency management guide  
- TESTING.md - Testing instructions
- TODO.md - Active development tasks
- LEARNING_MODE.md - Feature documentation
- MIDI_VISUALIZATION_GUIDE.md - Feature documentation
- ML_TRAINING_GUIDE.md - Feature documentation

**Refactoring Summaries (Remove - No longer needed):**
- CODE_DEDUPLICATION_SUMMARY.md - Completed work summary
- MODULE_SPLIT_SUMMARY.md - Completed work summary
- DEPENDENCY_CONSOLIDATION.md - Completed work summary
- FCIS_TEST_COVERAGE_REPORT.md - Completed work summary
- stems_to_midi_helpers.py.results.md - Completed work summary
- stems_to_midi_split.results.md - Completed work summary
- stems_to_midi_fcis_refactor.results.md - Completed work summary

**Historical Plans (Archive, not delete - Reference for understanding decisions):**
- stems_to_midi_fcis_refactor.plan.md
- stems_to_midi_helpers.py.plan.md
- stems_to_midi_split.plan.md

**Implementation Documentation (Consolidate into guides):**
- STATISTICAL_FILTER_IMPLEMENTATION.md - Move to STEMS_TO_MIDI_GUIDE.md
- TIMING_DRIFT_FIX.md - Move to STEMS_TO_MIDI_GUIDE.md or TODO.md
- TIMING_OFFSET_FIX.md - Move to STEMS_TO_MIDI_GUIDE.md or TODO.md
- ANIMATION_SMOOTHNESS_FIXES.md - Move to MIDI_VISUALIZATION_GUIDE.md
- ATTACK_FREQUENCY_IMPLEMENTATION.md - Move to STEMS_TO_MIDI_GUIDE.md
- SIDECHAIN_CLEANUP_GUIDE.md - Evaluate relevance

**Analysis Documents (Remove - Debugging artifacts):**
- analyze_kick_snare_bleed.analysis.md - Debugging artifact
- DUPLICATE_CODE_ANALYSIS.md - Pre-refactor analysis

**Architecture Documentation (Consolidate):**
- ARCHITECTURE_REVIEW.md - Consolidate into README.md architecture section

**Other:**
- MIGRATION.md - Evaluate relevance for users
- TESTING_CHECKLIST.md - Internal tool, consider removing or archiving
- midi_visualizer_setup.md - Seems duplicate of MIDI_VISUALIZATION_GUIDE.md

## Approach

### Principles
1. Keep only forward-looking documentation (what the project is/does, not what it was)
2. Retain user-facing guides and feature documentation
3. Remove all "summary of work completed" documents
4. Archive historical plans in a separate directory for reference (not root)
5. Consolidate scattered implementation details into relevant guides
6. Maintain single source of truth for each topic

### Target Documentation Structure

```
/docs/
  /archive/          # Historical plans (readonly reference)
    stems_to_midi_fcis_refactor.plan.md
    stems_to_midi_helpers.py.plan.md
    stems_to_midi_split.plan.md
    
README.md            # Project overview, architecture, setup
CONTRIBUTING.md      # Testing, development workflow (NEW - consolidate)
STEMS_TO_MIDI_GUIDE.md  # Complete user guide
MIDI_VISUALIZATION_GUIDE.md  # Visualization features
DEPENDENCIES.md      # Dependency management
TODO.md              # Active development roadmap
```

## Phases

### Phase 1: Create Archive Structure
**Goal:** Preserve historical plans without cluttering root

**Actions:**
- Create `/docs/archive/` directory
- Move plan files to archive:
  - stems_to_midi_fcis_refactor.plan.md
  - stems_to_midi_helpers.py.plan.md
  - stems_to_midi_split.plan.md
  
**Success Criteria:**
- Archive directory exists
- 3 plan files moved successfully
- Files still accessible for reference

**Metrics:**
- Files moved: 3

---

### Phase 2: Consolidate Architecture Documentation
**Goal:** Single clear architecture description in README.md

**Actions:**
- Extract relevant architecture content from ARCHITECTURE_REVIEW.md
- Update README.md Code Architecture section with consolidated content
- Remove redundant architecture descriptions
- Delete ARCHITECTURE_REVIEW.md

**Success Criteria:**
- README.md has comprehensive architecture section
- No duplicate architecture documentation
- Architecture section reads as current state (not review/history)

**Metrics:**
- README.md updated
- 1 file deleted

---

### Phase 3: Consolidate Implementation Documentation
**Goal:** Move scattered implementation details into user guides

**Actions:**
- Review and merge STATISTICAL_FILTER_IMPLEMENTATION.md into STEMS_TO_MIDI_GUIDE.md
- Review and merge relevant content from ATTACK_FREQUENCY_IMPLEMENTATION.md
- Review and decide on TIMING_DRIFT_FIX.md and TIMING_OFFSET_FIX.md:
  - If user-relevant: add to STEMS_TO_MIDI_GUIDE.md
  - If developer note: add to TODO.md as "completed" context
  - Otherwise: delete
- Review and merge ANIMATION_SMOOTHNESS_FIXES.md into MIDI_VISUALIZATION_GUIDE.md
- Evaluate SIDECHAIN_CLEANUP_GUIDE.md relevance

**Success Criteria:**
- User guides contain all relevant implementation details
- No standalone implementation docs in root
- Information flows naturally in context

**Metrics:**
- Files consolidated: 4-5
- User guides updated: 2

---

### Phase 4: Remove Summary Documents
**Goal:** Delete all "work completed" summary documents

**Actions:**
- Delete refactoring summaries:
  - CODE_DEDUPLICATION_SUMMARY.md
  - MODULE_SPLIT_SUMMARY.md
  - DEPENDENCY_CONSOLIDATION.md
  - FCIS_TEST_COVERAGE_REPORT.md
  - stems_to_midi_helpers.py.results.md
  - stems_to_midi_split.results.md
  - stems_to_midi_fcis_refactor.results.md
- Delete analysis documents:
  - analyze_kick_snare_bleed.analysis.md
  - DUPLICATE_CODE_ANALYSIS.md

**Success Criteria:**
- All summary documents removed
- No broken references to removed files
- Repository cleaner and more focused

**Metrics:**
- Files deleted: 9

---

### Phase 5: Clean Up Duplicate and Obsolete Files
**Goal:** Remove duplicates and evaluate edge cases

**Actions:**
- Compare midi_visualizer_setup.md vs MIDI_VISUALIZATION_GUIDE.md
  - Keep the more comprehensive one
  - Merge any unique content
  - Delete duplicate
- Evaluate MIGRATION.md:
  - If relevant to users: keep
  - If obsolete: delete
- Evaluate TESTING_CHECKLIST.md:
  - Consider merging into CONTRIBUTING.md or TESTING.md
  - If internal only: delete

**Success Criteria:**
- No duplicate content
- All remaining docs serve clear purpose
- Obsolete docs removed

**Metrics:**
- Files evaluated: 3
- Files deleted: 1-3

---

### Phase 6: Create CONTRIBUTING.md
**Goal:** Consolidate developer workflow documentation

**Actions:**
- Create CONTRIBUTING.md with sections:
  - Development Setup (from README.md if needed)
  - Testing (from TESTING.md)
  - Architecture Overview (link to README.md)
  - Code Standards (FCIS pattern, config-driven)
  - Commit Guidelines
  - Pull Request Process
- Update README.md to reference CONTRIBUTING.md
- Keep TESTING.md as short reference or consolidate fully

**Success Criteria:**
- Clear developer onboarding document
- Testing instructions easily accessible
- No duplicate information

**Metrics:**
- 1 new file created
- Possibly 1 file deleted (TESTING.md if fully consolidated)

---

### Phase 7: Update All Cross-References
**Goal:** Fix all documentation links after reorganization

**Actions:**
- Search all markdown files for references to moved/deleted files
- Update links to point to new locations or consolidated sections
- Add deprecation notices if needed
- Verify all internal links work

**Success Criteria:**
- No broken links in documentation
- All references point to correct locations
- Documentation navigable

**Metrics:**
- Files checked: all remaining .md files
- Links updated: TBD

---

### Phase 8: Final Documentation Review
**Goal:** Ensure documentation reads as unified current state

**Actions:**
- Review each remaining doc for:
  - Present tense, active voice
  - No "we just completed X" language
  - Clear purpose and audience
  - Consistent formatting
  - No speculation or outdated info
- Update README.md table of contents if needed
- Add brief descriptions to docs/ if needed

**Success Criteria:**
- All docs read as "current state"
- Consistent tone and style
- Clear navigation

**Metrics:**
- Files reviewed: all remaining .md files

---

## Success Criteria (Overall)

### Quantitative Metrics
- Root directory .md files reduced from ~30 to ~6-8
- Archive directory created with 3 plan files
- Zero broken documentation links
- All remaining docs serve active purpose

### Qualitative Metrics
- Documentation reads as current state, not history
- Clear separation: user guides vs developer guides
- Easy to find information
- No redundant content
- Professional open-source appearance

## Risks and Mitigations

### Risk 1: Losing Important Information
**Impact:** Medium  
**Mitigation:**
- Use git - all history preserved
- Review each file before deletion
- Move to archive before deletion when uncertain
- Consolidate before removing

### Risk 2: Breaking External Links
**Impact:** Low (probably no external refs to internal docs)  
**Mitigation:**
- Keep README.md, main guides in same location
- Only move/delete internal reference docs

### Risk 3: Over-consolidation
**Impact:** Low  
**Mitigation:**
- Keep guides focused and modular
- Use clear sections/headers for navigation
- Link between docs rather than duplicate

## Timeline Estimate

- Phase 1 (Archive): 10 minutes
- Phase 2 (Architecture): 30 minutes
- Phase 3 (Implementation): 45 minutes
- Phase 4 (Summaries): 15 minutes
- Phase 5 (Duplicates): 20 minutes
- Phase 6 (CONTRIBUTING): 30 minutes
- Phase 7 (Cross-refs): 20 minutes
- Phase 8 (Review): 30 minutes

**Total: ~3 hours**

## Rollback Plan

All changes tracked in git. If issues arise:
```bash
git log --oneline  # Find commit before cleanup
git revert <commit>  # Revert specific changes
# OR
git reset --hard <commit>  # Full rollback
```

## Dependencies

- None - pure documentation work
- No code changes required
- No test updates needed
