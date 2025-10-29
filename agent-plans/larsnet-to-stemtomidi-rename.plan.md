# LarsNet to StemToMIDI Rename Plan

## Objective
Remove all references to the Lars model and rename the project from LarsNetMidi to StemToMIDI. The MDX23C model is now the primary and only separation model used.

## Context
- LarsNet was the original drum separation model used in this project
- MDX23C is significantly more modern and effective
- LarsNet model code and references should be removed, but model selection infrastructure preserved
- Future models may be added, so keep --model parameter and conditional structure
- The repository is being renamed from larsnet-midi to StemToMIDI

## Phases

### Phase 1: Documentation and Attribution
**Goal:** Update all markdown files, documentation, and attribution to reflect StemToMIDI branding and remove LarsNet references.

**Scope:**
- Update README.md with new branding and remove LarsNet mentions
- Preserve LARSNET.md as historical attribution file
- Update all guides (WEBUI_SETUP.md, SETUP_*.md, CONTRIBUTING.md, etc.)
- Update API documentation (WEBUI_API.md)
- Update inline documentation comments

**Files:**
- README.md
- LARSNET.md (preserve but clarify it's historical)
- WEBUI_SETUP.md
- WEBUI_API.md
- WEBUI_SETTINGS.md
- CONTRIBUTING.md
- SETUP_MAC_NATIVE.md
- SETUP_WINDOWS_GPU.md
- MDX23C_PERFORMANCE.md
- All other .md files referencing LarsNet

**Success Criteria:**
- All user-facing documentation uses StemToMIDI branding
- Historical attribution to LarsNet research preserved appropriately
- No misleading references to LarsNet as active model

### Phase 2: Python Core Removal
**Goal:** Remove LarsNet model code and imports, but KEEP model selection logic for future extensibility.

**Scope:**
- Remove larsnet.py module entirely
- Remove unet.py (Lars-specific architecture)
- Update separation_utils.py to remove LarsNet code paths (keep model parameter)
- Update separate.py to keep --model flag but remove 'larsnet' option (default to 'mdx23c')
- Update project_manager.py references
- Update main.py
- Remove pretrained_larsnet_models/ directory (preserve in git history)
- Update config.yaml to remove LarsNet model paths

**Files to Modify:**
- separation_utils.py (remove LarsNet imports, remove 'larsnet' case from conditionals, keep model parameter)
- separate.py (keep --model argument, remove 'larsnet' from choices, update help text)
- project_manager.py (update docstrings and metadata)
- main.py
- config.yaml
- device_utils.py (if LarsNet-specific)

**Files to Delete:**
- larsnet.py
- unet.py
- pretrained_larsnet_models/ (entire directory)

**Success Criteria:**
- No imports of larsnet module
- Model selection infrastructure preserved for future models
- 'larsnet' option removed from all model choices
- MDX23C is the only currently available model
- Code is cleaner without LarsNet-specific code paths

### Phase 3: Test Updates
**Goal:** Update all tests to remove LarsNet references and model selection logic.

**Scope:**
- Update test files that reference LarsNet
- Remove tests for model selection/switching
- Update test expectations for MDX23C only
- Ensure all tests pass

**Files:**
- test_separate.py
- test_project_manager.py
- Any other test files with LarsNet references
- webui/test_api.py
- webui/test_config_*.py

**Success Criteria:**
- All tests pass
- No tests reference LarsNet model
- Test coverage maintained for MDX23C path

### Phase 4: WebUI and Assets
**Goal:** Update web interface code, templates, and SVG branding.

**Scope:**
- Update webui/app.py references
- Update webui/templates/index.html
- Rename CSS classes from larsnet-* to stemtomidi-*
- Update SVG logo (larsnetmidi.svg → stemtomidi.svg)
- Update localStorage keys
- Update API documentation
- Update webui module docstrings

**Files:**
- webui/app.py
- webui/templates/index.html (CSS classes, branding)
- webui/static/img/larsnetmidi.svg → webui/static/img/stemtomidi.svg
- webui/api/operations.py
- webui/config*.py
- All webui test files

**Success Criteria:**
- Web interface displays "StemToMIDI" branding
- SVG logo updated with correct naming
- CSS classes use stemtomidi-* naming
- localStorage uses stemtomidi_settings key
- All webui tests pass

### Phase 5: Infrastructure and Configuration
**Goal:** Update docker, conda, and other infrastructure files.

**Scope:**
- docker-compose.yaml (service name, container name)
- Dockerfile references
- environment.yml
- conda-lock.yml (if needed)
- .gitattributes (pretrained_larsnet_models filter)
- pytest.ini
- Any other config files

**Files:**
- docker-compose.yaml
- Dockerfile
- environment.yml
- .gitattributes
- PROJECT_METADATA_FILE constant

**Success Criteria:**
- Docker container named stemtomidi-midi or similar
- All infrastructure references updated
- Git LFS filters updated/removed

## Risks and Mitigations

**Risk:** Breaking existing user projects with .larsnet_project.json metadata
**Mitigation:** Rename metadata file to .stemtomidi_project.json, consider migration script

**Risk:** Users with docker containers named larsnet-midi
**Mitigation:** Document the rename in README, provide migration instructions

**Risk:** Removing working LarsNet code that users might depend on
**Mitigation:** User confirmed MDX23C is the only model in use, LarsNet is deprecated

**Risk:** Removing model selection infrastructure needed for future models
**Mitigation:** Keep model parameter and selection logic, only remove LarsNet-specific code paths

**Risk:** Test failures during refactoring
**Mitigation:** Run tests after each phase, commit at phase boundaries

## Success Criteria (Overall)

1. **Zero LarsNet References:** No code imports or uses larsnet module
2. **Clean Model Path:** Only MDX23C separation logic remains (but model selection infrastructure preserved)
3. **Extensible Architecture:** Model parameter kept for future model additions
4. **Consistent Branding:** All user-facing materials say "StemToMIDI"
5. **Tests Pass:** 100% test passage after each phase
6. **Attribution Preserved:** Historical credit to LarsNet research maintained appropriately
7. **Documentation Complete:** All docs reflect new reality

## Rollback Plan

Each phase commits separately. If issues arise:
1. Identify the problematic phase via git log
2. git revert the specific commit
3. Address issues and recommit

## Notes

- The note in docker-compose.yaml has a typo: "larsmet-midi" should be fixed
- SVG already says "StemToMIDI" in the graphic, just needs filename change
- Project metadata file constant should be renamed throughout
- localStorage key in webui should be migrated for existing users
