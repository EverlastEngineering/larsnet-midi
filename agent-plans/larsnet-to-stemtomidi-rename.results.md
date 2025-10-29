# LarsNet to StemToMIDI Rename - Results

## Progress Tracking

### Phase 1: Documentation and Attribution
- [x] README.md updated with StemToMIDI branding
- [x] LARSNET.md preserved as historical attribution
- [x] WEBUI_SETUP.md updated
- [x] WEBUI_API.md updated
- [x] WEBUI_SETTINGS.md updated
- [x] CONTRIBUTING.md updated
- [x] SETUP_MAC_NATIVE.md updated
- [x] SETUP_WINDOWS_GPU.md updated
- [x] MDX23C_PERFORMANCE.md updated
- [x] Other markdown files checked and updated
  - ALTERNATE_AUDIO_FEATURE.md
  - DEPENDENCIES.md
  - MDX23C_GUIDE.md
  - SIDECHAIN_CLEANUP_GUIDE.md
  - MIDI_VISUALIZATION_GUIDE.md
  - LEARNING_MODE.md
  - TESTING_MDX_PERFORMANCE.md
  - .github/instructions/how-to-perform-testing.instructions.md
- [x] Phase 1 commit completed

### Phase 2: Python Core Removal
- [x] larsnet.py deleted
- [x] unet.py deleted
- [x] separation_utils.py cleaned (LarsNet removed, model parameter kept)
- [x] separate.py updated (--model kept but 'larsnet' choice removed)
- [x] project_manager.py updated (metadata filename changed)
- [x] main.py updated
- [x] config.yaml updated (legacy paths commented out)
- [x] pretrained_larsnet_models/ directory removed
- [x] .gitattributes updated (LFS filter removed)
- [x] Phase 2 commit completed

### Phase 3: Test Updates
- [ ] test_separate.py updated
- [ ] test_project_manager.py updated
- [ ] webui/test_api.py updated
- [ ] Other test files updated
- [ ] All tests passing
- [ ] Phase 3 commit completed

### Phase 4: WebUI and Assets
- [ ] webui/app.py updated
- [ ] webui/templates/index.html updated
- [ ] CSS classes renamed (larsnet-* → stemtomidi-*)
- [ ] SVG renamed and referenced correctly
- [ ] localStorage keys updated
- [ ] webui/api/operations.py updated
- [ ] Other webui files updated
- [ ] Phase 4 commit completed

### Phase 5: Infrastructure and Configuration
- [ ] docker-compose.yaml updated
- [ ] Dockerfile updated
- [ ] environment.yml updated (if needed)
- [ ] .gitattributes updated
- [ ] PROJECT_METADATA_FILE renamed
- [ ] Phase 5 commit completed

## Metrics

**Files Modified:** TBD
**Files Deleted:** TBD
**Lines Changed:** TBD
**Test Coverage:** TBD

## Decision Log

### Phase 1
- **Container naming**: Changed from `larsnet-midi` to `stemtomidi-midi` for consistency
- **Conda environment**: Changed from `larsnet` or `larsnet-midi` to `stemtomidi-midi` 
- **localStorage key**: Changed from `larsnet_settings` to `stemtomidi_settings`
- **LARSNET.md**: Preserved as historical attribution with clear note about MDX23C transition
- **Documentation tone**: Updated to past tense for LarsNet, present tense for StemToMIDI/MDX23C
- **Repository structure**: Updated from `larsnet/` to `stemtomidi/` in documentation

### Phase 2
- **Model parameter preserved**: Kept --model flag and model parameter for future extensibility
- **LarsNet removal**: Removed all LarsNet-specific code including imports, conditionals, and processing
- **Metadata filename**: Changed from `.larsnet_project.json` to `.stemtomidi_project.json`
- **Config.yaml**: Commented out LarsNet model paths, added note about MDX23C
- **Model directory**: Removed 562MB of pretrained LarsNet models (7 files)
- **Future-ready**: Code structure supports adding new models easily

## Issues Encountered

*Any issues or blockers will be documented here*

## Final Validation

- [ ] All tests passing
- [ ] No grep matches for "larsnet" in active code
- [ ] Documentation review complete
- [ ] Web UI loads and functions correctly
- [ ] Docker container builds and runs
