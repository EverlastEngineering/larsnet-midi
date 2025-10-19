# File Management Refactoring Plan

## Current State

### Problem
- Tools use `-i` and `-o` flags pointing to arbitrary directories
- Current structure: `input/`, `separated_stems/`, `cleaned_stems/`, `midi_output*/`
- Users must manually manage multiple directories
- Risk of user files being committed to repository
- No clear "project" concept per song
- Complex command-line workflows with long path arguments

### Current Tool Chain
1. **separate.py**: Input dir → `separated_stems/song_name/`
2. **sidechain_cleanup.py**: `separated_stems/` → `cleaned_stems/`
3. **stems_to_midi.py**: `cleaned_stems/` → `midi_output/`
4. **render_midi_to_video.py**: MIDI file → video file

### Existing .gitignore Coverage
- Already ignores: `input/`, `output/`, `separated_stems/`, `cleaned_stems/`, `midi_output*/`, `*.wav`, `*.mp3`, etc.

## Proposed Solution

### New Directory Structure
```
user-files/                    # Ignored in git, single location for all user content
├── .gitkeep                   # Keep folder in repo
└── 1 - bob's song/            # Auto-numbered project folders
    ├── bob's song.wav         # Original input (moved here)
    ├── config.yaml            # Project-specific copy of root config.yaml
    ├── midiconfig.yaml        # Project-specific copy of root midiconfig.yaml
    ├── eq.yaml                # Project-specific copy of root eq.yaml (if used)
    ├── stems/                 # Separated stems
    │   ├── bob's song-kick.wav
    │   ├── bob's song-snare.wav
    │   ├── bob's song-toms.wav
    │   ├── bob's song-hihat.wav
    │   └── bob's song-cymbals.wav
    ├── cleaned/               # Optional: sidechain-cleaned stems
    │   ├── bob's song-kick.wav
    │   └── ... (same structure)
    ├── midi/                  # Generated MIDI files
    │   ├── bob's song.mid
    │   └── bob's song_learning.mid
    └── video/                 # Rendered videos
        └── bob's song.mp4
```

### User Workflow

#### Initial Setup (First Time)
User drops `bob's song.wav` into `user-files/`:
```
user-files/
└── bob's song.wav
```

#### Step 1: Separation
```bash
python separate.py           # No args needed!
```
**Behavior**:
- Detects `bob's song.wav` in `user-files/`
- Creates `user-files/1 - bob's song/`
- Moves wav into folder
- Processes stems into `stems/` subfolder
- Updates project metadata (timestamp, status)

**Result**:
```
user-files/
└── 1 - bob's song/
    ├── bob's song.wav
    └── stems/
        ├── bob's song-kick.wav
        ├── bob's song-snare.wav
        └── ...
```

#### Step 2: Optional Cleanup
```bash
python sidechain_cleanup.py   # Auto-detects project
```
**Behavior**:
- Finds project(s) with `stems/` folder
- Processes into `cleaned/` subfolder
- If multiple projects, prompts or accepts `sidechain_cleanup.py 1`

#### Step 3: MIDI Generation
```bash
python stems_to_midi.py       # Auto-detects project
```
**Behavior**:
- Uses `cleaned/` if exists, else `stems/`
- Generates MIDI into `midi/` subfolder

#### Step 4: Video Rendering
```bash
python render_midi_to_video.py  # Auto-detects project
```
**Behavior**:
- Finds MIDI file(s) in project
- Renders to `video/` subfolder

### Smart Project Detection

#### Single Project
If only one project exists: all tools auto-select it without prompting.

#### Multiple Projects
If multiple projects exist:
```bash
$ python separate.py
Multiple projects found:
  1 - bob's song (last modified: 2025-10-19 14:32)
  2 - alice's track (last modified: 2025-10-18 09:15)
  
Select project [1]: _
```

Or specify directly:
```bash
python separate.py 1          # Work with project 1
python stems_to_midi.py 2     # Work with project 2
```

#### New Files While Projects Exist
If loose `.wav` files exist in `user-files/` root, tools prioritize them:
```bash
$ python separate.py
Found new file: charlie's drums.wav
Process new file? [Y/n]: _
```

### Project Metadata (Internal)
Each project folder contains `.larsnet_project.json`:
```json
{
  "project_name": "bob's song",
  "project_number": 1,
  "created": "2025-10-19T14:32:00",
  "last_modified": "2025-10-19T15:45:00",
  "original_file": "bob's song.wav",
  "status": {
    "separated": false,
    "cleaned": false,
    "midi_generated": false,
    "video_rendered": false
  }
}
```

### Project-Specific Settings
Each project gets its own copies of configuration files on creation:
- **`config.yaml`**: LarsNet separation settings (copied from root)
- **`midiconfig.yaml`**: MIDI conversion settings (copied from root)
- **`eq.yaml`**: EQ/cleanup settings (copied from root if exists)

**Benefits**:
- Users can tweak settings per song without affecting other projects
- Settings preserved with project for reproducibility
- Same YAML format and structure as root configs
- Tools look for project configs first, fall back to root only if missing
- Users can version control their tweaked settings by copying back to root

## Implementation Phases

### Phase 1: Core Infrastructure (Foundation)
**Goal**: Create project management system.

**Tasks**:
1. Create `user-files/` directory with `.gitkeep`
2. Update `.gitignore` to include `user-files/` (except `.gitkeep`)
3. Create `project_manager.py` module:
   - `discover_projects()`: List existing projects
   - `find_loose_files()`: Find wav files in root
   - `create_project()`: Initialize new project folder with config copies
   - `get_project_by_number()`: Retrieve project by number
   - `select_project()`: Interactive/auto project selection
   - `update_project_metadata()`: Track processing status
   - `get_project_config()`: Load config from project folder or fall back to root
   - `copy_configs_to_project()`: Copy root YAML configs to project
4. Write tests for project manager (functional core)
5. Document API in docstrings

**Success Criteria**:
- All project manager functions tested and working
- Can create/discover projects programmatically
- Config files properly copied to new projects
- Config loading with fallback works correctly

**Risks**:
- Project numbering conflicts if users manually create folders
- Mitigation: Robust parsing and validation of project numbers

### Phase 2: Refactor separate.py
**Goal**: Make `separate.py` use new project system exclusively.

**Tasks**:
1. Remove all `-i/-o` argument parsing
2. Add project detection logic to `separate.py`
3. Support parameterless invocation (auto-detect)
4. Support `separate.py <number>` syntax for multi-project scenarios
5. Move processed file into project structure
6. Load config from project folder (via `get_project_config()`)
7. Update project metadata after separation
8. Update tests
9. Update LARSNET.md documentation

**Success Criteria**:
- Can run `separate.py` without arguments
- Can specify project by number
- Uses project-specific config.yaml
- Tests pass
- Documentation updated

**New Usage**:
```bash
# Auto-detect single project or new file
python separate.py

# Specify project by number (multiple projects)
python separate.py 1

# That's it! No flags needed.
```

### Phase 3: Refactor stems_to_midi.py
**Goal**: Update MIDI generation to use project structure exclusively.

**Tasks**:
1. Remove all `-i/-o` argument parsing
2. Add project detection to `stems_to_midi.py`
3. Auto-detect `cleaned/` vs `stems/` folder within project
4. Output MIDI to `midi/` subfolder
5. Load midiconfig.yaml from project folder (via `get_project_config()`)
6. Update project metadata after MIDI generation
7. Update tests
8. Update STEMS_TO_MIDI_GUIDE.md

**Success Criteria**:
- Works without arguments
- Correctly uses cleaned stems if available, otherwise stems
- Uses project-specific midiconfig.yaml
- Tests pass
- Documentation updated

### Phase 4: Refactor sidechain_cleanup.py
**Goal**: Update cleanup tool to use project structure exclusively.

**Tasks**:
1. Remove all `-i/-o` argument parsing
2. Add project detection
3. Read from `stems/`, write to `cleaned/`
4. Load eq.yaml from project folder if exists (via `get_project_config()`)
5. Update project metadata after cleanup
6. Update tests
7. Update SIDECHAIN_CLEANUP_GUIDE.md

**Success Criteria**:
- Works without arguments
- Correctly processes stems within project
- Uses project-specific eq.yaml if present
- Tests pass
- Documentation updated

### Phase 5: Refactor render_midi_to_video.py
**Goal**: Update video rendering to use project structure exclusively.

**Tasks**:
1. Remove legacy argument parsing
2. Add project detection
3. Read from `midi/`, write to `video/`
4. Support optional video rendering settings (fps, resolution) as flags
5. Update project metadata after rendering
6. Update tests
7. Update MIDI_VISUALIZATION_GUIDE.md

**Success Criteria**:
- Works without arguments
- Can optionally specify video settings (--fps, --width, --height)
- Tests pass
- Documentation updated

### Phase 6: Polish and Documentation
**Goal**: Final cleanup and comprehensive documentation.

**Tasks**:
1. Review all tools for consistency
2. Ensure error messages are clear and helpful
3. Add example project to documentation
4. Update all documentation files
5. Create user guide for project workflow
6. Add troubleshooting section

**Success Criteria**:
- Clean, simple CLI interface across all tools
- All tools use unified project system
- Comprehensive user documentation
- Clear error messages guide users

### Phase 7: Enhanced Features (Optional)
**Goal**: Add quality-of-life improvements.

**Tasks**:
1. Add `larsnet.py` unified CLI:
   ```bash
   python larsnet.py process    # Run all steps on a project
   python larsnet.py status     # Show project status
   python larsnet.py list       # List all projects
   python larsnet.py config     # Edit project config in editor
   ```
2. Add project renaming/archiving commands
3. Add progress tracking across all steps
4. Create interactive tutorial/quickstart guide

## Risk Assessment

### High Risk
- **Data loss**: Moving/organizing user files incorrectly
  - Mitigation: Never delete, only move; log all operations; comprehensive testing

### Medium Risk
- **Complex project detection logic**: Edge cases in file discovery
  - Mitigation: Comprehensive test coverage
- **Metadata corruption**: Invalid JSON in `.larsnet_project.json`
  - Mitigation: Validate on read, recover gracefully
- **Config file conflicts**: User edits project configs while tool running
  - Mitigation: Read configs at start, document to not edit during processing

### Low Risk
- **Performance**: Scanning for projects on each invocation
  - Mitigation: Projects are typically <10, scanning is fast
- **Config sync**: Project configs diverging from root configs
  - Mitigation: This is intentional; document that project configs are independent copies

## Testing Strategy

### Unit Tests
- `test_project_manager.py`: All project management functions
- Each tool's test file: Updated for new behavior

### Integration Tests
- End-to-end workflow: WAV → stems → MIDI → video
- Multiple project handling
- Legacy flag compatibility

### Manual Testing
- Test on real audio files
- Test with multiple projects
- Test edge cases (no projects, corrupted metadata)

## Documentation Updates

### Files to Update
1. **LARSNET.md**: Main usage guide
2. **STEMS_TO_MIDI_GUIDE.md**: MIDI generation
3. **SIDECHAIN_CLEANUP_GUIDE.md**: Cleanup process
4. **MIDI_VISUALIZATION_GUIDE.md**: Video rendering
5. **README.md**: Quick start examples
6. **CONTRIBUTING.md**: Development workflow

### New Documentation
1. **PROJECT_MANAGEMENT.md**: Detailed project system guide
2. **MIGRATION_GUIDE.md**: Moving from old to new workflow

## Success Metrics

### User Experience
- ✓ Zero required command-line arguments for basic workflow
- ✓ Clear project organization in single location
- ✓ No risk of committing user files
- ✓ Intuitive numbering system

### Technical
- ✓ All tests passing
- ✓ No breaking changes until Phase 6
- ✓ Clean separation of concerns (functional core)
- ✓ Comprehensive error handling

### Documentation
- ✓ All guides updated
- ✓ Examples reflect new workflow
- ✓ Migration guide available

## Timeline Estimate

- **Phase 1**: 2-3 hours (foundation with config management)
- **Phase 2**: 1-2 hours (separate.py - simpler without backward compat)
- **Phase 3**: 1-2 hours (stems_to_midi.py)
- **Phase 4**: 1-2 hours (sidechain_cleanup.py)
- **Phase 5**: 1 hour (render_midi_to_video.py)
- **Phase 6**: 1 hour (polish and docs)
- **Phase 7**: 2-4 hours (optional enhancements)

**Total**: 7-11 hours (without Phase 7)

## Notes
- This plan prioritizes user experience while maintaining code quality
- Functional core / imperative shell architecture maintained throughout
- Clean break from old system - no backward compatibility complexity
- Project-specific configs enable per-song tuning without affecting other projects
- Can be implemented incrementally, testing at each phase
- Config files use same YAML format as root, making them familiar and easy to edit
