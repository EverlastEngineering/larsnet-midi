# Alternate Audio Tracks Feature Plan

## Overview
Add support for uploading and selecting alternative audio tracks for video rendering. This allows users to render drum videos synced to different audio mixes (original, no-drums version, mastered version, etc).

## Current State
- Video rendering supports a boolean `include_audio` toggle (on/off)
- When enabled, it searches for the original project audio file (e.g., `The Fate Of Ophelia.wav`)
- Audio file must be in the project root directory
- Web UI has a simple checkbox toggle for "Include Audio"

## Target State
- Projects have an `alternate_mix/` directory for additional audio files
- Users can upload WAV files to this directory via web UI
- Video rendering UI shows a dropdown selector instead of toggle
- Dropdown options:
  - "Don't Include Audio" (equivalent to current "off")
  - "{project_name}.wav" (original project audio)
  - "{uploaded_filename}.wav" (any uploaded alternate mixes)
- Backend accepts audio selection parameter and uses appropriate file

## Architecture

### Directory Structure
```
user_files/
  1 - Project Name/
    Project Name.wav          # Original audio
    alternate_mix/            # NEW: Alternate audio tracks
      no_drums.wav
      mastered_version.wav
    midi/
    video/
    stems/
    cleaned/
```

### Data Flow
1. User uploads WAV file to project via web UI
2. File is saved to `project/alternate_mix/{filename}`
3. When user opens video rendering section, UI fetches available audio files
4. User selects audio from dropdown (none, original, or alternate)
5. Selection is passed to backend as `audio_source` parameter
6. Backend resolves path and renders video with selected audio

## Implementation Phases

### Phase 1: Backend - File Management
**Goal**: Support uploading and listing alternate audio files

**Tasks**:
1. Add endpoint: `POST /api/projects/{number}/upload-alternate-audio`
   - Accepts multipart form upload of WAV file
   - Validates file type and size
   - Creates `alternate_mix/` directory if needed
   - Saves file with sanitized filename
   - Returns success/error response
2. Add endpoint: `GET /api/projects/{number}/audio-files`
   - Lists available audio files:
     - Original project audio (if exists)
     - Files in `alternate_mix/` directory
   - Returns JSON array with: `[{name, path, type: 'original'|'alternate', size}]`
3. Add tests for new endpoints
4. Update WEBUI_API.md documentation

**Success Criteria**:
- Can upload WAV files via API
- Can list all available audio files
- Files saved to correct location with proper error handling
- Tests pass

### Phase 2: Backend - Video Rendering Integration
**Goal**: Modify video rendering to accept audio file selection

**Tasks**:
1. Update `render_project_video()` signature:
   - Replace `include_audio: bool` with `audio_source: Optional[str]`
   - `audio_source` can be: `None`, `"original"`, or `"alternate_mix/{filename}"`
2. Update audio file resolution logic:
   - If `None`: render without audio
   - If `"original"`: use existing logic to find project audio
   - If starts with `"alternate_mix/"`: resolve to `project_dir / audio_source`
3. Update `/api/render-video` endpoint to accept `audio_source` parameter
4. Update `run_render_video()` job function signature
5. Maintain backward compatibility with boolean `include_audio` (deprecated)
6. Add tests for audio source resolution
7. Update WEBUI_API.md

**Success Criteria**:
- Video rendering accepts audio source parameter
- Correctly resolves audio file paths for all scenarios
- Tests pass for each audio source type
- Documentation updated

### Phase 3: Frontend - Upload UI
**Goal**: Add file upload interface for alternate audio

**Tasks**:
1. Add upload section in project details panel (when project selected):
   - Small file input area above "Include Audio" section
   - Label: "Upload Alternate Mix"
   - Accept: `.wav` files only
   - Display upload progress
   - Show uploaded files list with delete option
2. Implement upload handler in `operations.js`:
   - Use `FormData` for multipart upload
   - Show loading indicator during upload
   - Update audio files list after successful upload
   - Show error toast on failure
3. Implement file list display:
   - Show uploaded alternate mixes
   - Include file size
   - Add delete button (with confirmation)
4. Add delete handler:
   - Call delete endpoint (to be created)
   - Remove from UI on success

**Success Criteria**:
- Users can upload WAV files from project view
- Upload shows progress and feedback
- Uploaded files appear in list
- Users can delete uploaded files
- UI handles errors gracefully

### Phase 4: Frontend - Audio Selection Dropdown
**Goal**: Replace toggle with dropdown selector

**Tasks**:
1. Replace checkbox toggle in video settings:
   - Remove: `<input type="checkbox" id="setting-include-audio">`
   - Add: `<select id="setting-audio-source">` dropdown
2. Populate dropdown options dynamically:
   - Fetch audio files when project loads
   - Option: `value=""` → "Don't Include Audio"
   - Option: `value="original"` → "{project_name}.wav" (if exists)
   - Option: `value="alternate_mix/{filename}"` → "{filename}"
3. Update `settings.js`:
   - Change `includeAudio` boolean to `audioSource` string
   - Update settings getter to return selected option value
4. Update `operations.js` `startVideo()`:
   - Send `audio_source` parameter instead of `include_audio`
   - Maintain backward compatibility in API client

**Success Criteria**:
- Dropdown shows all available audio options
- Selection persists during session
- Sends correct audio source to backend
- UI updates when files are uploaded/deleted

### Phase 5: Backend - File Deletion
**Goal**: Allow deletion of alternate audio files

**Tasks**:
1. Add endpoint: `DELETE /api/projects/{number}/audio-files/{filename}`
   - Validates filename is in `alternate_mix/` only (protect original)
   - Deletes file from disk
   - Returns success/error
2. Add safety checks:
   - Prevent path traversal attacks
   - Only allow deletion from `alternate_mix/` directory
   - Return 404 if file doesn't exist
3. Add tests
4. Update WEBUI_API.md

**Success Criteria**:
- Can delete alternate audio files via API
- Cannot delete original project audio
- Path traversal protected
- Tests pass

### Phase 6: Documentation
**Goal**: Document the new feature

**Tasks**:
1. Update WEBUI_SETUP.md:
   - Add section on alternate audio tracks
   - Explain use cases (no-drums mixes, different masters, etc)
   - Document upload and selection workflow
2. Update MIDI_VISUALIZATION_GUIDE.md:
   - Add section on audio track selection
   - Update CLI docs if needed
3. Update README.md if feature is significant enough
4. Add user-facing help text in UI (tooltips/info icons)

**Success Criteria**:
- Documentation covers all aspects of feature
- Use cases clearly explained
- Screenshots or examples provided
- Help text available in UI

## Risks and Considerations

### Technical Risks
1. **File Upload Security**: 
   - Mitigation: Validate file types, check magic bytes, sanitize filenames
   - Set reasonable file size limits (e.g., 500MB max)

2. **Storage Space**: 
   - Large audio files can consume disk space
   - Mitigation: Document best practices, consider adding storage warnings

3. **Audio Format Compatibility**: 
   - FFmpeg should handle various WAV formats, but edge cases exist
   - Mitigation: Test with various bit depths, sample rates

4. **Race Conditions**: 
   - File deletion while video rendering in progress
   - Mitigation: Check file exists before rendering, handle missing file gracefully

### UX Risks
1. **Discovery**: Users may not know this feature exists
   - Mitigation: Add help text, tooltips, and documentation
   - Consider adding a "tip" banner first time they render video

2. **Confusion**: Purpose of alternate mixes may not be clear
   - Mitigation: Provide clear examples in help text

### Backward Compatibility
- Existing API calls with `include_audio: true/false` must continue working
- Deprecate old parameter but maintain support for 1-2 versions
- Add migration guide for API users

## Success Metrics
- Users can upload alternate audio files without errors
- Video rendering succeeds with all audio source types
- All tests pass (target: >90% coverage on new code)
- Documentation is clear and complete
- Zero security vulnerabilities in file upload/deletion

## Future Enhancements (Out of Scope)
- Audio preview/playback in UI before rendering
- Format conversion (MP3 → WAV) on upload
- Automatic normalization of audio levels
- Batch rendering with multiple audio sources
- Cloud storage integration for large files
