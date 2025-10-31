# Alternate Audio Tracks Feature

## Overview

The alternate audio tracks feature allows you to upload and select different audio files when rendering drum visualization videos. This enables use cases like:

- **No-Drums Mix**: Upload a version without drums and your video becomes the drum track
- **Mastered Version**: Use a final mastered mix instead of the raw recording
- **Practice Tracks**: Custom backing tracks for practice sessions

## Quick Start

### 1. Upload Alternate Audio

1. Select a project in the web UI
2. Click the "Video" operation button
3. Expand the "Settings" section
4. In the "Upload Alternate Mix" area, click "Choose WAV File" or drag and drop a WAV file
5. Wait for upload to complete
6. The file appears in the list below and in the "Audio Track" dropdown

### 2. Render Video with Selected Audio

1. Click the "Video" operation button
2. Expand the "Settings" section
3. In the "Audio Track" dropdown, select:
   - "Don't Include Audio" - Video only
   - "{project_name}.wav" - Original audio
   - "{filename}.wav" - Your uploaded alternate mix
4. Click "Video" to start rendering

### 3. Manage Alternate Audio Files

- **View**: All uploaded files appear in the upload section and "Audio Track" dropdown
- **Delete**: Click the trash icon next to any alternate file in the settings section
- **Re-upload**: Delete the old file and upload a new one

## Technical Details

### File Requirements

- **Format**: WAV only
- **Location**: Stored in `project/alternate_mix/` directory
- **Size Limit**: 500MB (configured in `webui/config.py`)

### API Endpoints

```bash
# List audio files
GET /api/projects/1/audio-files

# Upload alternate audio
POST /api/projects/1/upload-alternate-audio
  -F "file=@no_drums.wav"

# Delete alternate audio
DELETE /api/projects/1/audio-files/no_drums.wav

# Render video with audio
POST /api/render-video
  {"project_number": 1, "audio_source": "alternate_mix/no_drums.wav"}
```

See [WEBUI_API.md](WEBUI_API.md) for complete API documentation.

### Project Directory Structure

```
user_files/
  1 - Song Name/
    Song Name.wav           # Original audio
    alternate_mix/          # NEW: Alternate audio files
      no_drums.wav
      mastered_version.wav
    midi/
    video/
    stems/
    cleaned/
```

## Use Cases

### Scenario 1: Practice Video (No Drums)

1. Export your mix without drums from your DAW
2. Upload as `no_drums.wav` to your DrumToMIDI project
3. Render video selecting the no-drums mix
4. Play the video and drum along - you ARE the drums!

### Scenario 2: Multiple Mixes

1. Upload different mixes:
   - `rough_mix.wav`
   - `mastered.wav`
   - `instrumental.wav`
2. Render separate videos for each
3. Compare and choose your favorite

### Scenario 3: Custom Arrangements

1. Create a simplified backing track for practice
2. Upload as alternate audio
3. Render practice video
4. Progress to full mix when ready

## Security & Safety

- **Path Protection**: Path traversal attacks are prevented
- **Original Protection**: Cannot delete original project audio via API
- **WAV Only**: Only WAV files accepted to ensure compatibility
- **Validation**: File type validation on both client and server

## Backward Compatibility

The old `include_audio` boolean parameter is deprecated but still works:

```json
// Old way (still works)
{"project_number": 1, "include_audio": true}

// New way
{"project_number": 1, "audio_source": "original"}
```

## Troubleshooting

### "Only WAV files are supported"

**Problem**: Tried to upload MP3, FLAC, or other format  
**Solution**: Convert to WAV using your DAW or audio editor

### Upload fails silently

**Problem**: File might be too large  
**Solution**: Check file size (must be < 500MB). Compress or reduce quality if needed

### Audio not in dropdown

**Problem**: File uploaded but doesn't appear  
**Solution**: Refresh the project (click on project name in sidebar again)

### Video has no audio

**Problem**: Selected audio but video renders without sound  
**Solution**: Check that the audio file still exists and wasn't deleted

## Implementation Details

For developers working with this feature:

- **Backend**: `webui/api/projects.py` (endpoints), `render_midi_to_video.py` (video rendering)
- **Frontend**: `webui/templates/index.html` (UI), `webui/static/js/operations.js` (handlers)
- **Tests**: `webui/test_api.py` (API tests - 8 test cases)
- **Documentation**: `WEBUI_API.md`, `WEBUI_SETUP.md`

See [agent-plans/alternate-audio-tracks.plan.md](agent-plans/alternate-audio-tracks.plan.md) for complete implementation plan.
