# Web UI Setup Guide

This guide explains how to set up and run the DrumToMIDI web interface.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed
- At least 8GB RAM available
- 10GB free disk space

## Quick Start

### 1. Start Docker Container

From the project root directory:

```bash
docker compose up -d
```

This will:
- Build the Docker image with all dependencies (first time only, ~5-10 minutes)
- Create and start the `DrumToMIDI-midi` container
- Expose port 5000 for the web UI

### 2. Start the Web UI

Enter the container shell:

```bash
docker exec -it DrumToMIDI-midi bash
```

Start the web server:

```bash
python -m webui.app
```

You should see:

```
============================================================
DrumToMIDI Web UI v0.1.0
============================================================

Starting server at http://0.0.0.0:5000

Press Ctrl+C to stop
```

### 3. Access the Web UI

Open your browser and navigate to:

```
http://localhost:49152
```

You should see the DrumToMIDI web interface with:
- Left sidebar showing your projects
- Upload area in the main panel
- Operation controls for each processing step

## Using the Web UI

### Upload Audio File

1. **Drag and Drop**: Drag a drum track file (WAV, MP3, FLAC, AIFF) onto the upload zone
2. **Or Click**: Click the upload zone to select a file from your computer

The file will be uploaded and a new project will be created automatically.

### Process Audio

Once a project is created, you'll see operation buttons:

1. **Separate Stems**
   - Click "Separate" button
   - Optionally configure device (CPU/CUDA/MPS), separation model settings
   - Monitor progress in real-time
   - Stems are saved to `user_files/PROJECT/stems/`

2. **Clean Stems (Optional)**
   - Click "Clean" button
   - Adjust sidechain compression parameters if needed
   - Cleaned stems saved to `user_files/PROJECT/cleaned/`

3. **Generate MIDI**
   - Click "Convert to MIDI" button
   - Adjust onset detection and velocity parameters
   - MIDI file saved to `user_files/PROJECT/midi/`

4. **Render Video (Optional)**
   - Click "Render Video" button
   - Choose FPS and resolution
   - Video saved to `user_files/PROJECT/video/`

### Monitor Jobs

- **Active Jobs**: Display at the bottom of the screen
- **Progress Bar**: Shows completion percentage
- **Live Logs**: View operation output in real-time
- **Cancel**: Click X to cancel a running job

### Download Results

After processing, download buttons appear for:
- Separated stems (ZIP)
- Cleaned stems (ZIP)
- MIDI file (.mid)
- Visualization video (.mp4)

## Configuration

### Basic Options

Available in collapsible panels for each operation:

**Separation:**
- Device: CPU or CUDA (GPU)
- Wiener Filter: Noise reduction (1.0-3.0)
- EQ Cleanup: Frequency-based bleed reduction

**Cleanup:**
- Threshold: Sidechain trigger level (-40 to -20 dB)
- Ratio: Compression amount (2:1 to 20:1)
- Attack: How fast compression starts (0.1-10 ms)
- Release: How fast compression releases (10-500 ms)

**MIDI Conversion:**
- Onset Threshold: Detection sensitivity (0.1-0.5)
- Onset Delta: Peak picking sensitivity
- Velocity Range: MIDI velocity limits (1-127)
- Tempo: BPM (auto-detected if not specified)

**Video Rendering:**
- FPS: 30, 60, or 120
- Resolution: 1080p, 1440p, or 4K (landscape or portrait)
- Audio Track: Select which audio to include in video (see below)

**Alternate Audio Tracks:**

Upload alternative audio files to render your drum video with different backing tracks:

1. **Upload Audio**: In the Video settings panel, find "Upload Alternate Mix" section
   - Click "Choose WAV File" or drag and drop WAV files
2. **Select Track**: Choose from the "Audio Track" dropdown:
   - "Don't Include Audio" - Video only
   - Original project audio
   - Any uploaded alternate mix
3. **Use Cases**:
   - No-drums mix: Be the drum track in playback
   - Mastered version: Video with final mix
   - Practice track: Custom backing track for practice

**Note:** Supported audio formats: WAV, MP3, FLAC, AIFF, AAC, OGG, M4A. Files are stored in the project's `alternate_mix/` directory and can be deleted from the settings panel.

### Advanced Configuration (Future)

Phase 3 will add advanced panels to edit all YAML config parameters:
- Per-stem onset detection overrides
- Spectral filtering parameters
- EQ frequency ranges
- Model paths

## Troubleshooting

### Web UI Won't Start

**Error:** `ImportError: No module named flask`

**Solution:** Rebuild Docker container with updated dependencies:
```bash
docker compose down
docker compose up -d --build
```

### Cannot Access http://localhost:49152

**Check 1:** Verify container is running:
```bash
docker ps
```
You should see `DrumToMIDI-midi` in the list.

**Check 2:** Verify web server is running inside container:
```bash
docker exec -it DrumToMIDI-midi bash -c "ps aux | grep python"
```

**Check 3:** Verify port is exposed:
```bash
docker port DrumToMIDI-midi
```
Should show: `5000/tcp -> 0.0.0.0:49152`

### Upload Fails

**Error:** File size exceeds limit

**Solution:** Increase `MAX_CONTENT_LENGTH` in `webui/config.py`:
```python
MAX_CONTENT_LENGTH = 1000 * 1024 * 1024  # 1GB
```

**Error:** Invalid file format

**Solution:** Ensure file is one of: WAV, MP3, FLAC, AIFF, AIF

### Job Fails Immediately

**Check Logs:** View job logs in the web UI or API:
```bash
curl http://localhost:5000/api/jobs/JOB_ID
```

**Common Issues:**
- No audio file in project → Re-upload
- Stems don't exist → Run separation first
- MIDI doesn't exist → Run MIDI conversion first
- Out of memory → Reduce concurrent jobs or use smaller files

### Jobs Get Stuck

**Solution 1:** Cancel and restart:
- Click X on job card in UI
- Or via API: `curl -X POST http://localhost:5000/api/jobs/JOB_ID/cancel`

**Solution 2:** Restart web server:
- Press Ctrl+C in container shell
- Run `python -m webui.app` again

**Solution 3:** Restart container:
```bash
docker restart DrumToMIDI-midi
```

## API Access

The web UI is built on a REST API. You can also use it programmatically:

### Example: Upload and Process

```bash
# Upload file
curl -X POST http://localhost:49152/api/upload \
  -F "file=@drums.wav"
# Returns: {"project": {"number": 1, ...}}

# Separate stems
curl -X POST http://localhost:49152/api/separate \
  -H "Content-Type: application/json" \
  -d '{"project_number": 1, "device": "cpu"}'
# Returns: {"job_id": "uuid..."}

# Check status
curl http://localhost:49152/api/jobs/UUID
```

See [WEBUI_API.md](WEBUI_API.md) for complete API documentation.

## Performance

### Processing Times (Approximate)

On a typical CPU (Intel i7/AMD Ryzen 7):

| Operation | 3-minute song | 5-minute song |
|-----------|---------------|---------------|
| Separation | 5-8 minutes | 8-12 minutes |
| Cleanup | 30 seconds | 1 minute |
| MIDI | 1-2 minutes | 2-3 minutes |
| Video | 2-3 minutes | 3-5 minutes |

With CUDA GPU acceleration:
- Separation: 2-3x faster

### Concurrent Jobs

Default: 2 concurrent jobs maximum

To change, edit `webui/config.py`:
```python
MAX_CONCURRENT_JOBS = 4  # Allow 4 simultaneous operations
```

**Warning:** More concurrent jobs = more RAM usage. Monitor with:
```bash
docker stats DrumToMIDI-midi
```

## Development

### Running Tests

Inside the container:

```bash
# Test API
pytest webui/test_api.py -v

# Test with coverage
pytest webui/test_api.py --cov=webui --cov-report=html
```

### Debug Mode

Enable debug mode for detailed error messages:

```bash
FLASK_ENV=development python -m webui.app
```

Debug features:
- Detailed error traces in browser
- Auto-reload on code changes
- Pretty-printed JSON responses

### Logs

View application logs:

```bash
# Follow web server logs
docker logs -f DrumToMIDI-midi

# View last 100 lines
docker logs --tail 100 DrumToMIDI-midi
```

## Production Deployment (Future)

For production use, consider:

1. **Reverse Proxy**: Use nginx to handle static files
2. **WSGI Server**: Use gunicorn instead of Flask dev server
3. **Database**: Add PostgreSQL for persistent job history
4. **Authentication**: Add user accounts and API keys
5. **HTTPS**: Enable SSL/TLS encryption
6. **Monitoring**: Add application performance monitoring

## Next Steps

- Read [WEBUI_API.md](WEBUI_API.md) for API documentation
- See [STEMS_TO_MIDI_GUIDE.md](STEMS_TO_MIDI_GUIDE.md) for MIDI conversion details
- Check [MIDI_VISUALIZATION_GUIDE.md](MIDI_VISUALIZATION_GUIDE.md) for video rendering info
