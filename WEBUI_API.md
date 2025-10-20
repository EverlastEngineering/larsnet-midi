# LarsNet Web UI API Documentation

REST API for the LarsNet MIDI web interface. All operations run asynchronously via a job queue with real-time status updates through Server-Sent Events.

## Base URL

```
http://localhost:49152/api
```

## Response Format

All API responses follow this format:

**Success Response:**
```json
{
  "data_field": "value",
  ...
}
```

**Error Response:**
```json
{
  "error": "Error category",
  "message": "Detailed error message"
}
```

## Authentication

Currently no authentication required. Future versions may add API keys or OAuth.

---

## Projects

### List All Projects

Get all projects in the `user_files/` directory with status information.

**Endpoint:** `GET /api/projects`

**Response:** `200 OK`
```json
{
  "projects": [
    {
      "number": 1,
      "name": "The Fate Of Ophelia",
      "path": "/app/user_files/1 - The Fate Of Ophelia",
      "created": "2025-10-19T10:30:00",
      "metadata": {
        "created_at": "2025-10-19T10:30:00",
        "audio_file": "The Fate Of Ophelia.wav"
      },
      "has_stems": true,
      "has_cleaned": true,
      "has_midi": true,
      "has_video": false
    }
  ]
}
```

**Status Indicators:**
- `has_stems`: Separated drum stems exist
- `has_cleaned`: Cleaned stems (after sidechain) exist
- `has_midi`: MIDI file generated
- `has_video`: Visualization video rendered

---

### Get Project Details

Get detailed information about a specific project including all files.

**Endpoint:** `GET /api/projects/:project_number`

**Parameters:**
- `project_number` (path, integer): Project number

**Response:** `200 OK`
```json
{
  "project": {
    "number": 1,
    "name": "The Fate Of Ophelia",
    "path": "/app/user_files/1 - The Fate Of Ophelia",
    "created": "2025-10-19T10:30:00",
    "metadata": {...},
    "files": {
      "audio": ["The Fate Of Ophelia.wav"],
      "stems": ["kick.wav", "snare.wav", "hihat.wav", "cymbals.wav", "toms.wav"],
      "cleaned": ["kick_cleaned.wav", "snare_cleaned.wav"],
      "midi": ["The Fate Of Ophelia.mid"],
      "video": []
    }
  }
}
```

**Error Responses:**
- `404 Not Found`: Project doesn't exist

---

### Get Project Configuration

Get contents of a project's configuration file (config.yaml, midiconfig.yaml, or eq.yaml).

**Endpoint:** `GET /api/projects/:project_number/config/:config_name`

**Parameters:**
- `project_number` (path, integer): Project number
- `config_name` (path, string): One of: `config.yaml`, `midiconfig.yaml`, `eq.yaml`

**Response:** `200 OK`
```json
{
  "config_name": "midiconfig.yaml",
  "content": "# MIDI Configuration\naudio:\n  force_mono: true\n..."
}
```

**Error Responses:**
- `400 Bad Request`: Invalid config name
- `404 Not Found`: Project or config file not found

---

## File Upload

### Upload Audio File

Upload a new audio file and automatically create a project.

**Endpoint:** `POST /api/upload`

**Content-Type:** `multipart/form-data`

**Body:**
- `file` (file): Audio file (.wav, .mp3, .flac, .aiff, .aif)

**Example (curl):**
```bash
curl -X POST http://localhost:49152/api/upload \
  -F "file=@/path/to/drums.wav"
```

**Response:** `201 Created`
```json
{
  "message": "File uploaded successfully",
  "project": {
    "number": 2,
    "name": "drums",
    "path": "/app/user_files/2 - drums"
  }
}
```

**Error Responses:**
- `400 Bad Request`: No file, invalid format, or file size exceeds limit (500MB)
- `500 Internal Server Error`: Upload or project creation failed

---

## Operations

All operations are asynchronous and return a job ID immediately. Use the job status endpoints to track progress.

### Separate Stems

Separate drums into individual stems (kick, snare, hi-hat, cymbals, toms) using LarsNet deep learning models.

**Endpoint:** `POST /api/separate`

**Content-Type:** `application/json`

**Body:**
```json
{
  "project_number": 1,
  "device": "cpu",      // optional: "cpu" or "cuda" (default: "cpu")
  "wiener": 2.0,        // optional: Wiener filter exponent (default: null/disabled)
  "eq": false           // optional: Apply EQ cleanup (default: false)
}
```

**Response:** `202 Accepted`
```json
{
  "message": "Separation job started",
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Parameters:**
- `device`: Processing device
  - `"cpu"`: Use CPU (slower but always available)
  - `"cuda"`: Use GPU acceleration (requires CUDA)
- `wiener`: Wiener filter exponent for noise reduction (typically 1.0-3.0). Higher values = more aggressive filtering
- `eq`: Apply frequency-based cleanup to reduce bleed between stems

**Error Responses:**
- `400 Bad Request`: Invalid parameters
- `404 Not Found`: Project not found
- `500 Internal Server Error`: Failed to start job

---

### Cleanup Stems (Sidechain Compression)

Apply sidechain compression to reduce bleed between stems (typically snare bleed in kick).

**Endpoint:** `POST /api/cleanup`

**Body:**
```json
{
  "project_number": 1,
  "threshold_db": -30.0,   // optional: Trigger threshold in dB (default: -30.0)
  "ratio": 10.0,           // optional: Compression ratio (default: 10.0)
  "attack_ms": 1.0,        // optional: Attack time in milliseconds (default: 1.0)
  "release_ms": 100.0      // optional: Release time in milliseconds (default: 100.0)
}
```

**Response:** `202 Accepted`
```json
{
  "message": "Cleanup job started",
  "job_id": "550e8400-e29b-41d4-a716-446655440001"
}
```

**Parameters:**
- `threshold_db`: Volume threshold to trigger compression. Lower = more sensitive
- `ratio`: How much to reduce volume (10:1 = reduce by 90%)
- `attack_ms`: How quickly compression starts (1ms = very fast)
- `release_ms`: How quickly compression releases (100ms = smooth)

---

### Convert Stems to MIDI

Detect drum hits in stems and convert to MIDI notes.

**Endpoint:** `POST /api/stems-to-midi`

**Body:**
```json
{
  "project_number": 1,
  "onset_threshold": 0.3,      // optional: Detection sensitivity (default: 0.3)
  "onset_delta": 0.01,         // optional: Peak picking sensitivity (default: 0.01)
  "onset_wait": 3,             // optional: Min frames between peaks (default: 3)
  "hop_length": 512,           // optional: Samples between frames (default: 512)
  "min_velocity": 80,          // optional: Minimum MIDI velocity (default: 80)
  "max_velocity": 110,         // optional: Maximum MIDI velocity (default: 110)
  "tempo": 120.0,              // optional: Tempo in BPM (default: detected)
  "detect_hihat_open": false   // optional: Detect open hi-hat (default: false)
}
```

**Response:** `202 Accepted`
```json
{
  "message": "MIDI conversion job started",
  "job_id": "550e8400-e29b-41d4-a716-446655440002"
}
```

**Key Parameters:**
- `onset_threshold`: Lower = detects quieter hits (0.1-0.5 typical)
- `onset_delta`: Lower = more sensitive to small peaks
- `onset_wait`: Prevents detecting same hit multiple times (3 frames ≈ 35ms)
- `min_velocity`/`max_velocity`: MIDI velocity range (0-127)

---

### Render MIDI to Video

Create Rock Band-style falling notes visualization from MIDI file.

**Endpoint:** `POST /api/render-video`

**Body:**
```json
{
  "project_number": 1,
  "fps": 60,          // optional: Frame rate (default: 60)
  "width": 1920,      // optional: Video width (default: 1920)
  "height": 1080      // optional: Video height (default: 1080)
}
```

**Response:** `202 Accepted`
```json
{
  "message": "Video rendering job started",
  "job_id": "550e8400-e29b-41d4-a716-446655440003"
}
```

**Parameters:**
- `fps`: Higher = smoother but larger file (30, 60, or 120)
- `width`/`height`: Resolution (1920x1080, 2560x1440, 3840x2160)

---

## Job Status

### Get All Jobs

List all jobs in the queue (queued, running, completed, failed).

**Endpoint:** `GET /api/jobs`

**Response:** `200 OK`
```json
{
  "jobs": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "operation": "separate",
      "project_id": 1,
      "status": "completed",
      "progress": 100,
      "logs": [
        {
          "timestamp": "2025-10-19T12:00:00",
          "level": "info",
          "message": "Job queued: separate"
        },
        {
          "timestamp": "2025-10-19T12:00:05",
          "level": "info",
          "message": "Job started: separate"
        },
        {
          "timestamp": "2025-10-19T12:05:30",
          "level": "info",
          "message": "Job completed successfully"
        }
      ],
      "result": {
        "project_number": 1,
        "stems_created": true
      },
      "error": null,
      "created_at": "2025-10-19T12:00:00",
      "started_at": "2025-10-19T12:00:05",
      "completed_at": "2025-10-19T12:05:30"
    }
  ]
}
```

**Job Status Values:**
- `queued`: Waiting to start
- `running`: Currently executing
- `completed`: Finished successfully
- `failed`: Error occurred
- `cancelled`: Cancelled by user

---

### Get Job Status

Get detailed status of a specific job.

**Endpoint:** `GET /api/jobs/:job_id`

**Parameters:**
- `job_id` (path, string): Job UUID

**Response:** `200 OK`
```json
{
  "job": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "operation": "stems-to-midi",
    "project_id": 1,
    "status": "running",
    "progress": 65,
    "logs": [...],
    "result": null,
    "error": null,
    "created_at": "2025-10-19T12:00:00",
    "started_at": "2025-10-19T12:00:05",
    "completed_at": null
  }
}
```

**Error Responses:**
- `404 Not Found`: Job doesn't exist

---

### Stream Job Status (SSE)

Real-time job status updates using Server-Sent Events. Connect once and receive updates as job progresses.

**Endpoint:** `GET /api/jobs/:job_id/stream`

**Parameters:**
- `job_id` (path, string): Job UUID

**Response:** `200 OK` (SSE stream)

**Event Types:**

**`job_update`** - Periodic progress updates:
```
event: job_update
data: {"id": "...", "status": "running", "progress": 50, "logs": [...]}
```

**`job_complete`** - Job finished successfully:
```
event: job_complete
data: {"id": "...", "status": "completed", "result": {...}}
```

**`job_error`** - Job failed:
```
event: job_error
data: {"id": "...", "status": "failed", "error": "Error message"}
```

**Example (JavaScript):**
```javascript
const eventSource = new EventSource('http://localhost:49152/api/jobs/550e8400-e29b-41d4-a716-446655440000/stream');

eventSource.addEventListener('job_update', (e) => {
  const job = JSON.parse(e.data);
  console.log(`Progress: ${job.progress}%`);
});

eventSource.addEventListener('job_complete', (e) => {
  const job = JSON.parse(e.data);
  console.log('Job finished!', job.result);
  eventSource.close();
});

eventSource.addEventListener('job_error', (e) => {
  const job = JSON.parse(e.data);
  console.error('Job failed:', job.error);
  eventSource.close();
});
```

---

### Cancel Job

Cancel a queued or running job.

**Endpoint:** `POST /api/jobs/:job_id/cancel`

**Parameters:**
- `job_id` (path, string): Job UUID

**Response:** `200 OK`
```json
{
  "message": "Job cancelled",
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Note:** If the job is already running, it will continue until the current operation completes, but will be marked as cancelled.

**Error Responses:**
- `404 Not Found`: Job doesn't exist
- `400 Bad Request`: Job already completed/failed and cannot be cancelled

---

### Get Project Jobs

Get all jobs associated with a specific project.

**Endpoint:** `GET /api/projects/:project_number/jobs`

**Parameters:**
- `project_number` (path, integer): Project number

**Response:** `200 OK`
```json
{
  "jobs": [
    {
      "id": "...",
      "operation": "separate",
      "project_id": 1,
      "status": "completed",
      ...
    },
    {
      "id": "...",
      "operation": "stems-to-midi",
      "project_id": 1,
      "status": "running",
      ...
    }
  ]
}
```

---

## Health Check

### Check API Health

Simple endpoint to verify API is running.

**Endpoint:** `GET /health`

**Response:** `200 OK`
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

---

## Error Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created (upload successful) |
| 202 | Accepted (job queued) |
| 400 | Bad Request (invalid parameters) |
| 404 | Not Found (resource doesn't exist) |
| 500 | Internal Server Error |

---

## Rate Limiting

Currently no rate limiting. Future versions may add:
- Max 10 uploads per hour
- Max 5 concurrent jobs per user
- Max 100 API requests per minute

---

## Examples

### Complete Workflow

**1. Upload audio file:**
```bash
curl -X POST http://localhost:49152/api/upload \
  -F "file=@drums.wav"
# Returns: {"project": {"number": 1, ...}}
```

**2. Separate stems:**
```bash
curl -X POST http://localhost:49152/api/separate \
  -H "Content-Type: application/json" \
  -d '{"project_number": 1, "device": "cpu"}'
# Returns: {"job_id": "uuid-1"}
```

**3. Check job status:**
```bash
curl http://localhost:49152/api/jobs/uuid-1
# Returns: {"job": {"status": "completed", ...}}
```

**4. Convert to MIDI:**
```bash
curl -X POST http://localhost:49152/api/stems-to-midi \
  -H "Content-Type: application/json" \
  -d '{"project_number": 1}'
# Returns: {"job_id": "uuid-2"}
```

**5. Render video:**
```bash
curl -X POST http://localhost:49152/api/render-video \
  -H "Content-Type: application/json" \
  -d '{"project_number": 1, "fps": 60}'
# Returns: {"job_id": "uuid-3"}
```

---

## WebSocket Alternative (Future)

Current implementation uses SSE for server→client updates. Future versions may add WebSocket support for bidirectional communication and multiple stream subscriptions.
