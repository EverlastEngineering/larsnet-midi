# Web UI Development Plan

## Overview
Build a modern web-based interface for LarsNet MIDI with a left navigation for projects, drag-and-drop audio upload, and controls for all CLI operations.

## Architecture Approach

### Technology Stack
- **Backend**: Flask (Python) - lightweight, integrates seamlessly with existing Python codebase
- **Frontend**: Modern vanilla JavaScript with Tailwind CSS for styling
- **State Management**: Server-side session + client-side local state
- **File Upload**: Drag-and-drop with progress tracking
- **Real-time Updates**: Server-Sent Events (SSE) for long-running operations

### Design Principles
- **Functional Core, Imperative Shell**: Web API handlers orchestrate existing project_manager functions
- **Minimal Dependencies**: Leverage existing Python code, avoid heavy frameworks
- **Progressive Enhancement**: Start with core features, expand to advanced config editing
- **Responsive Design**: Mobile-friendly interface

## Phases

### Phase 1: Backend API Foundation
**Goal**: Create Flask API exposing existing CLI functionality with full documentation and error handling

**Tasks**:
1. Create `webui/` directory structure
2. Set up Flask app with blueprints:
   - `/api/projects` - List, create, get project details
   - `/api/upload` - Handle file uploads
   - `/api/separate` - Trigger stem separation
   - `/api/cleanup` - Trigger sidechain cleanup
   - `/api/stems-to-midi` - Convert stems to MIDI
   - `/api/render-video` - Render visualization
   - `/api/status` - Job status and logs (SSE)
3. Create job queue system for long-running operations
4. Add comprehensive error handling and user feedback
5. Write API documentation (WEBUI_API.md)
6. Write unit tests for API endpoints
7. Document Docker setup for web UI

**Success Criteria**:
- All API endpoints return correct responses
- Jobs execute asynchronously with status tracking
- All errors return helpful messages
- API documentation complete and accurate
- Tests pass with >80% coverage on new code
- No modifications to existing core modules

**Estimated Time**: 5-7 hours

### Phase 2: Frontend UI Components
**Goal**: Build responsive UI with project navigation and controls, with error handling and user feedback

**Tasks**:
1. Create HTML template with Tailwind CSS
2. Build left sidebar with project list:
   - Project number + name
   - Status indicators (stems, midi, video available)
   - Active project highlighting
3. Build main content area:
   - Drag-and-drop upload zone
   - Project details card
   - Operation buttons (Separate, Clean, MIDI, Video)
   - Progress indicators
4. Build log/console viewer
5. Add file download links for outputs
6. Add error handling and user feedback (toasts/alerts)
7. Add loading states and animations
8. Document UI components and user workflows

**Success Criteria**:
- UI is responsive on mobile and desktop
- All projects display correctly in sidebar
- Upload works via drag-and-drop and click
- Operation buttons trigger API calls correctly
- Progress updates in real-time
- All errors display helpful messages to user
- UI documentation complete

**Estimated Time**: 4-5 hours

### Phase 3: Configuration UI (Basic + Advanced)
**Goal**: Expose all configuration options through graphical interface

**Tasks - Basic Parameters (collapsible panels)**:
1. **Separate Operation Panel**:
   - Device (CPU/CUDA) dropdown
   - Wiener filter toggle + exponent slider
   - EQ cleanup toggle
   - Tooltips explaining each option
2. **Cleanup Operation Panel**:
   - Threshold slider (dB)
   - Ratio slider
   - Attack time (ms)
   - Release time (ms)
   - Tooltips with audio engineering context
3. **MIDI Operation Panel**:
   - Onset threshold slider
   - Onset delta slider
   - Onset wait (frames)
   - Hop length
   - Min/max velocity sliders
   - Tempo input (BPM)
   - Hi-hat open detection toggle
   - Tooltips explaining detection parameters
4. **Video Operation Panel**:
   - FPS dropdown (30/60/120)
   - Resolution dropdown (1080p/1440p/4K)
   - Note height slider
   - Color scheme selector

**Tasks - Advanced Parameters (collapsible "Advanced" section per operation)**:
5. **Advanced MIDI Config** (reads from midiconfig.yaml):
   - Per-stem onset overrides (kick, snare, hihat, etc.)
   - Spectral filtering parameters (geomean thresholds, frequency ranges)
   - Statistical filter settings
   - Timing offsets
   - All parameters as form controls with validation
6. **Advanced EQ Config** (reads from eq.yaml):
   - Per-stem highpass/lowpass frequency controls
   - Visual frequency response curves
7. **Advanced Separation Config** (reads from config.yaml):
   - Audio processing parameters (segment, shift, sample rate)
   - Model paths (read-only display)
   
**Tasks - Infrastructure**:
8. Store user preferences in browser localStorage
9. Add "Reset to Defaults" button per section
10. Add "Save to Project" button to persist advanced changes
11. Document all parameters in tooltips and help text
12. Add form validation and error messages
13. Document configuration system in WEBUI_CONFIG_GUIDE.md

**Success Criteria**:
- All CLI parameters accessible in basic panels
- All YAML config options accessible in advanced panels
- Parameters persist across sessions (localStorage)
- Project-level config changes save to project YAML files
- Tooltips explain every parameter clearly
- Defaults match CLI and YAML behavior
- Form validation prevents invalid values
- Configuration documentation complete

**Estimated Time**: 6-8 hours

### Phase 4: Testing & Polish
**Goal**: Production-ready, tested, fully documented web UI

**Tasks**:
1. Write comprehensive frontend tests
2. Test all workflows end-to-end
3. Add keyboard shortcuts (document in UI)
4. Optimize performance (lazy loading, caching)
5. Add accessibility features (ARIA labels, keyboard nav)
6. Write WEBUI_GUIDE.md (user guide)
7. Update README.md with web UI quick start
8. Add screenshots and demo video to documentation
9. Create usage examples and common workflows guide

**Success Criteria**:
- No console errors in browser
- All tests passing (backend + frontend)
- Keyboard shortcuts documented and working
- Accessible via keyboard navigation
- Complete user documentation with screenshots
- Docker setup works out-of-box
- README includes web UI quick start

**Estimated Time**: 4-5 hours

## Directory Structure
```
webui/
├── __init__.py
├── app.py                 # Flask application entry point
├── api/
│   ├── __init__.py
│   ├── projects.py        # Project management endpoints
│   ├── operations.py      # Separate, cleanup, midi, video
│   ├── upload.py          # File upload handling
│   └── jobs.py            # Job queue and SSE status
├── static/
│   ├── css/
│   │   └── tailwind.min.css
│   ├── js/
│   │   ├── app.js         # Main application logic
│   │   ├── api.js         # API client wrapper
│   │   ├── projects.js    # Project list component
│   │   └── operations.js  # Operation controls
│   └── icons/
├── templates/
│   └── index.html         # Single-page app template
├── jobs.py                # Job queue implementation
├── config.py              # Flask configuration
└── test_api.py            # API tests
```

## Risks & Mitigations

### Risk: Long-running operations block UI
**Mitigation**: Use threading/async with job queue and SSE for status updates

### Risk: Large file uploads timeout
**Mitigation**: Chunked uploads with progress tracking, configurable timeouts

### Risk: Concurrent operations on same project
**Mitigation**: Project-level locking, prevent concurrent ops on same project

### Risk: Docker networking complexity
**Mitigation**: Use host networking or properly expose ports in docker-compose

### Risk: Breaking existing CLI tools
**Mitigation**: Web UI only orchestrates existing functions, no core modifications

## Success Metrics
- All CLI operations accessible via UI
- <2 second page load time
- Real-time progress updates (<500ms latency)
- Zero modifications to core modules (separate.py, stems_to_midi.py, etc.)
- 100% of existing tests still pass
- Web UI tests achieve >80% coverage

## Development Philosophy

### Documentation First
- Document APIs, components, and workflows as they're built
- Include inline comments explaining complex logic
- Write user-facing docs alongside features
- Good docs enable seamless AI-driven development handoffs

### Error Handling First
- Add comprehensive error handling from the start
- Provide helpful user feedback for all error states
- Log errors properly for debugging
- Never leave user confused about what went wrong

### Test As You Go
- Write tests alongside implementation
- Validate behavior before moving to next phase
- Maintain >80% coverage on new code

## Future Enhancements (Post-Launch)
- WebSocket for bidirectional communication
- Multi-project batch operations
- Audio preview player in browser
- MIDI preview/editor in browser
- Admin panel for model management
- User authentication and multi-user support
- Cloud storage integration
