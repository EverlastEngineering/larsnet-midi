# Web UI Settings

## Overview

The Web UI provides collapsible settings panels for each operation, allowing users to configure parameters that match the command-line interface. Settings are automatically persisted to browser localStorage and restored on page reload.

## User Interface

### Progressive Disclosure

Each operation card has a **[Settings]** button beneath it with a chevron icon:
- Click to expand settings panel
- Click again to collapse
- Opening one panel automatically closes others
- Panel appears full-width beneath the operation buttons

### Settings Persistence

All settings are automatically saved to browser localStorage:
- Changes save immediately when you adjust any value
- Settings persist across browser sessions
- Each user's browser maintains their own preferences
- No server-side storage needed

## Operation Settings

### Separate (Stem Separation)

Splits drum track into individual components (kick, snare, toms, hi-hat, cymbals).

**Device**
- Options: `CPU`, `CUDA (GPU)`, `MPS (Mac GPU)`
- Default: Auto-detected
- Choose based on your hardware for optimal performance

### Clean (Sidechain Compression)

Reduces bleed between stems using sidechain compression (snare triggers ducking on kick).

**Threshold (dB)**
- Range: `-60` to `0`
- Default: `-30`
- Trigger level for compression
- Lower values = more sensitive (compresses more often)

**Ratio**
- Range: `1` to `20`
- Default: `10`
- Compression ratio (10:1 means 10dB input → 1dB output)
- Higher = more aggressive compression

**Attack (ms)**
- Range: `0.1` to `10`
- Default: `1`
- How quickly compression engages
- Faster = more aggressive response

**Release (ms)**
- Range: `10` to `500`
- Default: `100`
- How quickly compression releases
- Faster = more pumping effect

### MIDI (Stem to MIDI Conversion)

Converts drum stems to MIDI notes with velocity.

**Onset Threshold**
- Range: `0` to `1` (step 0.01)
- Default: `0.3`
- Detection sensitivity
- Lower = more sensitive (catches quieter hits)
- Same as `--threshold` flag in CLI

**Onset Delta**
- Range: `0` to `0.1` (step 0.001)
- Default: `0.01`
- Peak picking sensitivity
- Lower = more sensitive to variations
- Same as `--delta` flag in CLI

**Min/Max Velocity**
- Range: `0` to `127`
- Default: `40` (min), `127` (max)
- MIDI velocity range for detected hits
- Maps detected amplitude to MIDI velocity
- Same as `--min-vel` and `--max-vel` in CLI

**Tempo (BPM)**
- Range: `60` to `200`
- Default: Empty (auto-detect)
- Sets MIDI file tempo
- Leave empty for automatic detection
- Same as `--tempo` flag in CLI

**Detect Open Hi-Hat**
- Type: Toggle
- Default: `Off`
- Distinguishes open vs closed hi-hat
- Uses sustain analysis
- Same as `--detect-hihat-open` flag in CLI

**Advanced Per-Stem Settings** *(Coming Soon)*
- Button placeholder for Phase 3B
- Will expose per-stem configuration from `midiconfig.yaml`:
  - Spectral filtering (geomean thresholds)
  - Timing offsets
  - Statistical outlier detection
  - Per-stem onset overrides

### Video (MIDI Visualization)

Renders piano roll visualization video from MIDI file.

**Frame Rate (FPS)**
- Options: `30 FPS`, `60 FPS`, `120 FPS`
- Default: `60 FPS`
- Higher = smoother animation (larger file size)

**Resolution**
- Options: `1080p`, `1440p`, `4K`
- Default: `1080p (1920×1080)`
- Higher = better quality (slower rendering, larger file)

## How It Works

### Frontend (JavaScript)

**SettingsManager** (`settings.js`):
```javascript
// Automatically initialized on page load
window.settingsManager = new SettingsManager();

// Get settings for an operation
const settings = settingsManager.getSettingsForOperation('midi');
// Returns: { onset_threshold: 0.3, onset_delta: 0.01, ... }

// Reset to defaults
settingsManager.resetOperation('separate');
```

**Auto-binding**:
- All inputs with `id="setting-*"` are automatically tracked
- Changes save immediately to localStorage
- Values are type-aware (checkbox → bool, number → float/int)

### Integration with Operations

When you click an operation button (Separate, Clean, MIDI, Video), the operation reads current settings:

```javascript
// operations.js
async function startMidi() {
    const settings = settingsManager.getSettingsForOperation('midi');
    
    const result = await api.stemsToMidi(currentProject.number, {
        onset_threshold: settings.onset_threshold,
        onset_delta: settings.onset_delta,
        min_velocity: settings.min_velocity,
        // ... other settings
    });
}
```

## Typical Workflows

### Quick Start (Default Settings)
1. Select project
2. Click **Separate** (uses defaults: CPU, no Wiener, no EQ)
3. Click **MIDI** (uses defaults: 0.3 threshold)
4. Click **Video** (uses defaults: 60 FPS, 1080p)

### Custom Workflow
1. Select project
2. Click **[Settings]** under Separate button
3. Change Device to **CUDA** for GPU acceleration
4. Enable **Apply EQ cleanup**
5. Set **Wiener** to **2.0** for aggressive filtering
6. Click **Separate**
7. Click **[Settings]** under MIDI button
8. Lower **Onset Threshold** to **0.2** for more sensitive detection
9. Enable **Detect Open Hi-Hat**
10. Click **MIDI**

### Settings Reuse
- Your settings persist across sessions
- Works per-browser (other users see their own settings)
- No need to reconfigure each time

## Keyboard Shortcuts

*(Coming in Phase 4)*
- `S` - Toggle Separate settings
- `C` - Toggle Clean settings
- `M` - Toggle MIDI settings
- `V` - Toggle Video settings
- `Esc` - Close settings panel

## Advanced Configuration

### Per-Project Configuration *(Phase 3B - Coming Soon)*

The advanced settings button in MIDI panel will allow editing project-specific YAML files:
- `midiconfig.yaml` - Per-stem detection and filtering
- `eq.yaml` - Per-stem frequency filtering
- `config.yaml` - Global audio processing parameters

Changes will be saved to the project directory and persist for that project only.

### Browser Storage

Settings are stored in `localStorage` under key `DrumToMIDI_settings`:

```javascript
// View current settings
console.log(localStorage.getItem('DrumToMIDI_settings'));

// Clear settings (reset to defaults)
localStorage.removeItem('DrumToMIDI_settings');
location.reload();
```

## Troubleshooting

**Settings not saving?**
- Check browser console for errors
- Ensure localStorage is enabled (some privacy modes disable it)
- Try incognito/private mode to test with clean slate

**Settings different from CLI results?**
- Web UI settings map 1:1 to CLI flags
- Check console log when operation starts to see exact parameters sent
- Verify you're comparing same parameter values

**Can't see settings panel?**
- Click **[Settings]** button beneath operation card
- If hidden, scroll down - panel appears below operation buttons
- Try resizing browser window

**Want to reset everything?**
```javascript
// In browser console:
localStorage.removeItem('DrumToMIDI_settings');
location.reload();
```

## Future Enhancements

Phase 3B will add:
- [ ] Advanced YAML editor for per-stem configuration
- [ ] Reset to defaults buttons per section
- [ ] Save settings as presets (beginner/balanced/aggressive)
- [ ] Import/export settings profiles
- [ ] Per-project configuration overrides
- [ ] Visual feedback for changed values (highlight non-defaults)
- [ ] Validation with helpful error messages
- [ ] Tooltips with audio engineering context

Phase 4 will add:
- [ ] Keyboard shortcuts
- [ ] Settings search/filter
- [ ] Quick presets dropdown
- [ ] A/B comparison mode
- [ ] Settings history/undo
