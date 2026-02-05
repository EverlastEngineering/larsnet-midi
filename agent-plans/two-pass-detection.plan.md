# Two-Pass Detection System Plan

## Problem Statement

Current energy-based detection has timing issues:
- **Missed peaks**: 149.467s tom hit not detected (RMS smoothing blends nearby peaks)
- **False positives**: 149.641s detection in reverb tail, not at actual peak (90ms delay)
- **Root cause**: Using RMS energy (50-100ms smoothing) for both detection AND timing
- **Time resolution**: Limited to ~150-200ms in reverb-heavy material

Visual inspection (DAW waveform at 1px=2ms) clearly shows all peaks, suggesting raw amplitude with minimal smoothing is superior for timing precision.

## Architecture: Two-Pass Detection

### Pass 1: Coarse Detection (Energy-Based)
**Purpose**: Identify regions of interest where drum events occur

- Input: Raw audio
- Processing: Current RMS energy detection
- Output: List of **regions** (not exact times): `[(start_time, end_time, peak_energy), ...]`
- Function: Semantic detection - "Is this a drum hit vs. noise/silence?"
- Keeps: Noise robustness, spectral analysis capability

### Pass 2: Fine Detection (Amplitude-Based)  
**Purpose**: Find exact timing of stick impact within each region

- Input: Raw audio + regions from Pass 1
- Processing: Amplitude peak detection with minimal smoothing (2-5ms anti-noise)
- Output: Exact onset times for MIDI events
- Function: Timing precision - "When exactly did the stick hit?"
- Provides: DAW-like visual accuracy (1-2ms resolution)

### Benefits

1. **Timing accuracy**: Raw amplitude peaks like DAW waveforms
2. **Robustness**: Energy regions filter out noise and silence
3. **Multi-peak detection**: Can find multiple hits within broad energy region
4. **No false positives**: Amplitude peaks validated within energy regions
5. **Separation of concerns**: Detection (what) vs. timing (when)

## Implementation Plan

### Phase 1: Core Functions

**File**: `stems_to_midi/energy_detection_core.py`

1. **New function**: `find_amplitude_peaks_in_region()`
   - Input: audio, region_start, region_end, sr, min_spacing_ms
   - Use `scipy.ndimage.maximum_filter1d` for 2-5ms peak-hold envelope
   - Apply `scipy.signal.find_peaks` with prominence threshold
   - Return: List of sample indices for amplitude peaks

2. **Modify**: `detect_transient_peaks()`
   - Rename to `detect_energy_regions()` for clarity
   - Change return value from exact times to regions: `(onset_time, peak_time, energy)`
   - Remove amplitude peak snapping (that's now Pass 2)
   - Keep backtracking to find region start

3. **New function**: `refine_timing_with_amplitude_peaks()`
   - Input: energy regions, raw audio, sr
   - For each region, call `find_amplitude_peaks_in_region()`
   - Return: Refined onset times (one or more per region)

### Phase 2: Integration

**File**: `stems_to_midi/energy_detection_core.py`

4. **Modify**: `detect_stereo_transient_peaks()`
   - Call `detect_energy_regions()` for Pass 1
   - Call `refine_timing_with_amplitude_peaks()` for Pass 2
   - Merge results for stereo (if multiple amplitude peaks, pick loudest or split)

### Phase 3: Configuration

**File**: `midiconfig.yaml`

5. **Add parameters**:
   ```yaml
   energy_detection:
     # Pass 1: Energy regions
     threshold_db: 15.0  # existing
     
     # Pass 2: Amplitude refinement
     enable_amplitude_refinement: true  # toggle two-pass mode
     amplitude_smoothing_ms: 3.0  # peak-hold window (2-5ms)
     amplitude_prominence: 0.3  # minimum prominence (fraction of max in region)
     max_peaks_per_region: 3  # limit to prevent explosion in noisy regions
   ```

### Phase 4: Testing

6. **Test cases**:
   - Thunderstruck toms 148.5-149.7s: Should find 149.467s peak, not 149.641s
   - Reverb tail validation: No false positives in sustained reverb
   - Fast rolls: Should split merged energy peaks into individual hits
   - Noise robustness: Energy regions should still filter out background noise

## Success Criteria

- [ ] Detects missing peak at 149.467s
- [ ] Eliminates false detection at 149.641s  
- [ ] Maintains noise robustness from energy detection
- [ ] No increase in false positives from noise
- [ ] All existing tests pass
- [ ] Time resolution matches visual inspection (1-2ms)

## Risks & Mitigation

**Risk 1**: Too many false amplitude peaks within energy regions
- **Mitigation**: Prominence threshold, max_peaks_per_region limit

**Risk 2**: Noise spikes detected as peaks
- **Mitigation**: Energy regions pre-filter noise; 2-5ms smoothing rejects <2ms spikes

**Risk 3**: Breaking existing detection behavior
- **Mitigation**: Make two-pass optional via config flag; test extensively

## Rollback Plan

If two-pass detection causes issues:
1. Add config flag `enable_amplitude_refinement: false` (default true)
2. When disabled, falls back to current single-pass behavior
3. Users can toggle per-stem if needed

## Future Enhancements

- Adaptive amplitude prominence based on region energy
- Multi-scale detection (different smoothing windows)
- Machine learning to classify real peaks vs. reverb bumps
