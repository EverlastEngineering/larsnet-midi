"""
Centralized Settings Schema

Defines all application settings in one place with:
- Type information
- Default values
- Validation rules (min/max, allowed values)
- UI metadata (labels, descriptions, control hints)
- Grouping and organization

This is the single source of truth for all settings, used by:
- Python backend (validation, defaults)
- WebUI (form generation, validation)
- CLI (argument parsing, defaults)
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class SettingType(str, Enum):
    """Setting data types"""
    BOOL = 'bool'
    INT = 'int'
    FLOAT = 'float'
    STRING = 'string'
    PATH = 'path'
    CHOICE = 'choice'  # Dropdown/select from allowed_values


class UIControl(str, Enum):
    """UI control type hints"""
    CHECKBOX = 'checkbox'
    NUMBER = 'number'
    TEXT = 'text'
    SLIDER = 'slider'
    SELECT = 'select'
    FILE = 'file'


class SettingCategory(str, Enum):
    """Setting categories for grouping"""
    AUDIO = 'audio'
    ONSET_DETECTION = 'onset_detection'
    MIDI_OUTPUT = 'midi_output'
    KICK = 'kick'
    SNARE = 'snare'
    TOMS = 'toms'
    HIHAT = 'hihat'
    CYMBALS = 'cymbals'
    CLUSTERING = 'clustering'
    THRESHOLD_OPTIMIZATION = 'threshold_optimization'
    DEBUG = 'debug'
    LEARNING = 'learning_mode'
    SEPARATION = 'separation'
    CLEANUP = 'cleanup'
    VIDEO = 'video'


@dataclass
class SettingDefinition:
    """
    Complete definition of a single setting.
    
    Attributes:
        key: Setting identifier (snake_case)
        type: Data type
        default: Default value
        label: Human-readable label
        description: Help text / tooltip
        category: Grouping category
        ui_control: Preferred UI control type
        min_value: Minimum valid value (for numeric types)
        max_value: Maximum valid value (for numeric types)
        step: Step size for numeric inputs
        allowed_values: List of valid values (for CHOICE type)
        nullable: Whether null/None is acceptable (uses global default)
        unit: Unit label (e.g., 'Hz', 'ms', 'dB')
        advanced: Whether this is an advanced setting (collapsed by default)
        readonly: Whether this setting is read-only in UI
        yaml_path: Path in YAML file (e.g., ['kick', 'midi_note'])
        cli_flag: Command-line flag (e.g., '--onset-threshold')
    """
    key: str
    type: SettingType
    default: Any
    label: str
    description: str
    category: SettingCategory
    ui_control: UIControl
    
    # Validation
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    nullable: bool = False
    
    # Metadata
    unit: Optional[str] = None
    advanced: bool = False
    readonly: bool = False
    
    # Integration
    yaml_path: Optional[List[str]] = None
    cli_flag: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            k: v.value if isinstance(v, Enum) else v
            for k, v in asdict(self).items()
            if v is not None
        }
    
    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validate a value against this setting's rules.
        
        Returns:
            (is_valid, error_message)
        """
        if value is None:
            if not self.nullable:
                return False, "Value cannot be null"
            return True, None
        
        # Type validation
        if self.type == SettingType.BOOL:
            if not isinstance(value, bool):
                return False, "Value must be a boolean"
        
        elif self.type in (SettingType.INT, SettingType.FLOAT):
            if not isinstance(value, (int, float)):
                return False, "Value must be a number"
            
            if self.min_value is not None and value < self.min_value:
                return False, f"Value must be >= {self.min_value}"
            
            if self.max_value is not None and value > self.max_value:
                return False, f"Value must be <= {self.max_value}"
        
        elif self.type == SettingType.CHOICE:
            if self.allowed_values and value not in self.allowed_values:
                return False, f"Value must be one of: {', '.join(map(str, self.allowed_values))}"
        
        elif self.type == SettingType.STRING:
            if not isinstance(value, str):
                return False, "Value must be a string"
        
        return True, None


# ============================================================================
# SETTINGS REGISTRY
# ============================================================================

# All settings defined here in one place
SETTINGS_REGISTRY: List[SettingDefinition] = [
    
    # =========================
    # Audio Processing Settings
    # =========================
    
    SettingDefinition(
        key='force_mono',
        type=SettingType.BOOL,
        default=True,
        label='Force Mono',
        description='Convert stereo files to mono before analysis (recommended). Uses average of left/right channels.',
        category=SettingCategory.AUDIO,
        ui_control=UIControl.CHECKBOX,
        yaml_path=['audio', 'force_mono'],
    ),
    
    SettingDefinition(
        key='silence_threshold',
        type=SettingType.FLOAT,
        default=0.001,
        label='Silence Threshold',
        description='Amplitude threshold for silence detection',
        category=SettingCategory.AUDIO,
        ui_control=UIControl.NUMBER,
        min_value=0.0,
        max_value=1.0,
        step=0.0001,
        unit='amplitude (-60dB)',
        advanced=True,
        yaml_path=['audio', 'silence_threshold'],
    ),
    
    SettingDefinition(
        key='min_segment_length',
        type=SettingType.INT,
        default=512,
        label='Min Segment Length',
        description='Minimum audio segment length for analysis',
        category=SettingCategory.AUDIO,
        ui_control=UIControl.NUMBER,
        min_value=128,
        max_value=2048,
        step=128,
        unit='samples',
        advanced=True,
        yaml_path=['audio', 'min_segment_length'],
    ),
    
    SettingDefinition(
        key='peak_window_sec',
        type=SettingType.FLOAT,
        default=0.10,
        label='Peak Window',
        description='Window size for peak detection',
        category=SettingCategory.AUDIO,
        ui_control=UIControl.NUMBER,
        min_value=0.01,
        max_value=1.0,
        step=0.01,
        unit='seconds',
        advanced=True,
        yaml_path=['audio', 'peak_window_sec'],
    ),
    
    SettingDefinition(
        key='sustain_window_sec',
        type=SettingType.FLOAT,
        default=0.2,
        label='Sustain Window',
        description='Window size for sustain analysis',
        category=SettingCategory.AUDIO,
        ui_control=UIControl.NUMBER,
        min_value=0.01,
        max_value=2.0,
        step=0.1,
        unit='seconds',
        advanced=True,
        yaml_path=['audio', 'sustain_window_sec'],
    ),
    
    SettingDefinition(
        key='envelope_threshold',
        type=SettingType.FLOAT,
        default=0.1,
        label='Envelope Threshold',
        description='Threshold for sustain detection (fraction of peak)',
        category=SettingCategory.AUDIO,
        ui_control=UIControl.NUMBER,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        advanced=True,
        yaml_path=['audio', 'envelope_threshold'],
    ),
    
    SettingDefinition(
        key='envelope_smooth_kernel',
        type=SettingType.INT,
        default=51,
        label='Envelope Smooth Kernel',
        description='Median filter kernel size for envelope smoothing (must be odd)',
        category=SettingCategory.AUDIO,
        ui_control=UIControl.NUMBER,
        min_value=3,
        max_value=101,
        step=2,
        unit='samples',
        advanced=True,
        yaml_path=['audio', 'envelope_smooth_kernel'],
    ),
    
    SettingDefinition(
        key='default_note_duration',
        type=SettingType.FLOAT,
        default=0.1,
        label='Default Note Duration',
        description='Default duration for last note',
        category=SettingCategory.AUDIO,
        ui_control=UIControl.NUMBER,
        min_value=0.01,
        max_value=2.0,
        step=0.01,
        unit='seconds',
        advanced=True,
        yaml_path=['audio', 'default_note_duration'],
    ),
    
    SettingDefinition(
        key='very_short_duration',
        type=SettingType.FLOAT,
        default=0.01,
        label='Very Short Duration',
        description='Very short note duration for MIDI output',
        category=SettingCategory.AUDIO,
        ui_control=UIControl.NUMBER,
        min_value=0.001,
        max_value=0.1,
        step=0.001,
        unit='seconds',
        advanced=True,
        yaml_path=['audio', 'very_short_duration'],
    ),
    
    # ================================
    # Global Onset Detection Settings
    # ================================

    SettingDefinition(
        key='onset_threshold',
        type=SettingType.FLOAT,
        default=0.3,
        label='Onset Threshold',
        description='Detection sensitivity (lower = more sensitive, catches quieter hits)',
        category=SettingCategory.ONSET_DETECTION,
        ui_control=UIControl.SLIDER,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        yaml_path=['onset_detection', 'threshold'],
        cli_flag='--onset-threshold',
    ),

    SettingDefinition(
        key='onset_delta',
        type=SettingType.FLOAT,
        default=0.01,
        label='Onset Delta',
        description='Peak picking sensitivity (lower = more sensitive to variations)',
        category=SettingCategory.ONSET_DETECTION,
        ui_control=UIControl.NUMBER,
        min_value=0.0,
        max_value=0.1,
        step=0.001,
        yaml_path=['onset_detection', 'delta'],
        cli_flag='--onset-delta',
    ),

    SettingDefinition(
        key='onset_wait',
        type=SettingType.INT,
        default=3,
        label='Onset Wait',
        description='Minimum frames between peaks (1 frame ≈ 11ms, allows fast repeated hits)',
        category=SettingCategory.ONSET_DETECTION,
        ui_control=UIControl.NUMBER,
        min_value=1,
        max_value=20,
        step=1,
        unit='frames',
        advanced=True,
        yaml_path=['onset_detection', 'wait'],
        cli_flag='--onset-wait',
    ),

    SettingDefinition(
        key='hop_length',
        type=SettingType.INT,
        default=512,
        label='Hop Length',
        description='Samples between frames (affects time resolution)',
        category=SettingCategory.ONSET_DETECTION,
        ui_control=UIControl.NUMBER,
        min_value=128,
        max_value=2048,
        step=128,
        unit='samples',
        advanced=True,
        yaml_path=['onset_detection', 'hop_length'],
        cli_flag='--hop-length',
    ),

    SettingDefinition(
        key='detection_method',
        type=SettingType.CHOICE,
        default='both',
        label='Detection Method',
        description=(
            "Which detector's events become events_configured. The "
            "spectral detector and the energy detector BOTH always run; "
            "this only chooses which list is promoted to "
            "events_configured for the MIDI output."
        ),
        category=SettingCategory.ONSET_DETECTION,
        ui_control=UIControl.SELECT,
        allowed_values=['energy', 'spectral', 'both'],
        yaml_path=['onset_detection', 'detection_method'],
        cli_flag='--detection-method',
    ),

    # ======================
    # MIDI Output Settings
    # ======================

    SettingDefinition(
        key='min_velocity',
        type=SettingType.INT,
        default=80,
        label='Min Velocity',
        description='Minimum MIDI velocity for detected hits',
        category=SettingCategory.MIDI_OUTPUT,
        ui_control=UIControl.NUMBER,
        min_value=1,
        max_value=127,
        step=1,
        yaml_path=['midi', 'min_velocity'],
        cli_flag='--min-velocity',
    ),

    SettingDefinition(
        key='max_velocity',
        type=SettingType.INT,
        default=110,
        label='Max Velocity',
        description='Maximum MIDI velocity for detected hits',
        category=SettingCategory.MIDI_OUTPUT,
        ui_control=UIControl.NUMBER,
        min_value=1,
        max_value=127,
        step=1,
        yaml_path=['midi', 'max_velocity'],
        cli_flag='--max-velocity',
    ),

    SettingDefinition(
        key='tempo',
        type=SettingType.FLOAT,
        default=None,
        label='Tempo',
        description='Tempo in BPM (leave empty for auto-detection)',
        category=SettingCategory.MIDI_OUTPUT,
        ui_control=UIControl.NUMBER,
        min_value=60.0,
        max_value=200.0,
        step=1.0,
        unit='BPM',
        nullable=True,
        cli_flag='--tempo',
    ),

    # =================
    # Global filtering
    # =================
    SettingDefinition(
        key='reverb_continuation_attack_threshold',
        type=SettingType.FLOAT,
        default=0.4,
        label='Reverb Attack Threshold',
        description='Attack sharpness threshold for reverb continuation filtering (real hits >= 0.4, reverb/echo < 0.4)',
        category=SettingCategory.MIDI_OUTPUT,
        ui_control=UIControl.SLIDER,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        yaml_path=['filtering', 'reverb_continuation_attack_threshold'],
        cli_flag='--reverb-attack-threshold',
    ),
    
    # =================
    # Per-Stem Settings
    # =================
    
    # Kick settings (examples - would continue for all stems)
    SettingDefinition(
        key='kick_midi_note',
        type=SettingType.INT,
        default=36,
        label='MIDI Note',
        description='MIDI note number for kick drum',
        category=SettingCategory.KICK,
        ui_control=UIControl.NUMBER,
        min_value=0,
        max_value=127,
        step=1,
        yaml_path=['kick', 'midi_note'],
        cli_flag='--kick-midi-note',
        advanced=True,
    ),

    SettingDefinition(
        key='kick_onset_threshold',
        type=SettingType.FLOAT,
        default=0.1,
        label='Onset Threshold Override',
        description='Per-stem onset threshold (overrides global setting)',
        category=SettingCategory.KICK,
        ui_control=UIControl.NUMBER,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        nullable=True,
        advanced=True,
        yaml_path=['kick', 'onset_threshold'],
    ),

    SettingDefinition(
        key='kick_timing_offset',
        type=SettingType.FLOAT,
        default=-0.014,
        label='Timing Offset',
        description='Timing correction in seconds (positive = shift MIDI events later)',
        category=SettingCategory.KICK,
        ui_control=UIControl.NUMBER,
        min_value=-0.1,
        max_value=0.1,
        step=0.001,
        unit='seconds',
        advanced=True,
        yaml_path=['kick', 'timing_offset'],
    ),

    SettingDefinition(
        key='kick_geomean_threshold',
        type=SettingType.FLOAT,
        default=800.0,
        label='GeoMean Threshold',
        description='Spectral filtering threshold (rejects artifacts)',
        category=SettingCategory.KICK,
        ui_control=UIControl.NUMBER,
        min_value=0.0,
        max_value=3000.0,
        step=1.0,
        yaml_path=['kick', 'geomean_threshold'],
        cli_flag='--kick-geomean',
    ),

    SettingDefinition(
        key='kick_use_stereo',
        type=SettingType.BOOL,
        default=False,
        label='Use Stereo Processing',
        description='Process kick in stereo to use pan position for identification (kick is typically centered, so usually disabled)',
        category=SettingCategory.KICK,
        ui_control=UIControl.CHECKBOX,
        yaml_path=['kick', 'use_stereo'],
    ),

    # Snare MIDI note
    SettingDefinition(
        key='snare_midi_note',
        type=SettingType.INT,
        default=38,
        label='MIDI Note (Snare)',
        description='MIDI note number for snare drum hit (primary sub-type)',
        category=SettingCategory.SNARE,
        ui_control=UIControl.NUMBER,
        min_value=0,
        max_value=127,
        step=1,
        yaml_path=['snare', 'midi_note'],
        cli_flag='--snare-midi-note',
        advanced=True,
    ),

    SettingDefinition(
        key='snare_midi_note_rimshot',
        type=SettingType.INT,
        default=37,
        label='MIDI Note (Rimshot)',
        description='MIDI note number for snare rimshot / side-stick',
        category=SettingCategory.SNARE,
        ui_control=UIControl.NUMBER,
        min_value=0,
        max_value=127,
        step=1,
        yaml_path=['snare', 'midi_note_rimshot'],
        cli_flag='--snare-midi-rimshot',
        advanced=True,
    ),

    SettingDefinition(
        key='snare_midi_note_clap',
        type=SettingType.INT,
        default=39,
        label='MIDI Note (Clap)',
        description='MIDI note number for hand clap detected from snare stem',
        category=SettingCategory.SNARE,
        ui_control=UIControl.NUMBER,
        min_value=0,
        max_value=127,
        step=1,
        yaml_path=['snare', 'midi_note_clap'],
        cli_flag='--snare-midi-clap',
        advanced=True,
    ),

    SettingDefinition(
        key='snare_use_stereo',
        type=SettingType.BOOL,
        default=True,
        label='Use Stereo Processing',
        description='Process snare in stereo to use pan position for identification (ghost notes and rimshots may be panned)',
        category=SettingCategory.SNARE,
        ui_control=UIControl.CHECKBOX,
        yaml_path=['snare', 'use_stereo'],
    ),

    SettingDefinition(
        key='snare_geomean_threshold',
        type=SettingType.FLOAT,
        default=40.0,
        label='GeoMean Threshold',
        description='Snare spectral filtering threshold (rejects artifacts)',
        category=SettingCategory.SNARE,
        ui_control=UIControl.NUMBER,
        min_value=0.0,
        max_value=500.0,
        step=1.0,
        yaml_path=['snare', 'geomean_threshold'],
        cli_flag='--snare-geomean',
    ),

    SettingDefinition(
        key='snare_cluster_feature',
        type=SettingType.CHOICE,
        default='auto',
        label='Cluster Feature',
        description='Feature used for snare sub-type clustering (auto = stereo_width, then spectral_centroid_hz)',
        category=SettingCategory.SNARE,
        ui_control=UIControl.SELECT,
        allowed_values=['auto', 'stereo_width', 'spectral_centroid_hz', 'pitch_hz', 'pan_confidence'],
        yaml_path=['snare', 'cluster_feature'],
        cli_flag='--snare-cluster-feature',
    ),
    
    # Toms MIDI notes
    SettingDefinition(
        key='toms_midi_note_low',
        type=SettingType.INT,
        default=45,
        label='MIDI Note (Low Tom)',
        description='MIDI note number for low tom',
        category=SettingCategory.TOMS,
        ui_control=UIControl.NUMBER,
        min_value=0,
        max_value=127,
        step=1,
        yaml_path=['toms', 'midi_note_low'],
        cli_flag='--toms-midi-low',
        advanced=True,
    ),

    SettingDefinition(
        key='toms_midi_note_mid',
        type=SettingType.INT,
        default=47,
        label='MIDI Note (Mid Tom)',
        description='MIDI note number for mid tom',
        category=SettingCategory.TOMS,
        ui_control=UIControl.NUMBER,
        min_value=0,
        max_value=127,
        step=1,
        yaml_path=['toms', 'midi_note_mid'],
        cli_flag='--toms-midi-mid',
        advanced=True,
    ),

    SettingDefinition(
        key='toms_midi_note_high',
        type=SettingType.INT,
        default=50,
        label='MIDI Note (High Tom)',
        description='MIDI note number for high tom',
        category=SettingCategory.TOMS,
        ui_control=UIControl.NUMBER,
        min_value=0,
        max_value=127,
        step=1,
        yaml_path=['toms', 'midi_note_high'],
        cli_flag='--toms-midi-high',
        advanced=True,
    ),

    SettingDefinition(
        key='toms_use_stereo',
        type=SettingType.BOOL,
        default=True,
        label='Use Stereo Processing',
        description='Process toms in stereo to use pan position for low/mid/high identification (toms are often panned left to right)',
        category=SettingCategory.TOMS,
        ui_control=UIControl.CHECKBOX,
        yaml_path=['toms', 'use_stereo'],
    ),

    SettingDefinition(
        key='toms_geomean_threshold',
        type=SettingType.FLOAT,
        default=80.0,
        label='GeoMean Threshold',
        description='Toms spectral filtering threshold (rejects artifacts)',
        category=SettingCategory.TOMS,
        ui_control=UIControl.NUMBER,
        min_value=0.0,
        max_value=500.0,
        step=1.0,
        yaml_path=['toms', 'geomean_threshold'],
        cli_flag='--toms-geomean',
    ),

    SettingDefinition(
        key='toms_cluster_feature',
        type=SettingType.CHOICE,
        default='auto',
        label='Cluster Feature',
        description='Feature used for tom low/mid/high clustering (auto = pitch_hz, then spectral_centroid_hz)',
        category=SettingCategory.TOMS,
        ui_control=UIControl.SELECT,
        allowed_values=['auto', 'pitch_hz', 'spectral_centroid_hz', 'stereo_width', 'pan_confidence'],
        yaml_path=['toms', 'cluster_feature'],
        cli_flag='--toms-cluster-feature',
    ),

    # Spectral snap settings — per-stem configuration of the
    # "head snap" frequency range for the spectral-transient
    # detector. The detector fires on the broadband percussive
    # transient at the attack onset (in the snap range) AND on
    # the per-band-dominant ring (in the full 5 bands). The
    # snap signal is what catches the attack onset within a few
    # ms instead of 50-100ms after (which is what the ring-only
    # detector does).
    #
    # User insight (2026-06-09): the toms attack onset is
    # broadband in 200-1200Hz (B1+B2). The B0 ring develops
    # 50-100ms later. Default for toms: snap_bands=[1, 2].
    SettingDefinition(
        key='toms_spectral_snap_bands',
        type=SettingType.STRING,
        default='1,2',
        label='Snap Bands (Toms)',
        description='Comma-separated band indices for the snap detection signal (e.g. "1,2" for 200-1200Hz). The detector fires on the broadband percussive transient in these bands at the attack onset. Use "0,1,2,3,4" to disable snap detection (fall back to ring-only).',
        category=SettingCategory.TOMS,
        ui_control=UIControl.TEXT,
        yaml_path=['toms', 'spectral_snap_bands'],
        cli_flag='--toms-snap-bands',
    ),
    SettingDefinition(
        key='toms_spectral_snap_min_delta',
        type=SettingType.FLOAT,
        default=0.05,
        label='Snap Min Delta (Toms)',
        description='find_peaks height for the toms snap signal. Lower = more sensitive (catches quieter snaps, but more FPs). Default 0.05 — calibrated on project 4 funk track (2026-06-09).',
        category=SettingCategory.TOMS,
        ui_control=UIControl.NUMBER,
        min_value=0.0,
        max_value=10.0,
        step=0.01,
        yaml_path=['toms', 'spectral_snap_min_delta'],
        cli_flag='--toms-snap-min-delta',
    ),

    # "Show Only Snap Events" toggle for toms (2026-06-10).
    # Replaces the 2026-06-09 snap_mask_enabled/threshold pair
    # (which had a lossy "0.001 default" that silently hid events
    # on first Save) and the Stage-1 snap/ring floor from the
    # 2026-06-10 advanced filter. When ON, any spectral event
    # whose snap_delta is zero (or null) is filtered from the
    # saved MIDI. snap_delta > 0 means the broadband attack
    # signal fired in the snap bands at the event's peak frame
    # — typical of a real percussive attack. snap_delta == 0 is
    # the classic wire-tail / decay signature (the RING signal
    # fired but the broadband attack had already decayed).
    #
    # Off by default — the user opts in. The filter is
    # idempotent across rebuilds (turning it off restores any
    # previously-filtered snap-zero events, unlike the old
    # snap-mask which was effectively a one-way ratchet).
    SettingDefinition(
        key='toms_show_only_snap_events',
        type=SettingType.BOOL,
        default=False,
        label='Show Only Snap Events (Toms)',
        description=(
            'When on, spectral events whose snap_delta is zero '
            'or null are filtered from the saved MIDI. This is '
            'the typical wire-tail / decay kill switch — events '
            'where the RING signal fired but the broadband '
            'attack had already decayed. Off by default. '
            'Idempotent: turning it off restores the filtered '
            'events on the next rebuild.'
        ),
        category=SettingCategory.TOMS,
        ui_control=UIControl.CHECKBOX,
        yaml_path=['toms', 'show_only_snap_events'],
        cli_flag='--toms-show-only-snap-events',
    ),

    # "Filter Events with Top/2nd Ratio Greater Than" slider for
    # toms (2026-06-10). Replaces the lossy
    # `advanced_filter_high_strength` toggle (which operated on
    # the strength field that was clamped to [0, 1] and therefore
    # could not distinguish a band_max_ratio of 11 from one of
    # 459). This slider reads the RAW band_max_ratio and drops
    # any spectral event whose top/second-highest-band ratio is
    # strictly greater than the threshold.
    #
    # The slider is a "ceiling on extreme dominance" gate — the
    # user's calibration case (a real hit with ratio 18.99 vs an
    # FP with ratio 459.12) is now expressible directly. The
    # slider max is the dataset's actual max ratio (set at UI
    # build time in threshold-tuning.js), and the step is
    # derived from the max so the user gets full resolution
    # across whatever range the data exhibits. The 0 value is
    # a special "Off / Disabled" sentinel — the filter is a
    # no-op when 0.
    SettingDefinition(
        key='toms_band_max_ratio_max',
        type=SettingType.FLOAT,
        default=0.0,
        label='Filter Events with Top/2nd Ratio Greater Than (Toms)',
        description=(
            'Spectral events whose band_max_ratio (top band / '
            'second-highest band at the event frame) is strictly '
            'greater than this value are filtered from the saved '
            'MIDI. Use to drop the "extreme dominance" FP '
            'signature — events where one band beats the others '
            'by 100x or more (real hits are typically <20x; the '
            'user\'s calibration case had FPs at 459x). 0 (the '
            'default) disables the filter — the slider in the '
            'sidecar shows "Off" at this position. The slider '
            'max is set to the dataset\'s max ratio so the full '
            'range is expressible without losing precision.'
        ),
        category=SettingCategory.TOMS,
        ui_control=UIControl.SLIDER,
        min_value=0.0,
        # Server-side upper bound. The WebUI overrides `max` at
        # build time to the actual dataset max; this default
        # just prevents the server from accepting absurdly huge
        # values via the CLI / config-file path.
        max_value=10000.0,
        step=0.1,
        yaml_path=['toms', 'band_max_ratio_max'],
        cli_flag='--toms-band-max-ratio-max',
    ),

    # Master onset-filter gate (2026-06-10). When ON (default),
    # the Geomean / Sustain / Strength filter sliders behave as
    # before. When OFF, those filter passes are skipped entirely —
    # every onset that the energy detector produced is treated as
    # KEPT regardless of the slider values. Use case: instead of
    # dragging the geomean threshold to a high value to see every
    # event (which clobbers the saved MIDI and is hard to recover
    # from), flip this toggle off for a one-off A/B comparison and
    # flip it back on to restore the strict filtering. The snap
    # mask is independent and still applies when this is off.
    # Onset events visibility gate (2026-06-10, renamed 2026-06-10
    # round 2). Originally called 'onset_events_enabled' and
    # implemented as a filter-bypass — wrong semantics. Correct
    # semantics: ON (default) = energy-detected onset events are
    # included in events_configured. OFF = energy-detected events
    # are REMOVED from events_configured entirely. Spectral events
    # are unaffected. Use case: when the energy detector is
    # producing too much noise, the user can suppress the entire
    # onset stream and rely on the spectral detector alone.
    # Direction is one-way per save: turning OFF drops the events,
    # and the user must re-run full detection to get them back
    # (the per-event override and the events_sensitive list remain
    # on disk, but events_configured is no longer the source of
    # truth for them). This aligns with the user's plan to
    # deprecate the energy detector for toms.
    SettingDefinition(
        key='toms_onset_events_enabled',
        type=SettingType.BOOL,
        default=True,
        label='Show Onset Events (Toms)',
        description=(
            'ON (default): energy-detected onset events are included '
            'in events_configured and the waveform view. OFF: '
            'energy-detected events are REMOVED from '
            'events_configured and the view (spectral events are '
            'unaffected). One-way per save — re-run full detection '
            'to restore dropped events. Use this when the energy '
            'detector is producing too much noise and you want to '
            'rely on the spectral detector alone. The snap mask, '
            'advanced filter, and other tuning are independent.'
        ),
        category=SettingCategory.TOMS,
        ui_control=UIControl.CHECKBOX,
        yaml_path=['toms', 'onset_events_enabled'],
        cli_flag='--toms-onset-events-enabled',
    ),

    # 2026-06-10: removed the legacy advanced-filter + snap-mask
    # settings (toms_advanced_filter_enabled, toms_advanced_min_snap_delta,
    # toms_advanced_snap_ring_threshold, toms_advanced_snap_ring_direction,
    # toms_advanced_filter_high_strength, toms_snap_mask_threshold). They
    # were replaced by `toms_show_only_snap_events` and
    # `toms_band_max_ratio_max` above. Back-compat: any existing project
    # YAML that still has the old keys is ignored — the new filters
    # are off by default, so nothing happens to the saved MIDI unless
    # the user explicitly sets the new keys.


    # Hi-hat settings
    SettingDefinition(
        key='hihat_midi_note_closed',
        type=SettingType.INT,
        default=42,
        label='MIDI Note (Closed)',
        description='MIDI note number for closed hi-hat',
        category=SettingCategory.HIHAT,
        ui_control=UIControl.NUMBER,
        min_value=0,
        max_value=127,
        step=1,
        yaml_path=['hihat', 'midi_note_closed'],
        cli_flag='--hihat-midi-closed',
        advanced=True,
    ),

    SettingDefinition(
        key='hihat_midi_note_open',
        type=SettingType.INT,
        default=46,
        label='MIDI Note (Open)',
        description='MIDI note number for open hi-hat',
        category=SettingCategory.HIHAT,
        ui_control=UIControl.NUMBER,
        min_value=0,
        max_value=127,
        step=1,
        yaml_path=['hihat', 'midi_note_open'],
        cli_flag='--hihat-midi-open',
        advanced=True,
    ),

    SettingDefinition(
        key='hihat_midi_note_foot_close',
        type=SettingType.INT,
        default=44,
        label='MIDI Note (Foot Close)',
        description='MIDI note number for foot close (pedal closes open hihat)',
        category=SettingCategory.HIHAT,
        ui_control=UIControl.NUMBER,
        min_value=0,
        max_value=127,
        step=1,
        yaml_path=['hihat', 'midi_note_foot_close'],
        cli_flag='--hihat-midi-foot-close',
        advanced=True,
    ),

    SettingDefinition(
        key='hihat_midi_note_handclap',
        type=SettingType.INT,
        default=39,
        label='MIDI Note (Handclap)',
        description='MIDI note number for hand clap detected from hihat bleed',
        category=SettingCategory.HIHAT,
        ui_control=UIControl.NUMBER,
        min_value=0,
        max_value=127,
        step=1,
        yaml_path=['hihat', 'midi_note_handclap'],
        cli_flag='--hihat-midi-handclap',
        advanced=True,
    ),

    SettingDefinition(
        key='hihat_use_stereo',
        type=SettingType.BOOL,
        default=True,
        label='Use Stereo Processing',
        description='Process hi-hat in stereo to use pan position for better detection (hi-hat may be off-center)',
        category=SettingCategory.HIHAT,
        ui_control=UIControl.CHECKBOX,
        yaml_path=['hihat', 'use_stereo'],
    ),

    # 2026-06-19: hihat_open_geomean_min and hihat_open_sustain_ms
    # removed from the schema. The slope rule
    # (hihat_open_decay_slope_max, below) is the only hihat
    # open/closed classifier on current sidecars. The
    # geomean+sustain rule in classify_hihat_notes is a
    # defensive fallback that only fires when decay_slope_db
    # is missing (older sidecars from before 2026-06-19), so
    # users never need to tune it.

    # 2026-06-19: broadband-envelope decay-slope classifier for
    # open/closed hihat. ``decay_slope_db`` is the mean per-frame
    # dB drop over the forward walk (positive = env dropped, larger
    # = sharper decay → closed hihat). An event with
    # ``decay_slope_db < threshold`` is open — its ring-out held
    # loud enough that the next strike cut in before the envelope
    # dropped to 50% of peak. Default 2.0 dB/frame is the
    # population p50 across all KEPT hihats in the Taylor Swift
    # project (p10=2.02, p90=3.41, closed hits 3.4-3.6, open
    # hits 0.7). The fallback 2.0 is intentionally conservative
    # — raise it to favor "open" calls, lower it to favor "closed".
    SettingDefinition(
        key='hihat_open_decay_slope_max',
        type=SettingType.FLOAT,
        default=2.0,
        label='Open Hi-Hat Decay Slope Max',
        description='Maximum broadband-envelope decay slope (dB/frame) for an open hihat hit. Closed hihats decay fast (slope 3.4-3.6) — open hihats ring out so the next strike cuts in before the envelope drops, giving a shallow slope (0.7-1.4). Events with decay_slope_db < threshold are classified open. Default 2.0 is the population p50 across all KEPT hihats in the Taylor Swift project.',
        category=SettingCategory.HIHAT,
        ui_control=UIControl.SLIDER,
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        unit='dB/frame',
        yaml_path=['hihat', 'open_decay_slope_max'],
        cli_flag='--hihat-open-decay-slope-max',
    ),

    SettingDefinition(
        key='hihat_geomean_threshold',
        type=SettingType.FLOAT,
        default=8.0,
        label='GeoMean Threshold',
        description='Hi-hat spectral filtering threshold (rejects artifacts)',
        category=SettingCategory.HIHAT,
        ui_control=UIControl.NUMBER,
        min_value=0.0,
        max_value=200.0,
        step=0.5,
        yaml_path=['hihat', 'geomean_threshold'],
        cli_flag='--hihat-geomean',
    ),

    # Cymbals MIDI note
    SettingDefinition(
        key='cymbals_midi_note',
        type=SettingType.INT,
        default=57,
        label='MIDI Note (Default)',
        description='MIDI note number used as fallback when no sub-type classifier matches',
        category=SettingCategory.CYMBALS,
        ui_control=UIControl.NUMBER,
        min_value=0,
        max_value=127,
        step=1,
        yaml_path=['cymbals', 'midi_note'],
        cli_flag='--cymbals-midi-note',
        advanced=True,
    ),

    SettingDefinition(
        key='cymbals_midi_note_crash',
        type=SettingType.INT,
        default=49,
        label='MIDI Note (Crash)',
        description='MIDI note number for crash cymbal (sub-type 0)',
        category=SettingCategory.CYMBALS,
        ui_control=UIControl.NUMBER,
        min_value=0,
        max_value=127,
        step=1,
        yaml_path=['cymbals', 'midi_note_crash'],
        cli_flag='--cymbals-midi-crash',
        advanced=True,
    ),

    SettingDefinition(
        key='cymbals_midi_note_ride',
        type=SettingType.INT,
        default=51,
        label='MIDI Note (Ride)',
        description='MIDI note number for ride cymbal (sub-type 1)',
        category=SettingCategory.CYMBALS,
        ui_control=UIControl.NUMBER,
        min_value=0,
        max_value=127,
        step=1,
        yaml_path=['cymbals', 'midi_note_ride'],
        cli_flag='--cymbals-midi-ride',
        advanced=True,
    ),

    SettingDefinition(
        key='cymbals_midi_note_chinese',
        type=SettingType.INT,
        default=52,
        label='MIDI Note (Chinese)',
        description='MIDI note number for chinese cymbal (sub-type 2)',
        category=SettingCategory.CYMBALS,
        ui_control=UIControl.NUMBER,
        min_value=0,
        max_value=127,
        step=1,
        yaml_path=['cymbals', 'midi_note_chinese'],
        cli_flag='--cymbals-midi-chinese',
        advanced=True,
    ),

    SettingDefinition(
        key='cymbals_use_stereo',
        type=SettingType.BOOL,
        default=True,
        label='Use Stereo Processing',
        description='Process cymbals in stereo to use pan position for crash/ride distinction (cymbals are often panned left/right)',
        category=SettingCategory.CYMBALS,
        ui_control=UIControl.CHECKBOX,
        yaml_path=['cymbals', 'use_stereo'],
    ),

    SettingDefinition(
        key='cymbals_geomean_threshold',
        type=SettingType.FLOAT,
        default=100.0,
        label='GeoMean Threshold',
        description='Cymbals spectral filtering threshold (rejects artifacts)',
        category=SettingCategory.CYMBALS,
        ui_control=UIControl.NUMBER,
        min_value=0.0,
        max_value=1000.0,
        step=5.0,
        yaml_path=['cymbals', 'geomean_threshold'],
        cli_flag='--cymbals-geomean',
    ),

    SettingDefinition(
        key='cymbals_cluster_feature',
        type=SettingType.CHOICE,
        default='auto',
        label='Cluster Feature',
        description='Feature used for cymbal sub-type clustering (auto = spectral_centroid_hz, then stereo_width)',
        category=SettingCategory.CYMBALS,
        ui_control=UIControl.SELECT,
        allowed_values=['auto', 'spectral_centroid_hz', 'stereo_width', 'pitch_hz', 'pan_confidence'],
        yaml_path=['cymbals', 'cluster_feature'],
        cli_flag='--cymbals-cluster-feature',
    ),
    
    # Kick Clustering
    SettingDefinition(
        key='kick_expected_clusters',
        type=SettingType.INT,
        default=1,
        label='Expected Clusters',
        description='Expected number of distinct kick sounds (1 = single kick)',
        category=SettingCategory.KICK,
        ui_control=UIControl.NUMBER,
        min_value=1,
        max_value=5,
        step=1,
        yaml_path=['kick', 'expected_clusters'],
        cli_flag='--kick-clusters',
        advanced=True,
    ),

    # Snare Clustering
    SettingDefinition(
        key='snare_expected_clusters',
        type=SettingType.INT,
        default=1,
        label='Expected Clusters',
        description='Expected number of distinct snare sounds (1 = single snare, 2 = snare + side-stick)',
        category=SettingCategory.SNARE,
        ui_control=UIControl.NUMBER,
        min_value=1,
        max_value=5,
        step=1,
        yaml_path=['snare', 'expected_clusters'],
        cli_flag='--snare-clusters',
        advanced=True,
    ),

    # Toms Clustering
    SettingDefinition(
        key='toms_expected_clusters',
        type=SettingType.INT,
        default=3,
        label='Expected Clusters',
        description='Expected number of distinct tom sounds (3 = low/mid/high)',
        category=SettingCategory.TOMS,
        ui_control=UIControl.NUMBER,
        min_value=1,
        max_value=5,
        step=1,
        yaml_path=['toms', 'expected_clusters'],
        cli_flag='--toms-clusters',
        advanced=True,
    ),

    # Hihat Clustering
    SettingDefinition(
        key='hihat_expected_clusters',
        type=SettingType.INT,
        default=2,
        label='Expected Clusters',
        description='Expected number of distinct hihat sounds (2 = open + closed)',
        category=SettingCategory.HIHAT,
        ui_control=UIControl.NUMBER,
        min_value=1,
        max_value=5,
        step=1,
        yaml_path=['hihat', 'expected_clusters'],
        cli_flag='--hihat-clusters',
        advanced=True,
    ),

    # Cymbals Clustering
    SettingDefinition(
        key='cymbals_expected_clusters',
        type=SettingType.INT,
        default=2,
        label='Expected Clusters',
        description='Expected number of distinct cymbals (e.g., 2 = left crash + right crash)',
        category=SettingCategory.CYMBALS,
        ui_control=UIControl.NUMBER,
        min_value=1,
        max_value=5,
        step=1,
        yaml_path=['cymbals', 'expected_clusters'],
        cli_flag='--cymbals-clusters',
        advanced=True,
    ),
    
    # ===================
    # Clustering Settings
    # ===================
    
    SettingDefinition(
        key='clustering_method',
        type=SettingType.CHOICE,
        default='dbscan',
        label='Clustering Method',
        description='Algorithm for grouping similar onsets (DBSCAN = density-based, k-means = centroid-based)',
        category=SettingCategory.CLUSTERING,
        ui_control=UIControl.SELECT,
        allowed_values=['dbscan', 'kmeans'],
        yaml_path=['clustering', 'method'],
        advanced=True,
    ),
    
    # =============================
    # Threshold Optimization Settings
    # =============================
    
    SettingDefinition(
        key='threshold_optimization_enabled',
        type=SettingType.BOOL,
        default=False,
        label='Enable Threshold Optimization',
        description='Automatically discover optimal thresholds by iterating until cluster count matches expected',
        category=SettingCategory.THRESHOLD_OPTIMIZATION,
        ui_control=UIControl.CHECKBOX,
        yaml_path=['threshold_optimization', 'enabled'],
    ),
    
    SettingDefinition(
        key='threshold_optimization_max_iterations',
        type=SettingType.INT,
        default=20,
        label='Max Iterations',
        description='Maximum number of optimization iterations before giving up',
        category=SettingCategory.THRESHOLD_OPTIMIZATION,
        ui_control=UIControl.NUMBER,
        min_value=5,
        max_value=100,
        step=5,
        yaml_path=['threshold_optimization', 'max_iterations'],
        advanced=True,
    ),
    
    SettingDefinition(
        key='threshold_optimization_tolerance',
        type=SettingType.INT,
        default=0,
        label='Cluster Count Tolerance',
        description='Stop when cluster count is within ±N of expected (0 = exact match required)',
        category=SettingCategory.THRESHOLD_OPTIMIZATION,
        ui_control=UIControl.NUMBER,
        min_value=0,
        max_value=5,
        step=1,
        yaml_path=['threshold_optimization', 'tolerance'],
        advanced=True,
    ),
    
    # ===================
    # Separation Settings
    # ===================
    
    SettingDefinition(
        key='device',
        type=SettingType.CHOICE,
        default='auto',
        label='Device',
        description='Processing device (auto-detect, CPU, or CUDA GPU)',
        category=SettingCategory.SEPARATION,
        ui_control=UIControl.SELECT,
        allowed_values=['auto', 'cpu', 'cuda'],
    ),
    
    SettingDefinition(
        key='overlap',
        type=SettingType.INT,
        default=4,
        label='Overlap',
        description='Model overlap factor (higher = better quality, slower)',
        category=SettingCategory.SEPARATION,
        ui_control=UIControl.NUMBER,
        min_value=1,
        max_value=8,
        step=1,
    ),
    
    SettingDefinition(
        key='wiener_exponent',
        type=SettingType.FLOAT,
        default=None,
        label='Wiener Filter',
        description='Wiener filter exponent for noise reduction (0 = disabled, 1.0-3.0 = light to aggressive)',
        category=SettingCategory.SEPARATION,
        ui_control=UIControl.NUMBER,
        min_value=0.0,
        max_value=5.0,
        step=0.1,
        nullable=True,
    ),
    
    # ================
    # Cleanup Settings
    # ================
    
    SettingDefinition(
        key='cleanup_threshold',
        type=SettingType.FLOAT,
        default=-30.0,
        label='Threshold',
        description='Sidechain trigger level',
        category=SettingCategory.CLEANUP,
        ui_control=UIControl.NUMBER,
        min_value=-40.0,
        max_value=-20.0,
        step=1.0,
        unit='dB',
    ),
    
    SettingDefinition(
        key='cleanup_ratio',
        type=SettingType.FLOAT,
        default=10.0,
        label='Ratio',
        description='Compression amount',
        category=SettingCategory.CLEANUP,
        ui_control=UIControl.NUMBER,
        min_value=2.0,
        max_value=20.0,
        step=1.0,
        unit=':1',
    ),
    
    SettingDefinition(
        key='cleanup_attack',
        type=SettingType.FLOAT,
        default=1.0,
        label='Attack',
        description='How fast compression starts',
        category=SettingCategory.CLEANUP,
        ui_control=UIControl.NUMBER,
        min_value=0.1,
        max_value=10.0,
        step=0.1,
        unit='ms',
    ),
    
    SettingDefinition(
        key='cleanup_release',
        type=SettingType.FLOAT,
        default=100.0,
        label='Release',
        description='How fast compression releases',
        category=SettingCategory.CLEANUP,
        ui_control=UIControl.NUMBER,
        min_value=10.0,
        max_value=500.0,
        step=10.0,
        unit='ms',
    ),
    
    # ==============
    # Video Settings
    # ==============
    
    SettingDefinition(
        key='video_fps',
        type=SettingType.INT,
        default=60,
        label='FPS',
        description='Video frame rate',
        category=SettingCategory.VIDEO,
        ui_control=UIControl.SELECT,
        allowed_values=[30, 60, 120],
    ),
    
    SettingDefinition(
        key='video_resolution',
        type=SettingType.CHOICE,
        default='1080p',
        label='Resolution',
        description='Video resolution',
        category=SettingCategory.VIDEO,
        ui_control=UIControl.SELECT,
        allowed_values=['1080p', '1440p', '4K', '1080p-portrait', '1440p-portrait', '4K-portrait'],
    ),
    
    SettingDefinition(
        key='video_fall_speed',
        type=SettingType.FLOAT,
        default=1.0,
        label='Fall Speed',
        description='Note fall speed multiplier',
        category=SettingCategory.VIDEO,
        ui_control=UIControl.SLIDER,
        min_value=0.5,
        max_value=3.0,
        step=0.1,
    ),
]


# ============================================================================
# SCHEMA ACCESS FUNCTIONS
# ============================================================================

def get_settings_by_category(category: SettingCategory) -> List[SettingDefinition]:
    """Get all settings for a specific category"""
    return [s for s in SETTINGS_REGISTRY if s.category == category]


def get_setting_by_key(key: str) -> Optional[SettingDefinition]:
    """Get a setting definition by its key"""
    for setting in SETTINGS_REGISTRY:
        if setting.key == key:
            return setting
    return None


def get_all_settings() -> List[SettingDefinition]:
    """Get all registered settings"""
    return SETTINGS_REGISTRY.copy()


def get_settings_schema() -> Dict[str, Any]:
    """
    Get complete settings schema for API consumption.
    
    Returns:
        Dictionary with categorized settings for frontend use
    """
    schema = {
        'version': '1.0',
        'categories': {},
        'settings': {}
    }
    
    # Group by category
    for category in SettingCategory:
        cat_settings = get_settings_by_category(category)
        if cat_settings:
            schema['categories'][category.value] = {
                'label': category.value.replace('_', ' ').title(),
                'settings': [s.key for s in cat_settings]
            }
    
    # Add all settings
    for setting in SETTINGS_REGISTRY:
        schema['settings'][setting.key] = setting.to_dict()
    
    return schema


def get_defaults_for_category(category: SettingCategory) -> Dict[str, Any]:
    """Get default values for all settings in a category"""
    return {
        s.key: s.default
        for s in get_settings_by_category(category)
    }


def get_cli_flags() -> Dict[str, str]:
    """Get mapping of setting keys to CLI flags"""
    return {
        s.key: s.cli_flag
        for s in SETTINGS_REGISTRY
        if s.cli_flag
    }
