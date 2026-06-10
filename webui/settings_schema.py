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

    # Snap-delta mask for toms (2026-06-09; toggle added 2026-06-10).
    # The mask is OFF by default — the user explicitly opted-in by
    # adding the toggle, because (a) masking is a one-way ratchet
    # (filtered events are physically removed from the saved MIDI
    # and there's no "unfilter" path that restores them), and (b) the
    # previous default of 0.001 silently hid legitimate spectral
    # events on the first Save. The threshold is the value the
    # snap_mask pass uses when enabled: filter any spectral event
    # whose snap_delta ≤ threshold. 0.001 (schema default) catches
    # floating-point zeros; 0.05–0.1 keeps only the strongest
    # broadband attacks.
    SettingDefinition(
        key='toms_snap_mask_enabled',
        type=SettingType.BOOL,
        default=False,
        label='Enable Snap Mask (Toms)',
        description=(
            'When on, spectral events whose snap_delta is below the '
            'Snap Mask Threshold are filtered from the saved MIDI. '
            'Off by default — turn on to clean up wire-tail / decay '
            'events that fired the RING signal but not the SNAP. '
            'WARNING: filtering is a one-way operation per Save; you '
            'cannot restore the masked events by turning the mask off '
            'in a later Save (the sidecar has already lost them).'
        ),
        category=SettingCategory.TOMS,
        ui_control=UIControl.CHECKBOX,
        yaml_path=['toms', 'snap_mask_enabled'],
        cli_flag='--toms-snap-mask-enabled',
    ),
    SettingDefinition(
        key='toms_snap_mask_threshold',
        type=SettingType.FLOAT,
        default=0.001,
        label='Snap Mask Threshold (Toms)',
        description=(
            'Filter spectral events whose snap_delta ≤ this threshold from '
            'the saved MIDI (toms only, requires the Snap Mask toggle to be '
            'on). 0 = filter only snap_delta==0 events (wire-tail / decay '
            'kill). 0.001 = same plus a tiny epsilon to catch floating-point '
            'zeros (default). 0.05–0.1 = keep only events with strong '
            'broadband attack.'
        ),
        category=SettingCategory.TOMS,
        ui_control=UIControl.SLIDER,
        min_value=0.0,
        max_value=0.5,
        step=0.01,
        yaml_path=['toms', 'snap_mask_threshold'],
        cli_flag='--toms-snap-mask',
    ),

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

    SettingDefinition(
        key='hihat_open_geomean_min',
        type=SettingType.FLOAT,
        default=262.0,
        label='Open Hi-Hat GeoMean Threshold',
        description='Minimum spectral geomean to classify a hi-hat hit as open (higher = stricter)',
        category=SettingCategory.HIHAT,
        ui_control=UIControl.SLIDER,
        min_value=50.0,
        max_value=1000.0,
        step=10.0,
        yaml_path=['hihat', 'open_geomean_min'],
        cli_flag='--hihat-open-geomean',
    ),

    SettingDefinition(
        key='hihat_open_sustain_ms',
        type=SettingType.FLOAT,
        default=100.0,
        label='Open Hi-Hat Sustain Threshold',
        description='Minimum sustain in milliseconds to classify a hi-hat hit as open (higher = stricter). Aligned with midiconfig.yaml default.',
        category=SettingCategory.HIHAT,
        ui_control=UIControl.SLIDER,
        min_value=20.0,
        max_value=500.0,
        step=5.0,
        unit='ms',
        yaml_path=['hihat', 'open_sustain_ms'],
        cli_flag='--hihat-open-sustain-ms',
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
