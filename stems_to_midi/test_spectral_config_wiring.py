"""Test that per-stem snap_bands / snap_min_delta flow from the project
config to the SpectralTransientConfig used by the detector.

The user (2026-06-09) wants to tune the detection per stem:
  - toms: snap_bands=(1, 2), snap_min_delta=0.05
  - hihat: snap_bands=(3, 4), snap_min_delta=0.01
  - snare: snap_bands=(1, 2), snap_min_delta=0.05
  - kick: snap_bands=(0,), snap_min_delta=2.0

This test verifies the wiring works: project config -> _build_spectral_config
-> SpectralTransientConfig -> detector behavior.
"""
import pytest

from stems_to_midi.spectral_transient_core import (
    SpectralTransientConfig,
)


def _build_spectral_config(stem_type, project_config):
    """Read per-stem snap_bands / snap_min_delta from project_config
    and build a SpectralTransientConfig. Import here to avoid a
    top-level import cycle."""
    from stems_to_midi.processing_shell import build_spectral_config_for_stem
    return build_spectral_config_for_stem(stem_type, project_config)


def test_build_spectral_config_uses_module_defaults_when_no_per_stem():
    """When the project config has no per-stem spectral_snap_bands,
    build_spectral_config_for_stem returns the module defaults
    (all 5 bands, snap_min_delta=0.05)."""
    project_config = {
        'toms': {'threshold_db': 15.0},
        'kick': {'threshold_db': 10.0},
    }
    cfg = _build_spectral_config('toms', project_config)
    assert tuple(cfg.snap_bands) == (0, 1, 2, 3, 4)
    assert cfg.snap_min_delta == 0.05


def test_build_spectral_config_reads_toms_snap_bands_from_project_config():
    """The project config for the toms stem must include
    spectral_snap_bands and spectral_snap_min_delta. When set,
    those flow into the SpectralTransientConfig."""
    project_config = {
        'toms': {
            'spectral_snap_bands': [1, 2],
            'spectral_snap_min_delta': 0.05,
        },
    }
    cfg = _build_spectral_config('toms', project_config)
    assert tuple(cfg.snap_bands) == (1, 2)
    assert cfg.snap_min_delta == 0.05


def test_build_spectral_config_reads_hihat_snap_bands_from_project_config():
    """Hihat snap_bands is (3, 4) — the 1200-8000Hz cymbal range."""
    project_config = {
        'hihat': {
            'spectral_snap_bands': [3, 4],
            'spectral_snap_min_delta': 0.01,
        },
    }
    cfg = _build_spectral_config('hihat', project_config)
    assert tuple(cfg.snap_bands) == (3, 4)
    assert cfg.snap_min_delta == 0.01


def test_build_spectral_config_per_stem_isolation():
    """The snap_bands for toms must NOT leak into the kick config.
    Each stem has its own per-stem settings."""
    project_config = {
        'toms': {'spectral_snap_bands': [1, 2]},
        'kick': {'spectral_snap_bands': [0]},
    }
    toms_cfg = _build_spectral_config('toms', project_config)
    kick_cfg = _build_spectral_config('kick', project_config)
    assert tuple(toms_cfg.snap_bands) == (1, 2)
    assert tuple(kick_cfg.snap_bands) == (0,)


def test_build_spectral_config_parses_string_snap_bands():
    """The schema delivers snap_bands as a comma-separated string
    (UI form input) or as a list (CLI/JSON). Both must be accepted."""
    # String form (WebUI form)
    project_config = {'toms': {'spectral_snap_bands': '1,2'}}
    cfg = _build_spectral_config('toms', project_config)
    assert tuple(cfg.snap_bands) == (1, 2)

    # String form with extra whitespace
    project_config = {'toms': {'spectral_snap_bands': ' 3 , 4 '}}
    cfg = _build_spectral_config('toms', project_config)
    assert tuple(cfg.snap_bands) == (3, 4)

    # String form with single value
    project_config = {'kick': {'spectral_snap_bands': '0'}}
    cfg = _build_spectral_config('kick', project_config)
    assert tuple(cfg.snap_bands) == (0,)

    # Empty string = no snap
    project_config = {'toms': {'spectral_snap_bands': ''}}
    cfg = _build_spectral_config('toms', project_config)
    assert tuple(cfg.snap_bands) == (0, 1, 2, 3, 4)  # falls back to default


def test_spectral_event_carries_band_delta_and_snap_delta():
    """Every SpectralTransientEvent must carry band_delta and
    snap_delta values (the detection signal values at the event
    frame) so the WebUI tooltip can show them. The user needs
    these to understand why the detector fired (or didn't)."""
    from stems_to_midi.spectral_transient_core import (
        SpectralTransientConfig,
        detect_spectral_transients,
    )
    from stems_to_midi.test_spectral_transient_core import (
        make_synthetic_drum_stem,
    )
    sr = 44100
    y = make_synthetic_drum_stem(sr=sr, hit_times_sec=(1.0,))
    cfg = SpectralTransientConfig(snap_bands=(1, 2))
    events, _ = detect_spectral_transients(y, sr, config=cfg)
    assert len(events) >= 1
    ev = events[0]
    assert hasattr(ev, 'band_delta'), (
        "SpectralTransientEvent needs a band_delta field (RING "
        "signal value at the event frame) so the WebUI tooltip "
        "can show it."
    )
    assert hasattr(ev, 'snap_delta'), (
        "SpectralTransientEvent needs a snap_delta field (SNAP "
        "signal value at the event frame) so the WebUI tooltip "
        "can show it."
    )
