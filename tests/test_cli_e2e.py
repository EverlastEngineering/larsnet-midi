"""End-to-end CLI tests for the stems_to_midi pipeline.

Validates that the PGA-universal cleanup didn't break the actual CLI flow.
Uses project 8 (user_files/8 - 2_funk_80_beat_4-4_4/) as the test fixture:
all 5 stems are present (kick, snare, hihat, toms, cymbals) and the
midiconfig.yaml is the calibrated baseline copied from project 4.

CLI behavior (verified by manual run on 2026-06-20):
  - The CLI writes ONE MIDI file at `<base_name>.mid` containing all
    detected events (not per-stem files).
  - The CLI writes ONE analysis sidecar at `<base_name>.analysis.json`
    with `events_pga` arrays per stem.
  - For project 8 specifically, PGA detects events for all 5 stems
    (kick ~106, snare ~50, hihat ~300+, cymbals ~6, toms ~0).

Test methodology:
  - Each test is idempotent: it cleans up the project's midi/ directory
    and resets midi_generated before running the CLI, so re-runs
    produce the same result.
  - These tests are SLOW (~5-10s each) so they're marked with
    `@pytest.mark.slow` (excluded by default via pytest.ini addopts).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROJECT_NUMBER = 8
PROJECT_DIR_NAME = f"{PROJECT_NUMBER} - 2_funk_80_beat_4-4_4"
PROJECT_DIR = ROOT / "user_files" / PROJECT_DIR_NAME
MIDI_DIR = PROJECT_DIR / "midi"
META_FILE = PROJECT_DIR / ".drumtomidi_project.json"
BASE_NAME = "2_funk_80_beat_4-4_4"
EXPECTED_MID = MIDI_DIR / f"{BASE_NAME}.mid"
EXPECTED_SIDECAR = MIDI_DIR / f"{BASE_NAME}.analysis.json"
CLI = ["python", "stems_to_midi_cli.py", str(PROJECT_NUMBER)]


def _reset_project() -> None:
    """Reset project 8 to a pre-CLI state so each test runs from clean."""
    if MIDI_DIR.exists():
        for child in MIDI_DIR.iterdir():
            if child.is_file():
                child.unlink()
            else:
                import shutil
                shutil.rmtree(child)
    if META_FILE.exists():
        meta = json.loads(META_FILE.read_text())
        meta.setdefault("status", {})["midi_generated"] = False
        META_FILE.write_text(json.dumps(meta, indent=2))


def _run_cli(extra_args: list[str] | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run the CLI on project 8 with optional extra args."""
    cmd = CLI + (extra_args or [])
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=timeout,
    )


def _read_sidecar() -> dict:
    """Read the analysis sidecar written by the CLI. Returns {} if missing."""
    if not EXPECTED_SIDECAR.exists():
        return {}
    return json.loads(EXPECTED_SIDECAR.read_text())


@pytest.fixture(autouse=True)
def _isolate_project():
    """Reset project 8 before and after each test."""
    if not PROJECT_DIR.exists():
        pytest.skip(f"project 8 not found at {PROJECT_DIR}")
    _reset_project()
    yield
    _reset_project()


# ─── A.1 Single-stem CLI smoke tests ──────────────────────────────────────


@pytest.mark.parametrize(
    "stem",
    ["kick", "snare", "toms", "hihat", "cymbals"],
)
@pytest.mark.slow
def test_cli_e2e_single_stem(stem: str):
    """stems_to_midi_cli.py 8 --stems <stem> should complete without error.

    The CLI writes ONE midi file (events for the requested stem only)
    and ONE sidecar.
    """
    result = _run_cli(["--stems", stem])
    assert result.returncode == 0, (
        f"CLI failed for {stem}:\n"
        f"  stdout: {result.stdout[-500:]}\n"
        f"  stderr: {result.stderr[-500:]}"
    )

    assert EXPECTED_MID.exists(), f"MIDI file not written: {EXPECTED_MID}"
    assert EXPECTED_MID.stat().st_size > 0, f"MIDI file is empty: {EXPECTED_MID}"

    assert EXPECTED_SIDECAR.exists(), (
        f"Sidecar not written: {EXPECTED_SIDECAR}"
    )

    sidecar = _read_sidecar()
    assert "stems" in sidecar, (
        f"Sidecar has no 'stems' key: keys={list(sidecar.keys())}"
    )


# ─── A.2 All-stems CLI smoke test ─────────────────────────────────────────


@pytest.mark.slow
def test_cli_e2e_all_stems():
    """stems_to_midi_cli.py 8 (no --stems arg) should process all 5 stems."""
    result = _run_cli()
    assert result.returncode == 0, (
        f"CLI failed for all-stems:\n"
        f"  stdout: {result.stdout[-500:]}\n"
        f"  stderr: {result.stderr[-500:]}"
    )

    assert EXPECTED_MID.exists(), f"MIDI file not written: {EXPECTED_MID}"
    assert EXPECTED_MID.stat().st_size > 0, f"MIDI file is empty: {EXPECTED_MID}"

    sidecar = _read_sidecar()
    assert "stems" in sidecar, (
        f"Sidecar has no 'stems' key: keys={list(sidecar.keys())}"
    )


# ─── A.3 Sidecar shape — no dead keys ────────────────────────────────────


@pytest.mark.slow
def test_cli_sidecar_no_dead_keys():
    """After the CLI runs, the analysis sidecar must NOT contain
    the keys that were removed in Phase 2/3 (geomean_threshold,
    open_geomean_min, etc.). This is the regression guard against
    accidentally re-introducing the dead surface.
    """
    result = _run_cli()
    assert result.returncode == 0, f"CLI failed: {result.stderr[-300:]}"

    sidecar_text = EXPECTED_SIDECAR.read_text()
    forbidden_keys = [
        # Phase 2 removed (per-stem thresholds)
        "geomean_threshold",
        "threshold_db",
        "min_peak_spacing_ms",
        "min_absolute_energy",
        "merge_window_ms",
        "energy_method",
        "peak_hold_ms",
        "onset_threshold",
        "onset_delta",
        "onset_wait",
        "min_strength_threshold",
        "min_sustain_ms",
        "enable_spectral_filter",
        # Phase 3 removed
        "open_geomean_min",
        "open_sustain_ms",
        "expected_clusters",
        "cluster_feature",
        # Phase 5 removed
        "events_spectral",
        # Phase 2 removed from onset_detection
        "detection_method",
    ]
    for key in forbidden_keys:
        assert f'"{key}"' not in sidecar_text, (
            f"Sidecar still contains dead key '{key}' — "
            f"Phase 5/3/2 cleanup left a writer behind"
        )


# ─── A.4 Sidecar has events for the live stems ───────────────────────────


@pytest.mark.slow
def test_cli_sidecar_kick_has_pga_events():
    """The kick stem is reliably detected by PGA in this song
    (106+ events). The sidecar must record them.
    """
    result = _run_cli()
    assert result.returncode == 0

    sidecar = _read_sidecar()
    events_pga = sidecar.get("stems", {}).get("kick", {}).get("events_pga", [])
    assert len(events_pga) > 0, (
        f"events_pga.kick is empty — PGA found events but sidecar lost them. "
        f"Sidecar stems: {list(sidecar.get('stems', {}).keys())}"
    )


# ─── A.5 Project metadata update ─────────────────────────────────────────


@pytest.mark.slow
def test_cli_updates_project_status():
    """After the CLI runs, project 8's metadata should show
    midi_generated: true (the CLI's update_project_metadata call)."""
    result = _run_cli()
    assert result.returncode == 0

    meta = json.loads(META_FILE.read_text())
    assert meta["status"]["midi_generated"] is True, (
        f"CLI did not flip midi_generated: status={meta['status']}"
    )
