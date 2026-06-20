"""Shared pytest fixtures for the tests/ package.

The ground-truth project (tests/assets/2_funk_80_beat_4-4_4.aif + .midi) is
registered as a real user_files project by scripts/register_ground_truth_project.py.
The marker file tests/ground_truth_project.txt records the assigned number.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# Repo root, computed relative to this conftest.py.
# tests/conftest.py → tests/ → repo root is parent.parent
TESTS_DIR = Path(__file__).parent
REPO_ROOT = TESTS_DIR.parent
ASSETS_DIR = TESTS_DIR / "assets"
MARKER_FILE = TESTS_DIR / "ground_truth_project.txt"


def _read_marker_project_number() -> int:
    """Read the project number from tests/ground_truth_project.txt."""
    if not MARKER_FILE.exists():
        raise RuntimeError(
            f"Marker file {MARKER_FILE} not found. Run "
            f"`python scripts/register_ground_truth_project.py` first."
        )
    content = MARKER_FILE.read_text(encoding="utf-8")
    first_line = content.splitlines()[0].strip()
    return int(first_line)


@pytest.fixture(scope="session")
def ground_truth_project_number() -> int:
    """The user_files project number assigned to the ground-truth AIFF."""
    return _read_marker_project_number()


@pytest.fixture(scope="session")
def ground_truth_project_dir(ground_truth_project_number: int) -> Path:
    """Path to the registered ground-truth project directory."""
    return REPO_ROOT / "user_files" / f"{ground_truth_project_number} - 2_funk_80_beat_4-4_4"


@pytest.fixture(scope="session")
def ground_truth_sidecar(ground_truth_project_dir: Path) -> dict:
    """The project's analysis.json sidecar as a Python dict.

    Returns an empty dict (with a 'stems' key) if the sidecar doesn't
    exist yet — the e2e test then reports that no data has been
    produced and asserts the test should re-run the pipeline.
    """
    sidecar_path = ground_truth_project_dir / "midi" / "2_funk_80_beat_4-4_4.analysis.json"
    if not sidecar_path.exists():
        return {"stems": {}}
    with open(sidecar_path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def ground_truth_midi_notes() -> dict:
    """Ground-truth per-stem note counts from tests/assets/2_funk_80_beat_4-4_4.midi.

    Returns a dict mapping stem type (rough mapping: kick / snare / hihat /
    toms / cymbals) to an integer count. The mapping is approximate —
    the ground-truth MIDI uses General MIDI numbers 22-55 across the 5
    stems, and we group them by typical drum mapping.
    """
    try:
        import mido
    except ImportError:
        pytest.skip("mido not available")
    midi_path = ASSETS_DIR / "2_funk_80_beat_4-4_4.midi"
    if not midi_path.exists():
        pytest.skip(f"ground truth MIDI not found at {midi_path}")
    mid = mido.MidiFile(str(midi_path))
    counts: dict[str, int] = {"kick": 0, "snare": 0, "hihat": 0, "toms": 0, "cymbals": 0}
    for track in mid.tracks:
        for msg in track:
            if msg.type != "note_on" or msg.velocity == 0:
                continue
            n = msg.note
            # General MIDI drum map grouping (rough):
            #   35-36: kick    38: snare    42: closed hh  46: open hh
            #   44: foot hh   37: rimshot  43, 50: toms
            #   49: crash     51: ride     52: china
            #   22-26: misc cymbals
            if n in (35, 36):
                counts["kick"] += 1
            elif n in (38, 37, 40, 39):
                counts["snare"] += 1
            elif n in (42, 44, 46):
                counts["hihat"] += 1
            elif n in (41, 43, 45, 47, 48, 50):
                counts["toms"] += 1
            else:
                counts["cymbals"] += 1
    return counts
