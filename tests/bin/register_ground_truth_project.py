#!/usr/bin/env python3
"""Register the ground-truth project (tests/assets/2_funk_80_beat_4-4_4.aif) as
a real project under user_files/. Records the assigned project number to
tests/ground_truth_project.txt so the e2e test can find it without re-registering.

Idempotent: if the project is already registered, prints the existing number
and exits 0. To re-register from scratch, delete the project directory first.

Usage:
    conda run -n drumtomidi python scripts/register_ground_truth_project.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
ASSET_AIF = ROOT / "tests" / "assets" / "2_funk_80_beat_4-4_4.aif"
USER_FILES = ROOT / "user_files"
MARKER_FILE = ROOT / "tests" / "ground_truth_project.txt"


def find_existing_project() -> int | None:
    """Return the project number if the 2_funk_80_beat_4-4_4 project already exists."""
    if not USER_FILES.exists():
        return None
    for entry in USER_FILES.iterdir():
        if not entry.is_dir():
            continue
        # Match "N - 2_funk_80_beat_4-4_4" pattern
        name = entry.name
        if " - 2_funk_80_beat_4-4_4" in name and entry.name.startswith(("4 -", "5 -", "6 -", "7 -", "8 -", "9 -")):
            try:
                num = int(name.split(" - ", 1)[0])
                return num
            except ValueError:
                continue
    return None


def main():
    if not ASSET_AIF.exists():
        print(f"error: ground-truth asset not found at {ASSET_AIF}", file=sys.stderr)
        sys.exit(2)

    # Check for existing registration
    existing = find_existing_project()
    if existing is not None:
        print(f"ground-truth project already registered: #{existing}")
        _write_marker(existing)
        sys.exit(0)

    # Copy the asset into user_files/ root (create_project requires the file
    # to be in user_files_dir, NOT in a subdirectory).
    USER_FILES.mkdir(exist_ok=True)
    staging = USER_FILES / ASSET_AIF.name
    if not staging.exists():
        shutil.copy(ASSET_AIF, staging)
        print(f"copied asset to {staging}")

    # Import project_manager after staging the file (it adds cwd to sys.path)
    sys.path.insert(0, str(ROOT))
    from project_manager import create_project

    project_info = create_project(staging)
    project_num = project_info["project_number"]
    print(f"registered ground-truth project: #{project_num}")
    print(f"  path: {project_info['path']}")

    # Clean up the staging copy (create_project moved the file into the project dir)
    if staging.exists():
        staging.unlink()

    _write_marker(project_num)
    print(f"wrote marker to {MARKER_FILE}")


def _write_marker(project_num: int) -> None:
    MARKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    MARKER_FILE.write_text(
        f"{project_num}\n"
        f"# Ground-truth project number (2_funk_80_beat_4-4_4.aif).\n"
        f"# Written by scripts/register_ground_truth_project.py on\n"
        f"# every run. Used by tests/test_ground_truth_e2e.py to\n"
        f"# locate the registered project without re-registering.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
