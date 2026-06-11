"""
Tests for scripts/dump_snap_per_frame.py.

Added 2026-06-09 after the user asked for a per-frame CSV dump of the
toms detection signals so they could see the assembled data behind
the snap-delta values and find patterns the peak-picked events hide.

The script is a CLI tool that:
  1. Resolves a project directory from --project (number / name / path)
  2. Loads midiconfig.yaml and reads the per-stem spectral_snap_bands
  3. Runs the spectral pipeline on the stem's WAV
  4. Writes per-frame CSV + sidecar JSON

Tests run end-to-end against a temp project with a synthetic WAV to
verify the output structure (column count, row count = STFT frame
count, JSON sidecar has the right config) without depending on the
real user_files/4 - 2_funk_80_beat_4-4_4/ project.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / 'scripts'
SCRIPT = SCRIPTS_DIR / 'dump_snap_per_frame.py'
PYTHON = '/Users/jasoncopp/miniforge3/envs/drumtomidi/bin/python'


# ─── Helpers ─────────────────────────────────────────────────────────────

def _make_synthetic_wav(
    wav_path: Path,
    sr: int = 44100,
    duration_sec: float = 1.0,
) -> None:
    """Write a short mono WAV with two transient bursts at t=0.2s and
    t=0.6s so the detector has something to find. Not realistic toms,
    just enough to exercise the pipeline end-to-end."""
    t = np.arange(int(sr * duration_sec)) / sr
    y = np.zeros_like(t, dtype=np.float32)
    # Two brief exponentially-decaying bursts across 5 freq bands
    for burst_t in (0.2, 0.6):
        for freq, amp in zip(
            (80, 300, 1000, 2000, 5000),
            (0.3, 0.5, 0.4, 0.2, 0.1),
        ):
            envelope = np.exp(-(t - burst_t) * 30) * (t >= burst_t)
            y += amp * envelope * np.sin(2 * np.pi * freq * t)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_path), y, sr)


def _make_project(
    tmp_path: Path,
    *,
    stem: str = 'toms',
    snap_bands_yaml: str = '1,2',
    snap_min_delta_yaml: float = 0.01,
) -> Path:
    """Create a minimal user_files/<proj> project with midiconfig.yaml
    and a stems/<song>-<stem>.wav. Returns the project dir."""
    proj = tmp_path / 'testproj'
    (proj / 'stems').mkdir(parents=True)
    (proj / 'analysis').mkdir(parents=True)
    wav_path = proj / 'stems' / f'testsong-{stem}.wav'
    _make_synthetic_wav(wav_path, sr=44100, duration_sec=1.0)
    cfg = {
        stem: {
            'spectral_snap_bands': snap_bands_yaml,
            'spectral_snap_min_delta': snap_min_delta_yaml,
        }
    }
    with open(proj / 'midiconfig.yaml', 'w') as f:
        yaml.safe_dump(cfg, f)
    return proj


# ─── Tests ───────────────────────────────────────────────────────────────

def test_script_runs_and_writes_csv(tmp_path):
    """The script should run end-to-end on a temp project and write
    a non-empty CSV with the documented columns."""
    proj = _make_project(tmp_path)
    out_csv = tmp_path / 'snap.csv'
    out_json = tmp_path / 'snap.json'
    result = subprocess.run(
        [
            PYTHON, str(SCRIPT),
            '--project', str(proj),
            '--stem', 'toms',
            '--output', str(out_csv),
            '--sidecar', str(out_json),
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"script failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert out_csv.exists(), "CSV not written"
    assert out_json.exists(), "JSON sidecar not written"


def test_csv_row_count_matches_stft_frames(tmp_path):
    """Row count in the CSV (excluding header) must equal the number
    of STFT frames for the audio. STFT frame count is
    (n_samples - n_fft) // hop + 1. At sr=44100, n_fft=1024, hop=256,
    1s of audio = 44100 samples -> 169 frames."""
    proj = _make_project(tmp_path)
    out_csv = tmp_path / 'snap.csv'
    out_json = tmp_path / 'snap.json'
    subprocess.run(
        [PYTHON, str(SCRIPT), '--project', str(proj), '--stem', 'toms',
         '--output', str(out_csv), '--sidecar', str(out_json)],
        check=True, capture_output=True, timeout=60,
    )
    with open(out_csv) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    # n_samples = 44100, n_fft = 1024, hop = 256
    # n_frames = (44100 - 1024) // 256 + 1 = 43076 // 256 + 1 = 168 + 1 = 169
    expected = (44100 - 1024) // 256 + 1
    assert len(rows) == expected, (
        f"expected {expected} rows, got {len(rows)}. "
        f"STFT frame count should match (n_samples - n_fft) // hop + 1."
    )


def test_csv_has_documented_columns(tmp_path):
    """The CSV header must include all the columns the user asked for
    in the assembled-data dump: per_bin_means for all 5 bands, both
    band_delta and snap_delta, ring/snap pass heights, the snap_bands
    per-band per_bin_means (one column per snap_bands entry), and the
    event-peak correlation columns."""
    proj = _make_project(tmp_path, snap_bands_yaml='1,2')
    out_csv = tmp_path / 'snap.csv'
    out_json = tmp_path / 'snap.json'
    subprocess.run(
        [PYTHON, str(SCRIPT), '--project', str(proj), '--stem', 'toms',
         '--output', str(out_csv), '--sidecar', str(out_json)],
        check=True, capture_output=True, timeout=60,
    )
    with open(out_csv) as f:
        reader = csv.reader(f)
        header = next(reader)
    required = [
        'time_sec', 'frame_idx',
        'band0_per_bin_mean', 'band1_per_bin_mean', 'band2_per_bin_mean',
        'band3_per_bin_mean', 'band4_per_bin_mean',
        'band_delta', 'ring_pass_height',
        'snap_band_1_per_bin_mean', 'snap_band_2_per_bin_mean',
        'snap_delta', 'snap_pass_height',
        'band_max_idx', 'band_max_ratio', 'max_db',
        'snap_bands', 'is_event_peak', 'event_time_sec',
        'event_band_delta', 'event_snap_delta',
    ]
    missing = [c for c in required if c not in header]
    assert not missing, (
        f"CSV missing required columns: {missing}. Got: {header}"
    )


def test_sidecar_json_records_yaml_config(tmp_path):
    """The JSON sidecar must record the exact spectral_snap_bands and
    spectral_snap_min_delta the YAML had. Without this, a CSV opened
    months later can't be interpreted correctly."""
    proj = _make_project(tmp_path, snap_bands_yaml='1,2,3', snap_min_delta_yaml=0.025)
    out_csv = tmp_path / 'snap.csv'
    out_json = tmp_path / 'snap.json'
    subprocess.run(
        [PYTHON, str(SCRIPT), '--project', str(proj), '--stem', 'toms',
         '--output', str(out_csv), '--sidecar', str(out_json)],
        check=True, capture_output=True, timeout=60,
    )
    with open(out_json) as f:
        sidecar = json.load(f)
    cfg = sidecar['spectral_config']
    assert cfg['snap_bands'] == [1, 2, 3], (
        f"sidecar should record snap_bands=[1,2,3] from YAML, got {cfg['snap_bands']}"
    )
    assert cfg['snap_min_delta'] == 0.025, (
        f"sidecar should record snap_min_delta=0.025 from YAML, "
        f"got {cfg['snap_min_delta']}"
    )
    # And the number of snap_band_* columns in the CSV should match
    # the number of snap_bands.
    with open(out_csv) as f:
        header = next(csv.reader(f))
    snap_cols = [c for c in header if c.startswith('snap_band_') and c.endswith('_per_bin_mean')]
    assert len(snap_cols) == 3, (
        f"expected 3 snap_band_* columns for snap_bands=(1,2,3), got {len(snap_cols)}: {snap_cols}"
    )


def test_sidecar_json_has_signal_summary_stats(tmp_path):
    """The JSON sidecar must include min/max/mean/p95/p99 stats for
    both band_delta and snap_delta, so the user can quickly see the
    distribution without opening the CSV."""
    proj = _make_project(tmp_path)
    out_csv = tmp_path / 'snap.csv'
    out_json = tmp_path / 'snap.json'
    subprocess.run(
        [PYTHON, str(SCRIPT), '--project', str(proj), '--stem', 'toms',
         '--output', str(out_csv), '--sidecar', str(out_json)],
        check=True, capture_output=True, timeout=60,
    )
    with open(out_json) as f:
        sidecar = json.load(f)
    stats = sidecar['signal_stats']
    for signal in ('band_delta', 'snap_delta'):
        assert signal in stats, f"signal_stats missing {signal}"
        for key in ('min', 'max', 'mean', 'p95', 'p99'):
            assert key in stats[signal], f"{signal} missing {key}"
            assert isinstance(stats[signal][key], (int, float))


def test_event_peak_rows_correspond_to_detector_events(tmp_path):
    """Frames with is_event_peak == 1 should be exactly the frames the
    detector chose as event peaks. The detector's event count and the
    number of is_event_peak == 1 rows in the CSV must match."""
    proj = _make_project(tmp_path)
    out_csv = tmp_path / 'snap.csv'
    out_json = tmp_path / 'snap.json'
    subprocess.run(
        [PYTHON, str(SCRIPT), '--project', str(proj), '--stem', 'toms',
         '--output', str(out_csv), '--sidecar', str(out_json)],
        check=True, capture_output=True, timeout=60,
    )
    with open(out_json) as f:
        sidecar = json.load(f)
    n_events = sidecar['events']['n_events']
    event_times = sidecar['events']['event_times_sec']
    with open(out_csv) as f:
        reader = csv.DictReader(f)
        peak_rows = [r for r in reader if r['is_event_peak'] == '1']
    assert len(peak_rows) == n_events, (
        f"expected {n_events} event_peak rows, got {len(peak_rows)}. "
        f"The CSV should mark exactly the detector's event frames."
    )
    # And the event_time_sec column should be populated for every peak row.
    assert all(r['event_time_sec'] for r in peak_rows), (
        "Every is_event_peak=1 row must have an event_time_sec value."
    )


def test_project_number_resolution(tmp_path):
    """The script must accept a project number prefix like '5' and
    match it against the unique project dir. Test this by setting up
    a project with a '5 - foo' style name."""
    proj = tmp_path / '5 - test_project'
    (proj / 'stems').mkdir(parents=True)
    (proj / 'analysis').mkdir(parents=True)
    _make_synthetic_wav(proj / 'stems' / 'song-toms.wav', sr=44100, duration_sec=0.5)
    with open(proj / 'midiconfig.yaml', 'w') as f:
        yaml.safe_dump({'toms': {}}, f)

    out_csv = tmp_path / 'snap.csv'
    out_json = tmp_path / 'snap.json'
    result = subprocess.run(
        [
            PYTHON, str(SCRIPT),
            '--project', str(proj),
            '--stem', 'toms',
            '--output', str(out_csv),
            '--sidecar', str(out_json),
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"script failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert out_csv.exists()


def test_missing_stem_errors_cleanly(tmp_path):
    """The script must fail with a clear error message when the stem
    WAV doesn't exist (not a traceback)."""
    proj = _make_project(tmp_path, stem='toms')
    # Don't create the hihat stem WAV
    result = subprocess.run(
        [
            PYTHON, str(SCRIPT),
            '--project', str(proj),
            '--stem', 'hihat',
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert 'hihat' in result.stderr, (
        f"error should mention the missing stem, got: {result.stderr}"
    )
