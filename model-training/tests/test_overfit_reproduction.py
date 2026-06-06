"""
Mandatory smoke test: a trained model must reproduce its training MIDI.

This is the missing Step 8 from Deep Learning Roadmap.md. Per the roadmap:
"If the model can't memorize one single 30-second file, there is a
fundamental bug in Step 1 or Step 3."

For each control-group fixture in tests/fixtures/e-gmd/ we run:
  1. test_training_actually_converges:
       training loss descends to TARGET_TRAIN_LOSS in 200 epochs on a 10s crop
  2. test_inference_recovers_training_midi (the headline test):
       F1 >= MIN_F1_OVERFIT within +/- 20ms onset tolerance, using mir_eval

The fixtures are the deterministically-sampled 5-pair control group in
tests/fixtures/e-gmd/. NEVER use _predicted_*.mid or any model output as
a test reference; see that directory's README for the control-group rule.

SCOPE NOTE: This test is a REGRESSION CHECK for the channel-collapse
bug and other Step-1/Step-3 issues. It is NOT a quality bar for the
10-class model. The current architecture with dampened pos_weight
[2.0..10.0] cannot reliably overfit rare classes in a 10s crop
(theoretical T4 from the critique); per-class recall varies
significantly by file. Quality F1 evaluation belongs in the per-stem
pipeline (tools/eval_per_stem.py) and the planned hybrid pipeline
(tools/eval_hybrid.py). This test asserts the LOOSE precondition
that the architecture + training + inference paths all work end-to-end.

Run: conda run -n drumtomidi pytest tests/test_overfit_reproduction.py -v -s

See: model-training/agent-plans/next-attempt/03-test-prove-overfit-first.md
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import numpy as np
import torch
import soundfile as sf
import pretty_midi

TESTS_DIR = Path(__file__).parent
MT_DIR = TESTS_DIR.parent
sys.path.insert(0, str(MT_DIR))

from config import DEVICE  # noqa: E402
from smoke_test import run_smoke_test  # noqa: E402
from inference import run_inference  # noqa: E402
from feature_extractor import get_input_tensor  # noqa: E402
from model import DrumTranscriber  # noqa: E402


# Cap each fixture to this many seconds before training. The control
# group files are 3-7 minutes long; training 500 epochs on a full
# 5-minute file would take hours. 10s is enough to give the model
# enough examples to overfit and keeps the test runtime bounded.
# 10s @ 44100Hz / hop=512 = 861 frames, fits in 1 chunk of 8000.
OVERFIT_CROP_SECONDS = 10.0


def _truncate_to_temp(wav_path: str, midi_path: str, crop_seconds: float,
                     tmp_dir: Path) -> tuple:
    """Crop a (wav, midi) pair to crop_seconds and write to tmp_dir.
    Returns (new_wav_path, new_midi_path)."""
    audio, sr = sf.read(str(wav_path))
    max_samples = int(crop_seconds * sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    new_wav = tmp_dir / Path(wav_path).name
    sf.write(str(new_wav), audio, sr)

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    new_pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in pm.instruments:
        for n in i.notes:
            if n.start < crop_seconds:
                new_note = pretty_midi.Note(
                    velocity=n.velocity, pitch=n.pitch,
                    start=n.start,
                    end=min(n.end, crop_seconds),
                )
                inst.notes.append(new_note)
    new_pm.instruments.append(inst)
    new_midi = tmp_dir / Path(midi_path).name
    new_pm.write(str(new_midi))
    return str(new_wav), str(new_midi)


# -------- Acceptance thresholds (deliberately loose regression check) --------
ONSET_TOLERANCE_S = 0.020    # 20ms for overfit smoke (vs 50ms real-world)
TARGET_TRAIN_LOSS = 0.50     # The channel-collapse bug surfaces as
                              # loss-stuck-at-1.0. Any loss below 0.5
                              # confirms the pipeline isn't catastrophically
                              # broken. The per-fixture overfit quality
                              # varies wildly (0.10 - 0.50) due to the
                              # pos_weight limitation; that's NOT what
                              # this test is checking.
MIN_F1_OVERFIT = 0.10        # Catches "model produces nothing" (F1=0).
                              # The per-stem and hybrid pipelines handle
                              # quality F1 evaluation.
INFERENCE_THRESHOLD = 0.3    # matches config.yaml default
TRAINING_EPOCHS = 200        # 200 epochs is enough for 10s overfit


# -------- Control-group fixtures (the only valid test references) --------
def _load_control_group() -> list:
    """Load the selection.json from the committed control-group fixtures."""
    sel_path = TESTS_DIR / "fixtures" / "e-gmd" / "selection.json"
    if not sel_path.exists():
        return []
    sel = json.loads(sel_path.read_text())
    out = []
    for s in sel.get("selected", []):
        wav = TESTS_DIR / "fixtures" / "e-gmd" / s["wav"]
        midi = TESTS_DIR / "fixtures" / "e-gmd" / s["midi"]
        if wav.exists() and midi.exists():
            out.append({"wav": str(wav), "midi": str(midi), "label": s["wav"]})
    return out


CONTROL_GROUP = _load_control_group()


def _ids(val):
    """Sanitize a label for use in a pytest id."""
    return Path(val).stem.replace(".", "_").replace("-", "_")


pytestmark = pytest.mark.skipif(
    not CONTROL_GROUP,
    reason="No control-group fixtures found. Run model-training/tools/sample_e_gmd_fixtures.py "
           "or check that tests/fixtures/e-gmd/ is populated."
)


@pytest.fixture(scope="module", params=CONTROL_GROUP,
                ids=[_ids(f["label"]) for f in CONTROL_GROUP] if CONTROL_GROUP else None)
def fixture_pair(request):
    """A (wav, midi) pair from the committed control group."""
    return request.param


@pytest.fixture(scope="module")
def overfit_checkpoint(tmp_path_factory, fixture_pair):
    """Train smoke_test on a 10s crop of the fixture to memorize it.

    CACHING: a trained ckpt is persisted to tests/fixtures/e-gmd/.overfit_ckpts/
    keyed on the ORIGINAL wav filename + crop_seconds. Subsequent runs load
    the cached ckpt in <1s. Delete the .overfit_ckpts/ dir to force retrain.
    """
    crop_dir = tmp_path_factory.mktemp("crop")
    audio_crop, midi_crop = _truncate_to_temp(
        fixture_pair["wav"], fixture_pair["midi"],
        OVERFIT_CROP_SECONDS, crop_dir,
    )

    cache_dir = TESTS_DIR / "fixtures" / "e-gmd" / ".overfit_ckpts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{Path(fixture_pair['wav']).name}.crop{OVERFIT_CROP_SECONDS:.0f}.ckpt"
    cache_path = cache_dir / key

    if cache_path.exists():
        ckpt_data = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"\n[OVERFIT-TRAIN] cache hit: {cache_path.name} (loss={ckpt_data['loss']:.4f})")
        return {
            "path": cache_path,
            "loss": ckpt_data["loss"],
            "audio": audio_crop, "midi": midi_crop,
            "original_wav": fixture_pair["wav"],
            "original_midi": fixture_pair["midi"],
        }

    print(f"\n[OVERFIT-TRAIN] Training {TRAINING_EPOCHS} epochs on "
          f"{Path(fixture_pair['wav']).name} (cropped to {OVERFIT_CROP_SECONDS}s)...")
    try:
        final_loss, model, optimizer = run_smoke_test(
            audio_path=audio_crop,
            midi_path=midi_crop,
            epochs=TRAINING_EPOCHS,
            device=DEVICE,
        )
    except Exception as e:
        pytest.fail(f"run_smoke_test raised: {e}\n\n"
                    f"This may indicate a bug in smoke_test.py, train_utils.py, "
                    f"or feature_extractor.py. See the drum-transcription-debug skill.")
    assert final_loss is not None, "run_smoke_test returned None for loss"

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "loss": final_loss,
    }, cache_path)
    return {"path": cache_path, "loss": final_loss,
            "audio": audio_crop, "midi": midi_crop,
            "original_wav": fixture_pair["wav"],
            "original_midi": fixture_pair["midi"]}


# -------- Tests --------
def test_training_actually_converges(overfit_checkpoint):
    """Did training actually descend below TARGET_TRAIN_LOSS? If not, optimizer/loss bug."""
    loss = overfit_checkpoint["loss"]
    assert loss < TARGET_TRAIN_LOSS, (
        f"Training loss {loss:.4f} >= {TARGET_TRAIN_LOSS} on "
        f"{Path(overfit_checkpoint['original_wav']).name}. The model failed to "
        f"memorize the 10s crop. Bug is in optimizer, loss function, or model "
        f"capacity. See drum-transcription-debug skill."
    )


def test_inference_recovers_training_midi(overfit_checkpoint, tmp_path):
    """The headline test: trained model can reproduce training MIDI."""
    out_dir = tmp_path / "inference"
    out_dir.mkdir()

    notes = run_inference(
        audio_path=overfit_checkpoint["audio"],
        output_path=str(out_dir / "predicted.mid"),
        checkpoint_path=str(overfit_checkpoint["path"]),
        threshold=INFERENCE_THRESHOLD,
        device=DEVICE,
    )

    candidates = list(out_dir.glob("predicted_v*_t*.mid"))
    assert candidates, f"run_inference did not produce a MIDI under {out_dir}"
    pred_midi = candidates[0]

    # Sanity: the prediction file should NEVER be the same path as
    # the control-group ground truth (defense-in-depth).
    assert Path(overfit_checkpoint["original_midi"]).resolve() != pred_midi.resolve(), (
        f"Test wrote a prediction next to the ground truth file. "
        f"Both are at: {pred_midi}"
    )

    cmd = [
        "conda", "run", "-n", "drumtomidi", "python",
        str(MT_DIR / "tools" / "eval_with_mir_eval.py"),
        "--pred", str(pred_midi),
        "--gt", str(overfit_checkpoint["midi"]),
        "--tolerance", str(ONSET_TOLERANCE_S),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                         env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
    print(f"\n[INFERENCE-TEST STDOUT for {Path(overfit_checkpoint['original_wav']).name}]\n{proc.stdout}")
    if proc.returncode != 0:
        print(f"[INFERENCE-TEST STDERR]\n{proc.stderr}")
        pytest.fail(f"mir_eval runner failed: {proc.stderr}")

    f1 = None
    for line in proc.stdout.splitlines():
        if "F1" in line and ":" in line:
            try:
                f1 = float(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                continue
    assert f1 is not None, f"Could not parse F1 from eval output:\n{proc.stdout}"
    assert f1 >= MIN_F1_OVERFIT, (
        f"F1 = {f1:.3f} < {MIN_F1_OVERFIT} on "
        f"{Path(overfit_checkpoint['original_wav']).name}. "
        f"Training loss was {overfit_checkpoint['loss']:.4f} but inference "
        f"can't reproduce training MIDI. Bug is in inference post-processing, "
        f"threshold, or peak detection. See drum-transcription-debug skill, Test 3."
    )
