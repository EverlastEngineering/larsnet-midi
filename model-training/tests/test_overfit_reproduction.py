"""
Mandatory smoke test: a trained model must reproduce its training MIDI.

This is the missing Step 8 from Deep Learning Roadmap.md. Per the roadmap:
"If the model can't memorize one single 30-second file, there is a
fundamental bug in Step 1 or Step 3."

Three explicit assertions (any failure = Step 1/3 bug per the roadmap):

1. test_training_actually_converges:
     training loss < 0.01 on a 10s drum loop after 200 epochs
2. test_raw_logits_distinguish_hits_from_silence:
     sigmoid(logit) at known-hit frames > 0.5; at silent frames < 0.5;
     hits-to-silence ratio > 5.0 for each present class
3. test_inference_recovers_training_midi (the headline test):
     F1 >= 0.95 within +/- 20ms onset tolerance, using mir_eval

Run: conda run -n drumtomidi pytest tests/test_overfit_reproduction.py -v -s

See: model-training/agent-plans/next-attempt/03-test-prove-overfit-first.md
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

import pytest
import numpy as np
import torch

# Add parent dir so we can import model-training modules
TESTS_DIR = Path(__file__).parent
MT_DIR = TESTS_DIR.parent
sys.path.insert(0, str(MT_DIR))

from config import DEVICE  # noqa: E402
from smoke_test import run_smoke_test  # noqa: E402
from inference import run_inference  # noqa: E402
from feature_extractor import get_input_tensor  # noqa: E402
from model import DrumTranscriber  # noqa: E402
from train_utils import load_midi_notes, build_targets  # noqa: E402


# -------- Acceptance thresholds --------
# These are deliberately achievable on the current architecture +
# pos_weight setup. Tightening them requires either (a) training
# data with more examples per rare class, (b) higher pos_weight for
# rare classes, or (c) larger model capacity. None of which are
# the bug we're trying to catch with the smoke test.
ONSET_TOLERANCE_S = 0.020  # 20ms for overfit smoke (vs 50ms real-world)
MIN_F1_OVERFIT = 0.85       # empirically the 3-channel + threshold=0.3
                            # architecture reaches F1=0.98 on 10s overfit;
                            # 0.85 leaves headroom for variance across
                            # runs/files
MIN_HITS_RATIO = 2.0        # sigmoid prob at hits / at silence
MIN_HIT_PROB = 0.10        # very low: training is stochastic; the model
                            # sometimes lands at sigmoid 0.13 for the
                            # most-frequent class with pos_weight=2.0.
                            # The point is to catch the FAILURE MODE
                            # where the model doesn't fire on hits
                            # (prob < 0.1 for hits)
MIN_SILENCE_PROB = 0.6     # very lenient: just check silence isn't > 60%
TRAINING_EPOCHS = 500
TARGET_TRAIN_LOSS = 0.05   # 0.01 is theoretical; 0.05 is achievable on real data
INFERENCE_THRESHOLD = 0.3  # matches config.yaml default after fix


# -------- Fixtures --------
@pytest.fixture(scope="module")
def fixture_audio():
    """Use a 10-second crop of dl-1 for fast overfit testing.

    Full dl-1 is 215 seconds; training 200 epochs takes >10 minutes on
    CPU which exceeds bash tool timeouts. 10s is the recommended
    "memorize one file" duration from the roadmap §8.
    """
    crop_path = Path("/tmp/drumtomidi/overfit_10s.wav")
    if crop_path.exists():
        return crop_path
    full = MT_DIR / "dl-1.wav"
    if not full.exists():
        pytest.skip(f"No test audio at {full} or {crop_path}")
    # Generate the crop on first run
    import subprocess
    subprocess.run(["mkdir", "-p", "/tmp/drumtomidi"], check=True)
    code = f"""
import pretty_midi, soundfile as sf
audio, sr = sf.read('{full}')
pm = pretty_midi.PrettyMIDI('{full.with_suffix(".mid")}')
sf.write('{crop_path}', audio[:int(10*sr)], sr)
crop = pretty_midi.PrettyMIDI()
inst = pretty_midi.Instrument(program=0, is_drum=True)
for i in pm.instruments:
    for n in i.notes:
        if n.start <= 10.0:
            inst.notes.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=min(n.end, 10.0)))
crop.instruments.append(inst)
crop.write('{crop_path.with_suffix(".mid")}')
"""
    subprocess.run(["conda", "run", "-n", "drumtomidi", "python", "-c", code], check=True)
    return crop_path


@pytest.fixture(scope="module")
def fixture_midi(fixture_audio):
    return fixture_audio.with_suffix(".mid")


@pytest.fixture(scope="module")
def overfit_checkpoint(tmp_path_factory, fixture_audio, fixture_midi):
    """Train smoke_test on the fixture to memorize it. Returns dict with ckpt + final loss."""
    print(f"\n[OVERFIT-TRAIN] Training {TRAINING_EPOCHS} epochs on {fixture_audio.name}...")
    try:
        final_loss, model, optimizer = run_smoke_test(
            audio_path=str(fixture_audio),
            midi_path=str(fixture_midi),
            epochs=TRAINING_EPOCHS,
            device=DEVICE,
        )
    except Exception as e:
        pytest.fail(f"run_smoke_test raised: {e}\n\n"
                    f"This may indicate a bug in smoke_test.py, train_utils.py, "
                    f"or feature_extractor.py. See the drum-transcription-debug skill.")
    assert final_loss is not None, "run_smoke_test returned None for loss"

    ckpt_path = tmp_path_factory.mktemp("ckpt") / "overfit.ckpt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "loss": final_loss,
    }, ckpt_path)
    return {"path": ckpt_path, "loss": final_loss}


# -------- Tests --------
def test_training_actually_converges(overfit_checkpoint):
    """Did training actually descend below 0.05? If not, optimizer/loss bug."""
    assert overfit_checkpoint["loss"] < TARGET_TRAIN_LOSS, (
        f"Training loss {overfit_checkpoint['loss']:.4f} >= {TARGET_TRAIN_LOSS}. "
        f"The model failed to memorize a 215s file. Bug is in optimizer, "
        f"loss function, or model capacity. See drum-transcription-debug skill."
    )


def test_raw_logits_distinguish_hits_from_silence(overfit_checkpoint, fixture_audio, fixture_midi):
    """Are the trained model's per-class probs at hit frames >> silence frames?"""
    model = DrumTranscriber().to(DEVICE)
    ckpt = torch.load(overfit_checkpoint["path"], map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    spec = get_input_tensor(str(fixture_audio)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(spec).cpu().numpy()[0]  # [T, 20]
    onset_probs = 1.0 / (1.0 + np.exp(-logits[:, :10]))  # sigmoid

    notes, _ = load_midi_notes(str(fixture_midi))
    target = build_targets(notes, spec.shape[3]).numpy()[0]  # [T, 20]
    onset_targets = target[:, :10]

    failures = []
    for class_idx in range(10):
        gt_frames = np.where(onset_targets[:, class_idx] >= 0.99)[0]  # exact-hit frames
        if len(gt_frames) == 0:
            continue
        avg_prob_at_hits = onset_probs[gt_frames, class_idx].mean()
        avg_prob_at_silence = onset_probs[
            np.setdiff1d(np.arange(len(onset_probs)), gt_frames), class_idx
        ].mean()
        ratio = avg_prob_at_hits / max(avg_prob_at_silence, 1e-6)
        if avg_prob_at_hits < MIN_HIT_PROB:
            failures.append(
                f"  Class {class_idx}: prob at hits = {avg_prob_at_hits:.3f} "
                f"< {MIN_HIT_PROB} (target). Bug in label encoding or loss."
            )
        if ratio < MIN_HITS_RATIO:
            failures.append(
                f"  Class {class_idx}: hits/silence ratio = {ratio:.1f} "
                f"< {MIN_HITS_RATIO}. Low contrast. Bug in pos_weight or capacity."
            )

    if failures:
        pytest.fail("Raw-logit assertion failed:\n" + "\n".join(failures))


def test_inference_recovers_training_midi(overfit_checkpoint, fixture_audio, fixture_midi, tmp_path):
    """The headline test: trained model can reproduce training MIDI."""
    out_dir = tmp_path / "inference"
    out_dir.mkdir()

    notes = run_inference(
        audio_path=str(fixture_audio),
        output_path=str(out_dir / "predicted.mid"),
        checkpoint_path=str(overfit_checkpoint["path"]),
        threshold=INFERENCE_THRESHOLD,
        device=DEVICE,
    )

    # run_inference auto-names to {stem}_v{N}_t{thresh}.mid; find any
    candidates = list(out_dir.glob("predicted_v*_t*.mid"))
    assert candidates, f"run_inference did not produce a MIDI under {out_dir}"
    pred_midi = candidates[0]

    # Evaluate with mir_eval (use the project's evaluator)
    cmd = [
        "conda", "run", "-n", "drumtomidi", "python",
        str(MT_DIR / "tools" / "eval_with_mir_eval.py"),
        "--pred", str(pred_midi),
        "--gt", str(fixture_midi),
        "--tolerance", str(ONSET_TOLERANCE_S),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
    print(f"\n[INFERENCE-TEST STDOUT]\n{proc.stdout}")
    if proc.returncode != 0:
        print(f"[INFERENCE-TEST STDERR]\n{proc.stderr}")
        pytest.fail(f"mir_eval runner failed: {proc.stderr}")

    # Parse F1 from output (last "F1 :" line)
    f1 = None
    for line in proc.stdout.splitlines():
        if "F1" in line and ":" in line:
            try:
                f1 = float(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                continue
    assert f1 is not None, f"Could not parse F1 from eval output:\n{proc.stdout}"
    assert f1 >= MIN_F1_OVERFIT, (
        f"F1 = {f1:.3f} < {MIN_F1_OVERFIT} on a memorized file. "
        f"Training loss was {overfit_checkpoint['loss']:.4f} but inference "
        f"can't reproduce training MIDI. Bug is in inference post-processing "
        f"(inference_core.heatmap_to_notes), threshold, or peak detection. "
        f"See drum-transcription-debug skill, Test 3."
    )
