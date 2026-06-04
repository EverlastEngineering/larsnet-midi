"""
Train.py — Full-Batch Training with Epoch Repeats

Trains on a batch of files for multiple epochs, repeating the entire
file list each epoch. Shows per-file loss during the run.

Saves checkpoint with pattern train_checkpoint_v{N}.ckpt

Usage:
    python train.py --list training_files.txt --epochs 100
    
    # Use with saved checkpoint
    python train.py --list training_files.txt --checkpoint train_checkpoint_v3.ckpt --epochs 100

    # With validation
    python train.py --list training_files.txt --val-list val_files.txt --epochs 100

Checkpoint format:
    'file_idx': current file index (0-based)
    'epoch_idx': current epoch index (0-based)
    'files': list of audio paths from the batch file
    'results': list of (audio_path, final_loss, status) per completed file
"""

import argparse
import time
import torch
from pathlib import Path

from config import DEVICE, get_models_dir, get_learning_rate, get_chunk_frames, get_training_config, SAMPLE_RATE, HOP_LENGTH
from io_utils import find_next_version, save_checkpoint, check_abort, clear_abort
from train_utils import setup_training, load_audio, load_midi_notes, build_targets, get_chunk, train_chunk, run_eval

from model import DrumTranscriber

MAX_TRAIN_SECONDS = 300
MAX_FRAMES = MAX_TRAIN_SECONDS * SAMPLE_RATE // HOP_LENGTH


def save_train_checkpoint(
    models_dir: Path,
    model,
    optimizer,
    loss,
    file_idx: int,
    epoch_idx: int,
    results: list,
    lines: list,
    version: int,
) -> int:
    """
    Save a training checkpoint with auto-incrementing version.
    
    Returns the new version number.
    """
    version += 1
    path = models_dir / f"train_checkpoint_v{version}.ckpt"
    save_checkpoint(path, model, optimizer, loss, {
        'file_idx': file_idx,
        'epoch_idx': epoch_idx,
        'files': [l.split('\t', 1)[0] for l in lines],
        'results': results
    })
    print(f"\n    [CHECKPOINT] v{version}: epoch={epoch_idx+1}, file={file_idx+1} → {path.name}")
    return version


def train_file(
    audio_path: str,
    midi_path: str,
    model=None,
    optimizer=None,
    criterion=None,
    clip_grad=1.0,
    device: str = None,
) -> tuple:
    """
    Train on one file, iterating over chunks.
    
    Args:
        audio_path: Path to audio file
        midi_path: Path to MIDI file
        model: Existing model
        optimizer: Existing optimizer
        criterion: Loss function (already initialized)
        clip_grad: Gradient clipping value
        device: Device string or None for auto-detect
        
    Returns:
        Tuple of (avg_loss, total_frames)
    """
    if device is None:
        device = DEVICE
    
    # STEP 1: Load audio
    # print("Loading audio. ", end='')
    try:
        input_tensor = load_audio(audio_path)
    except Exception as e:
        print(f"    ERROR: {e}")
        return None, None, None
    input_tensor = input_tensor.to(device)
    
    # STEP 2: Load MIDI
    # print("Loading MIDI. ", end='')
    try:
        notes, duration = load_midi_notes(midi_path)
        # print(f"    {len(notes)} notes, {duration:.2f}s")
    except Exception as e:
        print(f"    ERROR: {e}")
        return None, None, None

    # STEP 2b: Truncate to MAX_FRAMES
    input_tensor = input_tensor[:, :, :, :MAX_FRAMES]
    
    # STEP 3: Create labels
    # print("Creating labels. ", end='')
    try:
        total_frames = input_tensor.shape[3]
        target_tensor = build_targets(notes, total_frames)
        target_tensor = target_tensor.to(device)
        assert target_tensor.shape == (1, total_frames, 20), f"Target shape mismatch: {target_tensor.shape}"
    except Exception as e:
        print(f"    ERROR: {e}")
        return None, None, None
    
    # STEP 4: model and criterion are passed in — just move to device
    # Only initialize fresh model/criterion if not already provided
    if model is None:
        train_cfg = get_training_config()
        scheduler_patience = train_cfg.get('scheduler_patience', 10)
        scheduler_factor = train_cfg.get('scheduler_factor', 0.1)
        model, optimizer, criterion, scheduler, clip_grad = setup_training(
            device=device, learning_rate=lr,
            scheduler_patience=scheduler_patience, scheduler_factor=scheduler_factor,
        )
    else:
        if next(model.parameters()).device != torch.device(device):
            model = model.to(device)
        if criterion is None:
            from config import get_velocity_weight
            velocity_weight = get_velocity_weight()
            from train_utils import MultiTaskDrumLoss
            criterion = MultiTaskDrumLoss(velocity_weight=velocity_weight, device=device)
            criterion = criterion.to(device)

    # STEP 5: chunk_frames is always needed
    chunk_frames = get_chunk_frames()
    
    # STEP 6: Training loop (one pass through chunks, averaged loss returned)
    # print(f"\rTraining on {total_frames} frames...", end='')
    # file_loss = 0.0
    # step_count = 0
    # file_start = time.time()
    
    epoch_losses = []
    epoch_onset_losses = []
    epoch_velocity_losses = []
    
    for chunk_start in range(0, total_frames, chunk_frames):
        input_chunk, target_chunk = get_chunk(input_tensor, target_tensor, chunk_start, chunk_frames)
        loss_value, _ = train_chunk(model, input_chunk, target_chunk, optimizer, criterion, clip_grad)
        epoch_losses.append(loss_value)
        # file_loss += loss_value
        # step_count += 1
        # print("\033[1A\033[1A\033[1A")
        # print(f"    Loss: {loss_value:.6f} | Epoch {current_epoch_idx+1}/{total_epochs} | Processing file {idx+1}/{len(lines)}: {Path(audio_path).name}")
        print(f"       Chunk {chunk_start}-{min(chunk_start+chunk_frames, total_frames)}",end='\r')
    # avg_loss = file_loss / step_count
    # elapsed = time.time() - file_start
    # lr_val = optimizer.param_groups[0]['lr']
    # print(f"    {total_frames} frames | File Loss: {avg_loss:.6f} | LR: {lr_val:.0e} | Time: {elapsed:.1f}s")

    # Return average loss across chunks for scheduler
    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
    
    return avg_loss, model, optimizer, total_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Full-Batch Training')
    parser.add_argument('--list', '-l', required=True, help='File with lines of: audio.wav\\tmidi.mid\\t[epochs]')
    parser.add_argument('--epochs', '-e', type=int, default=200, help='Total passes through the file list (default 200)')
    parser.add_argument('--val-list', help='Validation file list (audio\\tMIDI per line)')
    parser.add_argument('--checkpoint', '-c', help='Checkpoint to resume from')
    
    args = parser.parse_args()
    
    total_epochs = args.epochs
    
    with open(args.list, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    print(f"=== Training: {len(lines)} files per epoch ===\n")
    
    val_lines = []
    if args.val_list:
        with open(args.val_list, 'r') as f:
            val_lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"=== Validation: {len(val_lines)} files ===\n")
    
    models_dir = get_models_dir()
    model = None
    optimizer = None
    criterion = None
    clip_grad = 1.0
    results = []
    version = find_next_version(models_dir)
    
    # Track resume position
    start_file_idx = 0
    start_epoch_idx = 0
    
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
        model = DrumTranscriber().to(DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_file_idx = ckpt.get('file_idx', 0)
        start_epoch_idx = ckpt.get('epoch_idx', 0)
        results = ckpt.get('results', [])
        print(f"  Resuming: epoch {start_epoch_idx}, file {start_file_idx}/{len(lines)}")
        print(f"  Results so far: {len(results)}")
    
    current_epoch_idx = start_epoch_idx
    current_file_idx = 0
    loss = None
    process_start = time.time()

    # Initialize model, optimizer, criterion once before the training loop
    lr = get_learning_rate()
    train_cfg = get_training_config()
    scheduler_patience = train_cfg.get('scheduler_patience', 10)
    scheduler_factor = train_cfg.get('scheduler_factor', 0.1)
    model, optimizer, criterion, scheduler, clip_grad = setup_training(
        device=DEVICE, learning_rate=lr,
        scheduler_patience=scheduler_patience, scheduler_factor=scheduler_factor,
    )
    print(f"  Starting LR: {lr:.0e}, clip_grad: {clip_grad}")

    while total_epochs is None or current_epoch_idx < total_epochs:
        # Check for abort signal
        if check_abort(models_dir):
            clear_abort(models_dir)
            version = save_train_checkpoint(
                models_dir, model, optimizer,
                results[-1][1] if results else None,
                current_file_idx, current_epoch_idx, results, lines, version - 1
            )
            print(f"\n=== ABORT at epoch {current_epoch_idx+1}, file {current_file_idx+1} ===")
            break
        
        # Track epoch-level losses for scheduler
        epoch_file_losses = []
        
        # Loop over files
        for idx in range(start_file_idx, len(lines)):
            current_file_idx = idx
            line = lines[idx]
            
            parts = line.split('\t', 1)
            if len(parts) < 2:
                results.append((line, None, "Malformed"))
                continue
            audio_path = parts[0]
            midi_path = parts[1]
            
            # print(f"--- File {idx+1}/{len(lines)}: {Path(audio_path).name} ---")
            if not Path(audio_path).exists():
                results.append((audio_path, None, "Audio not found"))
                continue
            
            try:
                file_process_start = time.time()

                # print epoch and file info
                # print(f"    Epoch {current_epoch_idx+1}/{total_epochs} | Processing file {idx+1}/{len(lines)}: {Path(audio_path).name}")
                loss, model, optimizer, total_frames = train_file(
                    audio_path,
                    midi_path,
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    clip_grad=clip_grad,
                    device=DEVICE
                )
                
                # accumulate for epoch-average scheduler
                if loss is not None:
                    epoch_file_losses.append(loss)

                # calculate elapsed time for this file
                elapsed_for_file = time.time() - file_process_start
                elapsed_for_process = time.time() - process_start
                lr_val = optimizer.param_groups[0]['lr']

                total_files_processed = (current_epoch_idx * len(lines)) + idx
                total_files = total_epochs * len(lines)
                files_remaining = total_files - total_files_processed

                eta = (elapsed_for_process / max(total_files_processed, 1)) * files_remaining

                files_in_epoch_processed = idx + 1
                eta_epoch = (elapsed_for_process / max(files_in_epoch_processed, 1)) * (len(lines) - idx - 1)

                print(f"Loss: {loss:.6f} | Epoch {current_epoch_idx+1}/{total_epochs} | LR: {lr_val:.0e} | Overall Time: {elapsed_for_process:.1f}s | Frames: {total_frames} | File {idx+1}/{len(lines)}: {Path(audio_path).name}")
                print(f"                              ETA for this epoch: {eta_epoch:.1f}s | Overall ETA: {eta:.1f}s", end='\r')
                results.append((audio_path, loss, "OK" if loss is not None else "FAILED"))
            except KeyboardInterrupt:
                version = save_train_checkpoint(
                    models_dir, model, optimizer, loss,
                    idx, current_epoch_idx, results, lines, version - 1
                )
                print(f"\n=== INTERRUPT at epoch {current_epoch_idx+1}, file {idx+1} ===")
                break
            
            # Extra checkpoint every 100 files (named snapshot, not version-bumped)
            if (idx + 1) % 100 == 0:
                extra_path = models_dir / f"train_checkpoint_v{version}_f{idx+1}.ckpt"
                save_checkpoint(extra_path, model, optimizer, loss, {
                    'file_idx': idx,
                    'epoch_idx': current_epoch_idx,
                    'files': [l.split('\t', 1)[0] for l in lines],
                    'results': results
                })
                print(f"\n    [CHECKPOINT] Snapshot at file {idx+1} → {extra_path.name}")
        
        else:
            # Completed one full epoch without break/interrupt
            current_epoch_idx += 1
            start_file_idx = 0  # Reset for next epoch

            # Validation pass (no gradients, usually faster)
            val_loss = None
            if val_lines:
                val_start = time.time()
                val_loss, _, _ = run_eval(model, val_lines, criterion, DEVICE)
                val_elapsed = time.time() - val_start
                print(f"  Validation: val_loss={val_loss:.6f} ({val_elapsed:.1f}s)")

            # STEP LR scheduler with validation loss (not training loss!)
            if val_loss is not None:
                scheduler.step(val_loss)
            elif epoch_file_losses:
                avg_epoch_loss = sum(epoch_file_losses) / len(epoch_file_losses)
                scheduler.step(avg_epoch_loss)

            lr_val = optimizer.param_groups[0]['lr']
            avg_train = sum(epoch_file_losses) / len(epoch_file_losses) if epoch_file_losses else 0.0
            val_str = f"{val_loss:.6f}" if val_loss is not None else "N/A"
            print(f"\r\033[K=== Epoch {current_epoch_idx} Complete | Val Loss: {val_str} | Train Avg: {avg_train:.6f} | LR: {lr_val:.0e} ===")

            # Epoch-level checkpoint (auto-increments version)
            version = save_train_checkpoint(
                models_dir, model, optimizer,
                val_loss if val_loss is not None else (results[-1][1] if results else None),
                len(lines) - 1, current_epoch_idx, results, lines, version
            )
            continue
        
        # Interrupted or aborted — break outer loop
        break
    
    # print("\n=== Summary ===")
    # for audio_path, loss, status in results:
    #     print(f"  {audio_path}: loss={loss:.6f} ({status})" if loss is not None else f"  {audio_path}: {status}")
