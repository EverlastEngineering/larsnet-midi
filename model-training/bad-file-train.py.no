"""
Train.py — Full-Batch Training with Epoch Repeats

Trains on a batch of files for multiple epochs, repeating the entire
file list each epoch. Shows per-file loss during the run.

Saves checkpoint with pattern train_checkpoint_v{N}.ckpt

Usage:
    python train.py --list training_files.txt --epochs 100
    
    # Use with saved checkpoint
    python train.py --list training_files.txt --checkpoint train_checkpoint_v3.ckpt --epochs 100

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

from config import DEVICE, get_models_dir, get_learning_rate, get_chunk_frames, get_training_config
from io_utils import find_next_version, save_checkpoint, check_abort, clear_abort
from train_utils import setup_training, load_audio, load_midi_notes, build_targets, get_chunk, train_chunk

from model import DrumTranscriber


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
    device: str = None,
) -> tuple:
    """
    Train on a batch of files. If model/optimizer provided, continue training.
    
    Args:
        audio_path: Path to audio file
        midi_path: Path to MIDI file
        epochs: Number of training epochs
        model: Existing model or None to create fresh
        optimizer: Existing optimizer or None
        device: Device string or None for auto-detect
        
    Returns:
        Tuple of (final_loss, model, optimizer)
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
    
    # STEP 4: Initialize model if not provided
    # print("Initializing model. ", end='')
    lr = get_learning_rate()
    chunk_frames = get_chunk_frames()
    train_cfg = get_training_config()
    scheduler_patience = train_cfg.get('scheduler_patience', 10)
    scheduler_factor = train_cfg.get('scheduler_factor', 0.1)
    model, optimizer, criterion, scheduler = setup_training(
        model=model, device=device, learning_rate=lr,
        scheduler_patience=scheduler_patience, scheduler_factor=scheduler_factor,
    )
    
    # STEP 5: Training loop (one pass through chunks, averaged loss returned)
    # print(f"\rTraining on {total_frames} frames...", end='')
    # file_loss = 0.0
    # step_count = 0
    # file_start = time.time()
    
    for chunk_start in range(0, total_frames, chunk_frames):
        input_chunk, target_chunk = get_chunk(input_tensor, target_tensor, chunk_start, chunk_frames)
        loss_value, _ = train_chunk(model, input_chunk, target_chunk, optimizer, criterion)
        # file_loss += loss_value
        # step_count += 1
        # print("\033[1A\033[1A\033[1A")
        # print(f"    Loss: {loss_value:.6f} | Epoch {current_epoch_idx+1}/{total_epochs} | Processing file {idx+1}/{len(lines)}: {Path(audio_path).name}")
        print(f"       Chunk {chunk_start}-{min(chunk_start+chunk_frames, total_frames)}",end='\r')
    # avg_loss = file_loss / step_count
    # elapsed = time.time() - file_start
    # lr_val = optimizer.param_groups[0]['lr']
    # print(f"    {total_frames} frames | File Loss: {avg_loss:.6f} | LR: {lr_val:.0e} | Time: {elapsed:.1f}s")

    if scheduler:
        scheduler.step(loss_value)
    
    return loss_value, model, optimizer, total_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Full-Batch Training')
    parser.add_argument('--list', '-l', required=True, help='File with lines of: audio.wav\\tmidi.mid\\t[epochs]')
    parser.add_argument('--epochs', '-e', type=int, default=200, help='Total passes through the file list (default 200)')
    parser.add_argument('--checkpoint', '-c', help='Checkpoint to resume from')
    
    args = parser.parse_args()
    
    total_epochs = args.epochs
    
    with open(args.list, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    print(f"=== Training: {len(lines)} files per epoch ===\n")
    
    models_dir = get_models_dir()
    model = None
    optimizer = None
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
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
                    device=DEVICE
                )

                # calculate elapsed time for this file
                elapsed_for_file = time.time() - file_process_start
                elapsed_for_process = time.time() - process_start
                lr_val = optimizer.param_groups[0]['lr']
                # eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
                # m, s = divmod(eta, 60)
                # h, m = divmod(int(m), 60)
                # eta_str = f"{h}h {m}m {s:.0f}s" if h > 0 else f"{m}m {s:.0f}s"
                # print(f"    Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f} | LR: {lr_val:.0e} | ETA: {eta_str}")
                # \033[1A
                # print(f"--- Loss: {loss:.6f} File {idx+1}/{len(lines)}: {Path(audio_path).name} ---")
                # print("\033[1A\033[1A\033[1A\033[1A")
       
                print(f"Loss: {loss:.6f} | Epoch {current_epoch_idx+1}/{total_epochs} | LR: {lr_val:.0e} | Overall Time: {elapsed_for_process:.1f}s | Frames: {total_frames} | File {idx+1}/{len(lines)}: {Path(audio_path).name}")
                # print(" ")
    
                # Print filename, loss, and elapsed time for this file                
                # print(f"    Loss: {loss:.6f} | Time For This File: {elapsed_for_file:.1f}s")

                #calculate ETA considering elapsed time, number of files and epochs
                eta = (elapsed_for_process / ((current_epoch_idx * len(lines)) + idx + 1)) * ((total_epochs * len(lines)) - ((current_epoch_idx * len(lines)) + idx + 1))
                
                # calculate eta for this epoch to complete based on elapsed time and remaining files
                eta_epoch = (elapsed_for_process / (idx + 1)) * (len(lines) - idx - 1)

                print(f"                              ETA for this epoch: {eta_epoch:.1f}s | Overall ETA: {eta:.1f}s", end='\r')

                # Print current file number, loss, elapsed time, and learning rate and epoch info
                # print(f"    Epoch {current_epoch_idx+1:3d}/{total_epochs} | Loss: {loss:.6f} ")    
                # print(f"\n\n  Current File loss: {loss:.6f} | Time: {elapsed:.1f}s | LR: {lr_val:.0e}" if loss is not None else "  FAILED")
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
            # Epoch-level checkpoint (auto-increments version)
            epoch_path = models_dir / f"train_checkpoint_v{version + 1}_epoch{current_epoch_idx}.ckpt"
            version = save_train_checkpoint(
                models_dir, model, optimizer,
                results[-1][1] if results else None,
                len(lines) - 1, current_epoch_idx, results, lines, version
            )
            print(f"\r\033[K\n=== Epoch {current_epoch_idx} Complete ({len(results)} files) ===")
            continue
        
        # Interrupted or aborted — break outer loop
        break
    
    # print("\n=== Summary ===")
    # for audio_path, loss, status in results:
    #     print(f"  {audio_path}: loss={loss:.6f} ({status})" if loss is not None else f"  {audio_path}: {status}")
