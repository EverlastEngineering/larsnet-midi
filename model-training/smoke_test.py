"""
Smoke Test - Training Verification

Trains on a single file (overfit) or batch of files to verify pipeline.
Saves checkpoint with pattern smoke_test_checkpoint_v{N}.ckpt

Usage:
    # Single file
    python smoke_test.py --audio file.wav --midi file.mid
    
    # Batch from file (tab-delimited: audio.wav\tmidi.mid\t[epochs])
    python smoke_test.py --list training_files.txt
    
    # Use with saved checkpoint
    python smoke_test.py --list training_files.txt --checkpoint smoke_test_checkpoint_v3.ckpt
"""

import argparse
import time
import torch
from pathlib import Path

from config import DEVICE, get_models_dir, get_learning_rate, get_chunk_frames, get_training_config
from io_utils import find_next_version, save_checkpoint, check_abort, clear_abort
from train_utils import setup_training, load_audio, load_midi_notes, build_targets, get_chunk, train_chunk

from model import DrumTranscriber


def run_smoke_test(
    audio_path: str,
    midi_path: str,
    epochs: int = 200,
    model=None,
    optimizer=None,
    device: str = None,
) -> tuple:
    """
    Train on a single file. If model/optimizer provided, continue training.
    
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
    print("\n[1] Loading audio...")
    try:
        input_tensor = load_audio(audio_path)
    except Exception as e:
        print(f"    ERROR: {e}")
        return None, None, None
    input_tensor = input_tensor.to(device)
    
    # STEP 2: Load MIDI
    print("[2] Loading MIDI...")
    try:
        notes, duration = load_midi_notes(midi_path)
        print(f"    {len(notes)} notes, {duration:.2f}s")
    except Exception as e:
        print(f"    ERROR: {e}")
        return None, None, None
    
    # STEP 3: Create labels
    print("[3] Creating labels...")
    try:
        total_frames = input_tensor.shape[3]
        target_tensor = build_targets(notes, total_frames)
        target_tensor = target_tensor.to(device)
        assert target_tensor.shape == (1, total_frames, 20), f"Target shape mismatch: {target_tensor.shape}"
    except Exception as e:
        print(f"    ERROR: {e}")
        return None, None, None
    
    # STEP 4: Initialize model if not provided
    lr = get_learning_rate()
    chunk_frames = get_chunk_frames()
    train_cfg = get_training_config()
    scheduler_patience = train_cfg.get('scheduler_patience', 10)
    scheduler_factor = train_cfg.get('scheduler_factor', 0.1)
    model, optimizer, criterion, scheduler, clip_grad = setup_training(
        model=model, device=device, learning_rate=lr,
        scheduler_patience=scheduler_patience, scheduler_factor=scheduler_factor,
    )
    
    # STEP 5: Training loop
    print(f"\n[4] Training {epochs} epochs on {total_frames} frames...")
    epoch_start = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        step_count = 0
        for chunk_start in range(0, total_frames, chunk_frames):
            # show progress per chunk with \r to overwrite line
            print(f"    Epoch {epoch+1:3d}/{epochs} | Chunk {chunk_start}-{min(chunk_start+chunk_frames, total_frames)}", end='\r')
            input_chunk, target_chunk = get_chunk(input_tensor, target_tensor, chunk_start, chunk_frames)
            loss_value, _ = train_chunk(model, input_chunk, target_chunk, optimizer, criterion, clip_grad)
            epoch_loss += loss_value
            step_count += 1
        
        avg_loss = epoch_loss / step_count
        elapsed = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
        m, s = divmod(eta, 60)
        h, m = divmod(int(m), 60)
        eta_str = f"{h}h {m}m {s:.0f}s" if h > 0 else f"{m}m {s:.0f}s"
        print(f"    Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f} | LR: {lr:.0e} | ETA: {eta_str}")

        if scheduler:
            scheduler.step(avg_loss)
    
    return avg_loss, model, optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Smoke Test Training')
    parser.add_argument('--audio', '-a', help='Single audio file (.wav)')
    parser.add_argument('--midi', '-m', help='Single MIDI file (.mid)')
    parser.add_argument('--list', '-l', help='File with lines of: audio.wav\\tmidi.mid\\t[epochs]')
    parser.add_argument('--epochs', '-e', type=int, default=200, help='Training epochs (default 200)')
    
    args = parser.parse_args()
    
    if args.list:
        # Batch: loop over files, accumulating into one model
        with open(args.list, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        print(f"=== Batch Smoke Test: {len(lines)} files ===")
        
        models_dir = get_models_dir()
        model = None
        optimizer = None
        results = []
        version = find_next_version(models_dir, prefix="smoke_test_checkpoint_v")
        
        for idx, line in enumerate(lines):
            if check_abort(models_dir):
                clear_abort(models_dir)
                save_checkpoint(models_dir / f"smoke_test_checkpoint_v{version}.ckpt", model, optimizer, results[-1][1] if results else None, {'file_idx': idx, 'files': [l.split('\t')[0] for l in lines], 'results': results})
                print(f"=== ABORT at file {idx+1}, checkpoint saved ===")
                break
            
            parts = line.split('\t', 1)
            if len(parts) < 2:
                results.append((line, None, "Malformed"))
                continue
            audio_path = parts[0]
            midi_path = parts[1]
            file_epochs = int(parts[2].strip()) if len(parts) > 2 and parts[2].strip().isdigit() else args.epochs
            
            print(f"\n--- File {idx+1}/{len(lines)}: {audio_path} ({file_epochs} epochs) ---")
            if not Path(audio_path).exists():
                results.append((audio_path, None, "Audio not found"))
                continue
            
            final_loss, model, optimizer = run_smoke_test(audio_path, midi_path, epochs=file_epochs, model=model, optimizer=optimizer, device=DEVICE)
            results.append((audio_path, final_loss, "OK" if final_loss is not None else "FAILED"))
            save_checkpoint(models_dir / f"smoke_test_checkpoint_v{version}.ckpt", model, optimizer, final_loss, {'file_idx': idx, 'files': [l.split('\t')[0] for l in lines], 'results': results})
            print(f"  Final loss: {final_loss:.6f}" if final_loss is not None else "  FAILED")
        else:
            print(f"\n=== Batch Complete ({len(results)} files) ===")
        
        print("\n=== Summary ===")
        for audio_path, loss, status in results:
            print(f"  {audio_path}: loss={loss:.6f} ({status})" if loss is not None else f"  {audio_path}: {status}")
    elif args.audio and args.midi:
        device = DEVICE
        models_dir = get_models_dir()
        
        model = DrumTranscriber().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        lr = get_learning_rate()
        chunk_frames = get_chunk_frames()
        train_cfg = get_training_config()
        scheduler_patience = train_cfg.get('scheduler_patience', 10)
        scheduler_factor = train_cfg.get('scheduler_factor', 0.1)
        model, optimizer, criterion, scheduler, clip_grad = setup_training(
            model=model, device=device, learning_rate=lr,
            scheduler_patience=scheduler_patience, scheduler_factor=scheduler_factor,
        )
        
        try:
            result = run_smoke_test(args.audio, args.midi, epochs=args.epochs, model=model, optimizer=optimizer, device=device)
            if result[0] is not None:
                print(f"\nFinal loss: {result[0]:.6f}")
                # Save on success
                version = find_next_version(models_dir, prefix="smoke_test_checkpoint_v")
                ckpt_path = models_dir / f"smoke_test_checkpoint_v{version}.ckpt"
                save_checkpoint(ckpt_path, result[1], result[2], result[0], {
                    'epochs': args.epochs,
                    'source': args.audio,
                })
                print(f"[CHECKPOINT] Saved to {ckpt_path}")
        except (KeyboardInterrupt, Exception) as e:
            print(f"\nTraining interrupted: {e}")
            # Save on interrupt
            version = find_next_version(models_dir, prefix="smoke_test_checkpoint_v")
            ckpt_path = models_dir / f"smoke_test_checkpoint_v{version}.ckpt"
            save_checkpoint(ckpt_path, model, optimizer, None, {
                'epochs': args.epochs,
                'source': args.audio,
                'interrupted': True,
            })
            print(f"[CHECKPOINT] Saved on exit to {ckpt_path}")
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python smoke_test.py --audio file.wav --midi file.mid")
        print("  python smoke_test.py --list training_files.txt --epochs 100")
        print("  python smoke_test.py --list training_files.txt --checkpoint smoke_test_checkpoint_v2.ckpt")
        print("\nBatch mode: Create 'model-training/models/abort' to stop gracefully")
