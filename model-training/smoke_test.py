"""
Smoke Test - Training Verification

Trains on a single file (overfit) or batch of files to verify pipeline.
Saves checkpoint with pattern smoke_test_checkpoint_v{N}.ckpt

Usage:
    # Single file
    conda run -n drumtomidi python smoke_test.py --audio file.wav --midi file.mid
    
    # Batch from file (tab-delimited: audio.wav\tmidi.mid)
    conda run -n drumtomidi python smoke_test.py --list training_files.txt
    
    # Use with saved checkpoint
    conda run -n drumtomidi python smoke_test.py --list training_files.txt --checkpoint smoke_test_checkpoint_v3.ckpt
"""

import sys
sys.path.insert(0, '/Users/jasoncopp/Source/GitHub/larsnet')

import argparse
import torch
import torch.nn as nn
import time
from pathlib import Path
from datetime import datetime

# Import our modules
from feature_extractor import get_input_tensor
from label_encoder import midi_to_frame_array, NoteAdapter, LABEL_NAMES
from model import DrumTranscriber
from midi_shell import load_midi_file
from midi_core import extract_midi_notes_from_tracks, build_tempo_map_from_tracks


def find_next_version(models_dir: Path, prefix: str = "smoke_test_checkpoint_v") -> int:
    """Find the next version number for checkpoint naming."""
    existing = list(models_dir.glob(f"{prefix}*.ckpt"))
    versions = []
    for f in existing:
        try:
            v = int(f.name.replace(prefix, "").replace(".ckpt", ""))
            versions.append(v)
        except:
            pass
    return max(versions) + 1 if versions else 1


def run_smoke_test(audio_path: str, midi_path: str, epochs: int = 200, model=None, optimizer=None, checkpoint_path=None):
    """
    Train on a single file. If model/optimizer provided, continue training.
    
    Returns:
        tuple: (final_loss, model, optimizer)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # STEP 1: Load audio
    print("\n[1] Loading audio...")
    try:
        input_tensor = get_input_tensor(audio_path)
        assert input_tensor.shape[0] == 1, "Should have 1 channel"
        assert input_tensor.shape[1] == 128, "Should have 128 mel bins"
    except Exception as e:
        print(f"    ERROR: {e}")
        return None, None, None
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    # STEP 2: Load MIDI
    print("[2] Loading MIDI...")
    try:
        midi_file = load_midi_file(midi_path)
        tempo_map = build_tempo_map_from_tracks(midi_file.tracks, midi_file.ticks_per_beat)
        midi_notes, duration = extract_midi_notes_from_tracks(
            midi_file.tracks, midi_file.ticks_per_beat, tempo_map
        )
        notes = [NoteAdapter(pitch=n.midi_note, start_time=n.time, velocity=n.velocity) for n in midi_notes]
        print(f"    {len(midi_notes)} notes, {duration:.2f}s")
    except Exception as e:
        print(f"    ERROR: {e}")
        return None, None, None
    
    # STEP 3: Create labels
    print("[3] Creating labels...")
    try:
        total_frames = input_tensor.shape[3]
        target_tensor = midi_to_frame_array(notes, total_frames, 512, 44100)
        target_tensor = target_tensor.unsqueeze(0).permute(0, 2, 1).to(device)
        assert target_tensor.shape == (1, total_frames, 10), "Target shape mismatch"
    except Exception as e:
        print(f"    ERROR: {e}")
        return None, None, None
    
    # STEP 4: Initialize model if not provided
    if model is None:
        model = DrumTranscriber().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10
        )
    else:
        criterion = nn.BCELoss()
        scheduler = None  # Will be recreated if needed
    
    # STEP 5: Training loop
    chunk_frames = 2000
    print(f"\n[5] Training {epochs} epochs on {total_frames} frames...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        step_count = 0
        for chunk_start in range(0, total_frames, chunk_frames):
            chunk_end = min(chunk_start + chunk_frames, total_frames)
            input_chunk = input_tensor[:, :, :, chunk_start:chunk_end]
            target_chunk = target_tensor[:, chunk_start:chunk_end, :]
            
            optimizer.zero_grad()
            output = model(input_chunk)
            loss = criterion(output, target_chunk)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            step_count += 1
        
        avg_loss = epoch_loss / step_count
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f}")
        
        if scheduler:
            scheduler.step(avg_loss)
    
    return avg_loss, model, optimizer


def run_batch_smoke_test(list_path: str, epochs: int = 200, checkpoint_path: str = None):
    """
    Train on multiple files in sequence, accumulating into one model.
    Saves checkpoint after each file. Create .abort_training file to stop gracefully.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Load file list
    with open(list_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    
    print(f"=== Batch Smoke Test: {len(lines)} files ===")
    print(f"  checkpoints saved to: {models_dir}")
    print(f"  create .abort_training file to stop and save progress")
    
    # Load existing checkpoint or start fresh
    model = None
    optimizer = None
    start_file_idx = 0
    results = []
    version = find_next_version(models_dir)  # Start new version for this batch
    
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model = DrumTranscriber().to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_file_idx = ckpt.get('file_idx', 0) + 1  # Resume after last completed file
        results = ckpt.get('results', [])
        version = find_next_version(models_dir)  # New version for resumed batch
        print(f"  Resuming from file {start_file_idx}, previous results: {len(results)}")
    
    for idx, line in enumerate(lines):
        if idx < start_file_idx:
            continue  # Skip already processed files
        
        # Check abort flag at start of each file
        abort_file = models_dir / ".abort_training"
        if abort_file.exists():
            abort_file.unlink()
            ckpt_path = models_dir / f"smoke_test_checkpoint_v{version}.ckpt"
            print(f"\n=== ABORT: Stopping at file {idx+1}, saving checkpoint ===")
            torch.save({
                'file_idx': idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': results[-1][1] if results else None,
                'files': [l.split('\t')[0].strip() for l in lines],
                'results': results,
            }, ckpt_path)
            print(f"  Resume with:")
            print(f"  python smoke_test.py --list {list_path} --checkpoint {ckpt_path} --epochs {epochs}")
            print(f"  Files completed: {len(results)}/{len(lines)}")
            break
        
        parts = line.split('\t')
        if len(parts) < 2:
            print(f"  Line {idx+1}: Malformed, skipping")
            continue
        
        audio_path = parts[0].strip()
        midi_path = parts[1].strip()
        file_epochs = int(parts[2].strip()) if len(parts) > 2 and parts[2].strip().isdigit() else epochs
        
        print(f"\n--- File {idx+1}/{len(lines)}: {audio_path} ({file_epochs} epochs) ---")
        
        if not Path(audio_path).exists():
            print(f"  ERROR: Audio not found")
            results.append((audio_path, None, "Audio not found"))
            continue
        
        final_loss, model, optimizer = run_smoke_test(
            audio_path, midi_path, epochs=file_epochs,
            model=model, optimizer=optimizer
        )
        
        if final_loss is not None:
            results.append((audio_path, final_loss, "OK"))
            print(f"  Final loss: {final_loss:.6f}")
        else:
            results.append((audio_path, None, "FAILED"))
        
        # Save checkpoint after each file
        torch.save({
            'file_idx': idx,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': final_loss,
            'files': [l.split('\t')[0].strip() for l in lines],
            'results': results,
        }, models_dir / f"smoke_test_checkpoint_v{version}.ckpt")
        print(f"    [CHECKPOINT] Saved")
    
    else:
        # Loop completed normally
        print(f"\n=== Batch Complete ({len(results)} files) ===")
    
    # Summary
    print("\n=== Summary ===")
    for audio_path, loss, status in results:
        if loss is not None:
            print(f"  {audio_path}: loss={loss:.6f} ({status})")
        else:
            print(f"  {audio_path}: {status}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Smoke Test Training')
    parser.add_argument('--audio', help='Single audio file (.wav)')
    parser.add_argument('--midi', help='Single MIDI file (.mid)')
    parser.add_argument('--list', '-l', help='File with lines of: audio.wav\\tmidi.mid\\t[epochs]')
    parser.add_argument('--checkpoint', '-c', help='Checkpoint to load/resume (auto-saved after each file)')
    parser.add_argument('--epochs', '-e', type=int, default=200, help='Training epochs (default 200)')
    
    args = parser.parse_args()
    
    if args.list:
        run_batch_smoke_test(args.list, epochs=args.epochs, checkpoint_path=args.checkpoint)
    elif args.audio and args.midi:
        result = run_smoke_test(args.audio, args.midi, epochs=args.epochs)
        if result[0] is not None:
            print(f"\nFinal loss: {result[0]:.6f}")
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python smoke_test.py --audio file.wav --midi file.mid")
        print("  python smoke_test.py --list training_files.txt --epochs 100")
        print("  python smoke_test.py --list training_files.txt --checkpoint smoke_test_checkpoint_v2.ckpt")
        print("\nBatch mode: Create 'model-training/models/.abort_training' to stop gracefully")