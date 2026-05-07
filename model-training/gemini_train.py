"""
train.py — Production-Grade Multi-Task Training Loop

Features:
- Robust path handling for filenames with spaces and shell escapes
- 3-Channel Stereo expansion for Mono inputs
- Independent Validation Loop for Scheduler tracking
- Explicit Gradient Clipping and component loss logging
"""

import argparse
import time
import torch
import sys
import os
from pathlib import Path

from config import DEVICE, get_models_dir
from io_utils import save_checkpoint
from train_utils import setup_training, load_audio, load_midi_notes, build_targets
from model import DrumTranscriber

def clean_path(p):
    """
    Cleans shell-escaped paths (like 'My\\ Drive') into standard python paths.
    Also handles trailing/leading whitespace from the manifest.
    """
    if not p:
        return ""
    # 1. Strip whitespace
    p = p.strip()
    # 2. Convert shell-escaped spaces '\ ' to actual spaces ' '
    p = p.replace('\\ ', ' ')
    # 3. Clean up potential double backslashes if any
    p = p.replace('\\\\', '\\')
    return p

def run_eval(model, criterion, manifest_lines, device):
    """
    Runs a non-gradient pass over validation files to get a true loss metric.
    """
    model.eval()
    val_loss_sum = 0.0
    processed_count = 0
    
    with torch.no_grad():
        for line in manifest_lines:
            try:
                # Use tab splitting only
                parts = line.strip().split('\t')
                if len(parts) < 2: continue
                
                wav_path = clean_path(parts[0])
                midi_path = clean_path(parts[1])
                
                if not os.path.exists(wav_path):
                    continue
                
                audio_tensor = load_audio(wav_path).to(device)
                midi_notes, _ = load_midi_notes(midi_path)
                
                if audio_tensor.dim() == 3:
                    audio_tensor = audio_tensor.repeat(3, 1, 1)
                # audio_tensor = audio_tensor.unsqueeze(0) 
                
                num_frames = audio_tensor.shape[-1]
                target_tensor = build_targets(midi_notes, num_frames).to(device).unsqueeze(0)

                predictions = model(audio_tensor)
                loss = criterion(predictions, target_tensor)
                val_loss_sum += loss.item()
                processed_count += 1
                print(f"Val File {processed_count}: Loss {loss.item():.4f}")
            except Exception:
                continue
                
    return (val_loss_sum / processed_count) if processed_count > 0 else 1e6

def main():
    parser = argparse.ArgumentParser(description='DrumToMIDI Explicit Training Loop')
    parser.add_argument('--list', type=str, required=True, help='Path to training manifest')
    parser.add_argument('--val-list', type=str, help='Path to validation manifest')
    parser.add_argument('--epochs', type=int, default=10, help='Number of full passes')
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--lr', type=float, default=1e-4, help='Starting Learning Rate')
    args = parser.parse_args()

    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Manifests
    with open(args.list, 'r', encoding='utf-8') as f:
        train_lines = [line.strip() for line in f if line.strip()]
    
    val_lines = []
    if args.val_list:
        with open(args.val_list, 'r', encoding='utf-8') as f:
            val_lines = [line.strip() for line in f if line.strip()]

    # 2. Setup
    model = DrumTranscriber().to(DEVICE)
    model, optimizer, criterion, scheduler = setup_training(
        model=model, 
        learning_rate=args.lr, 
        device=DEVICE,
        scheduler_patience=3 
    )

    start_epoch = 0
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            print(f"Loading checkpoint: {ckpt_path.name}")
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt.get('epoch_idx', 0) + 1

    # =================================================================
    # MASTER LOOP
    # =================================================================
    print(f"Starting training on {len(train_lines)} files...")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss_sum = 0.0
        num_files = len(train_lines)
        files_actually_processed = 0

        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")

        for i, line in enumerate(train_lines):
            try:
                # Splitting strictly on the Tab
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    print(f"Line {i+1}: Missing Tab delimiter. Skipping.")
                    continue
                
                wav_path = clean_path(parts[0])
                midi_path = clean_path(parts[1])

                if not os.path.exists(wav_path):
                    print(f"File not found: {wav_path}")
                    continue
                
                # --- Data Loading ---
                audio_tensor = load_audio(wav_path).to(DEVICE)
                if audio_tensor.dim() == 3:
                    audio_tensor = audio_tensor.repeat(3, 1, 1)
                # audio_tensor = audio_tensor.unsqueeze(0)
                
                midi_notes, _ = load_midi_notes(midi_path)
                num_frames = audio_tensor.shape[-1]
                target_tensor = build_targets(midi_notes, num_frames).to(DEVICE).unsqueeze(0)

                # --- Training Step ---
                optimizer.zero_grad()
                predictions = model(audio_tensor)
                loss = criterion(predictions, target_tensor)
                
                # Get sub-losses for logging
                onset_l = getattr(criterion, 'last_onset_loss', 0.0)
                vel_l = getattr(criterion, 'last_velocity_loss', 0.0)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss_sum += loss.item()
                files_actually_processed += 1
                
                # if (i + 1) % 10 == 0 or i == 0:
                current_lr = optimizer.param_groups[0]['lr']
                avg_train_loss = epoch_loss_sum / max(1, files_actually_processed)
                print(f"[{i+1}/{num_files}] Loss: {loss.item():.4f} (Avg: {avg_train_loss:.4f}) | "
                        f"On: {onset_l:.4f} Vel: {vel_l:.4f} | LR: {current_lr:.1e}")

            except Exception as e:
                print(f"Error on file {i+1} ({wav_path[:30]}...): {e}")
                optimizer.zero_grad() 
                continue

        # --- End of Epoch Logic ---
        epoch_avg_train_loss = epoch_loss_sum / max(1, files_actually_processed)
        print(f"\nEnd of Epoch {epoch+1} Training Loss: {epoch_avg_train_loss:.5f}")
        
        val_loss = 0.0
        if val_lines:
            print("Starting Validation Pass...")
            val_loss = run_eval(model, criterion, val_lines, DEVICE)
            print(f"Validation Loss: {val_loss:.5f}")
            scheduler.step(val_loss)
            monitor_metric = val_loss
        else:
            scheduler.step(epoch_avg_train_loss)
            monitor_metric = epoch_avg_train_loss

        # Save Checkpoint
        checkpoint_name = models_dir / f"drum_train_v49_epoch{epoch+1}.ckpt"
        save_checkpoint(checkpoint_name, model, optimizer, monitor_metric, {
            'epoch_idx': epoch,
            'train_loss': epoch_avg_train_loss,
            'val_loss': val_loss if val_lines else None
        })
        print(f"Saved Checkpoint: {checkpoint_name.name}")

    print("\nAll epochs completed.")

if __name__ == "__main__":
    main()