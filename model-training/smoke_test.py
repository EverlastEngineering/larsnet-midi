"""
Smoke Test - Overfit Verification

Proves that the data pipe is leak-proof. If the model can't memorize 
one single 30-second file, there is a fundamental bug in the pipeline.

Running this test verifies:
1. Feature extractor produces correct shapes
2. Label encoder produces correct shapes  
3. Model can overfit a single sample to loss < 0.01

Usage:
    conda run -n drumtomidi python smoke_test.py
"""

import sys
sys.path.insert(0, '/Users/jasoncopp/Source/GitHub/larsnet')

import torch
import torch.nn as nn
import time
from pathlib import Path

# Import our modules
from feature_extractor import get_input_tensor
from label_encoder import midi_to_frame_array, NoteAdapter, LABEL_NAMES
from model import DrumTranscriber
from midi_shell import load_midi_file
from midi_core import extract_midi_notes_from_tracks, build_tempo_map_from_tracks


def run_smoke_test(audio_path: str, midi_path: str, epochs: int = 200):
    """
    A minimal training loop to prove the model can 'memorize' a single track.
    If loss < 0.01, the data flow is mathematically verified.
    
    Args:
        audio_path: Path to audio file
        midi_path: Path to MIDI file
        epochs: Number of training epochs (default 200)
    
    Returns:
        "SUCCESS" if loss < 0.01, "FAILURE" otherwise
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # =====================================================================
    # STEP 1: Load and verify audio
    # =====================================================================
    print("\n[1] Loading audio...")
    step_start = time.time()
    try:
        input_tensor = get_input_tensor(audio_path, hop_length=hop_length)
        print(f"    Input shape: {input_tensor.shape}")
        assert input_tensor.shape[0] == 3, "Should have 3 channels"
        assert input_tensor.shape[1] == 128, "Should have 128 mel bins"
    except Exception as e:
        return f"FAILURE: Audio loading failed: {e}"
    print(f"    Time: {time.time() - step_start:.2f}s")
    
    # Add batch dimension: [3, 128, T] -> [1, 3, 128, T]
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    # =====================================================================
    # STEP 2: Load and verify MIDI
    # =====================================================================
    print("\n[2] Loading MIDI...")
    try:
        # Use low-level functions to get raw MidiNote objects
        midi_file = load_midi_file(midi_path)
        tempo_map = build_tempo_map_from_tracks(midi_file.tracks, midi_file.ticks_per_beat)
        midi_notes, duration = extract_midi_notes_from_tracks(
            midi_file.tracks,
            midi_file.ticks_per_beat,
            tempo_map
        )
        print(f"    Found {len(midi_notes)} MIDI notes, duration: {duration:.2f}s")
        
        # Convert MidiNote objects to adapter format
        # MidiNote has .midi_note (=pitch) and .time (=start_time)
        notes = [
            NoteAdapter(pitch=note.midi_note, start_time=note.time, velocity=note.velocity)
            for note in midi_notes
        ]
    except Exception as e:
        return f"FAILURE: MIDI loading failed: {e}"
    
    # =====================================================================
    # STEP 3: Create labels
    # =====================================================================
    print("\n[3] Creating labels...")
    try:
        total_frames = input_tensor.shape[3]  # Time dimension
        target = midi_to_frame_array(notes, total_frames, hop_length, 44100)
        
        # Reshape for loss: [C, T] -> [1, T, C]
        target = target.unsqueeze(0).permute(0, 2, 1).to(device)
        print(f"    Target shape: {target.shape}")
        assert target.shape == (1, total_frames, 10), "Target shape mismatch"
    except Exception as e:
        return f"FAILURE: Label encoding failed: {e}"
    
    # =====================================================================
    # STEP 4: Initialize model
    # =====================================================================
    print("\n[4] Initializing model...")
    try:
        model = DrumTranscriber().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10
        )
        print("    Model initialized successfully")
    except Exception as e:
        return f"FAILURE: Model initialization failed: {e}"
    
    # Save directory
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # =====================================================================
    # STEP 5: Training loop
    # =====================================================================
    print(f"\n[5] Running overfit test ({epochs} epochs)...")
    print("    Starting training loop...", flush=True)
    sys.stdout.flush()
    epoch_start = time.time()
    last_print = epoch_start
    
    # Chunked training parameters
    chunk_frames = 2000  # Process N frames at a time (reduces memory pressure)
    accumulation_steps = max(1, total_frames // chunk_frames)
    print(f"    Processing {total_frames} frames in chunks of {chunk_frames} (~{accumulation_steps} steps/epoch)")
    
    try:
        for epoch in range(epochs):
            epoch_loss = 0.0
            step_count = 0
            for chunk_start in range(0, total_frames, chunk_frames):
                print(f"    chunk start: {chunk_start}/{total_frames} frames", end='\r', flush=True)
                chunk_end = min(chunk_start + chunk_frames, total_frames)
                
                # Extract chunk from input/target
                input_chunk = input_tensor[:, :, :, chunk_start:chunk_end]
                target_chunk = target[:, chunk_start:chunk_end, :]
                
                optimizer.zero_grad()
                output = model(input_chunk)
                loss = criterion(output, target_chunk)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                step_count += 1
            
            avg_loss = epoch_loss / step_count
            
            now = time.time()
            elapsed = now - epoch_start
            ms_per_epoch = elapsed * 1000 / (epoch + 1)
            rate = (epoch + 1) / elapsed if elapsed > 0 else 0
            remaining = (epochs - epoch - 1) / rate if rate > 0 else 0
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e} | {ms_per_epoch:.0f}ms/ep | ETA: {remaining:.0f}s | {step_count} steps     ", flush=True)
            last_print = now
            
            # Step the scheduler (ReduceLROnPlateau)
            scheduler.step(avg_loss)
    except Exception as e:
        import traceback
        print(f"\n    ERROR during training: {e}")
        print(traceback.format_exc())
        raise
    
    total_time = time.time() - epoch_start
    print(f"\n    Training completed: {total_time:.1f}s total, {total_time/epochs*1000:.1f}ms/epoch")
    
    # Save model checkpoint
    checkpoint_path = models_dir / "smoke_test.ckpt"
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, checkpoint_path)
    print(f"    Model saved to: {checkpoint_path}")
    
    # =====================================================================
    # STEP 6: Final verification
    # =====================================================================
    print("\n[6] Final verification...")
    final_loss = avg_loss
    
    if final_loss < 0.02:
        print(f"    ✓ Loss {final_loss:.6f} < 0.02 threshold")
        print("    SUCCESS: Pipeline verified!")
        return "SUCCESS"
    else:
        print(f"    ✗ Loss {final_loss:.6f} >= 0.02 threshold")
        print("    FAILURE: Check data alignment")
        return "FAILURE"


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Smoke test for drum transcription')
    parser.add_argument('--epochs', '-e', type=int, default=1, help='Number of epochs (default: 1)')
    parser.add_argument('--hop', type=int, default=512, help='Hop length for spectrogram (default: 512)')
    args = parser.parse_args()
    
    hop_length = args.hop
    
    base = "/Users/jasoncopp/Source/GitHub/larsnet/model-training"
    audio_path = f"{base}/dl-1.wav"
    midi_path = f"{base}/dl-1.mid"
    
    print(f"Audio: {audio_path}")
    print(f"MIDI:  {midi_path}")
    
    if not Path(audio_path).exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)
    if not Path(midi_path).exists():
        print(f"ERROR: MIDI file not found: {midi_path}")
        sys.exit(1)
    
    result = run_smoke_test(audio_path, midi_path, epochs=args.epochs)
    print(f"\nResult: {result}")