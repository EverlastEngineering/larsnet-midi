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
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # =====================================================================
    # STEP 1: Load and verify audio
    # =====================================================================
    print("\n[1] Loading audio...")
    try:
        input_tensor = get_input_tensor(audio_path)
        print(f"    Input shape: {input_tensor.shape}")
        assert input_tensor.shape[0] == 3, "Should have 3 channels"
        assert input_tensor.shape[1] == 128, "Should have 128 mel bins"
    except Exception as e:
        return f"FAILURE: Audio loading failed: {e}"
    
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
        target_tensor = midi_to_frame_array(notes, total_frames, 512, 44100)
        
        # Reshape target for loss: [11, T] -> [1, T, 11]
        target_tensor = target_tensor.unsqueeze(0).permute(0, 2, 1).to(device)
        print(f"    Target shape: {target_tensor.shape}")
        assert target_tensor.shape == (1, total_frames, 11), "Target shape mismatch"
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
        print("    Model initialized successfully")
    except Exception as e:
        return f"FAILURE: Model initialization failed: {e}"
    
    # =====================================================================
    # STEP 5: Training loop
    # =====================================================================
    print(f"\n[5] Running overfit test ({epochs} epochs)...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch:3d} | Loss: {loss.item():.6f}")
    
    # =====================================================================
    # STEP 6: Final verification
    # =====================================================================
    print("\n[6] Final verification...")
    final_loss = loss.item()
    
    if final_loss < 0.01:
        print(f"    ✓ Loss {final_loss:.6f} < 0.01 threshold")
        print("    SUCCESS: Pipeline verified!")
        return "SUCCESS"
    else:
        print(f"    ✗ Loss {final_loss:.6f} >= 0.01 threshold")
        print("    FAILURE: Check data alignment")
        return "FAILURE"


if __name__ == "__main__":
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
    
    result = run_smoke_test(audio_path, midi_path)
    print(f"\nResult: {result}")