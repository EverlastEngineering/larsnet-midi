"""
train_utils.py — Training utilities for model-training.

Provides pure training helpers: loss computation, chunk iteration,
target preparation, and training loop orchestration.
"""

import torch
import torch.nn as nn
from typing import Any, List, Tuple, Optional
from config import HOP_LENGTH, SAMPLE_RATE


def build_targets(
    midi_notes: List[Any],
    total_frames: int,
) -> torch.Tensor:
    """
    Convert MIDI notes to a target tensor for training.
    
    Args:
        midi_notes: List of note objects with .pitch, .start_time attributes
        total_frames: Total spectrogram frames
        
    Returns:
        Target tensor of shape [1, total_frames, 10]
    """
    from label_encoder import midi_to_frame_array
    
    target_tensor = midi_to_frame_array(midi_notes, total_frames, HOP_LENGTH, SAMPLE_RATE)
    target_tensor = target_tensor.unsqueeze(0).permute(0, 2, 1)
    return target_tensor


def get_chunk(
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    chunk_start: int,
    chunk_frames: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Slice input and target tensors for a chunk.
    
    Args:
        input_tensor: Shape [1, 1, 128, T]
        target_tensor: Shape [1, T, 10]
        chunk_start: Starting frame
        chunk_frames: Number of frames to include
        
    Returns:
        Tuple of (input_chunk, target_chunk)
    """
    chunk_end = min(chunk_start + chunk_frames, input_tensor.shape[3])
    input_chunk = input_tensor[:, :, :, chunk_start:chunk_end]
    target_chunk = target_tensor[:, chunk_start:chunk_end, :]
    return input_chunk, target_chunk


def train_chunk(
    model: torch.nn.Module,
    input_chunk: torch.Tensor,
    target_chunk: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> Tuple[float, torch.Tensor]:
    """
    Run one training step on a chunk.
    
    Args:
        model: PyTorch model
        input_chunk: Input tensor slice
        target_chunk: Target tensor slice
        optimizer: Optimizer
        criterion: Loss function
        
    Returns:
        Tuple of (loss_value, output_tensor)
    """
    optimizer.zero_grad()
    output = model(input_chunk)
    loss = criterion(output, target_chunk)
    loss.backward()
    optimizer.step()
    return loss.item(), output


def compute_loss(
    model: torch.nn.Module,
    input_chunk: torch.Tensor,
    target_chunk: torch.Tensor,
    criterion: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute loss without weight updates (for evaluation).
    
    Args:
        model: PyTorch model
        input_chunk: Input tensor
        target_chunk: Target tensor
        criterion: Loss function
        
    Returns:
        Tuple of (loss_value, output_tensor)
    """
    with torch.no_grad():
        output = model(input_chunk)
        loss = criterion(output, target_chunk)
    return loss.item(), output


def setup_training(
    model: Optional[torch.nn.Module] = None,
    learning_rate: float = 1e-3,
    device: Optional[str] = None,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.1,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, nn.Module, Optional[torch.optim.lr_scheduler.ReduceLROnPlateau]]:
    """
    Initialize or reset training components.
    
    Args:
        model: Existing model to reuse, or None to create fresh
        learning_rate: Learning rate for optimizer
        device: Device string ('mps', 'cuda', 'cpu')
        scheduler_patience: ReduceLROnPlateau patience
        scheduler_factor: ReduceLROnPlateau factor
        
    Returns:
        Tuple of (model, optimizer, criterion, scheduler)
    """
    if device is None:
        from config import get_device
        device = get_device()
    
    if model is None:
        from model import DrumTranscriber
        model = DrumTranscriber()
    
    model = model.to(device)
    
    # 1. POS_WEIGHT: This is the 'Contrast' knob.
    # We tell the model that a drum hit (1) is 25x more important 
    # than a silent frame (0). This kills the 0.51 'gray smear'.
    # [Class 0 (Blue), Class 1 (Orange), Class 2 (Green)]
    # We give Orange and Blue more 'Gain' to help them stand out
    weights = [150.0, 15.0, 2.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0]
    pos_weight = torch.tensor(weights).to(device)
    
    # 2. CRITERION: Swap BCELoss for BCEWithLogitsLoss.
    # This expects raw values from your model (no sigmoid) 
    # and is numerically stable.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=scheduler_factor, 
        patience=scheduler_patience
    )
    
    return model, optimizer, criterion, scheduler


def load_audio(path: str) -> torch.Tensor:
    """
    Load audio file and return spectrogram tensor.
    
    Args:
        path: Path to audio file
        
    Returns:
        Tensor of shape [1, 1, 128, T]
    """
    from feature_extractor import get_input_tensor
    
    tensor = get_input_tensor(path)
    assert tensor.shape[0] == 1, "Should have 1 channel"
    assert tensor.shape[1] == 128, "Should have 128 mel bins"
    return tensor.unsqueeze(0)


def load_midi_notes(path: str) -> Tuple[List[Any], float]:
    """
    Load MIDI file and return drum notes.
    
    Args:
        path: Path to MIDI file
        
    Returns:
        Tuple of (notes_list, duration_seconds)
    """
    from midi_shell import load_midi_file
    from midi_core import extract_midi_notes_from_tracks, build_tempo_map_from_tracks
    from label_encoder import NoteAdapter
    
    midi_file = load_midi_file(path)
    tempo_map = build_tempo_map_from_tracks(midi_file.tracks, midi_file.ticks_per_beat)
    midi_notes, duration = extract_midi_notes_from_tracks(
        midi_file.tracks, midi_file.ticks_per_beat, tempo_map
    )
    notes = [NoteAdapter(pitch=n.midi_note, start_time=n.time, velocity=n.velocity) for n in midi_notes]
    return notes, duration
