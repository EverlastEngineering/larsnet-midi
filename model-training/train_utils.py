"""
train_utils.py — Training utilities for model-training.

Provides pure training helpers: loss computation, chunk iteration,
target preparation, and training loop orchestration.
"""

import torch
import torch.nn as nn
from typing import Any, List, Tuple, Optional
from config import HOP_LENGTH, SAMPLE_RATE


class MultiTaskDrumLoss(nn.Module):
    """
    Multi-task loss for drum transcription.
    
    Splits the 20-dim output into:
    - Channels 0-9: onset classification (BCEWithLogitsLoss)
    - Channels 10-19: velocity regression (masked MSE on onset frames only)
    """
    
    def __init__(self, velocity_weight: float = 2.0, device: str = 'cpu'):
        super().__init__()
        self.velocity_weight = velocity_weight
        
        # Pos weight for onset classification: [Class 0 (Kick), Class 1 (Snare), ...]
        # Kick/Snare/HHC need more weight to stand out
        pos_weight = torch.tensor([150.0, 15.0, 2.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0]).to(device)
        self.onset_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [Batch, Time, 20] raw logits from model
            target: [Batch, Time, 20] target tensor
            
        Returns:
            Scalar loss = onset_loss + (weight * velocity_loss)
        """
        onset_pred = pred[:, :, :10]
        onset_target = target[:, :, :10]
        
        velocity_pred = pred[:, :, 10:]
        velocity_target = target[:, :, 10:]
        
        # Onset loss: standard BCEWithLogitsLoss
        onset_loss = self.onset_criterion(onset_pred, onset_target)
        
        # Velocity loss: masked MSE — only compute on frames where GT onset is active
        # Create mask from onset target: [Batch, Time, 10]
        onset_mask = (onset_target > 0.5).float()
        
        # Expand mask to velocity channels [Batch, Time, 10] → [Batch, Time, 10] (same shape as velocity_target)
        # Each velocity channel corresponds to its onset channel
        velocity_squared_error = (velocity_pred - velocity_target) ** 2
        
        # Apply mask: only compute MSE where onset target > 0.5
        masked_squared_error = velocity_squared_error * onset_mask
        
        # Sum over velocity channels first, then divide by count of valid frames
        num_valid_frames = onset_mask.sum() + 1e-8
        velocity_loss = masked_squared_error.sum() / num_valid_frames
        
        total_loss = onset_loss + (self.velocity_weight * velocity_loss)
        return total_loss


def build_targets(
    midi_notes: List[Any],
    total_frames: int,
) -> torch.Tensor:
    """
    Convert MIDI notes to a target tensor for training.
    
    Args:
        midi_notes: List of note objects with .pitch, .start_time, .velocity attributes
        total_frames: Total spectrogram frames
        
    Returns:
        Target tensor of shape [1, total_frames, 20]
        Channels 0-9: binary onset heatmap
        Channels 10-19: normalized velocity targets
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
        target_tensor: Shape [1, T, 20]
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
    
    # Multi-task loss: onset classification (BCEWithLogitsLoss) + velocity regression (masked MSE)
    from config import get_velocity_weight
    velocity_weight = get_velocity_weight()
    criterion = MultiTaskDrumLoss(velocity_weight=velocity_weight, device=device)
    
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
