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
    
    # Velocity importance multiplier per class.
    # HHC (index 2) has 2.5x weight because high-frequency transients
    # are more prone to velocity compression.
    VELOCITY_CLASS_WEIGHTS = [1.0, 1.0, 2.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    def __init__(self, velocity_weight: float = 2.0, device: str = 'cpu'):
        super().__init__()
        self.velocity_weight = velocity_weight
        
        # Register buffers once — they move to GPU with the model and stay there.
        # No per-forward-pass tensor allocation.
        #
        # Weights computed from Roland e-GMD dataset frequency analysis.
        # Aggregated from Roland pitches to our 10-class mapping:
        #   Kick(36): 88067, Snare(38+40+37): 134745, HHC(42+22+44): 118798,
        #   HHO(46+26): 14148, TomHigh(50+48): 14706, TomMid(47+45): 5257,
        #   TomLow(43+58): 12263, Crash1(49+55): 6287, Crash2(57+52): 2878,
        #   Ride(51+59+53): 51634. Total ~449283.
        # Inverse frequency weighting: weight = total / count, then normalized so sum=10.
        pos_weight = torch.tensor([5.10, 3.34, 3.78, 31.77, 30.56, 85.45, 36.65, 71.49, 156.13, 8.70])
        self.register_buffer('pos_weight', pos_weight)
        
        velocity_class_weights = torch.tensor(self.VELOCITY_CLASS_WEIGHTS)
        self.register_buffer('velocity_class_weights', velocity_class_weights)
        
        self.onset_criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred: [Batch, Time, 20] raw logits from model
            target: [Batch, Time, 20] target tensor
            
        Returns:
            Tuple of (scalar_loss, dict with onset_loss and velocity_loss for logging)
        """
        onset_pred = pred[:, :, :10]
        onset_target = target[:, :, :10]
        
        velocity_pred = pred[:, :, 10:]
        velocity_target = target[:, :, 10:]
        
        # Onset loss: standard BCEWithLogitsLoss
        onset_loss = self.onset_criterion(onset_pred, onset_target)
        
        # Velocity loss: masked MSE — only compute on frames where GT onset is active
        onset_mask = (onset_target > 0.5).float()
        
        # Apply sigmoid to velocity predictions to match 0.0-1.0 target range
        vel_prob = torch.sigmoid(velocity_pred)
        velocity_squared_error = (vel_prob - velocity_target) ** 2
        
        # Apply per-class weight before averaging
        weights_expanded = self.velocity_class_weights.view(1, 1, 10)
        weighted_squared_error = velocity_squared_error * weights_expanded
        
        # Apply onset mask: only compute MSE where onset target > 0.5
        masked_squared_error = weighted_squared_error * onset_mask
        
        # Sum over velocity channels, then divide by count of valid frames
        num_valid_frames = onset_mask.sum() + 1e-8
        velocity_loss = masked_squared_error.sum() / num_valid_frames
        
        total_loss = onset_loss + (self.velocity_weight * velocity_loss)
        return total_loss, {'onset_loss': onset_loss.item(), 'velocity_loss': velocity_loss.item()}


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
    clip_grad: Optional[float] = 1.0,
) -> Tuple[float, torch.Tensor]:
    """
    Run one training step on a chunk.
    
    Args:
        model: PyTorch model
        input_chunk: Input tensor slice
        target_chunk: Target tensor slice
        optimizer: Optimizer
        criterion: Loss function
        clip_grad: Max gradient norm for clipping (None to skip)
        
    Returns:
        Tuple of (loss_value, output_tensor)
    """
    optimizer.zero_grad()
    output = model(input_chunk)
    loss, _ = criterion(output, target_chunk)
    loss.backward()
    if clip_grad is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    optimizer.step()
    return loss.item(), output


def compute_loss(
    model: torch.nn.Module,
    input_chunk: torch.Tensor,
    target_chunk: torch.Tensor,
    criterion: nn.Module,
) -> Tuple[float, torch.Tensor]:
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
        loss, _ = criterion(output, target_chunk)
    return loss.item(), output


def run_eval(
    model: torch.nn.Module,
    val_lines: list,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float, float]:
    """
    Run validation pass over all files in val_lines.

    Returns:
        Tuple of (avg_loss, avg_onset_loss, avg_velocity_loss)
    """
    from pathlib import Path
    from config import get_chunk_frames
    model.eval()
    total_loss = 0.0
    total_onset = 0.0
    total_velocity = 0.0
    count = 0

    for line in val_lines:
        parts = line.split('\t', 1)
        if len(parts) < 2:
            continue
        audio_path = parts[0].strip()
        midi_path = parts[1].strip()

        if not Path(audio_path).exists():
            continue

        try:
            input_tensor = load_audio(audio_path).to(device)
            notes, duration = load_midi_notes(midi_path)
            total_frames = input_tensor.shape[3]
            target_tensor = build_targets(notes, total_frames).to(device)
        except Exception:
            continue

        file_losses = []
        file_onset = []
        file_velocity = []

        for chunk_start in range(0, total_frames, get_chunk_frames()):
            input_chunk, target_chunk = get_chunk(input_tensor, target_tensor, chunk_start, get_chunk_frames())
            loss_val, loss_dict = compute_loss(model, input_chunk, target_chunk, criterion)
            file_losses.append(loss_val)
            file_onset.append(loss_dict['onset_loss'])
            file_velocity.append(loss_dict['velocity_loss'])

        if file_losses:
            total_loss += sum(file_losses) / len(file_losses)
            total_onset += sum(file_onset) / len(file_onset)
            total_velocity += sum(file_velocity) / len(file_velocity)
            count += 1

    model.train()
    avg_loss = total_loss / count if count > 0 else 0.0
    avg_onset = total_onset / count if count > 0 else 0.0
    avg_velocity = total_velocity / count if count > 0 else 0.0
    return avg_loss, avg_onset, avg_velocity


def setup_training(
    model: Optional[torch.nn.Module] = None,
    learning_rate: float = 1e-4,
    device: Optional[str] = None,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.1,
    clip_grad: Optional[float] = 1.0,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, nn.Module, Optional[torch.optim.lr_scheduler.ReduceLROnPlateau], float]:
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
    
    return model, optimizer, criterion, scheduler, clip_grad


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
