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
    def __init__(self, velocity_weight: float = 2.0, device: str = 'cpu'):
        super().__init__()
        self.velocity_weight = velocity_weight
        
        # Pos weight for onset classification
        pos_weight = torch.tensor([15.0, 15.0, 15.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0])
        self.register_buffer('pos_weight', pos_weight)
        
        # Velocity importance multiplier per class
        velocity_class_weights = torch.tensor([1.0, 1.0, 2.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.register_buffer('velocity_class_weights', velocity_class_weights)
        
        self.onset_criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
        # Placeholders for logging
        self.last_onset_loss = 0.0
        self.last_velocity_loss = 0.0
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Expected pred/target shape: [Batch, Time, 20]
        onset_pred = pred[:, :, :10]
        onset_target = target[:, :, :10]
        velocity_pred = pred[:, :, 10:]
        velocity_target = target[:, :, 10:]
        
        # 1. Onset Loss
        onset_loss = self.onset_criterion(onset_pred, onset_target)
        
        # 2. Velocity Loss (Masked to hit frames only)
        # Even more inclusive: capture any frame with target > 0.01
        onset_mask = (onset_target > 0.01).float()
        
        # Apply Sigmoid to velocity predictions to match 0.0-1.0 target range
        vel_prob = torch.sigmoid(velocity_pred)
        
        # Compute MSE
        velocity_squared_error = (vel_prob - velocity_target) ** 2
        
        # Apply class-specific velocity weighting (expanded to match batch/time)
        weighted_error = velocity_squared_error * self.velocity_class_weights.view(1, 1, 10)
        
        # Masked selection
        masked_error = weighted_error * onset_mask
        
        num_hits = onset_mask.sum()
        
        if num_hits > 0:
            velocity_loss = masked_error.sum() / num_hits
        else:
            velocity_loss = torch.tensor(0.0, device=pred.device)
            # DEBUG: If velocity is 0, let's see if the target actually had hits
            if target.max() > 0:
                 # This would mean hits exist but maybe in the wrong channels?
                 pass

        self.last_onset_loss = onset_loss.item()
        self.last_velocity_loss = velocity_loss.item()
        
        return onset_loss + (self.velocity_weight * velocity_loss)


def setup_training(
    model: nn.Module,
    learning_rate: float,
    device: str,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 5
) -> Tuple[nn.Module, torch.optim.Optimizer, nn.Module, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = MultiTaskDrumLoss(device=device)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=scheduler_factor, 
        patience=scheduler_patience
    )
    
    return model, optimizer, criterion, scheduler


def load_audio(path: str) -> torch.Tensor:
    from feature_extractor import get_input_tensor
    tensor = get_input_tensor(path)
    
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
        
    return tensor


def load_midi_notes(path: str) -> Tuple[List[Any], float]:
    from midi_shell import load_midi_file
    from midi_core import extract_midi_notes_from_tracks, build_tempo_map_from_tracks
    from label_encoder import NoteAdapter
    
    midi_file = load_midi_file(path)
    tempo_map = build_tempo_map_from_tracks(midi_file.tracks, midi_file.ticks_per_beat)
    notes, duration = extract_midi_notes_from_tracks(midi_file.tracks, midi_file.ticks_per_beat, tempo_map)
    
    adapted_notes = []
    for n in notes:
        start_val = getattr(n, 'start', getattr(n, 'start_time', None))
        if start_val is None:
            if isinstance(n, dict):
                start_val = n.get('start', n.get('start_time', 0.0))
            else:
                start_val = 0.0
            
        adapted_notes.append(NoteAdapter(n, start_val))
    
    return adapted_notes, duration


def build_targets(midi_notes: List[Any], num_frames: int) -> torch.Tensor:
    # Initialize targets: [Frames, 20]
    targets = torch.zeros((num_frames, 20))
    frames_per_second = SAMPLE_RATE / HOP_LENGTH
    
    pitch_to_idx = {
        36: 0, 38: 1, 42: 2, 46: 3, 47: 4, 
        43: 5, 50: 6, 49: 7, 57: 8, 51: 9
    }
    
    # We'll track if we actually added anything for a quick sanity check
    hits_added = 0
    
    for note in midi_notes:
        pitch = getattr(note, 'pitch', None)
        if pitch is None and isinstance(note, dict):
            pitch = note.get('pitch')
            
        if pitch in pitch_to_idx:
            idx = pitch_to_idx[pitch]
            
            start_time = getattr(note, 'start', getattr(note, 'start_time', None))
            if start_time is None and isinstance(note, dict):
                start_time = note.get('start', note.get('start_time', 0.0))
            elif start_time is None:
                start_time = 0.0
                
            frame = int(start_time * frames_per_second)
            
            if 0 <= frame < num_frames:
                hits_added += 1
                # Onset Channel
                targets[frame, idx] = 1.0
                
                # Velocity Channel (idx + 10)
                vel = getattr(note, 'velocity', 100)
                if isinstance(note, dict):
                    vel = note.get('velocity', 100)
                
                scaled_vel = (float(vel) / 127.0) ** 0.8
                targets[frame, idx + 10] = scaled_vel
                
                # Expand window to 3-5 frames to ensure the loss function mask catches it
                # even if the transient is slightly offset from the exact frame center.
                for offset in [-2, -1, 1, 2]:
                    f_off = frame + offset
                    if 0 <= f_off < num_frames:
                        # Onset target needs to be > 0.01 for the mask to work
                        targets[f_off, idx] = max(targets[f_off, idx], 0.5)
                        targets[f_off, idx + 10] = scaled_vel

    return targets