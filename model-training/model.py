"""
DrumTranscriber - CRNN Model Architecture

Convolutional layers extract frequency-domain features (transients).
Bi-directional GRU processes temporal sequences.
Multi-head output: gatekeeper, groupings, precision, velocity.
"""

import torch
import torch.nn as nn


# 24 unique Roland TD-17 pitches
ROLAND_PITCHES = [22, 26, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 57, 58, 59]
ROLAND_TO_IDX = {p: i for i, p in enumerate(ROLAND_PITCHES)}
IDX_TO_ROLAND = {i: p for p, i in ROLAND_TO_IDX.items()}


class GatekeeperHead(nn.Module):
    """Head 1: 3-class instrument family (Kick/Snare/Toms/Cymbals)"""
    def __init__(self, in_features=256):
        super().__init__()
        self.fc = nn.Linear(in_features, 3)
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))


class GroupingsHead(nn.Module):
    """Head 2: 10-class drum categories"""
    def __init__(self, in_features=256):
        super().__init__()
        self.fc = nn.Linear(in_features, 10)
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))


class PrecisionHead(nn.Module):
    """Head 3: 24-channel per-Roland-pitch trigger detection"""
    def __init__(self, in_features=256):
        super().__init__()
        self.fc = nn.Linear(in_features, 24)
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))


class VelocityHead(nn.Module):
    """Head 4: 24-channel per-Roland-pitch velocity regression"""
    def __init__(self, in_features=256):
        super().__init__()
        self.fc = nn.Linear(in_features, 24)
    
    def forward(self, x):
        # Constrain to 0.0-1.0 range (MIDI velocity normalized)
        return torch.sigmoid(self.fc(x))


class DrumTranscriber(nn.Module):
    """
    CRNN for drum transcription with multi-task learning heads.
    
    Architecture:
    - Conv2d blocks extract frequency-domain features
    - MaxPool2d((2,1)) reduces frequency height, preserves time
    - Bi-directional GRU for temporal processing
    - Four heads: gatekeeper, groupings, precision, velocity
    """
    
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        
        self.rnn = nn.GRU(2048, 128, batch_first=True, bidirectional=True)
        
        self.gatekeeper = GatekeeperHead(256)
        self.groupings = GroupingsHead(256)
        self.precision = PrecisionHead(256)
        self.velocity = VelocityHead(256)
        
    def forward(self, x):
        """
        Args:
            x: [Batch, 1, Freq, Time] - Freq=128 mel bins
        Returns:
            Dict with 4 heads:
              - gatekeeper: [Batch, Time, 3]
              - groupings: [Batch, Time, 10]
              - precision: [Batch, Time, 24]
              - velocity: [Batch, Time, 24]
        """
        x = self.conv(x)
        x = x.permute(0, 3, 1, 2).flatten(2)
        x, _ = self.rnn(x)
        
        return {
            'gatekeeper': self.gatekeeper(x),
            'groupings': self.groupings(x),
            'precision': self.precision(x),
            'velocity': self.velocity(x)
        }


if __name__ == "__main__":
    device = torch.device("cpu")
    model = DrumTranscriber().to(device)
    
    # Dummy input: [Batch, 1, Freq, Time] = [1, 1, 128, 100]
    dummy = torch.randn(1, 1, 128, 100).to(device)
    
    output = model(dummy)
    print(f"Input shape:       {dummy.shape}")
    print(f"gatekeeper shape:  {output['gatekeeper'].shape}")
    print(f"groupings shape:   {output['groupings'].shape}")
    print(f"precision shape:   {output['precision'].shape}")
    print(f"velocity shape:    {output['velocity'].shape}")
    
    assert output['gatekeeper'].shape == (1, 100, 3), f"gatekeeper shape mismatch"
    assert output['groupings'].shape == (1, 100, 10), f"groupings shape mismatch"
    assert output['precision'].shape == (1, 100, 24), f"precision shape mismatch"
    assert output['velocity'].shape == (1, 100, 24), f"velocity shape mismatch"
    print("Model forward pass: OK")