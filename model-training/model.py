"""
DrumTranscriber - CRNN Model Architecture

Convolutional layers extract frequency-domain features (transients).
Bi-directional GRU processes temporal sequences.
Linear layer maps 256 GRU features to 10 drum class probabilities.
"""

import torch
import torch.nn as nn


class DrumTranscriber(nn.Module):
    """
    CRNN for drum transcription.
    
    Architecture:
    - Conv2d blocks extract frequency-domain features
    - MaxPool2d((2,1)) reduces frequency height, preserves time
    - Bi-directional GRU for temporal processing
    - Linear layer for 10-class probability output
    """
    
    def __init__(self):
        super().__init__()
        
        # THE EYES: Conv block extracts frequency-domain features (transients)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # Reduce frequency height, preserve time resolution
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        
        # THE BRAIN: GRU processes the timeline
        # Bi-directional allows the model to look 'ahead' to see if a cymbal tail follows
        # Input size (2048) comes from 64 filters * (128 / 2 / 2) freq bins
        self.rnn = nn.GRU(2048, 128, batch_first=True, bidirectional=True)
        
        # Linear layer maps 256 GRU features to 10 drum class probabilities.
        self.fc = nn.Linear(256, 10)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [Batch, Channels, Freq, Time]
               Channels should be 3 (L, R, Width)
               Freq should be 128 (mel bins)
        
        Returns:
            Output tensor of shape [Batch, Time, 10]
            Each of the 10 values is a probability 0.0-1.0
        """
        # Conv block: [B, 3, 128, T] -> [B, 64, 32, T]
        x = self.conv(x)
        
        # Prepare for RNN: Reorder to [Batch, Time, Features]
        # [B, 64, 32, T] -> [B, T, 64*32] = [B, T, 2048]
        x = x.permute(0, 3, 1, 2).flatten(2)
        
        # Temporal processing: [B, T, 2048] -> [B, T, 256]
        x, _ = self.rnn(x)
        
        # Return probability (0.0 - 1.0) for every frame
        return torch.sigmoid(self.fc(x))


if __name__ == "__main__":
    # Smoke test
    device = torch.device("cpu")
    model = DrumTranscriber().to(device)
    
    # Dummy input: [Batch, Channels, Freq, Time] = [1, 3, 128, 100]
    dummy = torch.randn(1, 3, 128, 100).to(device)
    
    output = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected:     [1, 100, 10]")
    
    # Verify shapes
    assert output.shape == (1, 100, 10), f"Shape mismatch: {output.shape}"
    print("Model forward pass: OK")