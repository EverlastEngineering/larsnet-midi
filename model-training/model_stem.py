"""
Per-stem transcriber model.

Architecturally identical to DrumTranscriber (3-channel Conv + BiGRU +
Linear), but with a smaller output dim. Per-stem models only emit
`num_classes * 2` channels: half for onset probability, half for velocity.

Default num_classes=1: binary "is there a hit of this stem here?" detection.
This is the simplest per-stem problem and the natural first step.

A 5-class model (snare center, snare rim, hihat closed, hihat open, ...)
is a future enhancement.
"""

import torch
import torch.nn as nn


class StemTranscriber(nn.Module):
    """
    Per-stem onset + velocity model.

    Input:  [B, 3, 128, T]   (3-channel mel-spec; same as DrumTranscriber)
    Output: [B, T, 2 * num_classes]
              channels 0..N-1:    onset logits
              channels N..2N-1:   velocity logits (sigmoid for [0,1])
    """

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # 128 -> 64 freq
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # 64 -> 32 freq
        )
        # 64 * 32 = 2048 features per timestep
        self.rnn = nn.GRU(2048, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, num_classes * 2)

    def forward(self, x):
        """
        Args:
            x: [B, 3, 128, T] mel-spec
        Returns:
            logits: [B, T, 2*num_classes] raw logits (no sigmoid)
        """
        x = self.conv(x)                          # [B, 64, 32, T]
        x = x.permute(0, 3, 1, 2).flatten(2)      # [B, T, 2048]
        x, _ = self.rnn(x)                        # [B, T, 256]
        return self.fc(x)                         # [B, T, 2*num_classes]


class PerStemLoss(nn.Module):
    """
    Multi-task loss for per-stem model: BCEWithLogits(onset) + masked MSE(velocity).
    Identical formulation to MultiTaskDrumLoss but smaller (2*num_classes outputs).
    """

    def __init__(self, velocity_weight: float = 1.0):
        super().__init__()
        self.velocity_weight = velocity_weight
        self.onset_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple:
        """
        Args:
            pred:   [B, T, 2*num_classes]  raw logits
            target: [B, T, 2*num_classes]  (channels 0..N-1 binary, N..2N-1 velocity in [0,1])
        Returns:
            (total_loss, loss_dict)
        """
        n = pred.shape[-1] // 2
        onset_pred = pred[..., :n]
        onset_target = target[..., :n]
        onset_loss = self.onset_criterion(onset_pred, onset_target)

        vel_pred = pred[..., n:]
        vel_target = target[..., n:]
        onset_mask = (onset_target > 0.5).float()
        vel_prob = torch.sigmoid(vel_pred)
        vel_mse = (vel_prob - vel_target) ** 2
        masked_mse = (vel_mse * onset_mask).sum() / (onset_mask.sum() + 1e-8)

        total = onset_loss + self.velocity_weight * masked_mse
        return total, {
            "onset_loss": onset_loss.item(),
            "velocity_loss": masked_mse.item(),
        }


if __name__ == "__main__":
    # Sanity check
    model = StemTranscriber(num_classes=1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"StemTranscriber (num_classes=1) params: {n_params:,}")
    # vs DrumTranscriber ~600k
    dummy = torch.randn(2, 3, 128, 1000)
    out = model(dummy)
    print(f"Output shape: {out.shape}  (expected: [2, 1000, 2])")
    assert out.shape == (2, 1000, 2), f"Shape mismatch: {out.shape}"
    print("Forward pass: OK")
