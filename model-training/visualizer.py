"""
Verification Visualizer - Alignment Check

Overlays ground truth labels on the spectrogram.
Alignment check: Vertical lines in top plot must match hot spots in bottom plot.
"""

import torch
import matplotlib.pyplot as plt


LABEL_NAMES = ['Kick', 'Snare', 'HHC', 'HHO', 'TH', 'TM', 'TL', 'Cr', 'Ri', 'Ch', 'Sp']


def plot_alignment_check(spec_tensor: torch.Tensor, label_tensor: torch.Tensor):
    """
    Overlays ground truth labels on the spectrogram.
    
    Args:
        spec_tensor: Input feature tensor [3, 128, Time] (dB values)
        label_tensor: Target label tensor [11, Time] (0.0-1.0 probabilities)
    """
    # Detach and convert to numpy if needed
    spec_np = spec_tensor.detach().numpy()
    label_np = label_tensor.detach().numpy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # Plot Channel 0 (Left Spec)
    im1 = ax1.imshow(spec_np[0], aspect='auto', origin='lower')
    ax1.set_title("Input Feature: Left Spectrogram (dB)")
    ax1.set_ylabel("Mel Frequency Bins")
    fig.colorbar(im1, ax=ax1, label="dB")
    
    # Plot 11-Channel Label Matrix
    im2 = ax2.imshow(label_np, aspect='auto', origin='lower', cmap='magma')
    ax2.set_yticks(range(11))
    ax2.set_yticklabels(LABEL_NAMES)
    ax2.set_title("Target: 11-Channel Heatmap")
    ax2.set_xlabel("Time Frames")
    ax2.set_ylabel("Drum Class")
    fig.colorbar(im2, ax=ax2, label="Probability")
    
    plt.tight_layout()
    plt.savefig('/tmp/alignment_check.png', dpi=150)
    print("Saved to /tmp/alignment_check.png")
    plt.show()


if __name__ == "__main__":
    # Quick demo with random data
    print("Generating demo visualization...")
    
    # Random spectrogram
    spec = torch.randn(3, 128, 500)
    
    # Random labels with some structure
    labels = torch.zeros(11, 500)
    labels[0, 50] = 1.0   # Kick at frame 50
    labels[1, 120] = 1.0  # Snare at frame 120
    labels[2, 200:204] = torch.tensor([1.0, 0.8, 0.5, 0.2])  # HH with smear
    
    plot_alignment_check(spec, labels)