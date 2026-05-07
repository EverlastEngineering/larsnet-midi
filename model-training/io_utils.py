"""
io_utils.py — Checkpoint, file I/O, and abort management for model-training.

Provides utilities for versioned checkpoint save/load, abort file lifecycle,
and general file operations used across training scripts.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any, Tuple


def get_models_dir() -> Path:
    """Return the models directory, creating it if needed."""
    from config import MODELS_DIR
    MODELS_DIR.mkdir(exist_ok=True)
    return MODELS_DIR


def find_next_version(models_dir: Path, prefix: str = "train_checkpoint_v") -> int:
    """
    Find the next version number for checkpoint naming.
    
    Args:
        models_dir: Directory containing checkpoint files
        prefix: Checkpoint filename prefix
        
    Returns:
        Next version number (1 if no existing checkpoints)
    """
    existing = list(models_dir.glob(f"{prefix}*.ckpt"))
    versions = []
    for f in existing:
        try:
            v = int(f.name.replace(prefix, "").replace(".ckpt", ""))
            versions.append(v)
        except ValueError:
            pass
    return max(versions) + 1 if versions else 1


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    loss: Optional[float],
    extras: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a model checkpoint.
    
    Args:
        path: Destination file path
        model: PyTorch model
        optimizer: Optional optimizer state
        loss: Optional loss value
        extras: Optional additional metadata dict
    """
    data = {
        'model_state_dict': model.state_dict(),
    }
    if optimizer is not None:
        data['optimizer_state_dict'] = optimizer.state_dict()
    if loss is not None:
        data['loss'] = loss
    if extras is not None:
        data.update(extras)
    
    torch.save(data, path)


def load_checkpoint(path: Path, model: torch.nn.Module, device: str = "cpu") -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Dict[str, Any]]:
    """
    Load a model checkpoint.
    
    Args:
        path: Path to checkpoint file
        model: Model instance to load state into
        device: Target device for loading
        
    Returns:
        Tuple of (model, optimizer, metadata)
        
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If checkpoint is corrupted
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    optimizer = None
    if 'optimizer_state_dict' in ckpt:
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    
    # Collect remaining metadata
    metadata = {k: v for k, v in ckpt.items() if k not in ('model_state_dict', 'optimizer_state_dict')}
    
    return model, optimizer, metadata


def inspect_checkpoint(path: Path) -> None:
    """
    Print human-readable summary of a checkpoint file.
    """
    ckpt = torch.load(path, map_location='cpu')
    print(f"=== Checkpoint: {path.name} ===")
    for k, v in ckpt.items():
        if k == 'model_state_dict':
            print(f"  model_state_dict: {len(v)} layers")
            for layer_name, tensor in v.items():
                print(f"    {layer_name}: {tensor.shape} mean={tensor.mean().item():.4f}")
        elif k == 'optimizer_state_dict':
            print(f"  optimizer_state_dict: state + param_groups")
        elif hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}")
        else:
            print(f"  {k}: {v}")


def pulse_check(model, audio_path: str, midi_path: str = None, threshold: float = 0.5, device: str = None) -> dict:
    """
    Quick sanity check after a few epochs — is the model producing any output at all?
    Loads audio + optional MIDI, runs inference, and prints a health report.
    """
    from train_utils import load_audio, load_midi_notes, build_targets

    if device is None:
        from config import DEVICE
        device = DEVICE

    model.eval()
    stats = {
        "max_prob": 0.0,
        "avg_prob_on_hits": 0.0,
        "trigger_count": 0,
        "ground_truth_count": 0,
        "audio_path": audio_path,
        "threshold": threshold,
    }

    spec = load_audio(audio_path).to(device)
    target_tensor = None

    if midi_path:
        notes, _ = load_midi_notes(midi_path)
        total_frames = spec.shape[3]
        target_tensor = build_targets(notes, total_frames).to(device)
        hit_mask = (target_tensor > 0.5)
        stats["ground_truth_count"] = hit_mask.sum().item()

    with torch.no_grad():
        output = model(spec)
        probs = torch.sigmoid(output)

        stats["max_prob"] = probs.max().item()
        stats["trigger_count"] = (probs > threshold).sum().item()

        if target_tensor is not None and stats["ground_truth_count"] > 0:
            # Flatten both to [1, T*10] for element-wise comparison
            stats["avg_prob_on_hits"] = probs.flatten()[hit_mask.flatten()].mean().item()

    print("\n" + "=" * 40)
    print(" PULSE CHECK")
    print("=" * 40)
    print(f"Audio:              {audio_path}")
    print(f"Threshold:           {threshold}")
    print(f"Highest Probability: {stats['max_prob']:.4f}")
    print(f"Avg Prob on Hits:    {stats['avg_prob_on_hits']:.4f}")
    print(f"Triggers Found:      {stats['trigger_count']}")
    print(f"Expected Hits:       {stats['ground_truth_count']}")
    print("-" * 40)

    if stats["max_prob"] < 0.1:
        print("RESULT: [DEAD] Model outputting near-zero. Check weights or init.")
    elif stats["trigger_count"] == 0:
        print(f"RESULT: [SHY] Max {stats['max_prob']:.2f} but no triggers above {threshold}.")
        print(f"SUGGEST: Lower threshold to {stats['max_prob'] * 0.8:.2f} for testing.")
    elif stats["trigger_count"] > stats["ground_truth_count"] * 10:
        print("RESULT: [NOISY] Way too many triggers. Check loss function.")
    else:
        print("RESULT: [HEALTHY] Model producing output. Proceed with training.")
    print("=" * 40 + "\n")

    return stats


def check_abort(models_dir: Path) -> bool:
    """
    Check if abort file exists.
    
    Args:
        models_dir: Directory containing abort file
        
    Returns:
        True if abort file exists
    """
    return (models_dir / "abort").exists()


def clear_abort(models_dir: Path) -> None:
    """
    Remove abort file if it exists.
    
    Args:
        models_dir: Directory containing abort file
    """
    abort_file = models_dir / "abort"
    if abort_file.exists():
        abort_file.unlink()


def latest_checkpoint(models_dir: Path, prefix: str = "smoke_test_checkpoint_v") -> Optional[Path]:
    """
    Find the latest checkpoint by version number.
    
    Args:
        models_dir: Directory containing checkpoint files
        prefix: Checkpoint filename prefix to filter by (default "smoke_test_checkpoint_v")
        
    Returns:
        Path to latest checkpoint, or None if none exist
    """
    checkpoints = [p for p in models_dir.glob("*.ckpt") if p.name.startswith(prefix)]
    if not checkpoints:
        return None
    
    def parse_version(p: Path) -> int:
        try:
            return int(p.name[len(prefix):].replace(".ckpt", ""))
        except ValueError:
            return -1
    
    checkpoints.sort(key=parse_version)
    return checkpoints[-1]
