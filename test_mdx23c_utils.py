"""
Test script for MDX23C model loading utilities.
"""
import pytest
import torch
from pathlib import Path
from mdx23c_utils import (
    load_mdx23c_checkpoint,
    get_checkpoint_hyperparameters,
    MDXLoadError,
    is_mdx_checkpoint,
)


def test_is_mdx_checkpoint():
    """Test checkpoint detection."""
    ckpt_path = Path("mdx_models/drumsep_5stems_mdx23c_jarredou.ckpt")
    if ckpt_path.exists():
        assert is_mdx_checkpoint(ckpt_path)
    
    # Non-existent file
    assert not is_mdx_checkpoint("nonexistent.ckpt")


def test_get_checkpoint_hyperparameters():
    """Test hyperparameter extraction."""
    ckpt_path = Path("mdx_models/drumsep_5stems_mdx23c_jarredou.ckpt")
    config_path = Path("mdx_models/config_mdx23c.yaml")
    
    if not ckpt_path.exists():
        pytest.skip("Checkpoint file not found")
    
    params = get_checkpoint_hyperparameters(ckpt_path, config_path)
    
    # Check that we got a dict with expected keys
    assert isinstance(params, dict)
    assert 'audio' in params
    assert 'model' in params
    assert 'training' in params
    
    # Check specific audio params
    assert params['audio']['sample_rate'] == 44100
    assert params['audio']['n_fft'] == 2048
    assert params['audio']['hop_length'] == 512


def test_load_mdx23c_checkpoint():
    """Test model loading (fast test - no inference)."""
    ckpt_path = Path("mdx_models/drumsep_5stems_mdx23c_jarredou.ckpt")
    config_path = Path("mdx_models/config_mdx23c.yaml")
    
    if not ckpt_path.exists():
        pytest.skip("Checkpoint file not found")
    
    # Load on CPU
    model = load_mdx23c_checkpoint(ckpt_path, config_path, device="cpu")
    
    # Check that we got a module in eval mode
    assert isinstance(model, torch.nn.Module)
    assert not model.training
    
    # Verify model has expected attributes
    assert hasattr(model, 'forward')
    assert hasattr(model, 'stft')


@pytest.mark.slow
def test_mdx23c_inference():
    """Test model inference (slow test - full chunk processing)."""
    ckpt_path = Path("mdx_models/drumsep_5stems_mdx23c_jarredou.ckpt")
    config_path = Path("mdx_models/config_mdx23c.yaml")
    
    if not ckpt_path.exists():
        pytest.skip("Checkpoint file not found")
    
    model = load_mdx23c_checkpoint(ckpt_path, config_path, device="cpu")
    params = get_checkpoint_hyperparameters(ckpt_path, config_path)
    chunk_size = params['audio']['chunk_size']
    
    # Use a smaller input for faster testing
    test_size = min(chunk_size, 44100)  # 1 second or chunk_size, whichever is smaller
    dummy_input = torch.randn(1, 2, test_size)
    
    # Pad to chunk_size if needed
    if test_size < chunk_size:
        dummy_input = torch.nn.functional.pad(dummy_input, (0, chunk_size - test_size))
    
    with torch.no_grad():
        output = model(dummy_input)
    
    # Check output shape
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == 1  # batch
    if len(output.shape) == 4:
        # Multi-instrument output: (batch, instruments, channels, time)
        assert output.shape[1] == 5  # 5 instruments
        assert output.shape[2] == 2  # stereo
    else:
        # Single-instrument output: (batch, channels, time)
        assert output.shape[1] == 2  # stereo
    

def test_load_without_config_path():
    """Test that config is auto-discovered."""
    ckpt_path = Path("mdx_models/drumsep_5stems_mdx23c_jarredou.ckpt")
    
    if not ckpt_path.exists():
        pytest.skip("Checkpoint file not found")
    
    # Should auto-find config_mdx23c.yaml in same directory
    model = load_mdx23c_checkpoint(ckpt_path, device="cpu")
    assert isinstance(model, torch.nn.Module)


def test_load_nonexistent_checkpoint():
    """Test error handling for missing checkpoint."""
    with pytest.raises(MDXLoadError, match="Checkpoint not found"):
        load_mdx23c_checkpoint("nonexistent.ckpt")


if __name__ == "__main__":
    # Run basic smoke test (fast version)
    print("Testing MDX23C utilities...")
    
    ckpt_path = Path("mdx_models/drumsep_5stems_mdx23c_jarredou.ckpt")
    config_path = Path("mdx_models/config_mdx23c.yaml")
    
    if not ckpt_path.exists():
        print(f"Checkpoint not found at {ckpt_path}")
        exit(1)
    
    print(f"\n1. Testing hyperparameter loading...")
    params = get_checkpoint_hyperparameters(ckpt_path, config_path)
    print(f"   ✓ Loaded config with keys: {list(params.keys())}")
    
    print(f"\n2. Testing model loading (fast)...")
    model = load_mdx23c_checkpoint(ckpt_path, config_path, device="cpu")
    print(f"   ✓ Loaded model: {type(model).__name__}")
    print(f"   ✓ Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    print(f"\n3. Testing quick inference (1 second)...")
    # Use just 1 second for smoke test
    test_size = params['audio']['sample_rate']
    chunk_size = params['audio']['chunk_size']
    dummy_input = torch.randn(1, 2, test_size)
    
    # Pad to chunk_size
    if test_size < chunk_size:
        dummy_input = torch.nn.functional.pad(dummy_input, (0, chunk_size - test_size))
    
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   ✓ Input shape: {tuple(dummy_input.shape)}")
    print(f"   ✓ Output shape: {tuple(output.shape)}")
    
    print("\n✅ All smoke tests passed!")
    print("\nRun 'pytest test_mdx23c_utils.py -v' for full test suite")
    print("Run 'pytest test_mdx23c_utils.py -v -m slow' to include slow inference tests")
