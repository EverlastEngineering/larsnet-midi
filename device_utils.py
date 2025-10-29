"""
Device detection and selection utilities for PyTorch.

Provides automatic device detection with priority: MPS → CUDA → CPU
"""
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Cache for device properties to avoid repeated queries
_device_info_cache = {}


def detect_best_device(prefer_gpu: bool = True, verbose: bool = True) -> str:
    """
    Detect the best available device for PyTorch processing.
    
    Priority order:
    1. MPS (Metal Performance Shaders) - macOS GPU
    2. CUDA - NVIDIA GPU
    3. CPU - fallback
    
    Args:
        prefer_gpu: If False, will return 'cpu' even if GPU available
        verbose: If True, logs device selection reasoning
        
    Returns:
        Device string: 'mps', 'cuda', or 'cpu'
    """
    if not prefer_gpu:
        if verbose:
            logger.info("GPU disabled by user preference, using CPU")
        return 'cpu'
    
    # Check for MPS (Apple Silicon / Metal)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        if verbose:
            logger.info("✅ MPS (Metal) available - using GPU acceleration on macOS")
            # Log PyTorch version for MPS capability reference
            pytorch_version = torch.__version__
            logger.info(f"   PyTorch version: {pytorch_version}")
            # MPS support improved significantly in PyTorch 2.0+
            if pytorch_version.startswith("1."):
                logger.warning("   ⚠️  PyTorch 1.x has limited MPS support. Consider upgrading to 2.0+")
        return 'mps'
    
    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        if verbose:
            logger.info(f"✅ CUDA available - using GPU acceleration ({device_name})")
        return 'cuda'
    
    # Fallback to CPU
    if verbose:
        if hasattr(torch.backends, 'mps'):
            logger.info("ℹ️  No GPU available (MPS not available, CUDA not found) - using CPU")
        else:
            logger.info("ℹ️  No GPU available (CUDA not found) - using CPU")
    return 'cpu'


def get_device_info(device: Optional[str] = None, use_cache: bool = True) -> dict:
    """
    Get detailed information about a device.
    
    Args:
        device: Device string ('mps', 'cuda', 'cpu', or None for auto-detect)
        use_cache: If True, use cached device properties (recommended for performance)
        
    Returns:
        Dictionary with device information
    """
    if device is None:
        device = detect_best_device(verbose=False)
    
    # Check cache first
    if use_cache and device in _device_info_cache:
        return _device_info_cache[device]
    
    info = {
        'device': device,
        'type': None,
        'available': False,
        'name': None,
        'properties': {}
    }
    
    if device == 'mps':
        info['type'] = 'GPU'
        info['available'] = torch.backends.mps.is_available()
        info['name'] = 'Apple Metal Performance Shaders'
        info['properties'] = {
            'built': torch.backends.mps.is_built(),
            'platform': 'macOS',
            'pytorch_version': torch.__version__
        }
        # Add MPS memory tracking (limited compared to CUDA, but useful)
        if info['available']:
            try:
                # MPS uses unified memory, so we report system memory as approximate limit
                import psutil
                vm = psutil.virtual_memory()
                info['properties']['system_memory'] = vm.total
                info['properties']['available_memory'] = vm.available
                info['properties']['memory_percent_used'] = vm.percent
            except ImportError:
                # psutil not available, skip memory info
                pass
            
            # Check for known MPS capability milestones
            pytorch_major = int(torch.__version__.split('.')[0])
            pytorch_minor = int(torch.__version__.split('.')[1].split('+')[0])
            # MPS does not support fp16 for audio I/O (soundfile, etc.)
            info['properties']['fp16_support'] = False
            if pytorch_major >= 2:
                info['properties']['stft_support'] = True
            else:
                info['properties']['stft_support'] = False
    elif device == 'cuda':
        info['type'] = 'GPU'
        info['available'] = torch.cuda.is_available()
        if info['available']:
            info['name'] = torch.cuda.get_device_name(0)
            info['properties'] = {
                'compute_capability': torch.cuda.get_device_capability(0),
                'total_memory': torch.cuda.get_device_properties(0).total_memory,
                'multi_processor_count': torch.cuda.get_device_properties(0).multi_processor_count,
            }
    elif device == 'cpu':
        info['type'] = 'CPU'
        info['available'] = True
        info['name'] = 'CPU'
        info['properties'] = {
            'threads': torch.get_num_threads()
        }
    
    # Cache the result for future calls
    if use_cache:
        _device_info_cache[device] = info
    
    return info


def validate_device(device: str, fallback: bool = True) -> str:
    """
    Validate that a requested device is available.
    
    Args:
        device: Requested device ('mps', 'cuda', or 'cpu')
        fallback: If True, fall back to best available device if requested unavailable
        
    Returns:
        Valid device string
        
    Raises:
        RuntimeError: If device unavailable and fallback=False
    """
    device = device.lower()
    
    if device == 'mps':
        if torch.backends.mps.is_available():
            return 'mps'
        elif fallback:
            logger.warning("MPS requested but not available, falling back to best available device")
            return detect_best_device(verbose=True)
        else:
            raise RuntimeError("MPS not available on this system")
    
    elif device == 'cuda':
        if torch.cuda.is_available():
            return 'cuda'
        elif fallback:
            logger.warning("CUDA requested but not available, falling back to best available device")
            return detect_best_device(verbose=True)
        else:
            raise RuntimeError("CUDA not available on this system")
    
    elif device == 'cpu':
        return 'cpu'
    
    else:
        if fallback:
            logger.warning(f"Unknown device '{device}', falling back to best available device")
            return detect_best_device(verbose=True)
        else:
            raise ValueError(f"Unknown device: {device}. Must be 'mps', 'cuda', or 'cpu'")


def print_device_info(device: Optional[str] = None):
    """
    Print human-readable device information.
    
    Args:
        device: Device to query, or None for auto-detect
    """
    info = get_device_info(device)
    
    print(f"\n{'='*60}")
    print(f"Device Information")
    print(f"{'='*60}")
    print(f"Device: {info['device']}")
    print(f"Type: {info['type']}")
    print(f"Available: {'✅' if info['available'] else '❌'}")
    if info['name']:
        print(f"Name: {info['name']}")
    if info['properties']:
        print(f"Properties:")
        for key, value in info['properties'].items():
            print(f"  - {key}: {value}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Test device detection
    print("Testing device detection...")
    best_device = detect_best_device(verbose=True)
    print(f"\nBest device: {best_device}")
    
    # Print detailed info
    print_device_info(best_device)
    
    # Show info for all device types
    for device_type in ['cpu', 'cuda', 'mps']:
        print(f"\n{device_type.upper()} Information:")
        try:
            info = get_device_info(device_type)
            print(f"  Available: {info['available']}")
            if info['name']:
                print(f"  Name: {info['name']}")
        except Exception as e:
            print(f"  Error: {e}")
