"""
Utility helpers to load MDX23C-style checkpoints and run inference.

This supports two formats:
1. Legacy ConvTDFNet checkpoints with 'hyper_parameters' dict
2. Modern TFC_TDF_v3 checkpoints with separate YAML config files

The file also provides simple inference helpers that support:
- PyTorch Modules (torch.nn.Module)
- ONNX runtimes (onnxruntime.InferenceSession)
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, Callable
import logging
import warnings

import numpy as np
import torch
import yaml

# Suppress PyTorch STFT deprecation warning from lib_v5
# The library uses return_complex=False which is deprecated but still functional
# This will need updating when PyTorch removes support (not urgent)
warnings.filterwarnings('ignore', message='.*stft with return_complex=False is deprecated.*')

# Support both old and new architectures
try:
    import lib_v5.mdxnet as MdxnetSet  # type: ignore
except ImportError:
    MdxnetSet = None  # type: ignore
    
import lib_v5.tfc_tdf_v3 as TFC_TDF_v3  # type: ignore

try:
    import onnxruntime as ort  # optional dependency
except Exception:  # pragma: no cover - optional
    ort = None  # type: ignore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


PathLike = Union[str, Path]


class MDXLoadError(RuntimeError):
    pass


def is_mdx_checkpoint(path: PathLike) -> bool:
    """
    Quick heuristic whether the file is an MDX checkpoint used by this project.
    Returns True if the torch.load(...) object is a valid checkpoint dict.
    """
    p = Path(path)
    if not p.exists():
        return False
    try:
        ckpt = torch.load(str(p), map_location=lambda storage, loc: storage)
    except Exception:
        return False
    # Either has hyper_parameters (old format) or is a state dict (new format)
    return isinstance(ckpt, dict)


def load_mdx23c_checkpoint(
    checkpoint_path: PathLike,
    config_path: Optional[PathLike] = None,
    device: Union[str, torch.device] = "cpu",
) -> torch.nn.Module:
    """
    Load an MDX23C checkpoint and return a ready-to-run torch.nn.Module on `device`.

    Supports two formats:
    1. Legacy format with 'hyper_parameters' in checkpoint (uses ConvTDFNet)
    2. Modern format with separate YAML config (uses TFC_TDF_v3.BSRoformer)
    
    Args:
        checkpoint_path: Path to the .ckpt file
        config_path: Optional path to YAML config. If None, will search for:
                    - config_mdx23c.yaml in same directory as checkpoint
                    - config.yaml in same directory as checkpoint
        device: Device to load model on
    
    Returns:
      torch.nn.Module (moved to device and in eval() mode)

    Raises:
      MDXLoadError if checkpoint doesn't match the expected MDX format or loading fails.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise MDXLoadError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading checkpoint metadata from %s", ckpt_path)
    try:
        ckpt = torch.load(str(ckpt_path), map_location=lambda storage, loc: storage, weights_only=False)
    except Exception as e:
        raise MDXLoadError(f"Failed to torch.load checkpoint: {e}")

    if not isinstance(ckpt, dict):
        raise MDXLoadError("Checkpoint is not a dict")

    # Try legacy format first
    if 'hyper_parameters' in ckpt:
        logger.info("Detected legacy format with hyper_parameters")
        return _load_legacy_checkpoint(ckpt, ckpt_path, device)
    
    # Modern format - need config file
    logger.info("Detected modern format, loading with config file")
    if config_path is None:
        # Search for config in same directory
        config_candidates = [
            ckpt_path.parent / "config_mdx23c.yaml",
            ckpt_path.parent / "config.yaml",
        ]
        for candidate in config_candidates:
            if candidate.exists():
                config_path = candidate
                logger.info(f"Found config at {config_path}")
                break
        
        if config_path is None:
            raise MDXLoadError(
                f"No config_path provided and could not find config_mdx23c.yaml or config.yaml "
                f"in {ckpt_path.parent}"
            )
    
    return _load_modern_checkpoint(ckpt, config_path, device)


def _load_legacy_checkpoint(
    ckpt: Dict[str, Any],
    ckpt_path: Path,
    device: Union[str, torch.device],
) -> torch.nn.Module:
    """Load legacy ConvTDFNet checkpoint with hyper_parameters."""
    if MdxnetSet is None:
        raise MDXLoadError(
            "Legacy checkpoint format requires lib_v5.mdxnet (with pytorch_lightning), "
            "but it could not be imported."
        )
    
    model_params: Dict[str, Any] = ckpt['hyper_parameters']
    logger.debug("Model hyper_parameters keys: %s", list(model_params.keys()))

    try:
        logger.info("Instantiating ConvTDFNet with hyperparameters.")
        model_cls = getattr(MdxnetSet, "ConvTDFNet")
    except Exception as e:
        raise MDXLoadError(f"Could not find ConvTDFNet in lib_v5.mdxnet: {e}")

    loader = getattr(model_cls, "load_from_checkpoint", None)
    try:
        if callable(loader):
            logger.info("Loading model with load_from_checkpoint(...)")
            model = loader(str(ckpt_path))
        else:
            logger.info("Building instance and loading state dict.")
            model = model_cls(**model_params)
            state_dict = ckpt.get("state_dict") or ckpt.get("model") or ckpt.get("model_state")
            if state_dict is None:
                raise MDXLoadError("Checkpoint lacks 'state_dict'")
            model.load_state_dict(state_dict)
    except Exception as exc:
        raise MDXLoadError(f"Failed to instantiate/load model from checkpoint: {exc}")

    model.to(device)
    model.eval()
    return model


def _load_modern_checkpoint(
    ckpt: Dict[str, Any],
    config_path: PathLike,
    device: Union[str, torch.device],
) -> torch.nn.Module:
    """Load modern TFC_TDF_v3 checkpoint with separate config file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise MDXLoadError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    except Exception as e:
        raise MDXLoadError(f"Failed to load config YAML: {e}")
    
    # Convert dict to object for attribute access
    class Config:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, Config(v))
                else:
                    setattr(self, k, v)
    
    config = Config(config_dict)
    
    try:
        logger.info("Instantiating TFC_TDF_net with config.")
        model = TFC_TDF_v3.TFC_TDF_net(config, device)
        
        # Load state dict
        logger.info("Loading state dict into model.")
        model.load_state_dict(ckpt, strict=False)
        
        model.to(device)
        model.eval()
        return model
    except Exception as exc:
        raise MDXLoadError(f"Failed to instantiate/load TFC_TDF_v3 model: {exc}")


def prepare_model_runner(
    model_path: PathLike,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[Any, str]:
    """
    Convenience: return (model_or_session, type) where type is one of:
     - "mdx_ckpt" : a torch.nn.Module loaded from MDX checkpoint (use load_mdx23c_checkpoint)
     - "onnx"     : an onnxruntime.InferenceSession
     - "callable" : a generic callable (if the path points to a Python module / function — NOT implemented here)

    This mirrors how the UI repo supports multiple backends (MDX ckpt, ONNX, or converted models).
    """
    p = Path(model_path)
    if not p.exists():
        raise MDXLoadError(f"Model path does not exist: {p}")

    suffix = p.suffix.lower()
    # Prefer torch checkpoint inspection because suffixes vary (.ckpt, .pth, .pt, .th)
    try:
        if is_mdx_checkpoint(p):
            model = load_mdx23c_checkpoint(p, device=device)
            return model, "mdx_ckpt"
    except MDXLoadError:
        logger.debug("Not an MDX-style checkpoint or failed to load as such.")

    # ONNX fallback
    if suffix == ".onnx" or (ort is not None and suffix in {".onnx"}):
        if ort is None:
            raise MDXLoadError("onnxruntime is not installed but model_path points to an ONNX file.")
        try:
            sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"] if str(device) == "cpu" else None)
            return sess, "onnx"
        except Exception as e:
            raise MDXLoadError(f"Failed to create ONNX InferenceSession: {e}")

    # Unknown format
    raise MDXLoadError(
        "Unsupported or unknown model format. This utility expects either an MDX checkpoint "
        "(contains 'hyper_parameters') or an ONNX file. Other formats require custom handling."
    )


def run_model_inference(
    model_or_session: Any,
    input_tensor: torch.Tensor,
    device: Union[str, torch.device] = "cpu",
) -> np.ndarray:
    """
    Run inference on an input tensor and return numpy array output.

    - For torch.nn.Module: performs `model(input_tensor.to(device))` under torch.no_grad().
      Expects input_tensor shaped like the repo uses (batch, channels, length) as torch.float32.

    - For onnxruntime.InferenceSession: converts to numpy and runs session.run(...).
      The input name is attempted as 'input' (matching the repo pattern). If that fails,
      the function will attempt to read ort_session.get_inputs()[0].name to map the input.

    Returns:
      numpy.ndarray (CPU numpy array)
    """
    # Torch module path
    if isinstance(model_or_session, torch.nn.Module):
        model = model_or_session
        model.to(device)
        model.eval()
        with torch.no_grad():
            inp = input_tensor.to(device)
            out = model(inp)
            if isinstance(out, (list, tuple)):
                # pick first output by default; adjust if your model returns other shapes
                out = out[0]
            return out.detach().cpu().numpy()

    # ONNX path
    if ort is not None and isinstance(model_or_session, ort.InferenceSession):
        sess: ort.InferenceSession = model_or_session
        arr = input_tensor.cpu().numpy()
        try:
            return sess.run(None, {"input": arr})[0]
        except Exception:
            # try using the engine input's actual name
            input_name = sess.get_inputs()[0].name
            return sess.run(None, {input_name: arr})[0]

    # Callable path (if you passed a numpy-returning callable)
    if callable(model_or_session):
        out = model_or_session(input_tensor)
        if isinstance(out, torch.Tensor):
            return out.detach().cpu().numpy()
        return np.asarray(out)

    raise MDXLoadError("Unsupported model/session type for inference.")


def get_checkpoint_hyperparameters(
    checkpoint_path: PathLike,
    config_path: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """
    Return the hyperparameters dict for a checkpoint.
    
    For legacy checkpoints, returns the 'hyper_parameters' dict.
    For modern checkpoints, loads and returns the config YAML as a dict.
    
    Useful when you need to inspect dim_c, hop_length, etc. before building other parts
    of the pipeline (e.g., to compute chunk sizes).
    """
    p = Path(checkpoint_path)
    if not p.exists():
        raise MDXLoadError(f"Checkpoint path not found: {p}")
    try:
        ckpt = torch.load(str(p), map_location=lambda storage, loc: storage, weights_only=False)
    except Exception as e:
        raise MDXLoadError(f"Failed to load checkpoint: {e}")
    
    if not isinstance(ckpt, dict):
        raise MDXLoadError("Checkpoint is not a dict")
    
    # Legacy format
    if 'hyper_parameters' in ckpt:
        return dict(ckpt['hyper_parameters'])
    
    # Modern format - load config
    if config_path is None:
        config_candidates = [
            p.parent / "config_mdx23c.yaml",
            p.parent / "config.yaml",
        ]
        for candidate in config_candidates:
            if candidate.exists():
                config_path = candidate
                break
        
        if config_path is None:
            raise MDXLoadError(
                f"No config file found in {p.parent} for modern checkpoint format"
            )
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise MDXLoadError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)