"""Device utilities for PyTorch."""

import torch


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        PyTorch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def should_use_pin_memory(device: torch.device, config_pin_memory: bool) -> bool:
    """Determine if pin_memory should be used (MPS doesn't support it).
    
    Args:
        device: PyTorch device.
        config_pin_memory: Pin memory setting from config.
        
    Returns:
        True if pin_memory should be used, False otherwise.
    """
    return config_pin_memory and device.type != "mps"
