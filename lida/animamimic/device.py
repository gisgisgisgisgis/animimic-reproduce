"""Device selection. CUDA is the primary target; MPS and CPU are fallbacks."""
import torch


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def physics_device(device: torch.device) -> torch.device:
    """FEM needs float64 and batched SVD, which MPS does not provide."""
    return torch.device("cpu") if device.type == "mps" else device
