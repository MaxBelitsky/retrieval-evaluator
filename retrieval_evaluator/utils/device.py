from dataclasses import dataclass


@dataclass(frozen=True)
class Device:
    """Small device descriptor that does not make PyTorch a core dependency."""

    type: str

    def __str__(self) -> str:
        return self.type


def set_device() -> Device:
    try:
        import torch
    except ImportError:
        return Device("cpu")

    if torch.cuda.is_available():
        return Device("cuda")
    if torch.backends.mps.is_available():
        return Device("mps")
    return Device("cpu")
