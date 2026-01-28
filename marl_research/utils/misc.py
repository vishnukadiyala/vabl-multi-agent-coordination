"""Miscellaneous utility functions."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    """Get the appropriate torch device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    """Soft update target network parameters."""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def hard_update(target: torch.nn.Module, source: torch.nn.Module) -> None:
    """Hard update target network parameters."""
    target.load_state_dict(source.state_dict())


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_directories(paths: list) -> None:
    """Create directories if they don't exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, window_size: Optional[int] = None):
        self.window_size = window_size
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.values = []

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.values.append(val)

        if self.window_size is not None and len(self.values) > self.window_size:
            self.values = self.values[-self.window_size:]
            self.avg = np.mean(self.values)
        else:
            self.avg = self.sum / self.count


class LinearSchedule:
    """Linear schedule for parameters."""

    def __init__(self, start: float, end: float, steps: int):
        self.start = start
        self.end = end
        self.steps = steps
        self.delta = (end - start) / steps

    def __call__(self, step: int) -> float:
        if step >= self.steps:
            return self.end
        return self.start + self.delta * step
