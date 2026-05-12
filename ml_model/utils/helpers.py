"""
General Utility Functions
==========================

Small, reusable helpers used across the ml_model package:
seed setting, device detection, parameter counting, and time formatting.
"""

from __future__ import annotations

import os
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility across all libraries.

    Configures Python's built-in random, NumPy, and PyTorch (CPU and
    CUDA) random number generators.  Also sets the ``PYTHONHASHSEED``
    environment variable and enables deterministic CuDNN behaviour.

    Args:
        seed: Integer seed value.

    Note:
        Deterministic mode may reduce performance due to disabling
        CuDNN auto-tuning and non-deterministic algorithms.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Detect and return the best available compute device.

    Checks for CUDA GPUs first, then Apple MPS, and falls back to CPU.

    Returns:
        ``torch.device`` for the best available backend.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model.

    Args:
        model: PyTorch module.
        trainable_only: If ``True``, count only parameters with
            ``requires_grad=True``.

    Returns:
        Total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_parameters(count: int) -> str:
    """Format a parameter count as a human-readable string.

    Args:
        count: Number of parameters.

    Returns:
        Formatted string like ``"109.5M"`` or ``"1.2B"``.
    """
    if count >= 1e9:
        return f"{count / 1e9:.1f}B"
    if count >= 1e6:
        return f"{count / 1e6:.1f}M"
    if count >= 1e3:
        return f"{count / 1e3:.1f}K"
    return str(count)


def format_elapsed(seconds: float) -> str:
    """Format elapsed time as a human-readable string.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        Formatted string like ``"2h 15m 30s"`` or ``"45.3s"``.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


class Timer:
    """Context manager for timing code blocks.

    Usage::

        with Timer("data loading"):
            data = load_data()
        # Prints: "data loading completed in 2.3s"

    Args:
        name: Description of the timed operation.
        logger: Optional logger instance.  If ``None``, prints to stdout.
    """

    def __init__(self, name: str = "operation", logger=None) -> None:
        """Store the operation name and logger."""
        self.name = name
        self.logger = logger
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        """Record the start time."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        """Compute elapsed time and log it."""
        self.elapsed = time.perf_counter() - self.start_time
        msg = f"{self.name} completed in {format_elapsed(self.elapsed)}"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
