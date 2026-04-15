"""Deterministic seeding for reproducibility.

Every experiment script must call ``set_global_seed`` before any model
construction, data shuffle, or split call. This is required by the
reproducibility commitments in Section 6.1 and Section 7 of the proposal.

Usage
-----
>>> from egh490.utils.seeding import set_global_seed
>>> set_global_seed(20260413)
"""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and (if installed) PyTorch.

    Parameters
    ----------
    seed
        Integer seed shared across all RNGs.
    deterministic
        When True, also force PyTorch and cuDNN into deterministic mode.
        This trades a small amount of speed for bit-exact reproducibility,
        which is the right default for a research codebase.

    Notes
    -----
    PyTorch is imported lazily so this module remains usable in lightweight
    contexts (e.g. data-only scripts) that do not need torch installed.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Newer PyTorch: opt-in to deterministic algorithms where available.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except (AttributeError, RuntimeError):
                pass
    except ImportError:
        pass