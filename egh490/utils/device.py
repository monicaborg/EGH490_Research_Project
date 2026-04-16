"""Device selection for PyTorch.

Automatically picks the best available device across environments:

- CUDA (NVIDIA GPUs) when available — the HPC and school desktop fallbacks.
- MPS (Apple Silicon) when available — for local development on the MacBook.
- CPU everywhere else — for the smoke-test environment and CI.

This is a thin convenience wrapper: calling code never hardcodes a device
string, so the same script runs unchanged across all four target environments
identified in the proposal risk register.
"""

from __future__ import annotations


def get_device(prefer: str | None = None) -> str:
    """Return the best available PyTorch device string.

    Parameters
    ----------
    prefer
        If given, try this device first (``"cuda"``, ``"mps"``, or ``"cpu"``).
        Falls back to auto-detection if the preferred device isn't available.
        Useful for forcing CPU-only smoke runs even on a GPU machine.

    Returns
    -------
    str
        One of ``"cuda"``, ``"mps"``, or ``"cpu"``.

    Notes
    -----
    PyTorch is imported lazily so this module can be imported in contexts
    where torch isn't installed (e.g. pure data-preparation scripts).
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    # Honour the user's preference if that device is actually available.
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    if prefer == "mps" and _mps_available(torch):
        return "mps"
    if prefer == "cpu":
        return "cpu"

    # Auto-detect — fastest available first.
    if torch.cuda.is_available():
        return "cuda"
    if _mps_available(torch):
        return "mps"
    return "cpu"


def _mps_available(torch) -> bool:  # type: ignore[no-untyped-def]
    """Check Apple Silicon Metal Performance Shaders availability."""
    # torch.backends.mps only exists on PyTorch 1.12+. The getattr guard
    # keeps this safe on older versions and non-Mac machines.
    mps = getattr(torch.backends, "mps", None)
    if mps is None:
        return False
    return bool(mps.is_available() and mps.is_built())