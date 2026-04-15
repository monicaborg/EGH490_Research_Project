"""Project-wide logger factory.

Use ``get_logger(__name__)`` at the top of every module instead of
``logging.getLogger(__name__)`` directly. This guarantees consistent
formatting and a single point to change verbosity later (e.g. switching
to JSON logs for HPC runs).
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _configure_root(level: int = logging.INFO) -> None:
    """Configure the root logger exactly once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet noisy third-party loggers
    for noisy in ("urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str, level: int | str = logging.INFO) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name
        Conventionally ``__name__``.
    level
        Either a logging level constant (``logging.DEBUG``) or its string name
        (``"DEBUG"``). Applied to the returned logger only, not the root.
    """
    _configure_root()
    logger = logging.getLogger(name)
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    logger.setLevel(level)
    return logger