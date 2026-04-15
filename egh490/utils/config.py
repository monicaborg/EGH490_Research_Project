"""Config loader with ``defaults:`` inheritance.

The YAML configs in ``configs/`` use a Hydra-style ``defaults:`` list to
compose a final config from a chain of base files. For example,
``configs/smoke.yaml`` looks like::

    defaults:
      - base
      - models/electra_small

    data:
      max_seq_length: 64

This module resolves that chain into a single flat dict.

Why a custom loader and not Hydra? Hydra is excellent but adds a heavy
dependency and a CLI integration we do not need at this stage. The 60 lines
below cover everything the project actually uses, with no external deps
beyond PyYAML.

Resolution rules
----------------
1. ``defaults:`` entries are loaded in order, each merged into the accumulator.
2. The current file's own keys (excluding ``defaults``) are merged last and
   therefore win on conflict.
3. Merging is recursive for dicts; lists and scalars are replaced wholesale.
4. ``defaults`` entries are interpreted as paths *relative to the configs
   root*, with no extension (``.yaml`` is appended automatically).
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from egh490.utils.io import load_yaml


def load_config(path: str | Path, configs_root: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config and recursively resolve any ``defaults:`` entries.

    Parameters
    ----------
    path
        Path to the leaf config file (e.g. ``configs/smoke.yaml``).
    configs_root
        Root directory under which ``defaults:`` entries are resolved.
        Defaults to the parent directory of ``path``.
    """
    path = Path(path)
    if configs_root is None:
        configs_root = path.parent
    configs_root = Path(configs_root)

    return _load_recursive(path, configs_root, _seen=set())


def _load_recursive(
    path: Path, configs_root: Path, _seen: set[Path]
) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in _seen:
        raise ValueError(f"Circular config inheritance detected at {resolved}")
    _seen = _seen | {resolved}

    raw = load_yaml(path)
    defaults = raw.pop("defaults", []) or []
    if not isinstance(defaults, list):
        raise ValueError(
            f"`defaults:` in {path} must be a list, got {type(defaults).__name__}"
        )

    merged: dict[str, Any] = {}
    for entry in defaults:
        if not isinstance(entry, str):
            raise ValueError(
                f"`defaults:` entries in {path} must be strings, got {entry!r}"
            )
        # Resolve relative to the configs root, not the current file.
        # That way "models/electra_small" works from any depth.
        default_path = (configs_root / f"{entry}.yaml").resolve()
        if not default_path.is_file():
            raise FileNotFoundError(
                f"Default `{entry}` referenced by {path} not found at {default_path}"
            )
        merged = _deep_merge(merged, _load_recursive(default_path, configs_root, _seen))

    # Current file's own keys win.
    merged = _deep_merge(merged, raw)
    return merged


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dicts, with ``override`` taking precedence."""
    out = deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out