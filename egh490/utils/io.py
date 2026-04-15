"""Lightweight I/O helpers for YAML and JSON.

Centralising load/save means the rest of the codebase never imports yaml
or json directly. Two benefits:

1. Path objects, not strings, used everywhere — fewer cross-platform bugs.
2. JSON output is always sorted and indented, so diffs in version control
   stay readable when configs or manifests change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict. Raises on missing file or parse error."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at {path}, got {type(data).__name__}")
    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Save a dict as YAML, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)


def load_json(path: str | Path) -> Any:
    """Load a JSON file. Returns whatever the file contains (dict, list, etc.)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(data: Any, path: str | Path, *, indent: int = 2) -> None:
    """Save data as pretty-printed, sorted JSON for diff-friendly output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, sort_keys=True, ensure_ascii=False)
        fh.write("\n")