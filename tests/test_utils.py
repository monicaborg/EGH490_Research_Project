"""Smoke tests for ``egh490.utils``.

These tests prove the foundation works before any data or model code is
written. They run on a CPU in well under a second and have no external
dependencies beyond what ``pyproject.toml`` already pins.

Run with::

    pytest tests/test_utils.py -v
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import pytest

from egh490.utils.config import load_config
from egh490.utils.io import load_json, load_yaml, save_json, save_yaml
from egh490.utils.logging import get_logger
from egh490.utils.seeding import set_global_seed


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------


def test_set_global_seed_makes_python_random_deterministic() -> None:
    set_global_seed(42)
    first = [random.random() for _ in range(5)]
    set_global_seed(42)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_set_global_seed_rejects_negative() -> None:
    with pytest.raises(ValueError):
        set_global_seed(-1)


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------


def test_yaml_roundtrip(tmp_path: Path) -> None:
    payload = {"a": 1, "b": {"c": [1, 2, 3], "d": "hello"}}
    target = tmp_path / "nested" / "out.yaml"
    save_yaml(payload, target)
    assert target.is_file()
    assert load_yaml(target) == payload


def test_json_roundtrip_is_sorted_and_indented(tmp_path: Path) -> None:
    payload = {"z": 1, "a": 2}
    target = tmp_path / "out.json"
    save_json(payload, target)
    text = target.read_text(encoding="utf-8")
    # Sorted keys means "a" appears before "z" in the file.
    assert text.index('"a"') < text.index('"z"')
    # Indented means there is a newline after the opening brace.
    assert "{\n" in text
    assert load_json(target) == payload


def test_load_yaml_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "does_not_exist.yaml")


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------


def test_get_logger_returns_logger_with_name() -> None:
    logger = get_logger("egh490.test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "egh490.test"


def test_get_logger_accepts_string_level() -> None:
    logger = get_logger("egh490.test_string_level", level="DEBUG")
    assert logger.level == logging.DEBUG


# ---------------------------------------------------------------------------
# config loader
# ---------------------------------------------------------------------------


def test_config_loads_single_file_with_no_defaults(tmp_path: Path) -> None:
    cfg_path = tmp_path / "leaf.yaml"
    save_yaml({"foo": 1, "bar": {"baz": 2}}, cfg_path)
    assert load_config(cfg_path) == {"foo": 1, "bar": {"baz": 2}}


def test_config_resolves_defaults_chain(tmp_path: Path) -> None:
    # Mimic the real layout: a base file plus a leaf that inherits from it.
    save_yaml({"a": 1, "shared": {"x": 10, "y": 20}}, tmp_path / "base.yaml")
    save_yaml(
        {"defaults": ["base"], "b": 2, "shared": {"y": 99, "z": 30}},
        tmp_path / "leaf.yaml",
    )

    cfg = load_config(tmp_path / "leaf.yaml")

    # Keys from base are kept, leaf adds new keys, leaf overrides on conflict,
    # nested dicts are merged recursively.
    assert cfg == {"a": 1, "b": 2, "shared": {"x": 10, "y": 99, "z": 30}}


def test_config_detects_circular_inheritance(tmp_path: Path) -> None:
    save_yaml({"defaults": ["b"], "x": 1}, tmp_path / "a.yaml")
    save_yaml({"defaults": ["a"], "y": 2}, tmp_path / "b.yaml")
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "a.yaml")


def test_config_missing_default_raises(tmp_path: Path) -> None:
    save_yaml({"defaults": ["nope"]}, tmp_path / "leaf.yaml")
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "leaf.yaml")